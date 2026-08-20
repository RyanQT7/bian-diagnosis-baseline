#!/usr/bin/env python3
"""BiAn two-stage diagnosis with case-native endpoint identifiers."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from llm_backend import Local32BBackend, numeric_leaves, status_number

METRICS = ("bias", "rxpower", "txpower", "media_snr", "host_snr", "serdes_snr")
STATUS_FIELDS = ("RxLOL", "TxLOL", "TxLOS", "RxLOS")
ENDPOINT_KEY_FIELDS = METRICS + STATUS_FIELDS + ("vendor", "vendor_sn", "Temperature", "Voltage")
SOP_SOFT_PRIOR = """WEAK EXPERT BACKGROUND (optional, possibly inaccurate):
- host_snr abnormalities may provide evidence about the endpoint itself.
- serdes_snr abnormalities may provide evidence about endpoint-side impairment.
- media_snr abnormalities may reflect endpoint or transmission-path degradation.
- rxpower abnormalities may originate from transmitter, receiver, optical module, lane, or fiber/path impairment.
- txpower abnormalities may provide evidence about the transmitting endpoint.
- combinations of metrics can be more informative than any single metric, but interpretation must come from the complete case.
This weak prior may be incomplete, dataset-specific, or inaccurate. Never apply a fixed same/opposite mapping, threshold rule, or implicit decision tree."""


def _prediction_cases(data_root: Path) -> list[dict[str, Any]]:
    roots = [data_root / "data1", data_root / "data2"]
    if not any(root.is_dir() for root in roots):
        roots = [data_root]
    paths = sorted(path for root in roots for path in root.glob("*/*.json"))
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"case is not an object: {path}")
        case_id = path.stem
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        # The only answer-bearing field is removed before any inference work.
        observable = {key: value for key, value in raw.items() if key != "label"}
        cases.append({"case_id": case_id, "source_path": path, "data": observable})
        seen.add(case_id)
    if not cases:
        raise ValueError(f"no cases below {data_root}")
    return sorted(cases, key=lambda item: item["case_id"])


def endpoint_keys(data: dict[str, Any]) -> tuple[str, ...]:
    mapping = data.get("link_side_ip_interface_map")
    if not isinstance(mapping, dict) or len(mapping) < 2:
        raise ValueError("link_side_ip_interface_map must define at least two endpoints")
    endpoints = tuple(str(key) for key in mapping)
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("duplicate endpoint identifier")
    return endpoints


def _lane_series(raw: Any) -> dict[str, list[float]]:
    if isinstance(raw, dict):
        lanes = {str(key): numeric_leaves(value) for key, value in raw.items()}
        if any(lanes.values()):
            return lanes
    values = numeric_leaves(raw)
    return {"aggregate": values} if values else {}


def _metric_summary(raw: Any, metric: str) -> dict[str, Any]:
    lanes = _lane_series(raw)
    lane_means = {lane: float(np.mean(values)) for lane, values in lanes.items() if values}
    values = [value for lane in lanes.values() for value in lane]
    if not values:
        return {"available": False}
    array = np.asarray(values, dtype=float)
    deltas = [lane[-1] - lane[0] for lane in lanes.values() if len(lane) > 1]
    lane_diff = max(lane_means.values()) - min(lane_means.values()) if len(lane_means) > 1 else 0.0
    result: dict[str, Any] = {
        "available": True,
        "lane_values_or_means": {key: round(value, 6) for key, value in lane_means.items()},
        "mean": round(float(np.mean(array)), 6), "median": round(float(np.median(array)), 6),
        "std": round(float(np.std(array)), 6), "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6), "lane_difference": round(float(lane_diff), 6),
        "sample_count": int(len(array)),
        "temporal_mean_delta": round(float(np.mean(deltas)), 6) if deltas else None,
        "temporal_max_abs_delta": round(float(max(map(abs, deltas))), 6) if deltas else None,
    }
    return result


def _status_summary(raw: Any) -> dict[str, Any]:
    items = list(raw.values()) if isinstance(raw, dict) else [raw]
    parsed = [status_number(value) for value in items]
    valid = [value for value in parsed if value is not None]
    return {"available": bool(valid), "abnormal_fraction": round(float(np.mean(valid)), 6) if valid else None}


def summarize_case(data: dict[str, Any]) -> dict[str, Any]:
    endpoints = endpoint_keys(data)
    mapping = data["link_side_ip_interface_map"]
    alarm_interface = data.get("alarm_ip_interface")
    alarm_endpoint = next((endpoint for endpoint in endpoints if mapping.get(endpoint) == alarm_interface), None)
    metadata = {
        endpoint: {
            "interface": mapping.get(endpoint),
            "interface_metadata_does_not_define_label": True,
            "vendor": data.get("vendor", {}).get(endpoint) if isinstance(data.get("vendor"), dict) else None,
            "vendor_sn": data.get("vendor_sn", {}).get(endpoint) if isinstance(data.get("vendor_sn"), dict) else None,
            "temperature": _metric_summary(data.get("Temperature", {}).get(endpoint) if isinstance(data.get("Temperature"), dict) else None, "Temperature"),
            "voltage": _metric_summary(data.get("Voltage", {}).get(endpoint) if isinstance(data.get("Voltage"), dict) else None, "Voltage"),
        }
        for endpoint in endpoints
    }
    evidence: dict[str, Any] = {}
    for endpoint in endpoints:
        evidence[endpoint] = {
            metric: _metric_summary(data.get(metric, {}).get(endpoint) if isinstance(data.get(metric), dict) else None, metric)
            for metric in METRICS
        }
        evidence[endpoint]["status"] = {
            field: _status_summary(data.get(field, {}).get(endpoint) if isinstance(data.get(field), dict) else None)
            for field in STATUS_FIELDS
        }
    transmission: dict[str, Any] = {}
    raw_transmission = data.get("transmission")
    for source in endpoints:
        for target in endpoints:
            if source != target:
                direction = f"{source}-{target}"
                transmission[direction] = _metric_summary(
                    raw_transmission.get(direction) if isinstance(raw_transmission, dict) else None,
                    "transmission",
                )
    coverage = {}
    for field in ENDPOINT_KEY_FIELDS:
        raw = data.get(field)
        coverage[field] = {endpoint: bool(isinstance(raw, dict) and endpoint in raw) for endpoint in endpoints}
    cross = {}
    for left_index, left in enumerate(endpoints):
        for right in endpoints[left_index + 1:]:
            for metric in METRICS:
                a = evidence[left][metric].get("mean"); b = evidence[right][metric].get("mean")
                cross[f"{metric}.{left}_minus_{right}_mean"] = round(float(a - b), 6) if a is not None and b is not None else None
    return {
        "endpoints": list(endpoints),
        "candidate_diagnoses": [*endpoints, "fiber"],
        "endpoint_metadata": metadata,
        "alarm_observation": {"alarm_name": data.get("alarm_name"), "alarm_time": data.get("alarm_time"), "alarm_endpoint": alarm_endpoint, "note": "alarm endpoint is evidence, not an automatic root cause"},
        "endpoint_field_coverage": coverage,
        "endpoint_evidence": evidence,
        "transmission_by_native_direction": transmission,
        "cross_endpoint_comparison": cross,
        "summary_contract": "objective values and statistics only; no SOP threshold flags or rule-derived anomaly labels",
    }


def _stage1_schema(endpoints: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "endpoint_facts": {"type": "object", "properties": {endpoint: {"type": "string"} for endpoint in endpoints}, "required": list(endpoints), "additionalProperties": False},
            "lane_facts": {"type": "string"},
            "transmission_facts": {"type": "string"},
            "status_and_metadata_facts": {"type": "string"},
            "cross_endpoint_facts": {"type": "string"},
            "data_limitations": {"type": "string"},
            "possible_interpretations": {"type": "string"},
        },
        "required": ["endpoint_facts", "lane_facts", "transmission_facts", "status_and_metadata_facts", "cross_endpoint_facts", "data_limitations", "possible_interpretations"],
        "additionalProperties": False,
    }


def _stage2_schema(endpoints: tuple[str, ...]) -> dict[str, Any]:
    candidates = [*endpoints, "fiber"]
    assessment = {
        "type": "object",
        "properties": {
            "supporting": {"type": "string"},
            "contradictory": {"type": "string"},
            "unexplained": {"type": "string"},
        },
        "required": ["supporting", "contradictory", "unexplained"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "hypothesis_assessments": {"type": "object", "properties": {candidate: assessment for candidate in candidates}, "required": candidates, "additionalProperties": False},
            "ignored_expert_prior": {"type": "string"},
            "diagnosis_root_cause": {"type": "string", "enum": candidates},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["hypothesis_assessments", "ignored_expert_prior", "diagnosis_root_cause", "confidence", "reasoning"],
        "additionalProperties": False,
    }


def _stage1_prompt(summary: dict[str, Any]) -> str:
    candidates = ", ".join(summary["candidate_diagnoses"])
    return (
        "BiAn Stage 1: extract physical evidence using the native endpoint identifiers exactly as provided. "
        f"Candidate diagnoses for THIS case: {candidates}. Do not rename endpoints as local, remote, side_a, or side_b. "
        "Report objective values, distributions, lane spreads, real temporal changes, native transmission directions, statuses, metadata, missing data, and cross-endpoint relationships. Null or unavailable is not abnormal; do not claim temporal behavior without a series. "
        "Do not select a diagnosis. Keep possible interpretations brief and evidence-led. The alarm endpoint is only an observation and interface rate never defines a label. "
        "The expert SOP below is only a weak engineering reference. It may be incomplete, dataset-specific, or inaccurate for some cases. Do NOT mechanically follow it or let it override stronger case evidence. If observations contradict it, trust observations. "
        "Use one short sentence per field and return only requested JSON.\n" + SOP_SOFT_PRIOR + "\nOBSERVABLE_SUMMARY:\n" + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def _stage2_prompt(summary: dict[str, Any], stage1: dict[str, Any]) -> str:
    candidates = ", ".join(summary["candidate_diagnoses"])
    return (
        "BiAn Stage 2: diagnose one root cause using native endpoint identity. "
        f"Candidate diagnoses for THIS case: {candidates}. Output exactly one listed candidate; never output another endpoint. "
        "Evaluate every listed hypothesis equally. For each endpoint and fiber, state supporting, contradictory, and unexplained observations. Select the hypothesis explaining the most key observations with the fewest contradictions. "
        "Fiber is an independent physical hypothesis grounded primarily in native bidirectional transmission, Tx-to-corresponding-Rx consistency, path-sensitive SNR/power/LOS/LOL evidence, lane patterns, and whether one endpoint alone explains the case. No metric combination or bilateral abnormality automatically means fiber. "
        "Raw observations have highest priority, cross-metric/cross-endpoint causal consistency second, and expert background last. The expert SOP below is only a weak engineering reference; it may be incomplete, dataset-specific, or inaccurate. Do NOT mechanically follow it. Do NOT let it override stronger case evidence. If evidence contradicts it, explicitly ignore it. You are not reproducing an expert decision tree. "
        "Do not infer a label from interface rate or assume the alarm endpoint is causal. Stage 1 made no diagnosis and is not an anchor. Keep every hypothesis field under 16 words and reasoning under 35 words. Return JSON only.\n"
        + SOP_SOFT_PRIOR + "\nSTAGE1_EVIDENCE:\n" + json.dumps(stage1, ensure_ascii=False, separators=(",", ":"))
        + "\nOBSERVABLE_SUMMARY:\n" + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def _valid_stage1(value: Any, endpoints: tuple[str, ...]) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    return {
        "endpoints": list(endpoints),
        "endpoint_facts": value.get("endpoint_facts", {}),
        "lane_facts": str(value.get("lane_facts", "unavailable")),
        "transmission_facts": str(value.get("transmission_facts", "unavailable")),
        "status_and_metadata_facts": str(value.get("status_and_metadata_facts", "unavailable")),
        "cross_endpoint_facts": str(value.get("cross_endpoint_facts", "unavailable")),
        "data_limitations": str(value.get("data_limitations", "unavailable")),
        "possible_interpretations": str(value.get("possible_interpretations", "unavailable")),
    }


def _valid_stage2(value: Any, candidates: list[str]) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    diagnosis = value.get("diagnosis_root_cause")
    if diagnosis not in candidates:
        diagnosis = candidates[0]
    confidence = value.get("confidence", 0.0)
    return {
        "diagnosis_root_cause": diagnosis,
        "hypothesis_assessments": value.get("hypothesis_assessments", {}),
        "ignored_expert_prior": str(value.get("ignored_expert_prior", "none stated")),
        "confidence": float(confidence) if isinstance(confidence, (int, float)) and math.isfinite(float(confidence)) else 0.0,
        "reasoning": str(value.get("reasoning", "format fallback")),
    }


def _truth_after_predictions(cases: list[dict[str, Any]]) -> dict[str, str]:
    truth: dict[str, str] = {}
    for case in cases:
        raw = json.loads(case["source_path"].read_text(encoding="utf-8"))
        label = raw.get("label")
        candidates = [*endpoint_keys(case["data"]), "fiber"]
        if label not in candidates:
            raise ValueError(f"JSON label outside native candidates for {case['case_id']}: {label}")
        truth[case["case_id"]] = str(label)
    return truth


def _score_labels(frame: pd.DataFrame) -> list[str]:
    endpoints = sorted((set(frame["true_label"]) | set(frame["diagnosis_root_cause"])) - {"fiber"})
    return endpoints + ["fiber"]


def score_frame_v3(frame: pd.DataFrame) -> str:
    labels = _score_labels(frame)
    y_true = frame["true_label"].tolist(); y_pred = frame["diagnosis_root_cause"].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels, digits=6, zero_division=0)
    lines = [
        "BiAn case-native endpoint scores",
        f"Accuracy: {accuracy_score(y_true, y_pred):.6f}",
        f"Macro Precision: {precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        f"Macro Recall: {recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        f"Macro F1: {f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        "", report.rstrip(), "", f"Confusion Matrix (rows=true, columns=predicted; order={','.join(labels)}):",
        "              " + "  ".join(f"{label:>6}" for label in labels),
    ]
    lines.extend(f"{label:>12}  " + "  ".join(f"{int(value):>6}" for value in row) for label, row in zip(labels, cm))
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    cases = _prediction_cases(args.data_root)
    summaries = [summarize_case(case["data"]) for case in cases]
    endpoints = [endpoint_keys(case["data"]) for case in cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "cache"
    os.environ["VLLM_CACHE_ROOT"] = str(cache / "vllm")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    backend = Local32BBackend(args.model_path, args.gpu_ids, args.seed, args.max_input_tokens, args.max_output_tokens, args.max_num_seqs)
    if args.smoke_test:
        selected: list[int] = []
        seen_signatures: set[tuple[str, ...]] = set()
        for index, signature in enumerate(endpoints):
            if signature not in seen_signatures:
                selected.append(index); seen_signatures.add(signature)
            if len(selected) >= args.smoke_cases:
                break
        selected.extend(index for index in range(len(cases)) if index not in selected and len(selected) < args.smoke_cases)
    else:
        selected = list(range(len(cases)))
    active_summaries = [summaries[index] for index in selected]
    active_endpoints = [endpoints[index] for index in selected]
    stage1_raw = backend.generate_json([{"prompt": _stage1_prompt(summary), "schema": _stage1_schema(eps)} for summary, eps in zip(active_summaries, active_endpoints)])
    stage1 = [_valid_stage1(item["value"], eps) for item, eps in zip(stage1_raw, active_endpoints)]
    stage2_raw = backend.generate_json([{"prompt": _stage2_prompt(summary, first), "schema": _stage2_schema(eps)} for summary, first, eps in zip(active_summaries, stage1, active_endpoints)])
    stage2 = [_valid_stage2(item["value"], [*eps, "fiber"]) for item, eps in zip(stage2_raw, active_endpoints)]
    if args.smoke_test:
        payload = [{"case_id": cases[index]["case_id"], "endpoints": endpoints[index], "stage1": first, "stage2": second} for index, first, second in zip(selected, stage1, stage2)]
        (args.output_dir / "smoke_test.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"native-endpoint smoke passed: {len(selected)} cases")
        return 0
    predictions = {case["case_id"]: result["diagnosis_root_cause"] for case, result in zip(cases, stage2)}
    for case, eps in zip(cases, endpoints):
        if predictions[case["case_id"]] not in {*eps, "fiber"}:
            raise AssertionError("prediction outside case-native candidates")
    truth = _truth_after_predictions(cases)
    frame = pd.DataFrame({
        "case_id": [case["case_id"] for case in cases],
        "diagnosis_root_cause": [predictions[case["case_id"]] for case in cases],
        "true_label": [truth[case["case_id"]] for case in cases],
    })
    frame.to_csv(args.output_dir / "bian_results.csv", index=False)
    (args.output_dir / "scores.txt").write_text(score_frame_v3(frame), encoding="utf-8")
    print((args.output_dir / "scores.txt").read_text(encoding="utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-cases", type=int, default=2)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
