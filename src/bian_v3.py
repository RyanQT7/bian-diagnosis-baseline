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
REFERENCE_THRESHOLDS = {
    "rxpower": {"lane_down": -40.0, "low": -2.5, "high": 4.6, "lane_diff": 1.0},
    "txpower": {"lane_down": -40.0, "low": -2.5, "high": 2.5, "lane_diff": 1.3},
    "host_snr": {"lane_down": 0.0, "low": 22.8, "high": 27.5, "lane_diff": 2.5},
    "media_snr": {"lane_down": 0.0, "low": 22.4, "high": 28.7, "lane_diff": 3.0},
    "serdes_snr": {"lane_down": 0.0, "low": 458750.0, "high": 947750.0, "lane_diff": 230000.0},
}

SOP_SOFT_PRIOR = """EXPERT SOP SOFT PRIORS (not mandatory rules):
- host_snr and serdes_snr abnormalities usually support the same endpoint.
- media_snr and rxpower abnormalities usually support the opposite endpoint.
- txpower abnormality, especially lane-down/extreme loss, strongly supports the same endpoint.
- combined serdes_snr + media_snr + rxpower anomalies require severity, lanes, time, and other metrics; when coherent they can strongly support the opposite endpoint.
- strong bilateral, similarly severe, directionally conflicting evidence increases fiber evidence; bilateral asymmetry favors the stronger coherent endpoint.
Apply same/opposite relative to the actual endpoint identifier where each anomaly occurs. Historical threshold flags are reference-only. Never turn these priors into a decision tree or infer a diagnosis from interface rate."""


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
        return {"available": False, "state_candidates": ["uncertain"]}
    array = np.asarray(values, dtype=float)
    deltas = [lane[-1] - lane[0] for lane in lanes.values() if len(lane) > 1]
    lane_diff = max(lane_means.values()) - min(lane_means.values()) if len(lane_means) > 1 else 0.0
    result: dict[str, Any] = {
        "available": True,
        "lane_values_or_means": {key: round(value, 6) for key, value in lane_means.items()},
        "mean": round(float(np.mean(array)), 6), "median": round(float(np.median(array)), 6),
        "std": round(float(np.std(array)), 6), "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6), "lane_difference": round(float(lane_diff), 6),
        "temporal_mean_delta": round(float(np.mean(deltas)), 6) if deltas else None,
        "temporal_max_abs_delta": round(float(max(map(abs, deltas))), 6) if deltas else None,
    }
    states: list[str] = []
    flags: list[str] = []
    reference = REFERENCE_THRESHOLDS.get(metric)
    if reference:
        checks = (
            (any(v <= reference["lane_down"] for v in lane_means.values()), "lane_down", "at_or_below_lane_down_reference"),
            (any(v < reference["low"] for v in lane_means.values()), "low_value", "below_low_reference"),
            (any(v > reference["high"] for v in lane_means.values()), "high_value", "above_high_reference"),
            (lane_diff > reference["lane_diff"], "lane_difference", "above_lane_difference_reference"),
        )
        for matched, state, flag in checks:
            if matched:
                states.append(state); flags.append(flag)
    result["state_candidates"] = states or ["normal_or_uncertain"]
    result["reference_threshold_flag"] = {"scope": "expert_reference_only", "flags": flags}
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
        endpoint: {"interface": mapping.get(endpoint), "rate_is_metadata_only": True}
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
        "threshold_note": "reference flags are expert_reference_only and cannot directly determine diagnosis",
    }


def _stage1_schema(endpoints: tuple[str, ...]) -> dict[str, Any]:
    candidates = [*endpoints, "fiber"]
    return {
        "type": "object",
        "properties": {
            "endpoint_evidence": {"type": "array", "items": {"type": "object", "properties": {"endpoint": {"type": "string", "enum": list(endpoints)}, "evidence": {"type": "string"}}, "required": ["endpoint", "evidence"], "additionalProperties": False}},
            "lane_temporal_findings": {"type": "string"},
            "transmission_findings": {"type": "string"},
            "cross_endpoint_comparison": {"type": "string"},
            "directional_evidence": {"type": "array", "items": {"type": "object", "properties": {"diagnosis": {"type": "string", "enum": candidates}, "evidence": {"type": "string"}}, "required": ["diagnosis", "evidence"], "additionalProperties": False}},
            "conflicting_evidence": {"type": "string"},
            "preliminary_diagnosis": {"type": "string", "enum": candidates},
        },
        "required": ["endpoint_evidence", "lane_temporal_findings", "transmission_findings", "cross_endpoint_comparison", "directional_evidence", "conflicting_evidence", "preliminary_diagnosis"],
        "additionalProperties": False,
    }


def _stage2_schema(endpoints: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "diagnosis_root_cause": {"type": "string", "enum": [*endpoints, "fiber"]},
            "confidence": {"type": "number"},
            "evidence_summary": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["diagnosis_root_cause", "confidence", "evidence_summary", "reasoning"],
        "additionalProperties": False,
    }


def _stage1_prompt(summary: dict[str, Any]) -> str:
    candidates = ", ".join(summary["candidate_diagnoses"])
    return (
        "BiAn Stage 1: extract physical evidence using the native endpoint identifiers exactly as provided. "
        f"Candidate diagnoses for THIS case: {candidates}. Do not rename endpoints as local, remote, side_a, or side_b. "
        "For every endpoint, assess lane-down/low/high/lane differences, single versus multi-lane behavior, time changes, multi-metric combinations, directional transmission, symmetry, and conflicts. "
        "The alarm endpoint is only an observation. Interface rate is metadata only and never defines an endpoint label. Apply same-side/opposite-side priors dynamically to the named endpoint. "
        "Use short fields and return only requested JSON.\n" + SOP_SOFT_PRIOR + "\nOBSERVABLE_SUMMARY:\n" + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def _stage2_prompt(summary: dict[str, Any], stage1: dict[str, Any]) -> str:
    candidates = ", ".join(summary["candidate_diagnoses"])
    return (
        "BiAn Stage 2: diagnose one root cause using native endpoint identity. "
        f"Candidate diagnoses for THIS case: {candidates}. Output exactly one listed candidate; never output another endpoint. "
        "Reconcile Stage 1 with both endpoints, native transmission directions, severity, temporal persistence, lane consistency, multi-metric support, and fiber conflicts. "
        "Do not infer a label from 400G/200G metadata and do not assume the alarm endpoint is causal. Keep evidence and reasoning under 45 words each. Return JSON only.\n"
        + SOP_SOFT_PRIOR + "\nSTAGE1_EVIDENCE:\n" + json.dumps(stage1, ensure_ascii=False, separators=(",", ":"))
        + "\nOBSERVABLE_SUMMARY:\n" + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def _valid_stage1(value: Any, endpoints: tuple[str, ...]) -> dict[str, Any]:
    candidates = [*endpoints, "fiber"]
    value = value if isinstance(value, dict) else {}
    preliminary = value.get("preliminary_diagnosis")
    if preliminary not in candidates:
        preliminary = endpoints[0]
    return {
        "endpoints": list(endpoints),
        "endpoint_evidence": value.get("endpoint_evidence", []),
        "lane_temporal_findings": str(value.get("lane_temporal_findings", "unavailable")),
        "transmission_findings": str(value.get("transmission_findings", "unavailable")),
        "cross_endpoint_comparison": str(value.get("cross_endpoint_comparison", "unavailable")),
        "directional_evidence": value.get("directional_evidence", []),
        "conflicting_evidence": str(value.get("conflicting_evidence", "unavailable")),
        "preliminary_diagnosis": preliminary,
    }


def _valid_stage2(value: Any, candidates: list[str], fallback: str) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    diagnosis = value.get("diagnosis_root_cause")
    if diagnosis not in candidates:
        diagnosis = fallback
    confidence = value.get("confidence", 0.0)
    return {
        "diagnosis_root_cause": diagnosis,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) and math.isfinite(float(confidence)) else 0.0,
        "evidence_summary": str(value.get("evidence_summary", "format fallback")),
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
    stage2 = [_valid_stage2(item["value"], [*eps, "fiber"], first["preliminary_diagnosis"]) for item, first, eps in zip(stage2_raw, stage1, active_endpoints)]
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
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-cases", type=int, default=2)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
