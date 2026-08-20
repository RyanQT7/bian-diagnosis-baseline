#!/usr/bin/env python3
"""BiAn v3: v1-style two-stage diagnosis with expert SOP soft priors."""
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

PHYSICAL_LOCATIONS = ("local", "remote", "fiber")
OUTPUT_LABELS = ("l1", "l2", "fiber")

METRICS = ("bias", "rxpower", "txpower", "media_snr", "host_snr", "serdes_snr")
STATUS_FIELDS = ("RxLOL", "TxLOL", "TxLOS", "RxLOS")
REFERENCE_THRESHOLDS = {
    "rxpower": {"lane_down": -40.0, "low": -2.5, "high": 4.6, "lane_diff": 1.0},
    "txpower": {"lane_down": -40.0, "low": -2.5, "high": 2.5, "lane_diff": 1.3},
    "host_snr": {"lane_down": 0.0, "low": 22.8, "high": 27.5, "lane_diff": 2.5},
    "media_snr": {"lane_down": 0.0, "low": 22.4, "high": 28.7, "lane_diff": 3.0},
    "serdes_snr": {"lane_down": 0.0, "low": 458750.0, "high": 947750.0, "lane_diff": 230000.0},
}

STAGE1_V3_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "local_side_evidence": {"type": "string"},
        "remote_side_evidence": {"type": "string"},
        "lane_temporal_findings": {"type": "string"},
        "cross_side_comparison": {"type": "string"},
        "local_fault_evidence": {"type": "string"},
        "remote_fault_evidence": {"type": "string"},
        "fiber_fault_evidence": {"type": "string"},
        "conflicting_evidence": {"type": "string"},
        "preliminary_physical_location": {"type": "string", "enum": list(PHYSICAL_LOCATIONS)},
    },
    "required": [
        "local_side_evidence", "remote_side_evidence", "lane_temporal_findings",
        "cross_side_comparison", "local_fault_evidence", "remote_fault_evidence",
        "fiber_fault_evidence", "conflicting_evidence", "preliminary_physical_location",
    ],
    "additionalProperties": False,
}

STAGE2_V3_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "physical_fault_location": {"type": "string", "enum": list(PHYSICAL_LOCATIONS)},
        "fault_endpoint_rate": {"type": "string", "enum": ["400G", "200G", "not_applicable"]},
        "diagnosis_root_cause": {"type": "string", "enum": list(OUTPUT_LABELS)},
        "confidence": {"type": "number"},
        "evidence_summary": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["physical_fault_location", "fault_endpoint_rate", "diagnosis_root_cause", "confidence", "evidence_summary", "reasoning"],
    "additionalProperties": False,
}

SOP_SOFT_PRIOR = """EXPERT DIAGNOSTIC PRIORS (soft evidence, never mandatory rules):
- host_snr or serdes_snr abnormality usually supports a fault on that same side.
- media_snr or rxpower abnormality usually supports a fault on the opposite side.
- txpower abnormality, especially lane-down/extreme power loss, strongly supports the same side.
- simultaneous serdes_snr + media_snr + rxpower anomalies on one side can strongly support the opposite side, but only when type, severity, lane consistency, time behavior, and other metrics agree.
- strong bilateral abnormalities that imply conflicting directions with similar strength increase fiber evidence. Bilateral but asymmetric evidence should favor the stronger coherent side.
- fiber is not a default third class, and weak evidence must not default to local.
Threshold flags are historical engineering references only. Do not mechanically apply thresholds, decide from one metric, or turn this SOP into a decision tree. Compare both sides, lane consistency, temporal behavior, severity, multi-metric combinations, and contradictions."""


def _discover_prediction_cases(data_root: Path) -> list[dict[str, Any]]:
    roots = [data_root / "data1", data_root / "data2"]
    paths = sorted(path for root in roots for path in root.glob("*/*.json"))
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        case_id = path.stem
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"case is not an object: {path}")
        observable = {key: value for key, value in raw.items() if key != "label"}
        cases.append({"case_id": case_id, "source_path": path, "data": observable})
        seen.add(case_id)
    cases.sort(key=lambda item: item["case_id"])
    if not cases:
        raise ValueError(f"no cases below {data_root}")
    return cases


def _lane_series(raw: Any) -> dict[str, list[float]]:
    if isinstance(raw, dict):
        result = {str(key): numeric_leaves(value) for key, value in raw.items()}
        if any(result.values()):
            return result
    values = numeric_leaves(raw)
    return {"aggregate": values} if values else {}


def _status_v3(value: Any) -> float | None:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"abnormal", "alarm", "down", "true", "yes", "on", "异常", "是"}:
            return 1.0
        if text in {"normal", "up", "false", "no", "off", "正常", "否"}:
            return 0.0
    return status_number(value)


def _metric_summary(raw: Any, metric: str) -> dict[str, Any]:
    lanes = _lane_series(raw)
    lane_means = {lane: float(np.mean(values)) for lane, values in lanes.items() if values}
    all_values = [value for values in lanes.values() for value in values]
    if not all_values:
        return {"available": False, "state_candidates": ["uncertain"]}
    array = np.asarray(all_values, dtype=float)
    temporal_deltas = [values[-1] - values[0] for values in lanes.values() if len(values) > 1]
    lane_diff = float(max(lane_means.values()) - min(lane_means.values())) if len(lane_means) > 1 else 0.0
    summary: dict[str, Any] = {
        "available": True,
        "lane_values_or_means": {key: round(value, 6) for key, value in lane_means.items()},
        "mean": round(float(np.mean(array)), 6),
        "median": round(float(np.median(array)), 6),
        "std": round(float(np.std(array)), 6),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
        "lane_difference": round(lane_diff, 6),
        "temporal_mean_delta": round(float(np.mean(temporal_deltas)), 6) if temporal_deltas else None,
        "temporal_max_abs_delta": round(float(max(map(abs, temporal_deltas))), 6) if temporal_deltas else None,
    }
    reference = REFERENCE_THRESHOLDS.get(metric)
    candidates: list[str] = []
    flags: list[str] = []
    if reference:
        if any(value <= reference["lane_down"] for value in lane_means.values()):
            candidates.append("lane_down")
            flags.append("at_or_below_lane_down_reference")
        if any(value < reference["low"] for value in lane_means.values()):
            candidates.append("low_value")
            flags.append("below_low_reference")
        if any(value > reference["high"] for value in lane_means.values()):
            candidates.append("high_value")
            flags.append("above_high_reference")
        if lane_diff > reference["lane_diff"]:
            candidates.append("lane_difference")
            flags.append("above_lane_difference_reference")
    if not candidates:
        candidates.append("normal_or_uncertain")
    summary["state_candidates"] = candidates
    summary["reference_threshold_flag"] = {
        "scope": "expert_reference_only",
        "flags": flags,
    }
    return summary


def _endpoint_rate(endpoint: Any) -> str | None:
    text = str(endpoint).upper()
    if "400G" in text:
        return "400G"
    if "200G" in text:
        return "200G"
    return None


def _role_sides(data: dict[str, Any]) -> tuple[str, str, str]:
    mapping = data.get("link_side_ip_interface_map")
    alarm = data.get("alarm_ip_interface")
    if isinstance(mapping, dict) and alarm is not None:
        matched = [str(side) for side, endpoint in mapping.items() if endpoint == alarm]
        if len(matched) == 1:
            local = matched[0]
            others = [str(side) for side in mapping if str(side) != local]
            if others:
                return local, others[0], "alarm_endpoint_match"
    if isinstance(mapping, dict) and len(mapping) >= 2:
        sides = [str(side) for side in mapping]
        return sides[0], sides[1], "canonical_endpoint_order_when_alarm_endpoint_unavailable"
    return "l1", "l2", "canonical_endpoint_order_when_mapping_unavailable"


def summarize_v3(data: dict[str, Any]) -> dict[str, Any]:
    local_raw, remote_raw, mapping_basis = _role_sides(data)
    endpoint_map = data.get("link_side_ip_interface_map") if isinstance(data.get("link_side_ip_interface_map"), dict) else {}
    local_rate = _endpoint_rate(endpoint_map.get(local_raw))
    remote_rate = _endpoint_rate(endpoint_map.get(remote_raw))
    if local_rate not in {"400G", "200G"} or remote_rate not in {"400G", "200G"}:
        raise ValueError("endpoint rate is not observable as 400G/200G")
    local_metrics: dict[str, Any] = {}
    remote_metrics: dict[str, Any] = {}
    for metric in METRICS:
        outer = data.get(metric)
        local_metrics[metric] = _metric_summary(outer.get(local_raw) if isinstance(outer, dict) else None, metric)
        remote_metrics[metric] = _metric_summary(outer.get(remote_raw) if isinstance(outer, dict) else None, metric)
    status: dict[str, Any] = {"local_side": {}, "remote_side": {}}
    for field in STATUS_FIELDS:
        outer = data.get(field)
        for role, side in (("local_side", local_raw), ("remote_side", remote_raw)):
            raw = outer.get(side) if isinstance(outer, dict) else None
            parsed = [_status_v3(value) for value in (raw.values() if isinstance(raw, dict) else [raw])]
            valid = [value for value in parsed if value is not None]
            status[role][field] = {"available": bool(valid), "abnormal_fraction": round(float(np.mean(valid)), 6) if valid else None}
    cross: dict[str, Any] = {}
    for metric in METRICS:
        left = local_metrics[metric].get("mean")
        right = remote_metrics[metric].get("mean")
        cross[metric] = {"local_minus_remote_mean": round(float(left - right), 6)} if left is not None and right is not None else {"local_minus_remote_mean": None}
    return {
        "endpoint_metadata": {
            "source_field": "link_side_ip_interface_map",
            "local": {"raw_side": local_raw, "rate": local_rate},
            "remote": {"raw_side": remote_raw, "rate": remote_rate},
            "role_mapping_basis": mapping_basis,
            "semantic_note": "l1/l2 are rate classes, never aliases for local/remote",
        },
        "observable_context": {"alarm_name": data.get("alarm_name"), "alarm_time": data.get("alarm_time"), "link_location": data.get("link_location")},
        "local_side": local_metrics,
        "remote_side": remote_metrics,
        "status": status,
        "cross_side_comparison": cross,
        "threshold_note": "all reference_threshold_flag values are expert_reference_only and cannot directly determine diagnosis",
    }


def _stage1_prompt(summary: dict[str, Any]) -> str:
    return (
        "You are BiAn v3 Stage 1. Extract structured anomaly evidence from observable optical-link telemetry; do not mechanically diagnose from thresholds. "
        "Local and remote are physical directions only. l1/l2 are NOT endpoint directions: l1 means a 400G fault class and l2 means a 200G fault class. "
        "For local and remote sides, identify clear metric anomalies, single-lane versus all-lane patterns, down/extreme versus merely low/high values, temporal change, symmetry, and multi-metric combinations. "
        "Read each endpoint rate only from endpoint_metadata. Explicitly list evidence for local, remote, fiber, and conflicts, then give preliminary_physical_location only. Reference flags are historical expert_reference_only hints, not truth. "
        "Use one short sentence per field, at most 16 words. Return only the requested JSON.\nOBSERVABLE_SUMMARY:\n"
        + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def _stage2_prompt(summary: dict[str, Any], stage1: dict[str, Any]) -> str:
    return (
        "You are BiAn v3 Stage 2. First diagnose physical_fault_location as local, remote, or fiber. Then convert it to the official class using observable endpoint_metadata: fiber stays fiber; a fault on a 400G endpoint becomes l1; a fault on a 200G endpoint becomes l2. "
        "Never map local->l1 or remote->l2 directly because local/remote direction and endpoint rate vary by case. Return physical location, observed fault endpoint rate, and final l1/l2/fiber class for audit. Reconcile Stage 1 with the observable summary.\n"
        + SOP_SOFT_PRIOR
        + "\nThese expert rules are diagnostic priors, NOT mandatory decision rules. Keep evidence_summary and reasoning under 45 words each. Return only the requested JSON.\nSTAGE1_EVIDENCE:\n"
        + json.dumps(stage1, ensure_ascii=False, separators=(",", ":"))
        + "\nOBSERVABLE_SUMMARY:\n"
        + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def _valid_stage1(value: Any, summary: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    diagnosis = value.get("preliminary_physical_location")
    if diagnosis not in PHYSICAL_LOCATIONS:
        # Parsing fallback only; it never reads ground truth or converts a rate
        # class. Stage 2 still performs the physical evidence diagnosis.
        diagnosis = "remote"
    result = {
        "local_side_evidence": str(value.get("local_side_evidence", "unavailable")),
        "remote_side_evidence": str(value.get("remote_side_evidence", "unavailable")),
        "lane_temporal_findings": str(value.get("lane_temporal_findings", "unavailable")),
        "cross_side_comparison": str(value.get("cross_side_comparison", "unavailable")),
        "preliminary_physical_location": diagnosis,
    }
    for key in ("local_fault_evidence", "remote_fault_evidence", "fiber_fault_evidence", "conflicting_evidence"):
        result[key] = str(value.get(key, "unavailable"))
    return result


def _label_from_physical(physical: str, summary: dict[str, Any]) -> tuple[str, str]:
    if physical == "fiber":
        return "fiber", "not_applicable"
    endpoint = summary["endpoint_metadata"][physical]
    rate = endpoint["rate"]
    if rate == "400G":
        return "l1", rate
    if rate == "200G":
        return "l2", rate
    raise ValueError(f"unsupported endpoint rate: {rate}")


def _valid_stage2(value: Any, fallback: str, summary: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    physical = value.get("physical_fault_location")
    if physical not in PHYSICAL_LOCATIONS:
        physical = fallback
    diagnosis, observed_rate = _label_from_physical(physical, summary)
    confidence = value.get("confidence", 0.0)
    return {
        "physical_fault_location": physical,
        "fault_endpoint_rate": observed_rate,
        "model_diagnosis_root_cause": value.get("diagnosis_root_cause"),
        "diagnosis_root_cause": diagnosis,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) and math.isfinite(float(confidence)) else 0.0,
        "evidence_summary": str(value.get("evidence_summary", "format fallback")),
        "reasoning": str(value.get("reasoning", "format fallback")),
    }


def _ground_truth_after_predictions(cases: list[dict[str, Any]]) -> dict[str, str]:
    truth: dict[str, str] = {}
    for case in cases:
        raw = json.loads(case["source_path"].read_text(encoding="utf-8"))
        source_label = raw.get("label")
        if source_label == "fiber":
            truth[case["case_id"]] = "fiber"
            continue
        endpoint_map = raw.get("link_side_ip_interface_map")
        endpoint = endpoint_map.get(source_label) if isinstance(endpoint_map, dict) else None
        rate = _endpoint_rate(endpoint)
        if rate == "400G":
            truth[case["case_id"]] = "l1"
        elif rate == "200G":
            truth[case["case_id"]] = "l2"
        else:
            raise ValueError(f"JSON label cannot be mapped to a 400G/200G endpoint for {case['case_id']}")
    return truth


def score_frame_v3(frame: pd.DataFrame) -> str:
    y_true = frame["true_label"].tolist()
    y_pred = frame["diagnosis_root_cause"].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=list(OUTPUT_LABELS))
    report = classification_report(y_true, y_pred, labels=list(OUTPUT_LABELS), target_names=list(OUTPUT_LABELS), digits=6, zero_division=0)
    lines = [
        "BiAn v3 two-stage 32B scores",
        f"Accuracy: {accuracy_score(y_true, y_pred):.6f}",
        f"Macro Precision: {precision_score(y_true, y_pred, labels=list(OUTPUT_LABELS), average='macro', zero_division=0):.6f}",
        f"Macro Recall: {recall_score(y_true, y_pred, labels=list(OUTPUT_LABELS), average='macro', zero_division=0):.6f}",
        f"Macro F1: {f1_score(y_true, y_pred, labels=list(OUTPUT_LABELS), average='macro', zero_division=0):.6f}",
        "",
        report.rstrip(),
        "",
        "Confusion Matrix (rows=true, columns=predicted; order=l1,l2,fiber):",
        "                  l1      l2   fiber",
    ]
    lines.extend(f"{label:>12}  " + "  ".join(f"{int(value):>6}" for value in row) for label, row in zip(OUTPUT_LABELS, cm))
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    cases = _discover_prediction_cases(args.data_root)
    summaries = [summarize_v3(case["data"]) for case in cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "cache"
    os.environ["VLLM_CACHE_ROOT"] = str(cache / "vllm")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    backend = Local32BBackend(args.model_path, args.gpu_ids, args.seed, args.max_input_tokens, args.max_output_tokens, args.max_num_seqs)

    if args.smoke_test:
        stage1_raw = backend.generate_json([{"prompt": _stage1_prompt(summaries[0]), "schema": STAGE1_V3_SCHEMA}])[0]
        stage1 = _valid_stage1(stage1_raw["value"], summaries[0])
        stage2_raw = backend.generate_json([{"prompt": _stage2_prompt(summaries[0], stage1), "schema": STAGE2_V3_SCHEMA}])[0]
        stage2 = _valid_stage2(stage2_raw["value"], stage1["preliminary_physical_location"], summaries[0])
        (args.output_dir / "smoke_test.json").write_text(json.dumps({"stage1": stage1, "stage2": stage2, "calls": backend.call_count, "stage1_error": stage1_raw["error"], "stage2_error": stage2_raw["error"], "stage1_raw_preview": stage1_raw["raw_preview"], "stage2_raw_preview": stage2_raw["raw_preview"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("BiAn v3 smoke passed")
        return 0

    stage1_raw = backend.generate_json([{"prompt": _stage1_prompt(summary), "schema": STAGE1_V3_SCHEMA} for summary in summaries])
    stage1_values = [_valid_stage1(item["value"], summary) for item, summary in zip(stage1_raw, summaries)]
    stage2_raw = backend.generate_json([{"prompt": _stage2_prompt(summary, stage1), "schema": STAGE2_V3_SCHEMA} for summary, stage1 in zip(summaries, stage1_values)])
    stage2_values = [_valid_stage2(item["value"], stage1["preliminary_physical_location"], summary) for item, stage1, summary in zip(stage2_raw, stage1_values, summaries)]

    predictions = {case["case_id"]: stage2["diagnosis_root_cause"] for case, stage2 in zip(cases, stage2_values)}
    if len(predictions) != len(cases) or set(predictions.values()) - set(OUTPUT_LABELS):
        raise AssertionError("predictions are incomplete")
    audit = {
        "version": "v3",
        "model": "DeepSeek-R1-Distill-Qwen-32B",
        "sop": "expert soft priors; thresholds reference-only",
        "stages": {"stage1_cases": len(stage1_values), "stage2_cases": len(stage2_values), "model_calls": backend.call_count, "retries": backend.retry_count},
        "cases": [
            {"case_id": case["case_id"], "stage1": stage1, "stage2": stage2, "stage1_error": raw1["error"], "stage2_error": raw2["error"]}
            for case, stage1, stage2, raw1, raw2 in zip(cases, stage1_values, stage2_values, stage1_raw, stage2_raw)
        ],
    }
    (args.output_dir / "inference_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    truth = _ground_truth_after_predictions(cases)
    frame = pd.DataFrame({
        "case_id": [case["case_id"] for case in cases],
        "diagnosis_root_cause": [predictions[case["case_id"]] for case in cases],
        "true_label": [truth[case["case_id"]] for case in cases],
    })
    frame.to_csv(args.output_dir / "bian_v3_results.csv", index=False)
    (args.output_dir / "scores.txt").write_text(score_frame_v3(frame), encoding="utf-8")
    (args.output_dir / "experiment_metadata.json").write_text(json.dumps({"version": "v3", "model": "DeepSeek-R1-Distill-Qwen-32B", "n_cases": len(cases), "load_seconds": backend.load_seconds, **audit["stages"]}, indent=2) + "\n", encoding="utf-8")
    print((args.output_dir / "scores.txt").read_text(encoding="utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
