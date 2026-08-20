#!/usr/bin/env python3
"""Data-driven, two-stage BiAn diagnosis with case-native endpoint names."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from llm_backend import Local32BBackend, numeric_leaves, status_number


NUMERIC_FIELDS = (
    "bias", "rxpower", "txpower", "media_snr", "host_snr", "serdes_snr",
    "Temperature", "Voltage",
)
STATUS_FIELDS = ("RxLOL", "TxLOL", "TxLOS", "RxLOS")
STRUCTURAL_FIELDS = (
    "Lane number", "vendor", "vendor_sn", "region", "link_location", "syslog",
)


def _case_paths(data_root: Path) -> list[Path]:
    roots = [data_root / "data1", data_root / "data2"]
    if not any(root.is_dir() for root in roots):
        roots = [data_root]
    return sorted(path for root in roots for path in root.glob("*/*.json"))


def _prediction_cases(data_root: Path, selected_ids: set[str] | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _case_paths(data_root):
        case_id = path.stem
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        if selected_ids is not None and case_id not in selected_ids:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"case is not an object: {path}")
        # The answer-bearing field is removed before any summary, prompt, or model call.
        observable = {key: value for key, value in raw.items() if key != "label"}
        cases.append({"case_id": case_id, "source_path": path, "data": observable})
    if not cases:
        raise ValueError(f"no selected cases below {data_root}")
    return sorted(cases, key=lambda item: item["case_id"])


def endpoint_keys(data: dict[str, Any]) -> tuple[str, ...]:
    mapping = data.get("link_side_ip_interface_map")
    if not isinstance(mapping, dict) or len(mapping) < 2:
        raise ValueError("link_side_ip_interface_map must define at least two endpoints")
    endpoints = tuple(str(key) for key in mapping)
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("duplicate endpoint identifier")
    return endpoints


def _round(value: float) -> float:
    return round(float(value), 6)


def _null_count(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, dict):
        return sum(_null_count(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_null_count(child) for child in value)
    return 0


def _lane_values(value: Any) -> dict[str, list[float]]:
    if isinstance(value, dict):
        lanes = {str(key): numeric_leaves(child) for key, child in value.items()}
        if any(lanes.values()):
            return {key: values for key, values in lanes.items() if values}
    values = numeric_leaves(value)
    return {"aggregate": values} if values else {}


def _lane_scalar_values(value: Any) -> dict[str, float]:
    return {lane: _round(float(np.mean(values))) for lane, values in _lane_values(value).items() if values}


def _metric_summary(value: Any) -> dict[str, Any]:
    lanes = _lane_values(value)
    values = [item for lane in lanes.values() for item in lane]
    missing = _null_count(value)
    if not values:
        return {
            "available": False,
            "lane_values": {},
            "value_count": 0,
            "missing_value_count": missing,
        }
    array = np.asarray(values, dtype=float)
    lane_means = [float(np.mean(lane)) for lane in lanes.values() if lane]
    return {
        "available": True,
        "lane_values": {lane_name: [_round(item) for item in lane_values] for lane_name, lane_values in lanes.items()},
        "value_count": int(array.size),
        "missing_value_count": missing,
        "mean": _round(float(np.mean(array))),
        "median": _round(float(np.median(array))),
        "std": _round(float(np.std(array))),
        "min": _round(float(np.min(array))),
        "max": _round(float(np.max(array))),
        "q25": _round(float(np.quantile(array, 0.25))),
        "q75": _round(float(np.quantile(array, 0.75))),
        "lane_spread": _round(max(lane_means) - min(lane_means)) if len(lane_means) > 1 else 0.0,
        "zero_count": int(np.sum(array == 0)),
        "negative_count": int(np.sum(array < 0)),
        "minus40_count": int(np.sum(array == -40)),
    }


def _status_summary(value: Any) -> dict[str, Any]:
    raw_values: list[Any] = []

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
        else:
            raw_values.append(item)

    collect(value)
    parsed = [status_number(item) for item in raw_values]
    valid = [item for item in parsed if item is not None]
    return {
        "available": bool(valid),
        "observed_values": [_round(item) for item in valid],
        "value_count": len(valid),
        "nonzero_count": int(sum(item != 0 for item in valid)),
        "missing_or_unparsed_count": len(raw_values) - len(valid) + _null_count(value),
    }


def _endpoint_value(data: dict[str, Any], field: str, endpoint: str) -> Any:
    raw = data.get(field)
    return raw.get(endpoint) if isinstance(raw, dict) else None


def _direction_summary(data: dict[str, Any], source: str, target: str) -> dict[str, Any]:
    direction = f"{source}->{target}"
    transmission = data.get("transmission")
    tx_raw = _endpoint_value(data, "txpower", source)
    rx_raw = _endpoint_value(data, "rxpower", target)
    transmission_raw = transmission.get(f"{source}-{target}") if isinstance(transmission, dict) else None
    tx_lanes = _lane_scalar_values(tx_raw)
    rx_lanes = _lane_scalar_values(rx_raw)
    transmission_lanes = _lane_scalar_values(transmission_raw)
    tx_rx = {
        lane: _round(tx_lanes[lane] - rx_lanes[lane])
        for lane in sorted(set(tx_lanes) & set(rx_lanes))
    }
    transmission_rx = {
        lane: _round(transmission_lanes[lane] - rx_lanes[lane])
        for lane in sorted(set(transmission_lanes) & set(rx_lanes))
    }
    return {
        "direction": direction,
        "source_txpower": _metric_summary(tx_raw),
        "destination_rxpower": _metric_summary(rx_raw),
        "transmission": _metric_summary(transmission_raw),
        "lane_txpower_minus_destination_rxpower": tx_rx,
        "lane_transmission_minus_destination_rxpower": transmission_rx,
    }


def _pair_comparison(data: dict[str, Any], endpoints: tuple[str, ...]) -> dict[str, Any]:
    if len(endpoints) < 2:
        return {}
    left, right = endpoints[:2]
    comparison: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        left_summary = _metric_summary(_endpoint_value(data, field, left))
        right_summary = _metric_summary(_endpoint_value(data, field, right))
        item: dict[str, Any] = {
            "left": left,
            "right": right,
            "left_median": left_summary.get("median"),
            "right_median": right_summary.get("median"),
            "left_mean": left_summary.get("mean"),
            "right_mean": right_summary.get("mean"),
            "left_lane_spread": left_summary.get("lane_spread"),
            "right_lane_spread": right_summary.get("lane_spread"),
        }
        if left_summary.get("median") is not None and right_summary.get("median") is not None:
            difference = float(left_summary["median"]) - float(right_summary["median"])
            item["median_difference"] = _round(difference)
            item["absolute_median_difference"] = _round(abs(difference))
        if left_summary.get("mean") is not None and right_summary.get("mean") is not None:
            difference = float(left_summary["mean"]) - float(right_summary["mean"])
            item["mean_difference"] = _round(difference)
            item["absolute_mean_difference"] = _round(abs(difference))
        comparison[field] = item
    return comparison


def summarize_case(data: dict[str, Any]) -> dict[str, Any]:
    endpoints = endpoint_keys(data)
    mapping = data["link_side_ip_interface_map"]
    alarm_interface = data.get("alarm_ip_interface")
    alarm_endpoint = next((endpoint for endpoint in endpoints if mapping.get(endpoint) == alarm_interface), None)
    endpoint_observations: dict[str, Any] = {}
    for endpoint in endpoints:
        endpoint_observations[endpoint] = {
            "numeric": {field: _metric_summary(_endpoint_value(data, field, endpoint)) for field in NUMERIC_FIELDS},
            "status": {field: _status_summary(_endpoint_value(data, field, endpoint)) for field in STATUS_FIELDS},
            "metadata": {
                "interface": mapping.get(endpoint),
                "lane_number": _endpoint_value(data, "Lane number", endpoint),
                "vendor": _endpoint_value(data, "vendor", endpoint),
                "vendor_sn": _endpoint_value(data, "vendor_sn", endpoint),
                "temperature": _metric_summary(_endpoint_value(data, "Temperature", endpoint)),
                "voltage": _metric_summary(_endpoint_value(data, "Voltage", endpoint)),
            },
        }
    directions = [_direction_summary(data, source, target) for source in endpoints for target in endpoints if source != target]
    available = {
        field: data.get(field) is not None and _null_count(data.get(field)) == 0
        for field in NUMERIC_FIELDS + STATUS_FIELDS + STRUCTURAL_FIELDS + ("transmission",)
    }
    return {
        "endpoints": list(endpoints),
        "candidate_diagnoses": [*endpoints, "fiber"],
        "endpoint_metadata": {endpoint: {"interface": mapping.get(endpoint), "rate_is_metadata_only": True} for endpoint in endpoints},
        "alarm_observation": {
            "alarm_name": data.get("alarm_name"),
            "alarm_time": data.get("alarm_time"),
            "alarm_ip_interface": alarm_interface,
            "alarm_endpoint_if_exactly_matched": alarm_endpoint,
            "note": "alarm endpoint is an observation, not an automatic cause",
        },
        "available_fields": available,
        "endpoint_observations": endpoint_observations,
        "directional_links": directions,
        "cross_endpoint_comparison": _pair_comparison(data, endpoints),
        "analysis_note": (
            "Lane-keyed arrays are spatial lane observations, not time series. Null is unavailable; "
            "zero and -40 are retained as observed values. No threshold flag is a diagnosis."
        ),
    }


def _stage1_schema(endpoints: tuple[str, ...]) -> dict[str, Any]:
    directions = [f"{source}->{target}" for source in endpoints for target in endpoints if source != target]
    return {
        "type": "object",
        "properties": {
            "endpoint_observations": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "endpoint": {"type": "string", "enum": list(endpoints)},
                    "observations": {"type": "string"},
                }, "required": ["endpoint", "observations"], "additionalProperties": False},
            },
            "directional_links": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "direction": {"type": "string", "enum": directions},
                    "observations": {"type": "string"},
                }, "required": ["direction", "observations"], "additionalProperties": False},
            },
            "cross_endpoint_comparison": {"type": "string"},
            "data_quality": {"type": "string"},
            "notable_inconsistencies": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["endpoint_observations", "directional_links", "cross_endpoint_comparison", "data_quality", "notable_inconsistencies"],
        "additionalProperties": False,
    }


def _stage2_schema(endpoints: tuple[str, ...]) -> dict[str, Any]:
    candidates = [*endpoints, "fiber"]
    comparison_item = {
        "type": "object",
        "properties": {
            "candidate": {"type": "string", "enum": candidates},
            "supporting_observations": {"type": "string"},
            "contradictory_observations": {"type": "string"},
            "unexplained_observations": {"type": "string"},
        },
        "required": ["candidate", "supporting_observations", "contradictory_observations", "unexplained_observations"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "candidate_comparison": {"type": "array", "items": comparison_item},
            "diagnosis_root_cause": {"type": "string", "enum": candidates},
            "reason": {"type": "string"},
        },
        "required": ["candidate_comparison", "diagnosis_root_cause", "reason"],
        "additionalProperties": False,
    }


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _stage2_case_view(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep Stage 2 within context while retaining lane detail for path metrics."""
    view = copy.deepcopy(summary)
    for endpoint in view["endpoints"]:
        numeric = view["endpoint_observations"][endpoint]["numeric"]
        for field, item in numeric.items():
            if field not in {"rxpower", "txpower"}:
                item.pop("lane_values", None)
        for item in view["endpoint_observations"][endpoint]["status"].values():
            item.pop("observed_values", None)
        view["endpoint_observations"][endpoint]["metadata"].pop("vendor_sn", None)
    for link in view["directional_links"]:
        for field in ("source_txpower", "destination_rxpower", "transmission"):
            link[field].pop("lane_values", None)
    return view


def _stage1_prompt(summary: dict[str, Any]) -> str:
    endpoints = ", ".join(summary["endpoints"])
    return (
        "BiAn Stage 1 is an objective evidence extraction step for one optical-link case. "
        f"Keep the native endpoint identifiers exactly: {endpoints}. Do not rename them or invent another endpoint. "
        "Do not output a diagnosis in this stage. Describe observed facts concisely: lane values and spread, missing values, "
        "zeros or -40 values, endpoint telemetry, status fields, and each native directional link. "
        "The arrays keyed by lane are spatial lane observations, not a time series, so do not invent temporal trends. "
        "Treat null as unavailable and do not turn any numeric threshold into a fault rule. "
        "The alarm interface is only an observation; interface strings and rate metadata do not define labels. "
        "Prefer facts and explicit inconsistencies over early interpretations. Return only the requested JSON.\n"
        "The optional expert background is weak and may be wrong for this dataset. It does not prescribe a direction: "
        "endpoint-internal and path-sensitive measurements must be interpreted from the complete observed case.\n"
        "OBSERVABLE_CASE_SUMMARY:\n" + _json_compact(summary)
    )


def _stage2_prompt(summary: dict[str, Any], stage1: dict[str, Any]) -> str:
    candidates = ", ".join(summary["candidate_diagnoses"])
    stage2_view = _stage2_case_view(summary)
    return (
        "BiAn Stage 2: choose exactly one diagnosis for THIS case from: " + candidates + ". "
        "The endpoint names are native identifiers and must not be renamed; do not use physical-side aliases in the output. "
        "Compare the complete observation under every candidate: "
        "supporting observations, contradictory observations, and unexplained observations. "
        "Use the aligned native directional records (source TxPower, destination RxPower, transmission) and the robust "
        "cross-endpoint comparisons explicitly; development analysis found these relationships more informative than isolated fields, "
        "but this is not a threshold or a fiber rule. Check lane-level consistency, missingness, status values, and alarm context. "
        "Do not infer a label from interface rate, do not treat the alarm endpoint as causal without corroborating telemetry, and do not use a single metric alone. "
        "The expert background is only a weak, possibly inaccurate engineering reference; observed evidence has priority and may override it. "
        "Do not force a fiber diagnosis merely because both endpoints have abnormalities, and do not make fiber a default. "
        "Return concise JSON only; diagnosis_root_cause must be one of the listed candidates.\n"
        "STAGE1_OBJECTIVE_EVIDENCE:\n" + _json_compact(stage1) +
        "\nOBSERVABLE_CASE_SUMMARY:\n" + _json_compact(stage2_view)
    )


def _compact_fact(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "available", "mean", "median", "min", "max", "lane_spread",
        "zero_count", "negative_count", "minus40_count", "value_count", "missing_value_count",
    )
    return {key: value[key] for key in keys if key in value}


def _fallback_stage1(summary: dict[str, Any]) -> dict[str, Any]:
    """Retain objective programmatic facts if the model emits an empty optional list."""
    endpoint_observations = []
    for endpoint in summary["endpoints"]:
        numeric = {}
        for field, item in summary["endpoint_observations"][endpoint]["numeric"].items():
            numeric[field] = _compact_fact(item)
        status = {
            field: _compact_fact(item)
            for field, item in summary["endpoint_observations"][endpoint]["status"].items()
        }
        endpoint_observations.append({"endpoint": endpoint, "observations": _json_compact({
            "numeric_fact_summary": numeric,
            "status_fact_summary": status,
        })})
    directional_links = []
    for link in summary["directional_links"]:
        directional_links.append({"direction": link["direction"], "observations": _json_compact({
            "source_txpower": _compact_fact(link["source_txpower"]),
            "destination_rxpower": _compact_fact(link["destination_rxpower"]),
            "transmission": _compact_fact(link["transmission"]),
            "lane_txpower_minus_destination_rxpower": link["lane_txpower_minus_destination_rxpower"],
            "lane_transmission_minus_destination_rxpower": link["lane_transmission_minus_destination_rxpower"],
        })})
    missing = [field for field, available in summary["available_fields"].items() if not available]
    return {
        "endpoint_observations": endpoint_observations,
        "directional_links": directional_links,
        "cross_endpoint_comparison": _json_compact(summary["cross_endpoint_comparison"]),
        "data_quality": "unavailable or partially missing fields: " + ", ".join(missing[:20]) if missing else "all summarized fields available",
        "notable_inconsistencies": [],
        "endpoints": list(summary["endpoints"]),
    }


def _valid_stage1(value: Any, endpoints: tuple[str, ...], summary: dict[str, Any]) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    fallback = _fallback_stage1(summary)
    endpoint_observations = value.get("endpoint_observations")
    directional_links = value.get("directional_links")
    return {
        "endpoint_observations": endpoint_observations if isinstance(endpoint_observations, list) and endpoint_observations else fallback["endpoint_observations"],
        "directional_links": directional_links if isinstance(directional_links, list) and directional_links else fallback["directional_links"],
        "cross_endpoint_comparison": str(value.get("cross_endpoint_comparison") or fallback["cross_endpoint_comparison"]),
        "data_quality": str(value.get("data_quality") or fallback["data_quality"]),
        "notable_inconsistencies": value.get("notable_inconsistencies") if isinstance(value.get("notable_inconsistencies"), list) else fallback["notable_inconsistencies"],
        "endpoints": list(endpoints),
    }


def _valid_stage2(value: Any, candidates: list[str]) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    diagnosis = value.get("diagnosis_root_cause")
    if diagnosis not in candidates:
        # Deterministic format fallback; it never reads ground truth.
        diagnosis = candidates[0]
    comparisons = value.get("candidate_comparison", [])
    if not isinstance(comparisons, list):
        comparisons = []
    return {
        "candidate_comparison": comparisons,
        "diagnosis_root_cause": diagnosis,
        "reason": str(value.get("reason", "format fallback")),
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
    y_true = frame["true_label"].tolist()
    y_pred = frame["diagnosis_root_cause"].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels, digits=6, zero_division=0)
    fiber_true = sum(value == "fiber" for value in y_true)
    fiber_pred = sum(value == "fiber" for value in y_pred)
    fiber_tp = sum(true == "fiber" and pred == "fiber" for true, pred in zip(y_true, y_pred))
    fiber_fp = fiber_pred - fiber_tp
    fiber_fn = fiber_true - fiber_tp
    lines = [
        "BiAn data-driven native-endpoint scores",
        f"Cases: {len(frame)}",
        f"Accuracy: {accuracy_score(y_true, y_pred):.6f}",
        f"Macro Precision: {precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        f"Macro Recall: {recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        f"Macro F1: {f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        f"Fiber support: {fiber_true}",
        f"Fiber predicted: {fiber_pred}",
        f"Fiber TP: {fiber_tp}",
        f"Fiber FP: {fiber_fp}",
        f"Fiber FN: {fiber_fn}",
        "",
        report.rstrip(),
        "",
        f"Confusion Matrix (rows=true, columns=predicted; order={','.join(labels)}):",
        "              " + "  ".join(f"{label:>6}" for label in labels),
    ]
    lines.extend(f"{label:>12}  " + "  ".join(f"{int(value):>6}" for value in row) for label, row in zip(labels, cm))
    return "\n".join(lines) + "\n"


def _selected_ids(args: argparse.Namespace) -> set[str] | None:
    if args.case_list:
        ids = {line.strip() for line in args.case_list.read_text(encoding="utf-8").splitlines() if line.strip()}
        if not ids:
            raise ValueError("case list is empty")
        return ids
    if args.split_manifest:
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        if args.subset == "development":
            return set(manifest["development_case_ids"])
        if args.subset == "holdout":
            return set(manifest["holdout_case_ids"])
    return None


def run(args: argparse.Namespace) -> int:
    selected_ids = _selected_ids(args)
    cases = _prediction_cases(args.data_root, selected_ids)
    summaries = [summarize_case(case["data"]) for case in cases]
    endpoints = [endpoint_keys(case["data"]) for case in cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (args.output_dir / "cache")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache / "vllm")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    backend = Local32BBackend(args.model_path, args.gpu_ids, args.seed, args.max_input_tokens, args.max_output_tokens, args.max_num_seqs)
    if args.smoke_test:
        selected: list[int] = []
        seen_signatures: set[tuple[str, ...]] = set()
        for index, signature in enumerate(endpoints):
            if signature not in seen_signatures:
                selected.append(index)
                seen_signatures.add(signature)
            if len(selected) >= args.smoke_cases:
                break
        selected.extend(index for index in range(len(cases)) if index not in selected and len(selected) < args.smoke_cases)
    else:
        selected = list(range(len(cases)))
    active_summaries = [summaries[index] for index in selected]
    active_endpoints = [endpoints[index] for index in selected]
    stage1_raw = backend.generate_json([
        {"prompt": _stage1_prompt(summary), "schema": _stage1_schema(eps)}
        for summary, eps in zip(active_summaries, active_endpoints)
    ])
    stage1 = [_valid_stage1(item["value"], eps, summary) for item, eps, summary in zip(stage1_raw, active_endpoints, active_summaries)]
    stage2_raw = backend.generate_json([
        {"prompt": _stage2_prompt(summary, first), "schema": _stage2_schema(eps)}
        for summary, first, eps in zip(active_summaries, stage1, active_endpoints)
    ])
    stage2 = [_valid_stage2(item["value"], [*eps, "fiber"]) for item, eps in zip(stage2_raw, active_endpoints)]
    if args.smoke_test:
        payload = [
            {"case_id": cases[index]["case_id"], "endpoints": endpoints[index], "stage1": first, "stage2": second}
            for index, first, second in zip(selected, stage1, stage2)
        ]
        (args.output_dir / "smoke_test.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"data-driven native-endpoint smoke passed: {len(selected)} cases")
        return 0
    if len(stage2) != len(cases):
        raise RuntimeError(f"missing model output: {len(stage2)}/{len(cases)}")
    predictions = {case["case_id"]: result["diagnosis_root_cause"] for case, result in zip(cases, stage2)}
    for case, eps in zip(cases, endpoints):
        if predictions[case["case_id"]] not in {*eps, "fiber"}:
            raise AssertionError("prediction outside case-native candidates")
    # Inference is complete before this function reopens JSON labels for evaluation.
    truth = _truth_after_predictions(cases)
    frame = pd.DataFrame({
        "case_id": [case["case_id"] for case in cases],
        "diagnosis_root_cause": [predictions[case["case_id"]] for case in cases],
        "true_label": [truth[case["case_id"]] for case in cases],
    })
    frame.to_csv(args.output_dir / args.results_name, index=False)
    scores = score_frame_v3(frame)
    (args.output_dir / args.scores_name).write_text(scores, encoding="utf-8")
    print(scores)
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
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--subset", choices=("all", "development", "holdout"), default="all")
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--results-name", default="bian_results.csv")
    parser.add_argument("--scores-name", default="scores.txt")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-cases", type=int, default=4)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
