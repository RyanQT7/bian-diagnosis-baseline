#!/usr/bin/env python3
"""API-backed BiAn native-endpoint diagnosis (the former native-endpoint strategy)."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from api_backend import OpenAIResponsesBackend, UsageLedger


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


def numeric_leaves(value: Any) -> list[float]:
    result: list[float] = []
    if isinstance(value, dict):
        for child in value.values():
            result.extend(numeric_leaves(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(numeric_leaves(child))
    elif not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)):
        result.append(float(value))
    return result


def status_number(value: Any) -> float | None:
    if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "up", "alarm", "异常", "是"}:
            return 1.0
        if text in {"0", "false", "no", "off", "down", "normal", "正常", "否"}:
            return 0.0
    return None


def endpoint_keys(data: dict[str, Any]) -> tuple[str, ...]:
    mapping = data.get("link_side_ip_interface_map")
    if not isinstance(mapping, dict) or len(mapping) < 2:
        raise ValueError("link_side_ip_interface_map must define at least two endpoints")
    endpoints = tuple(str(key) for key in mapping)
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("duplicate endpoint identifier")
    return endpoints


def load_cases(data_root: Path, selected_ids: set[str] | None = None) -> list[dict[str, Any]]:
    roots = [data_root / "data1", data_root / "data2"]
    if not any(root.is_dir() for root in roots):
        roots = [data_root]
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for path in sorted(root.glob("*/*.json")):
            case_id = path.stem
            if case_id in seen:
                raise ValueError(f"duplicate case_id: {case_id}")
            seen.add(case_id)
            if selected_ids is not None and case_id not in selected_ids:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"case is not an object: {path}")
            # Ground truth is intentionally removed before any summary or API prompt.
            observable = {key: value for key, value in raw.items() if key != "label"}
            endpoint_keys(observable)
            cases.append({"case_id": case_id, "source_path": path, "data": observable})
    if not cases:
        raise ValueError(f"no selected cases below {data_root}")
    return sorted(cases, key=lambda item: item["case_id"])


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
                states.append(state)
                flags.append(flag)
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
    metadata = {endpoint: {"interface": mapping.get(endpoint), "rate_is_metadata_only": True} for endpoint in endpoints}
    evidence: dict[str, Any] = {}
    for endpoint in endpoints:
        evidence[endpoint] = {
            metric: _metric_summary(
                data.get(metric, {}).get(endpoint) if isinstance(data.get(metric), dict) else None, metric
            ) for metric in METRICS
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
                a = evidence[left][metric].get("mean")
                b = evidence[right][metric].get("mean")
                cross[f"{metric}.{left}_minus_{right}_mean"] = round(float(a - b), 6) if a is not None and b is not None else None
    return {
        "endpoints": list(endpoints),
        "candidate_diagnoses": [*endpoints, "fiber"],
        "endpoint_metadata": metadata,
        "alarm_observation": {
            "alarm_name": data.get("alarm_name"), "alarm_time": data.get("alarm_time"),
            "alarm_endpoint": alarm_endpoint,
            "note": "alarm endpoint is evidence, not an automatic root cause",
        },
        "endpoint_field_coverage": coverage,
        "endpoint_evidence": evidence,
        "transmission_by_native_direction": transmission,
        "cross_endpoint_comparison": cross,
        "threshold_note": "reference flags are expert_reference_only and cannot directly determine diagnosis",
    }


def _local32b_stage1_schema(endpoints: tuple[str, ...]) -> dict[str, Any]:
    directions = [f"{source}->{target}" for source in endpoints for target in endpoints if source != target]
    return {
        "title": "bian_stage1",
        "type": "object",
        "properties": {
            "endpoint_observations": {"type": "array", "items": {"type": "object", "properties": {
                "endpoint": {"type": "string", "enum": list(endpoints)},
                "observations": {"type": "string"},
            }, "required": ["endpoint", "observations"], "additionalProperties": False}},
            "directional_links": {"type": "array", "items": {"type": "object", "properties": {
                "direction": {"type": "string", "enum": directions},
                "observations": {"type": "string"},
            }, "required": ["direction", "observations"], "additionalProperties": False}},
            "cross_endpoint_comparison": {"type": "string"},
            "data_quality": {"type": "string"},
            "notable_inconsistencies": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["endpoint_observations", "directional_links", "cross_endpoint_comparison",
                      "data_quality", "notable_inconsistencies"],
        "additionalProperties": False,
    }


def _local32b_stage2_schema(endpoints: tuple[str, ...]) -> dict[str, Any]:
    candidates = [*endpoints, "fiber"]
    comparison_item = {
        "type": "object",
        "properties": {
            "candidate": {"type": "string", "enum": candidates},
            "supporting_observations": {"type": "string"},
            "contradictory_observations": {"type": "string"},
            "unexplained_observations": {"type": "string"},
        },
        "required": ["candidate", "supporting_observations", "contradictory_observations",
                      "unexplained_observations"],
        "additionalProperties": False,
    }
    return {
        "title": "bian_stage2",
        "type": "object",
        "properties": {
            "candidate_comparison": {"type": "array", "items": comparison_item},
            "diagnosis_root_cause": {"type": "string", "enum": candidates},
            "reason": {"type": "string"},
        },
        "required": ["candidate_comparison", "diagnosis_root_cause", "reason"],
        "additionalProperties": False,
    }


def stage1_schema(endpoints: tuple[str, ...], profile: str = "compact") -> dict[str, Any]:
    if profile == "local32b":
        return _local32b_stage1_schema(endpoints)
    endpoint_properties = {endpoint: {"type": "string"} for endpoint in endpoints}
    return {
        "title": "bian_stage1",
        "type": "object",
        "properties": {
            "endpoint_evidence": {"type": "object", "properties": endpoint_properties,
                                   "required": list(endpoints), "additionalProperties": False},
            "cross_endpoint_evidence": {"type": "string"},
            "fiber_evidence": {"type": "string"},
            "uncertainty": {"type": "string"},
        },
        "required": ["endpoint_evidence", "cross_endpoint_evidence", "fiber_evidence", "uncertainty"],
        "additionalProperties": False,
    }


def stage2_schema(endpoints: tuple[str, ...], profile: str = "compact") -> dict[str, Any]:
    if profile == "local32b":
        return _local32b_stage2_schema(endpoints)
    return {
        "title": "bian_stage2",
        "type": "object",
        "properties": {
            "diagnosis_root_cause": {"type": "string", "enum": [*endpoints, "fiber"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["diagnosis_root_cause"],
        "additionalProperties": True,
    }


def stage1_prompt(summary: dict[str, Any], profile: str = "compact") -> str:
    candidates = ", ".join(summary["candidate_diagnoses"])
    if profile == "local32b":
        return (
            "BiAn Stage 1 is objective evidence extraction for one optical-link case. "
            f"Keep native endpoint identifiers exactly: {', '.join(summary['endpoints'])}; never rename them. "
            "Do not output a diagnosis in Stage 1. Summarize observed endpoint facts and each native directional link "
            "concisely, including lane pattern, missingness, status, cross-endpoint differences, and inconsistencies. "
            "Lane-keyed values are spatial observations, not a time series. Null is unavailable; do not invent values "
            "or threshold decisions. Return ONLY the required JSON object, without markdown, code fences, repeated input, "
            "chain-of-thought, or text outside JSON. Keep each text field to one or two short sentences.\n"
            "Use exactly these Stage 1 fields: endpoint_observations (array of endpoint/observations objects), "
            "directional_links (array of direction/observations objects), cross_endpoint_comparison, data_quality, "
            "and notable_inconsistencies (array of strings).\n"
            + SOP_SOFT_PRIOR + "\nOBSERVABLE_SUMMARY:\n"
            + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        )
    return (
        "BiAn Stage 1: extract physical evidence using the native endpoint identifiers exactly as provided. "
        f"Candidate diagnoses for THIS case: {candidates}. Do not rename endpoints as local, remote, side_a, or side_b. "
        "For every endpoint, assess lane-down/low/high/lane differences, single versus multi-lane behavior, time changes, multi-metric combinations, directional transmission, symmetry, and conflicts. "
        "The alarm endpoint is only an observation. Interface rate is metadata only and never defines an endpoint label. Apply same-side/opposite-side priors dynamically to the named endpoint. "
        "Return ONLY one valid JSON object with these fields: endpoint_evidence (an object with one short string per endpoint), "
        "cross_endpoint_evidence, fiber_evidence, uncertainty. Do not use markdown, code fences, or prose outside JSON.\n" + SOP_SOFT_PRIOR
        + "\nOBSERVABLE_SUMMARY:\n" + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def stage2_prompt(summary: dict[str, Any], stage1: dict[str, Any], profile: str = "compact") -> str:
    candidates = ", ".join(summary["candidate_diagnoses"])
    if profile == "local32b":
        return (
            "BiAn Stage 2: choose exactly one diagnosis for THIS case from: " + candidates + ". "
            "Preserve native endpoint names; never output local, remote, side_a, or side_b. Compare each candidate "
            "against the complete observed case, including directional Tx/Rx/transmission relationships, lane consistency, "
            "missingness, status, alarm context, and contradictions. Interface/rate metadata is not a label. The expert "
            "background is weak and may be wrong; observed evidence has priority and may override it. Do not force fiber "
            "or use it as a default. Return ONLY the required JSON object, with concise one-sentence text fields, no markdown, "
            "code fences, repeated input, chain-of-thought, or text outside JSON.\n"
            "Use exactly these Stage 2 fields: candidate_comparison (candidate/supporting_observations/"
            "contradictory_observations/unexplained_observations objects), diagnosis_root_cause, and reason.\n"
            + SOP_SOFT_PRIOR + "\nSTAGE1_OBJECTIVE_EVIDENCE:\n"
            + json.dumps(stage1, ensure_ascii=False, separators=(",", ":"))
            + "\nOBSERVABLE_SUMMARY:\n"
            + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        )
    return (
        "BiAn Stage 2: diagnose one root cause using native endpoint identity. "
        f"Candidate diagnoses for THIS case: {candidates}. Output exactly one listed candidate; never output another endpoint. "
        "Reconcile Stage 1 with both endpoints, native transmission directions, severity, temporal persistence, lane consistency, multi-metric support, and fiber conflicts. "
        "Do not infer a label from 400G/200G metadata and do not assume the alarm endpoint is causal. "
        "Return ONLY one valid JSON object with diagnosis_root_cause, confidence, and reason. "
        "Do not use markdown, code fences, or prose outside JSON.\n" + SOP_SOFT_PRIOR
        + "\nSTAGE1_EVIDENCE:\n" + json.dumps(stage1, ensure_ascii=False, separators=(",", ":"))
        + "\nOBSERVABLE_SUMMARY:\n" + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def fallback_stage1(endpoints: tuple[str, ...], profile: str = "compact") -> dict[str, Any]:
    if profile == "local32b":
        return {
            "endpoints": list(endpoints), "endpoint_observations": [], "directional_links": [],
            "cross_endpoint_comparison": "unavailable", "data_quality": "unavailable",
            "notable_inconsistencies": [],
        }
    return {
        "endpoints": list(endpoints), "endpoint_evidence": {endpoint: "unavailable" for endpoint in endpoints},
        "cross_endpoint_evidence": "unavailable", "fiber_evidence": "unavailable", "uncertainty": "unavailable",
    }


def normalize_stage1(value: Any, endpoints: tuple[str, ...], profile: str = "compact") -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    if profile == "local32b":
        endpoint_observations = value.get("endpoint_observations")
        if not isinstance(endpoint_observations, list):
            endpoint_facts = value.get("endpoint_facts")
            endpoint_observations = [
                {"endpoint": endpoint, "observations": str(endpoint_facts[endpoint])}
                for endpoint in endpoints
                if isinstance(endpoint_facts, dict) and endpoint in endpoint_facts
            ]
        else:
            endpoint_observations = [
                {
                    "endpoint": item.get("endpoint"),
                    "observations": str(item.get("observations", item.get("evidence", ""))),
                }
                for item in endpoint_observations
                if isinstance(item, dict) and item.get("endpoint") in endpoints
            ]

        directional_links = value.get("directional_links")
        if not isinstance(directional_links, list):
            link_facts = value.get("directional_link_facts")
            directional_links = [
                {"direction": _canonical_direction(direction, endpoints), "observations": str(link_facts[direction])}
                for direction in _expected_raw_directions(endpoints)
                if isinstance(link_facts, dict) and direction in link_facts
            ]
        else:
            directional_links = [
                {
                    "direction": _canonical_direction(item.get("direction"), endpoints),
                    "observations": str(item.get("observations", item.get("evidence", ""))),
                }
                for item in directional_links
                if isinstance(item, dict) and _canonical_direction(item.get("direction"), endpoints)
            ]

        cross = value.get("cross_endpoint_comparison", value.get("cross_endpoint_differences"))
        quality = value.get("data_quality", value.get("missingness_and_inconsistencies"))
        inconsistencies = value.get("notable_inconsistencies")
        if not isinstance(inconsistencies, list):
            if isinstance(inconsistencies, str):
                inconsistencies = [inconsistencies]
            elif isinstance(quality, str) and quality:
                # Gemini's equivalent field combines data-quality and
                # inconsistency text; retaining the same text is lossless.
                inconsistencies = [quality]
            else:
                inconsistencies = []
        return {
            "endpoints": list(endpoints),
            "endpoint_observations": endpoint_observations,
            "directional_links": directional_links,
            "cross_endpoint_comparison": str(cross or "unavailable"),
            "data_quality": str(quality or "unavailable"),
            "notable_inconsistencies": inconsistencies,
        }
    raw_evidence = value.get("endpoint_evidence", {})
    evidence: dict[str, str] = {endpoint: "unavailable" for endpoint in endpoints}
    if isinstance(raw_evidence, dict):
        for endpoint in endpoints:
            if endpoint in raw_evidence:
                evidence[endpoint] = str(raw_evidence[endpoint])
    elif isinstance(raw_evidence, list):
        for item in raw_evidence:
            if isinstance(item, dict) and item.get("endpoint") in evidence:
                evidence[str(item["endpoint"])] = str(item.get("evidence", "unavailable"))
    return {
        "endpoints": list(endpoints),
        "endpoint_evidence": evidence,
        "cross_endpoint_evidence": str(value.get("cross_endpoint_evidence", value.get("cross_endpoint_comparison", "unavailable"))),
        "fiber_evidence": str(value.get("fiber_evidence", "unavailable")),
        "uncertainty": str(value.get("uncertainty", value.get("conflicting_evidence", "unavailable"))),
    }


def _expected_raw_directions(endpoints: tuple[str, ...]) -> list[str]:
    return [f"{source}-{target}" for source in endpoints for target in endpoints if source != target]


def _canonical_direction(direction: Any, endpoints: tuple[str, ...]) -> str | None:
    if not isinstance(direction, str):
        return None
    for source in endpoints:
        for target in endpoints:
            if source == target:
                continue
            if direction in {f"{source}->{target}", f"{source}-{target}", f"{source}→{target}"}:
                return f"{source}->{target}"
    return None


def _stage1_has_canonical_shape(value: dict[str, Any], endpoints: tuple[str, ...]) -> bool:
    observations = value.get("endpoint_observations")
    if isinstance(observations, list):
        observed_endpoints = {item.get("endpoint") for item in observations if isinstance(item, dict)}
        has_endpoints = set(endpoints) <= observed_endpoints
    else:
        facts = value.get("endpoint_facts")
        has_endpoints = isinstance(facts, dict) and set(endpoints) <= set(facts)
    links = value.get("directional_links")
    if isinstance(links, list):
        directions = {_canonical_direction(item.get("direction"), endpoints) for item in links if isinstance(item, dict)}
    else:
        facts = value.get("directional_link_facts")
        directions = {_canonical_direction(item, endpoints) for item in facts} if isinstance(facts, dict) else set()
    expected = {f"{source}->{target}" for source in endpoints for target in endpoints if source != target}
    has_links = expected <= directions
    has_cross = isinstance(value.get("cross_endpoint_comparison", value.get("cross_endpoint_differences")), str)
    quality = value.get("data_quality", value.get("missingness_and_inconsistencies"))
    has_quality = isinstance(quality, str)
    has_notable = isinstance(value.get("notable_inconsistencies"), list) or isinstance(value.get("missingness_and_inconsistencies"), str)
    return has_endpoints and has_links and has_cross and has_quality and has_notable


def _stage2_diagnosis(value: dict[str, Any], candidates: list[str]) -> Any:
    for key in ("diagnosis_root_cause", "root_cause", "diagnosis"):
        if key in value:
            return value[key]
    return None


def _canonical_comparisons(value: dict[str, Any], candidates: list[str]) -> list[dict[str, Any]] | None:
    raw = value.get("candidate_comparison", value.get("candidate_comparisons", value.get("comparisons")))
    if not isinstance(raw, list):
        return None
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        candidate = item.get("candidate", item.get("diagnosis"))
        supporting = item.get("supporting_observations", item.get("supporting", item.get("support")))
        contradictory = item.get("contradictory_observations", item.get("contradictory", item.get("conflicts")))
        unexplained = item.get("unexplained_observations", item.get("unexplained"))
        if candidate not in candidates or not all(isinstance(text, str) for text in (supporting, contradictory, unexplained)):
            return None
        normalized.append({
            "candidate": candidate,
            "supporting_observations": supporting,
            "contradictory_observations": contradictory,
            "unexplained_observations": unexplained,
        })
    return normalized


def normalize_stage2(value: Any, candidates: list[str], fallback: str, profile: str = "compact") -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    diagnosis = _stage2_diagnosis(value, candidates)
    if diagnosis not in candidates:
        diagnosis = fallback if fallback in candidates else candidates[0]
    confidence = value.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    confidence = min(1.0, max(0.0, confidence))
    reason = value.get("reason", value.get("reasoning", value.get("rationale", value.get("explanation", "format fallback"))))
    if profile == "local32b":
        comparisons = _canonical_comparisons(value, candidates) or []
        return {"candidate_comparison": comparisons, "diagnosis_root_cause": diagnosis, "reason": str(reason)}
    return {"diagnosis_root_cause": diagnosis, "confidence": confidence, "reason": str(reason)}


def valid_stage1(value: Any, endpoints: tuple[str, ...], profile: str = "compact") -> bool:
    if not isinstance(value, dict):
        return False
    if profile == "local32b":
        return _stage1_has_canonical_shape(value, endpoints)
    evidence = value.get("endpoint_evidence")
    if isinstance(evidence, dict):
        return any(endpoint in evidence for endpoint in endpoints)
    if isinstance(evidence, list):
        return any(isinstance(item, dict) and item.get("endpoint") in endpoints for item in evidence)
    return False


def valid_stage2(value: Any, candidates: list[str], profile: str = "compact") -> bool:
    if not isinstance(value, dict) or _stage2_diagnosis(value, candidates) not in candidates:
        return False
    if profile != "local32b":
        return True
    reason = value.get("reason", value.get("reasoning", value.get("rationale", value.get("explanation"))))
    return isinstance(reason, str) and _canonical_comparisons(value, candidates) is not None


def truth_after_predictions(cases: list[dict[str, Any]]) -> dict[str, str]:
    truth: dict[str, str] = {}
    for case in cases:
        raw = json.loads(case["source_path"].read_text(encoding="utf-8"))
        label = raw.get("label")
        candidates = [*endpoint_keys(case["data"]), "fiber"]
        if label not in candidates:
            raise ValueError(f"JSON label outside native candidates for {case['case_id']}: {label}")
        truth[case["case_id"]] = str(label)
    return truth


def score_frame(frame: pd.DataFrame) -> str:
    labels = sorted((set(frame["true_label"]) | set(frame["diagnosis_root_cause"])) - {"fiber"}) + ["fiber"]
    y_true = frame["true_label"].tolist()
    y_pred = frame["diagnosis_root_cause"].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels, digits=6, zero_division=0)
    lines = [
        "BiAn API case-native endpoint scores",
        f"Cases: {len(frame)}",
        f"Accuracy: {accuracy_score(y_true, y_pred):.6f}",
        f"Macro Precision: {precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        f"Macro Recall: {recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        f"Macro F1: {f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.6f}",
        "", report.rstrip(), "", f"Confusion Matrix (rows=true, columns=predicted; order={','.join(labels)}):",
        "              " + "  ".join(f"{label:>6}" for label in labels),
    ]
    lines.extend(f"{label:>12}  " + "  ".join(f"{int(value):>6}" for value in row) for label, row in zip(labels, cm))
    return "\n".join(lines) + "\n"


def _write_evaluation_reports(
    output_dir: Path,
    records: list[dict[str, Any]],
    total_cases: int,
    model: str,
    base_url: str | None,
) -> None:
    failed = [record for record in records if record.get("status") != "SUCCESS"]
    successful = total_cases - len(failed)
    failed_path = output_dir / "failed_cases.json"
    failed_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    coverage = successful / total_cases if total_cases else 0.0
    lines = [
        "BiAn API evaluation run",
        f"Model: {model}",
        f"Base URL: {base_url or ''}",
        f"Test cases: {total_cases}",
        f"Successful cases: {successful}",
        f"Failed cases: {len(failed)}",
        f"Coverage: {coverage:.6f}",
        "Failed cases are not assigned fallback predictions and are excluded from score metrics.",
        "Ground truth is read only after prediction processing is complete.",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ledger_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in (
            "calls", "input_tokens", "output_tokens", "total_tokens",
            "cached_input_tokens", "reasoning_tokens", "unknown_usage_requests",
        )
    }


def _process_case(
    case: dict[str, Any],
    summary: dict[str, Any],
    eps: tuple[str, ...],
    backend: OpenAIResponsesBackend,
    debug_dir: Path | None,
    stage1_max_output_tokens: int,
    stage2_max_output_tokens: int,
    schema_profile: str,
) -> tuple[dict[str, Any], UsageLedger]:
    """Run one case; Stage 2 remains ordered after Stage 1."""
    ledger = UsageLedger()
    case_started = time.monotonic()
    candidates = [*eps, "fiber"]
    stage1_before = dict(ledger.stages.get("stage1", {}))
    stage1_started = time.monotonic()
    first_response = backend.request_json(
        stage1_prompt(summary, schema_profile), stage1_schema(eps, schema_profile), "stage1",
        lambda value: valid_stage1(value, eps, schema_profile),
        case_id=case["case_id"],
        max_output_tokens=stage1_max_output_tokens,
        ledger=ledger,
    )
    stage1_latency = time.monotonic() - stage1_started
    stage1_after = dict(ledger.stages.get("stage1", {}))
    _write_debug_response(debug_dir, f"{case['case_id']}_stage1", first_response.get("raw_text", ""))
    stage1_delta = _ledger_delta(stage1_after, stage1_before)

    if first_response["value"] is None:
        return {
            "case_id": case["case_id"], "endpoints": list(eps),
            "status": "STAGE1_FAILED", "diagnosis_root_cause": None,
            "stage1_attempt": first_response["attempt"], "stage2_attempt": 0,
            "stage1_error": first_response["error"], "stage2_error": "not_run_due_stage1_failure",
            "stage1_parsed": False, "stage2_parsed": False,
            "stage1_input_tokens": stage1_delta["input_tokens"],
            "stage1_output_tokens": stage1_delta["output_tokens"],
            "stage2_input_tokens": 0, "stage2_output_tokens": 0,
            "input_tokens": stage1_delta["input_tokens"],
            "output_tokens": stage1_delta["output_tokens"],
            "total_tokens": stage1_delta["total_tokens"],
            "unknown_usage_requests": stage1_delta["unknown_usage_requests"],
            "api_calls": stage1_delta["calls"],
            "retries": max(0, stage1_delta["calls"] - 1),
            "stage1_latency_seconds": round(stage1_latency, 3), "stage2_latency_seconds": 0.0,
            "latency_seconds": round(time.monotonic() - case_started, 3),
        }, ledger

    first = normalize_stage1(first_response["value"], eps, schema_profile)
    stage2_before = dict(ledger.stages.get("stage2", {}))
    stage2_started = time.monotonic()
    second_response = backend.request_json(
        stage2_prompt(summary, first, schema_profile), stage2_schema(eps, schema_profile), "stage2",
        lambda value: valid_stage2(value, candidates, schema_profile),
        case_id=case["case_id"],
        max_output_tokens=stage2_max_output_tokens,
        ledger=ledger,
    )
    stage2_latency = time.monotonic() - stage2_started
    stage2_after = dict(ledger.stages.get("stage2", {}))
    _write_debug_response(debug_dir, f"{case['case_id']}_stage2", second_response.get("raw_text", ""))
    stage2_delta = _ledger_delta(stage2_after, stage2_before)
    second = normalize_stage2(second_response["value"], candidates, eps[0], schema_profile) if second_response["value"] is not None else None
    if second is not None:
        diagnosis = second["diagnosis_root_cause"]
        status = "SUCCESS"
    else:
        diagnosis = None
        status = "STAGE2_FAILED"
    attempted_stages = 2
    return {
        "case_id": case["case_id"], "endpoints": list(eps),
        "status": status, "diagnosis_root_cause": diagnosis,
        "stage1_attempt": first_response["attempt"], "stage2_attempt": second_response["attempt"],
        "stage1_error": first_response["error"], "stage2_error": second_response["error"],
        "stage1_parsed": True, "stage2_parsed": second_response["value"] is not None,
        "stage1_input_tokens": stage1_delta["input_tokens"],
        "stage1_output_tokens": stage1_delta["output_tokens"],
        "stage2_input_tokens": stage2_delta["input_tokens"],
        "stage2_output_tokens": stage2_delta["output_tokens"],
        "input_tokens": stage1_delta["input_tokens"] + stage2_delta["input_tokens"],
        "output_tokens": stage1_delta["output_tokens"] + stage2_delta["output_tokens"],
        "total_tokens": stage1_delta["total_tokens"] + stage2_delta["total_tokens"],
        "unknown_usage_requests": stage1_delta["unknown_usage_requests"] + stage2_delta["unknown_usage_requests"],
        "api_calls": stage1_delta["calls"] + stage2_delta["calls"],
        "retries": max(0, stage1_delta["calls"] + stage2_delta["calls"] - attempted_stages),
        "stage1_latency_seconds": round(stage1_latency, 3),
        "stage2_latency_seconds": round(stage2_latency, 3),
        "latency_seconds": round(time.monotonic() - case_started, 3),
    }, ledger


def load_split_ids(split_dir: Path, subset: str, all_case_ids: set[str]) -> set[str]:
    if subset == "all":
        return all_case_ids
    filename = "train_case_ids.json" if subset == "train" else "test_case_ids.json"
    ids = set(json.loads((split_dir / filename).read_text(encoding="utf-8")))
    if not ids <= all_case_ids:
        raise ValueError(f"split contains unknown case IDs: {sorted(ids - all_case_ids)[:3]}")
    return ids


def _read_resume(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict) and isinstance(record.get("case_id"), str):
            records[record["case_id"]] = record
    return records


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_debug_response(debug_dir: Path | None, stage: str, raw_text: str) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    safe_text = (raw_text or "").replace(api_key, "[redacted]") if api_key else (raw_text or "")
    (debug_dir / f"{stage}_raw.txt").write_text(safe_text, encoding="utf-8")


def _write_usage(output_dir: Path, ledger: UsageLedger, completed_cases: int) -> None:
    payload = ledger.as_dict(completed_cases)
    (output_dir / "token_usage.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overall = payload["overall"]
    lines = [
        "API token usage (actual response usage when a run is performed)",
        f"Completed cases: {completed_cases}",
    ]
    for stage in ("stage1", "stage2"):
        stats = payload["stages"].get(stage, {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "unknown_usage_requests": 0, "cached_input_tokens": 0,
        })
        lines.append(
            f"{stage}: calls={stats['calls']}, input_tokens={stats['input_tokens']}, "
            f"output_tokens={stats['output_tokens']}, cached_input_tokens={stats.get('cached_input_tokens', 0)}, "
            f"unknown_usage_requests={stats.get('unknown_usage_requests', 0)}"
        )
    lines.extend([
        f"Retries: calls={payload['retries']['calls']}, input_tokens={payload['retries']['input_tokens']}, output_tokens={payload['retries']['output_tokens']}",
        f"Overall: calls={overall['calls']}, input_tokens={overall['input_tokens']}, output_tokens={overall['output_tokens']}, "
        f"total_tokens={overall['total_tokens']}, unknown_usage_requests={overall.get('unknown_usage_requests', 0)}",
    ])
    if completed_cases:
        average = payload["average_per_case"]
        lines.append("Average per case: " + ", ".join(f"{key}={value:.2f}" for key, value in average.items()))
    text = "\n".join(lines) + "\n"
    (output_dir / "token_usage_summary.txt").write_text(text, encoding="utf-8")
    (output_dir / "token_usage.txt").write_text(text, encoding="utf-8")


def _write_reliability_reports(
    output_dir: Path,
    records: list[dict[str, Any]],
    ledger: UsageLedger,
    model: str,
    base_url: str | None,
) -> None:
    payload = ledger.as_dict(len(records))
    request_log_path = output_dir / "request_log.jsonl"
    request_logs = [
        json.loads(line) for line in request_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if request_log_path.exists() else []
    status_counts = Counter(str(record.get("status", "UNKNOWN")) for record in records)
    error_counts = Counter(
        str(item.get("error_category"))
        for item in request_logs
        if item.get("error_category")
    )
    summary = {
        "model": model,
        "base_url": base_url,
        "cases": len(records),
        "successful_cases": status_counts.get("SUCCESS", 0),
        "failed_cases": len(records) - status_counts.get("SUCCESS", 0),
        "api_calls": payload["overall"]["calls"],
        "retries": payload["retries"]["calls"],
        "known_input_tokens": payload["overall"]["input_tokens"],
        "known_output_tokens": payload["overall"]["output_tokens"],
        "known_total_tokens": payload["overall"]["total_tokens"],
        "unknown_usage_requests": payload["overall"].get("unknown_usage_requests", 0),
        "error_counts": dict(sorted(error_counts.items())),
        "per_case": records,
        "full_test_not_started": True,
    }
    (output_dir / "request_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    error_lines = [
        "BiAn API reliability error summary",
        f"Cases: {len(records)}",
        f"Successful cases: {summary['successful_cases']}",
        f"Failed cases: {summary['failed_cases']}",
        "",
        "Request failure categories:",
    ]
    if error_counts:
        error_lines.extend(f"{category}: {count}" for category, count in sorted(error_counts.items()))
    else:
        error_lines.append("none")
    error_lines.extend([
        "",
        f"Unknown-usage requests: {summary['unknown_usage_requests']}",
        "Timeout/transport requests with no provider usage are not counted as zero-consumed.",
        "True labels were not read; full 484-case evaluation was not started.",
    ])
    (output_dir / "error_summary.txt").write_text("\n".join(error_lines) + "\n", encoding="utf-8")
    usage_text = (output_dir / "token_usage.txt").read_text(encoding="utf-8")
    usage_lines = [usage_text.rstrip(), "", "Per-case reliability usage:"]
    for record in records:
        usage_lines.append(
            f"{record.get('case_id')}: status={record.get('status')}, "
            f"input={record.get('input_tokens', 0)}, output={record.get('output_tokens', 0)}, "
            f"total={record.get('total_tokens', 0)}, calls={record.get('api_calls', 0)}, "
            f"retries={record.get('retries', 0)}, latency={record.get('latency_seconds', 0)}s"
        )
    (output_dir / "token_usage.txt").write_text("\n".join(usage_lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = load_cases(args.data_root)
    all_ids = {case["case_id"] for case in all_cases}
    if args.split_dir:
        selected_ids = load_split_ids(args.split_dir, args.subset, all_ids)
    elif args.subset != "all":
        raise ValueError("--split-dir is required unless --subset all is selected")
    else:
        selected_ids = all_ids
    cases = [case for case in all_cases if case["case_id"] in selected_ids]
    summaries = [summarize_case(case["data"]) for case in cases]
    endpoints = [endpoint_keys(case["data"]) for case in cases]

    if args.dry_run:
        rows = []
        for case, summary, eps in zip(cases, summaries, endpoints):
            first = fallback_stage1(eps, args.schema_profile)
            rows.append({
                "case_id": case["case_id"], "endpoints": list(eps),
                "stage1_prompt_chars": len(stage1_prompt(summary, args.schema_profile)),
                "stage2_prompt_chars": len(stage2_prompt(summary, first, args.schema_profile)),
                "schema_profile": args.schema_profile,
                "candidate_diagnoses": [*eps, "fiber"],
            })
        if args.limit is not None:
            rows = rows[:args.limit]
        (args.output_dir / "dry_run_summary.json").write_text(json.dumps({"api_calls": 0, "cases": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"dry-run complete: cases={len(rows)}, api_calls=0")
        return 0

    if not args.model:
        raise ValueError("--model or OPENAI_MODEL is required for a real API run")
    if args.limit is not None:
        cases = cases[:args.limit]
        summaries = summaries[:args.limit]
        endpoints = endpoints[:args.limit]
    if not 1 <= args.workers <= 3:
        raise ValueError("--workers must be between 1 and 3")
    shared_max = args.max_output_tokens
    stage1_max_output_tokens = (
        args.stage1_max_output_tokens if args.stage1_max_output_tokens is not None else shared_max
    )
    stage2_max_output_tokens = (
        args.stage2_max_output_tokens if args.stage2_max_output_tokens is not None else shared_max
    )
    if stage1_max_output_tokens <= 0 or stage2_max_output_tokens <= 0:
        raise ValueError("stage-specific max output tokens must be positive")
    # The checkpoint is the canonical resumable record for each selected run.
    # Keep legacy api_records.jsonl runs readable when explicitly resuming them.
    records_path = args.output_dir / "checkpoint.jsonl"
    if args.resume and not records_path.exists() and (args.output_dir / "api_records.jsonl").exists():
        records_path = args.output_dir / "api_records.jsonl"
    resume_records = _read_resume(records_path) if args.resume else {}
    usage_path = args.output_dir / "token_usage.json"
    ledger = UsageLedger.from_dict(json.loads(usage_path.read_text(encoding="utf-8"))) if args.resume and usage_path.exists() else UsageLedger()
    read_timeout = args.timeout_seconds if args.timeout_seconds is not None else args.read_timeout_seconds
    backend = OpenAIResponsesBackend(
        args.model,
        args.max_output_tokens,
        args.max_retries,
        args.reasoning_effort,
        args.thinking,
        ledger,
        base_url=args.base_url,
        timeout_seconds=read_timeout,
        connect_timeout_seconds=args.connect_timeout_seconds,
        read_timeout_seconds=read_timeout,
        write_timeout_seconds=args.write_timeout_seconds,
        pool_timeout_seconds=args.pool_timeout_seconds,
        request_log_path=args.output_dir / "request_log.jsonl",
        request_start_interval_seconds=args.request_start_interval_seconds,
        request_start_jitter_seconds=args.request_start_jitter_seconds,
        request_start_lock_path=args.request_start_lock_path,
        structured_output_mode=args.structured_output,
    )
    predictions: dict[str, str] = {}
    normalized_records: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]] = []
    for case, summary, eps in zip(cases, summaries, endpoints):
        candidates = [*eps, "fiber"]
        existing = resume_records.get(case["case_id"])
        if (
            existing
            and existing.get("status") == "SUCCESS"
            and tuple(existing.get("endpoints", [])) == eps
            and existing.get("diagnosis_root_cause") in candidates
        ):
            predictions[case["case_id"]] = existing["diagnosis_root_cause"]
            normalized_records.append(existing)
            continue
        pending.append((case, summary, eps))

    def persist_record(record: dict[str, Any], case_ledger: UsageLedger) -> None:
        normalized_records.append(record)
        if record.get("status") == "SUCCESS" and isinstance(record.get("diagnosis_root_cause"), str):
            predictions[record["case_id"]] = record["diagnosis_root_cause"]
        ledger.merge(case_ledger)
        _append_record(records_path, record)
        _write_usage(args.output_dir, ledger, len(predictions))

    if args.workers == 1:
        for case, summary, eps in pending:
            record, case_ledger = _process_case(
                case, summary, eps, backend, args.debug_dir,
                stage1_max_output_tokens, stage2_max_output_tokens,
                args.schema_profile,
            )
            persist_record(record, case_ledger)
    else:
        # One client is initialized before worker creation; request/log and
        # usage updates are protected inside the backend/ledger.
        backend.client
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(
                _process_case, case, summary, eps, backend, args.debug_dir,
                stage1_max_output_tokens, stage2_max_output_tokens,
                args.schema_profile,
            ) for case, summary, eps in pending]
            for future in as_completed(futures):
                record, case_ledger = future.result()
                persist_record(record, case_ledger)

    if args.no_evaluate:
        _write_usage(args.output_dir, ledger, len(predictions))
        _write_reliability_reports(args.output_dir, normalized_records, ledger, args.model, args.base_url)
        print(f"API inference complete: cases={len(predictions)}, api_calls={ledger.as_dict()['overall']['calls']}")
        return 0

    # No ground truth is read until every prediction in this selected run is fixed.
    # Failed cases intentionally have no prediction and are excluded from scoring.
    successful_cases = [case for case in cases if case["case_id"] in predictions]
    truth = truth_after_predictions(successful_cases)
    frame = pd.DataFrame({
        "case_id": [case["case_id"] for case in successful_cases],
        "diagnosis_root_cause": [predictions[case["case_id"]] for case in successful_cases],
        "true_label": [truth[case["case_id"]] for case in successful_cases],
    })
    if frame["case_id"].duplicated().any() or len(frame) != len(successful_cases):
        raise AssertionError("duplicate API prediction set")
    frame.to_csv(args.output_dir / args.results_name, index=False)
    if len(frame):
        (args.output_dir / "scores.txt").write_text(score_frame(frame), encoding="utf-8")
    else:
        (args.output_dir / "scores.txt").write_text("No successful predictions; scores unavailable.\n", encoding="utf-8")
    _write_usage(args.output_dir, ledger, len(predictions))
    _write_evaluation_reports(args.output_dir, normalized_records, len(cases), args.model, args.base_url)
    print((args.output_dir / "scores.txt").read_text(encoding="utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--subset", choices=("train", "test", "all"), default="test")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("CHATANYWHERE_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--output-dir", type=Path, required=True)
    # Covers the largest successful default-thinking completion observed in
    # the validated reliability run (Stage 1 max 7720; Stage 2 max 5383).
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--stage1-max-output-tokens", type=int)
    parser.add_argument("--stage2-max-output-tokens", type=int)
    parser.add_argument("--timeout-seconds", type=float, help="legacy alias for read timeout")
    parser.add_argument("--connect-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=150.0)
    parser.add_argument("--write-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--pool-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--request-start-interval-seconds", type=float, default=0.0,
                        help="Minimum gap between shared-client request starts")
    parser.add_argument("--request-start-jitter-seconds", type=float, default=0.0,
                        help="Optional non-negative random pacing jitter")
    parser.add_argument("--request-start-lock-path", type=Path,
                        help="Optional file lock for pacing across independent shard processes")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--structured-output", choices=("prompt-json", "json-object"), default="json-object",
                        help="Prompt-only JSON or provider json_object mode")
    parser.add_argument("--schema-profile", choices=("compact", "local32b"), default="compact",
                        help="Structured response fields; local32b mirrors the final local vLLM schema")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--thinking", choices=("default", "disabled"), default="default",
                        help="Optional provider-specific thinking mode; default omits the parameter")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Sequential by default; supported range is 1..3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--results-name", default="bian_results.csv")
    parser.add_argument("--no-evaluate", action="store_true", help="Do not reopen JSON labels or write an evaluated CSV")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
