#!/usr/bin/env python3
"""Schema and development-only pattern analysis for the case-native BiAn data."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


METRIC_FIELDS = (
    "bias", "rxpower", "txpower", "media_snr", "host_snr", "serdes_snr",
)
STATUS_FIELDS = ("RxLOL", "TxLOL", "TxLOS", "RxLOS")
OTHER_FIELDS = (
    "transmission", "Temperature", "Voltage", "vendor", "vendor_sn",
    "alarm_ip_interface", "alarm_name", "region", "link_location",
    "link_side_ip_interface_map",
)
NUMERIC_METRICS = METRIC_FIELDS + ("Temperature", "Voltage")


def _paths(data_root: Path) -> list[Path]:
    roots = [data_root / "data1", data_root / "data2"]
    if not any(path.is_dir() for path in roots):
        roots = [data_root]
    return sorted(path for root in roots for path in root.glob("*/*.json"))


def _numeric(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        number = float(value)
        return [number] if math.isfinite(number) else []
    if isinstance(value, dict):
        values: list[float] = []
        for child in value.values():
            values.extend(_numeric(child))
        return values
    if isinstance(value, (list, tuple)):
        values: list[float] = []
        for child in value:
            values.extend(_numeric(child))
        return values
    return []


def _nulls(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, dict):
        return sum(_nulls(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nulls(child) for child in value)
    return 0


def _lane_values(value: Any) -> dict[str, list[float]]:
    if isinstance(value, dict):
        lanes = {str(key): _numeric(child) for key, child in value.items()}
        if any(lanes.values()):
            return {key: values for key, values in lanes.items() if values}
    values = _numeric(value)
    return {"aggregate": values} if values else {}


def _shape(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        keys = ",".join(sorted(str(key) for key in value)[:8])
        suffix = ",..." if len(value) > 8 else ""
        return f"object[{keys}{suffix}]"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def _interface_map(data: dict[str, Any]) -> dict[str, Any]:
    value = data.get("link_side_ip_interface_map")
    return value if isinstance(value, dict) else {}


def _endpoints(data: dict[str, Any]) -> tuple[str, ...]:
    mapping = _interface_map(data)
    return tuple(str(key) for key in mapping)


def _load(data_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _paths(data_root):
        raw = json.loads(path.read_text(encoding="utf-8"))
        case_id = path.stem
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        if not isinstance(raw, dict):
            raise ValueError(f"not an object: {path}")
        observable = {key: value for key, value in raw.items() if key != "label"}
        cases.append({
            "case_id": case_id,
            "path": str(path),
            "data": observable,
            "label": raw.get("label"),
        })
    return sorted(cases, key=lambda item: item["case_id"])


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summary(value: Any) -> dict[str, float | int | None]:
    lanes = _lane_values(value)
    values = [number for lane in lanes.values() for number in lane]
    if not values:
        return {"present": 0, "leaves": 0, "lanes": 0, "mean": None, "std": None,
                "min": None, "max": None, "median": None, "q25": None, "q75": None,
                "lane_spread": None, "zero_fraction": None, "negative_fraction": None,
                "minus40_fraction": None, "first_last_delta": None}
    array = np.asarray(values, dtype=float)
    lane_means = [float(np.mean(lane)) for lane in lanes.values() if lane]
    deltas = [lane[-1] - lane[0] for lane in lanes.values() if len(lane) > 1]
    return {
        "present": 1,
        "leaves": len(values),
        "lanes": len(lanes),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "lane_spread": float(max(lane_means) - min(lane_means)) if len(lane_means) > 1 else 0.0,
        "zero_fraction": float(np.mean(array == 0)),
        "negative_fraction": float(np.mean(array < 0)),
        "minus40_fraction": float(np.mean(array == -40)),
        "first_last_delta": _safe_mean(deltas),
    }


def _add_summary(features: dict[str, float], prefix: str, value: Any) -> None:
    summary = _summary(value)
    for key, item in summary.items():
        if isinstance(item, (int, float)) and math.isfinite(float(item)):
            features[f"{prefix}.{key}"] = float(item)
        elif key == "present":
            features[f"{prefix}.present"] = 0.0


def _status_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "up", "down", "loss", "los", "lol"}:
            return 1.0
        if text in {"0", "false", "no", "off", "normal", "ok", "none", "null"}:
            return 0.0
    return None


def _status_stats(value: Any) -> dict[str, float]:
    values: list[float] = []
    if isinstance(value, dict):
        for child in value.values():
            values.extend(number for number in (_status_number(v) for v in _numeric_or_items(child)) if number is not None)
    else:
        number = _status_number(value)
        if number is not None:
            values.append(number)
    return {
        "present": float(bool(values)),
        "mean": float(np.mean(values)) if values else 0.0,
        "nonzero": float(np.mean(np.asarray(values) != 0)) if values else 0.0,
    }


def _numeric_or_items(value: Any) -> list[Any]:
    if isinstance(value, dict):
        result: list[Any] = []
        for child in value.values():
            result.extend(_numeric_or_items(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_numeric_or_items(child))
        return result
    return [value]


def _features(data: dict[str, Any]) -> dict[str, float]:
    endpoints = _endpoints(data)
    features: dict[str, float] = {}
    for endpoint in endpoints:
        for field in NUMERIC_METRICS:
            raw = data.get(field)
            value = raw.get(endpoint) if isinstance(raw, dict) else None
            _add_summary(features, f"endpoint.{endpoint}.{field}", value)
        for field in STATUS_FIELDS:
            raw = data.get(field)
            value = raw.get(endpoint) if isinstance(raw, dict) else None
            stats = _status_stats(value)
            for key, item in stats.items():
                features[f"endpoint.{endpoint}.{field}.{key}"] = item
    for source in endpoints:
        for target in endpoints:
            if source == target:
                continue
            direction = f"{source}-{target}"
            transmission = data.get("transmission")
            tx = data.get("txpower")
            rx = data.get("rxpower")
            _add_summary(features, f"direction.{direction}.transmission", transmission.get(direction) if isinstance(transmission, dict) else None)
            tx_value = tx.get(source) if isinstance(tx, dict) else None
            rx_value = rx.get(target) if isinstance(rx, dict) else None
            tx_summary = _summary(tx_value)
            rx_summary = _summary(rx_value)
            for name, summary in (("tx_source", tx_summary), ("rx_destination", rx_summary)):
                for key in ("present", "mean", "std", "min", "max", "median", "lane_spread", "leaves", "lanes"):
                    item = summary.get(key)
                    if isinstance(item, (int, float)) and math.isfinite(float(item)):
                        features[f"direction.{direction}.{name}.{key}"] = float(item)
            if tx_summary["mean"] is not None and rx_summary["mean"] is not None:
                features[f"direction.{direction}.tx_minus_rx_mean"] = float(tx_summary["mean"] - rx_summary["mean"])
                features[f"direction.{direction}.abs_tx_minus_rx_mean"] = abs(float(tx_summary["mean"] - rx_summary["mean"]))
    if len(endpoints) >= 2:
        left, right = endpoints[0], endpoints[1]
        for field in NUMERIC_METRICS:
            left_summary = _summary(data.get(field, {}).get(left) if isinstance(data.get(field), dict) else None)
            right_summary = _summary(data.get(field, {}).get(right) if isinstance(data.get(field), dict) else None)
            if left_summary["mean"] is not None and right_summary["mean"] is not None:
                diff = float(left_summary["mean"] - right_summary["mean"])
                features[f"cross.{field}.mean_diff"] = diff
                features[f"cross.{field}.abs_mean_diff"] = abs(diff)
    return features


def _schema_section(cases: list[dict[str, Any]]) -> list[str]:
    lines = ["SCHEMA (computed from observable fields only; JSON label was removed)"]
    field_counts: Counter[str] = Counter()
    null_counts: Counter[str] = Counter()
    shape_counts: dict[str, Counter[str]] = defaultdict(Counter)
    endpoint_key_counts: dict[str, Counter[str]] = defaultdict(Counter)
    lane_counts: dict[str, Counter[int]] = defaultdict(Counter)
    exact_values: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        data = case["data"]
        endpoints = _endpoints(data)
        for field, value in data.items():
            field_counts[field] += 1
            null_counts[field] += _nulls(value)
            shape_counts[field][_shape(value)] += 1
            if isinstance(value, dict):
                for endpoint in endpoints:
                    if endpoint in value:
                        endpoint_key_counts[field][endpoint] += 1
            if field in {"alarm_name", "alarm_ip_interface", "region", "link_location", "vendor", "vendor_sn"}:
                if isinstance(value, (str, int, float)) or value is None:
                    exact_values[field][str(value)] += 1
        for field in METRIC_FIELDS + STATUS_FIELDS + ("Temperature", "Voltage"):
            raw = data.get(field)
            if isinstance(raw, dict):
                for endpoint in endpoints:
                    if endpoint in raw:
                        lane_counts[field][len(_lane_values(raw[endpoint]))] += 1
        mapping = _interface_map(data)
        exact_values["endpoint_pair"]["/".join(endpoints)] += 1
        exact_values["interface_pair"][" | ".join(str(mapping.get(ep)) for ep in endpoints)] += 1
        transmission = data.get("transmission")
        if isinstance(transmission, dict):
            exact_values["transmission_keys"][" | ".join(sorted(map(str, transmission)))] += 1
    lines.append(f"cases={len(cases)}")
    lines.append("field presence:")
    for field in sorted(field_counts):
        shapes = ", ".join(f"{key}:{value}" for key, value in shape_counts[field].most_common(4))
        lines.append(f"  {field}: {field_counts[field]}/{len(cases)}; null_leaves={null_counts[field]}; shapes={shapes}")
    lines.append("endpoint key coverage for nested fields:")
    for field in sorted(endpoint_key_counts):
        coverage = ", ".join(f"{key}:{value}" for key, value in endpoint_key_counts[field].most_common())
        lines.append(f"  {field}: {coverage}")
    lines.append("lane-container counts (endpoint values):")
    for field in sorted(lane_counts):
        counts = ", ".join(f"{key}:{value}" for key, value in sorted(lane_counts[field].items()))
        lines.append(f"  {field}: {counts}")
    lines.append("important observed categorical/structural values:")
    for field in ("endpoint_pair", "transmission_keys", "interface_pair", "alarm_name", "alarm_ip_interface", "region", "link_location"):
        if exact_values[field]:
            values = ", ".join(f"{key} ({value})" for key, value in exact_values[field].most_common(12))
            lines.append(f"  {field}: {values}")
    return lines


def _effect_table(records: list[dict[str, Any]], labels: list[str], group_a: set[str], group_b: set[str], title: str) -> list[str]:
    names = sorted({name for record in records for name in record["features"]})
    rows: list[tuple[float, str, float, float, int, int]] = []
    for name in names:
        a = [record["features"][name] for record, label in zip(records, labels) if label in group_a and name in record["features"]]
        b = [record["features"][name] for record, label in zip(records, labels) if label in group_b and name in record["features"]]
        if len(a) < 3 or len(b) < 3:
            continue
        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        scale = float(np.sqrt((np.var(a) + np.var(b)) / 2.0))
        effect = (mean_a - mean_b) / scale if scale > 1e-12 else (0.0 if abs(mean_a - mean_b) < 1e-12 else math.copysign(999.0, mean_a - mean_b))
        rows.append((abs(effect), name, mean_a, mean_b, len(a), len(b)))
    lines = [title, "  feature | group-A mean | group-B mean | standardized difference | nA/nB"]
    for _, name, mean_a, mean_b, n_a, n_b in sorted(rows, reverse=True)[:18]:
        lines.append(f"  {name} | {mean_a:.6g} | {mean_b:.6g} | {(mean_a - mean_b):.6g} | {n_a}/{n_b}")
    if len(lines) == 2:
        lines.append("  insufficient numeric coverage")
    return lines


def _categorical_dev(cases: list[dict[str, Any]], dev_ids: set[str]) -> list[str]:
    dev = [case for case in cases if case["case_id"] in dev_ids]
    lines = ["DEVELOPMENT CATEGORICAL EVIDENCE (descriptive; not hard rules)"]
    for field in ("alarm_name", "alarm_ip_interface", "region", "link_location"):
        by_value: dict[str, Counter[str]] = defaultdict(Counter)
        for case in dev:
            value = str(case["data"].get(field))
            by_value[value][str(case["label"])] += 1
        lines.append(f"{field}:")
        rows = []
        for value, counts in by_value.items():
            total = sum(counts.values())
            fiber = counts.get("fiber", 0)
            rows.append((fiber / total if total else 0.0, total, value, counts))
        for rate, total, value, counts in sorted(rows, reverse=True)[:12]:
            if total >= 2:
                lines.append(f"  {value}: n={total}, fiber_rate={rate:.3f}, labels={dict(counts)}")
    lines.append("alarm endpoint match (exact observable interface lookup):")
    match_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case in dev:
        data = case["data"]
        mapping = _interface_map(data)
        match = any(data.get("alarm_ip_interface") == value for value in mapping.values())
        match_counts["matched" if match else "unmatched_or_null"] += 1
        label_counts["matched" if match else "unmatched_or_null"][str(case["label"])] += 1
    lines.append(f"  {dict(match_counts)}; labels_by_status={dict(label_counts)}")
    return lines


def _discovery_model(records: list[dict[str, Any]], labels: list[str]) -> list[str]:
    names = sorted({name for record in records for name in record["features"]})
    if len(set(labels)) < 2 or len(records) < 20:
        return ["DEVELOPMENT-ONLY DISCOVERY MODEL", "  skipped: insufficient development data"]
    matrix = np.full((len(records), len(names)), np.nan, dtype=float)
    for row, record in enumerate(records):
        for col, name in enumerate(names):
            if name in record["features"]:
                matrix[row, col] = record["features"][name]
    medians = np.nanmedian(matrix, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    missing = ~np.isfinite(matrix)
    matrix[missing] = np.take(medians, np.where(missing)[1])
    model = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=1)
    model.fit(matrix, np.asarray(labels))
    importance = sorted(zip(model.feature_importances_, names), reverse=True)
    lines = ["DEVELOPMENT-ONLY DISCOVERY MODEL (RandomForest feature importance; not the BiAn classifier)"]
    for score, name in importance[:15]:
        lines.append(f"  {name}: {score:.6f}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-fraction", type=float, default=0.25)
    args = parser.parse_args()
    cases = _load(args.data_root)
    if not cases:
        raise SystemExit("no cases")
    labels = [str(case["label"]) for case in cases]
    ids = [case["case_id"] for case in cases]
    dev_ids, holdout_ids, dev_labels, _ = train_test_split(
        ids, labels, test_size=1.0 - args.development_fraction, random_state=args.seed, stratify=labels,
    )
    dev_id_set = set(dev_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "development_fraction": args.development_fraction,
        "development_case_ids": sorted(dev_ids),
        "holdout_case_ids": sorted(holdout_ids),
        "development_label_counts": dict(sorted(Counter(dev_labels).items())),
        "holdout_case_count": len(holdout_ids),
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    records = [{"case_id": case["case_id"], "features": _features(case["data"])} for case in cases if case["case_id"] in dev_id_set]
    records.sort(key=lambda item: item["case_id"])
    record_labels = [str(next(case["label"] for case in cases if case["case_id"] == record["case_id"])) for record in records]
    endpoint_labels = {label for label in record_labels if label != "fiber"}
    endpoint_records = [record for record, label in zip(records, record_labels) if label != "fiber"]
    endpoint_record_labels = [label for label in record_labels if label != "fiber"]
    lines = [
        "BiAn data-driven development analysis",
        f"seed={args.seed}; development_cases={len(dev_ids)}; holdout_cases={len(holdout_ids)}",
        "Holdout labels are not summarized here; they are reserved until after frozen inference.",
        "",
    ]
    lines.extend(_schema_section(cases))
    lines.extend(["", "DEVELOPMENT LABEL COUNTS", f"  {dict(sorted(Counter(record_labels).items()))}"])
    lines.extend(["", *_effect_table(records, record_labels, {"fiber"}, endpoint_labels, "FIBER VS ENDPOINT-FAULT DEVELOPMENT PATTERNS")])
    lines.extend(["", *_effect_table(endpoint_records, endpoint_record_labels, {label for label in endpoint_labels if label == "l1"}, {label for label in endpoint_labels if label != "l1"}, "EXAMPLE NATIVE-ENDPOINT CONTRAST (l1 vs other endpoint labels; descriptive)")])
    lines.extend(["", *_categorical_dev(cases, dev_id_set)])
    lines.extend(["", *_discovery_model(records, record_labels)])
    lines.extend([
        "", "INTERPRETATION GUARDRAILS",
        "  Numeric differences are aggregate development descriptions, not thresholds or production rules.",
        "  Endpoint identifiers remain native case keys; no physical-side or rate-to-label mapping is used.",
        "  The discovery model is not used for BiAn predictions.",
    ])
    (args.output_dir / "development_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"development analysis written: {args.output_dir / 'development_summary.txt'}")
    print(f"development cases: {len(dev_ids)}; holdout cases: {len(holdout_ids)}")
    print(f"development labels: {dict(sorted(Counter(record_labels).items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
