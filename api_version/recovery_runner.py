#!/usr/bin/env python3
"""Bounded technical recovery runner for the API BiAn pipeline.

This runner reuses successful records from a seed run, keeps the existing
native-endpoint prompts/schema, and only changes the output-limit tier for
cases that reached ``finish_reason=length``.  It never reads ``label`` during
inference; evaluation is performed only after inference is complete.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from api_backend import OpenAIResponsesBackend, UsageLedger  # noqa: E402
from bian_api import (  # noqa: E402
    _append_record,
    _ledger_delta,
    _read_resume,
    _write_debug_response,
    endpoint_keys,
    load_cases,
    load_split_ids,
    normalize_stage1,
    normalize_stage2,
    score_frame,
    stage1_prompt,
    stage1_schema,
    stage2_prompt,
    stage2_schema,
    summarize_case,
    truth_after_predictions,
    valid_stage1,
    valid_stage2,
)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_usage(output_dir: Path, ledger: UsageLedger, completed_cases: int) -> None:
    payload = ledger.as_dict(completed_cases)
    (output_dir / "token_usage.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    overall = payload["overall"]
    lines = [
        "API token usage (actual response usage when a run is performed)",
        f"Completed cases: {completed_cases}",
    ]
    for stage in ("stage1", "stage2"):
        stats = payload["stages"].get(stage, {})
        lines.append(
            f"{stage}: calls={stats.get('calls', 0)}, input_tokens={stats.get('input_tokens', 0)}, "
            f"output_tokens={stats.get('output_tokens', 0)}, total_tokens={stats.get('total_tokens', 0)}, "
            f"reasoning_tokens={stats.get('reasoning_tokens', 0)}, "
            f"unknown_usage_requests={stats.get('unknown_usage_requests', 0)}"
        )
    retries = payload["retries"]
    lines.append(
        f"Retries: calls={retries.get('calls', 0)}, input_tokens={retries.get('input_tokens', 0)}, "
        f"output_tokens={retries.get('output_tokens', 0)}, total_tokens={retries.get('total_tokens', 0)}"
    )
    lines.append(
        f"Overall: calls={overall.get('calls', 0)}, input_tokens={overall.get('input_tokens', 0)}, "
        f"output_tokens={overall.get('output_tokens', 0)}, total_tokens={overall.get('total_tokens', 0)}, "
        f"unknown_usage_requests={overall.get('unknown_usage_requests', 0)}"
    )
    if completed_cases:
        average = payload.get("average_per_case", {})
        lines.append("Average per case: " + ", ".join(f"{key}={value:.2f}" for key, value in average.items()))
    text = "\n".join(lines) + "\n"
    (output_dir / "token_usage.txt").write_text(text, encoding="utf-8")


def copy_seed(seed_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        return
    for name in ("checkpoint.jsonl", "api_records.jsonl", "token_usage.json"):
        source = seed_dir / name
        if source.exists():
            shutil.copyfile(source, output_dir / name)


def _stage2_error_category(response: dict[str, Any]) -> str | None:
    return response.get("error_category") or response.get("error")


def process_case(
    case: dict[str, Any],
    summary: dict[str, Any],
    endpoints: tuple[str, ...],
    backend: OpenAIResponsesBackend,
    debug_dir: Path | None,
    stage1_max: int,
    stage2_tiers: list[int],
    schema_profile: str,
    prompt_suffix: str,
    first_tier_name: str,
) -> tuple[dict[str, Any], UsageLedger]:
    """Run Stage 1 once, then Stage 2 through bounded output tiers."""
    local_ledger = UsageLedger()
    case_started = time.monotonic()
    candidates = [*endpoints, "fiber"]

    before = dict(local_ledger.stages.get("stage1", {}))
    stage1_started = time.monotonic()
    first_response = backend.request_json(
        stage1_prompt(summary, schema_profile),
        stage1_schema(endpoints, schema_profile),
        "stage1",
        lambda value: valid_stage1(value, endpoints, schema_profile),
        case_id=case["case_id"],
        max_output_tokens=stage1_max,
        ledger=local_ledger,
    )
    stage1_latency = time.monotonic() - stage1_started
    after = dict(local_ledger.stages.get("stage1", {}))
    _write_debug_response(debug_dir, f"{case['case_id']}_stage1", first_response.get("raw_text", ""))
    stage1_delta = _ledger_delta(after, before)

    base_record = {
        "case_id": case["case_id"],
        "endpoints": list(endpoints),
        "stage1_max_output_tokens": stage1_max,
        "stage2_tier_limits": stage2_tiers,
        "recovery_policy": "BASE->RECOVERY->FINAL on OUTPUT_LIMIT_EXCEEDED only",
        "stage1_attempt": first_response["attempt"],
        "stage1_error": first_response["error"],
        "stage1_parsed": first_response["value"] is not None,
        "stage1_input_tokens": stage1_delta["input_tokens"],
        "stage1_output_tokens": stage1_delta["output_tokens"],
        "stage1_total_tokens": stage1_delta["total_tokens"],
        "stage1_latency_seconds": round(stage1_latency, 3),
    }
    if first_response["value"] is None:
        record = {
            **base_record,
            "status": "STAGE1_FAILED",
            "diagnosis_root_cause": None,
            "stage2_attempt": 0,
            "stage2_error": "not_run_due_stage1_failure",
            "stage2_parsed": False,
            "stage2_input_tokens": 0,
            "stage2_output_tokens": 0,
            "stage2_total_tokens": 0,
            "stage2_latency_seconds": 0.0,
            "stage2_recovery_tier": None,
            "stage2_tier_history": [],
            "input_tokens": stage1_delta["input_tokens"],
            "output_tokens": stage1_delta["output_tokens"],
            "total_tokens": stage1_delta["total_tokens"],
            "unknown_usage_requests": stage1_delta["unknown_usage_requests"],
            "api_calls": stage1_delta["calls"],
            "retries": max(0, stage1_delta["calls"] - 1),
            "latency_seconds": round(time.monotonic() - case_started, 3),
        }
        return record, local_ledger

    first = normalize_stage1(first_response["value"], endpoints, schema_profile)
    stage2_text = stage2_prompt(summary, first, schema_profile)
    if prompt_suffix:
        stage2_text += "\n" + prompt_suffix.strip()
    stage2_schema_value = stage2_schema(endpoints, schema_profile)

    stage2_started = time.monotonic()
    stage2_before = dict(local_ledger.stages.get("stage2", {}))
    responses: list[dict[str, Any]] = []
    tier_history: list[dict[str, Any]] = []
    chosen_tier: int | None = None
    second_response: dict[str, Any] | None = None
    for tier_index, tier in enumerate(stage2_tiers):
        response = backend.request_json(
            stage2_text,
            stage2_schema_value,
            "stage2",
            lambda value: valid_stage2(value, candidates, schema_profile),
            case_id=case["case_id"],
            max_output_tokens=tier,
            ledger=local_ledger,
        )
        responses.append(response)
        category = response.get("error_category")
        tier_history.append({
            "tier": tier,
            "tier_name": first_tier_name if tier_index == 0 else ("RECOVERY" if tier_index == 1 else "FINAL"),
            "attempt": response.get("attempt"),
            "error_category": category,
            "finish_reason": response.get("finish_reason"),
        })
        second_response = response
        if response["value"] is not None:
            chosen_tier = tier
            break
        if category != "OUTPUT_LIMIT_EXCEEDED":
            break
    stage2_latency = time.monotonic() - stage2_started
    stage2_after = dict(local_ledger.stages.get("stage2", {}))
    if second_response is None:
        second_response = {"value": None, "attempt": 0, "error": "not_run", "error_category": "not_run"}
    _write_debug_response(debug_dir, f"{case['case_id']}_stage2", second_response.get("raw_text", ""))
    stage2_delta = _ledger_delta(stage2_after, stage2_before)
    second = (
        normalize_stage2(second_response["value"], candidates, endpoints[0], schema_profile)
        if second_response["value"] is not None else None
    )
    status = "SUCCESS" if second is not None else "STAGE2_FAILED"
    diagnosis = second["diagnosis_root_cause"] if second is not None else None
    record = {
        **base_record,
        "status": status,
        "diagnosis_root_cause": diagnosis,
        "stage2_attempt": sum(int(item.get("attempt") or 0) for item in responses),
        "stage2_error": second_response.get("error"),
        "stage2_error_category": second_response.get("error_category"),
        "stage2_finish_reason": second_response.get("finish_reason"),
        "stage2_parsed": second is not None,
        "stage2_input_tokens": stage2_delta["input_tokens"],
        "stage2_output_tokens": stage2_delta["output_tokens"],
        "stage2_total_tokens": stage2_delta["total_tokens"],
        "stage2_latency_seconds": round(stage2_latency, 3),
        "stage2_recovery_tier": chosen_tier,
        "stage2_tier_history": tier_history,
        "input_tokens": stage1_delta["input_tokens"] + stage2_delta["input_tokens"],
        "output_tokens": stage1_delta["output_tokens"] + stage2_delta["output_tokens"],
        "total_tokens": stage1_delta["total_tokens"] + stage2_delta["total_tokens"],
        "unknown_usage_requests": stage1_delta["unknown_usage_requests"] + stage2_delta["unknown_usage_requests"],
        "api_calls": stage1_delta["calls"] + stage2_delta["calls"],
        "retries": max(0, stage1_delta["calls"] + stage2_delta["calls"] - 2),
        "latency_seconds": round(time.monotonic() - case_started, 3),
    }
    return record, local_ledger


def remaining_stage2_tiers(old_record: dict[str, Any] | None, configured: list[int]) -> list[int]:
    """Do not repeat an already exhausted output tier on resume."""
    if not old_record or old_record.get("status") == "SUCCESS":
        return list(configured)
    history = old_record.get("stage2_tier_history")
    if not isinstance(history, list) or not history:
        return list(configured)
    last = history[-1] if isinstance(history[-1], dict) else {}
    if last.get("error_category") != "OUTPUT_LIMIT_EXCEEDED":
        return list(configured)
    last_tier = int(last.get("tier") or 0)
    return [tier for tier in configured if tier > last_tier]


def write_summary(output_dir: Path, records: list[dict[str, Any]], total_cases: int, args: argparse.Namespace) -> None:
    status = Counter(str(r.get("status", "UNKNOWN")) for r in records)
    successful = status.get("SUCCESS", 0)
    failed = len(records) - successful
    lines = [
        "BiAn API technical recovery run",
        f"Model: {args.model}",
        f"Base URL: {args.base_url or ''}",
        f"Test cases: {total_cases}",
        f"Successful cases: {successful}",
        f"Failed cases: {failed}",
        f"Coverage: {successful / total_cases:.6f}" if total_cases else "Coverage: 0.000000",
        f"Stage 1 max output: {args.stage1_max_output_tokens}",
        f"Stage 2 tiers: {','.join(str(x) for x in args.stage2_tiers)}",
        f"Structured output: {args.structured_output}",
        f"Schema profile: {args.schema_profile}",
        "Ground truth is read only after prediction processing is complete.",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failed_rows = [r for r in records if r.get("status") != "SUCCESS"]
    (output_dir / "failed_cases.json").write_text(json.dumps(failed_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(output_dir: Path, cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    predictions = {str(r["case_id"]): r["diagnosis_root_cause"] for r in records if r.get("status") == "SUCCESS"}
    successful_cases = [case for case in cases if case["case_id"] in predictions]
    # This is the first point at which JSON labels are reopened.
    truth = truth_after_predictions(successful_cases)
    frame = pd.DataFrame({
        "case_id": [case["case_id"] for case in successful_cases],
        "diagnosis_root_cause": [predictions[case["case_id"]] for case in successful_cases],
        "true_label": [truth[case["case_id"]] for case in successful_cases],
    })
    frame.to_csv(output_dir / "bian_results.csv", index=False)
    (output_dir / "scores.txt").write_text(score_frame(frame) if len(frame) else "No successful predictions; scores unavailable.\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if args.stage2_tiers != sorted(args.stage2_tiers) or len(set(args.stage2_tiers)) != len(args.stage2_tiers):
        raise ValueError("--stage2-tiers must be strictly increasing")
    if args.seed_dir:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
            raise ValueError(f"output directory is non-empty; use --resume: {args.output_dir}")
        copy_seed(args.seed_dir, args.output_dir)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = load_cases(args.data_root)
    all_ids = {case["case_id"] for case in all_cases}
    selected_ids = load_split_ids(args.split_dir, args.subset, all_ids) if args.split_dir else all_ids
    cases = [case for case in all_cases if case["case_id"] in selected_ids]
    summaries = [summarize_case(case["data"]) for case in cases]
    endpoints = [endpoint_keys(case["data"]) for case in cases]
    resume_path = args.output_dir / "checkpoint.jsonl"
    resume_records = _read_resume(resume_path) if args.resume or resume_path.exists() else {}
    usage_path = args.output_dir / "token_usage.json"
    ledger = UsageLedger.from_dict(read_json(usage_path, {})) if usage_path.exists() else UsageLedger()

    backend = OpenAIResponsesBackend(
        args.model,
        args.stage1_max_output_tokens,
        args.max_retries,
        None,
        "default",
        ledger,
        base_url=args.base_url,
        timeout_seconds=args.read_timeout_seconds,
        connect_timeout_seconds=args.connect_timeout_seconds,
        read_timeout_seconds=args.read_timeout_seconds,
        write_timeout_seconds=args.write_timeout_seconds,
        pool_timeout_seconds=args.pool_timeout_seconds,
        request_log_path=args.output_dir / "request_log.jsonl",
        request_start_interval_seconds=args.request_start_interval_seconds,
        request_start_jitter_seconds=args.request_start_jitter_seconds,
        structured_output_mode=args.structured_output,
    )

    predictions: dict[str, str] = {}
    records_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        old = resume_records.get(case["case_id"])
        eps = endpoint_keys(case["data"])
        if old and tuple(old.get("endpoints", [])) == eps:
            records_by_id[case["case_id"]] = old
            if old.get("status") == "SUCCESS" and old.get("diagnosis_root_cause") in [*eps, "fiber"]:
                predictions[case["case_id"]] = old["diagnosis_root_cause"]

    pending = []
    for case, summary, eps in zip(cases, summaries, endpoints):
        if case["case_id"] in predictions:
            continue
        old = records_by_id.get(case["case_id"])
        if not remaining_stage2_tiers(old, args.stage2_tiers) and old and old.get("status") != "SUCCESS":
            # The final configured tier was already exhausted in an earlier
            # recovery invocation; preserve that failure without another API call.
            continue
        pending.append((case, summary, eps))
    if args.limit is not None:
        pending = pending[:args.limit]

    for case, summary, eps in pending:
        old = records_by_id.get(case["case_id"])
        tiers = remaining_stage2_tiers(old, args.stage2_tiers)
        record, case_ledger = process_case(
            case, summary, eps, backend, args.debug_dir,
            args.stage1_max_output_tokens, tiers,
            args.schema_profile, args.stage2_prompt_suffix,
            args.first_tier_name,
        )
        records_by_id[case["case_id"]] = record
        if record.get("status") == "SUCCESS":
            predictions[case["case_id"]] = record["diagnosis_root_cause"]
        ledger.merge(case_ledger)
        _append_record(resume_path, record)
        write_usage(args.output_dir, ledger, len(predictions))

    records = [records_by_id[case["case_id"]] for case in cases if case["case_id"] in records_by_id]
    write_usage(args.output_dir, ledger, len(predictions))
    write_summary(args.output_dir, records, len(cases), args)
    if not args.no_evaluate and len(records) == len(cases):
        evaluate(args.output_dir, cases, records)
    elif not args.no_evaluate:
        # A partial run is never scored, preventing feedback during recovery.
        (args.output_dir / "scores.txt").write_text("Inference incomplete; evaluation deferred.\n", encoding="utf-8")
    print(f"processed={len(pending)} successful={len(predictions)} total_records={len(records)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--subset", choices=("train", "test", "all"), default="test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--seed-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1-max-output-tokens", type=int, required=True)
    parser.add_argument("--stage2-tiers", type=int, nargs="+", required=True)
    parser.add_argument("--structured-output", choices=("prompt-json", "json-object"), default="json-object")
    parser.add_argument("--schema-profile", choices=("compact", "local32b"), default="local32b")
    parser.add_argument("--stage2-prompt-suffix", default="")
    parser.add_argument("--first-tier-name", default="RECOVERY")
    parser.add_argument("--connect-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=150.0)
    parser.add_argument("--write-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--pool-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--request-start-interval-seconds", type=float, default=0.0)
    parser.add_argument("--request-start-jitter-seconds", type=float, default=0.0)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--limit", type=int)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
