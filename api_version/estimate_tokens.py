#!/usr/bin/env python3
"""Estimate API tokens for the previous 608-case native-endpoint run."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from bian_api import fallback_stage1, load_cases, stage1_prompt, stage2_prompt, summarize_case


def _tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def _percentile(values: list[int], q: float) -> int:
    return int(round(float(np.percentile(np.asarray(values, dtype=float), q)))) if values else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--previous-audit", type=Path, required=True)
    parser.add_argument("--test-case-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    cases = load_cases(args.data_root)
    audit: dict[str, Any] = {}
    if args.previous_audit.exists():
        raw_audit = json.loads(args.previous_audit.read_text(encoding="utf-8"))
        audit = {str(item["case_id"]): item for item in raw_audit.get("cases", []) if isinstance(item, dict) and "case_id" in item}

    stage1_inputs: list[int] = []
    stage2_inputs: list[int] = []
    stage1_outputs: list[int] = []
    stage2_outputs: list[int] = []
    per_case: dict[str, dict[str, int]] = {}
    nonempty_s1: list[int] = []
    nonempty_s2: list[int] = []
    proxy_output_cases = 0
    for case in cases:
        summary = summarize_case(case["data"])
        eps = tuple(summary["endpoints"])
        audit_case = audit.get(case["case_id"], {})
        raw_s1 = str(audit_case.get("stage1_raw_preview") or "")
        raw_s2 = str(audit_case.get("stage2_raw_preview") or "")
        proxy_s1 = json.loads(raw_s1) if raw_s1.strip().startswith("{") else fallback_stage1(eps)
        if not isinstance(proxy_s1, dict):
            proxy_s1 = fallback_stage1(eps)
        s1_in = _tokens(tokenizer, stage1_prompt(summary))
        s2_in = _tokens(tokenizer, stage2_prompt(summary, proxy_s1))
        s1_out = min(_tokens(tokenizer, raw_s1), 256) if raw_s1 else 0
        s2_out = min(_tokens(tokenizer, raw_s2), 256) if raw_s2 else 0
        if s1_out:
            nonempty_s1.append(s1_out)
        if s2_out:
            nonempty_s2.append(s2_out)
        if not s1_out or not s2_out:
            proxy_output_cases += 1
        stage1_inputs.append(s1_in)
        stage2_inputs.append(s2_in)
        stage1_outputs.append(s1_out)
        stage2_outputs.append(s2_out)
        per_case[case["case_id"]] = {"input": s1_in + s2_in, "output": s1_out + s2_out}

    s1_fallback = int(round(statistics.median(nonempty_s1))) if nonempty_s1 else 0
    s2_fallback = int(round(statistics.median(nonempty_s2))) if nonempty_s2 else 0
    # The native-endpoint run did not retain raw responses. Missing proxy outputs
    # are imputed only for this estimate, never for an inference result.
    stage1_outputs = [value or s1_fallback for value in stage1_outputs]
    stage2_outputs = [value or s2_fallback for value in stage2_outputs]
    for case, s1_out, s2_out in zip(cases, stage1_outputs, stage2_outputs):
        per_case[case["case_id"]]["output"] = s1_out + s2_out
        per_case[case["case_id"]]["total"] = per_case[case["case_id"]]["input"] + per_case[case["case_id"]]["output"]

    retry_calls = 0
    base_calls = len(cases) * 2
    input_total = sum(stage1_inputs) + sum(stage2_inputs)
    output_total = sum(stage1_outputs) + sum(stage2_outputs)
    test_ids = set(json.loads(args.test_case_ids.read_text(encoding="utf-8")))
    test_rows = [per_case[case["case_id"]] for case in cases if case["case_id"] in test_ids]
    if len(test_rows) != len(test_ids):
        raise ValueError("time-split test IDs do not match data")
    test_inputs = [row["input"] for row in test_rows]
    test_outputs = [row["output"] for row in test_rows]
    test_totals = [row["total"] for row in test_rows]

    def range_line(label: str, q: float, input_values: list[int], output_values: list[int]) -> str:
        input_est = int(round(_percentile(input_values, q) * len(input_values)))
        output_est = int(round(_percentile(output_values, q) * len(output_values)))
        return f"{label}: input tokens={input_est}, output tokens={output_est}, total tokens={input_est + output_est}"

    lines = [
        "Previous native-endpoint run token estimate (ESTIMATED)",
        "Cases = 608",
        "Tokenizer = DeepSeek-R1-Distill-Qwen-32B local tokenizer; API model tokenizer may differ",
        "Input basis = reconstructed native-endpoint prompts; Stage 2 input uses saved v2 raw-response length as a proxy where native raw output was absent",
        "Output basis = saved compact v2 raw-response token lengths as a proxy, capped at the previous 256-token generation limit; missing previews use the non-empty median",
        f"Proxy output imputed cases = {proxy_output_cases}",
        "",
        "Stage 1:",
        f"calls = {len(cases)}",
        f"estimated input tokens = {sum(stage1_inputs)}",
        f"estimated output tokens = {sum(stage1_outputs)}",
        "",
        "Stage 2:",
        f"calls = {len(cases)}",
        f"estimated input tokens = {sum(stage2_inputs)}",
        f"estimated output tokens = {sum(stage2_outputs)}",
        "",
        "Retries:",
        "calls = 0 central estimate (native-endpoint retry metadata was not saved)",
        "estimated tokens = 0 central estimate; actual API runs record retry usage",
        "",
        "Total:",
        f"model calls = {base_calls}",
        f"input tokens = {input_total}",
        f"output tokens = {output_total}",
        f"total tokens = {input_total + output_total}",
        "",
        "Average per case:",
        f"input tokens/case = {input_total / len(cases):.2f}",
        f"output tokens/case = {output_total / len(cases):.2f}",
        f"total tokens/case = {(input_total + output_total) / len(cases):.2f}",
        f"P50 tokens/case = {_percentile([row['total'] for row in per_case.values()], 50)}",
        f"P95 tokens/case = {_percentile([row['total'] for row in per_case.values()], 95)}",
        "",
        f"Estimated API tokens for new Test split (cases={len(test_rows)}, central base-call estimate):",
        f"estimated input tokens = {sum(test_inputs)}",
        f"estimated output tokens = {sum(test_outputs)}",
        f"estimated total tokens = {sum(test_totals)}",
        "Reasonable range (P50 / mean / P95 per-case distribution; retry overhead excluded):",
        range_line("low / P50", 50, test_inputs, test_outputs),
        f"central / mean: input tokens={sum(test_inputs)}, output tokens={sum(test_outputs)}, total tokens={sum(test_totals)}",
        range_line("high / P95", 95, test_inputs, test_outputs),
        "",
        "Confidence = LOW: native-endpoint raw prompts/responses and retry metadata were not preserved; input reconstruction is direct, output is a documented proxy.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
