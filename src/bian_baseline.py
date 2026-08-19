#!/usr/bin/env python3
"""Two-stage local 32B BiAn-style case diagnosis baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
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

LABELS = ("remote", "local", "fiber")
SIDES = ("l1", "l2", "l3", "l4")
SERIES = ("bias", "rxpower", "txpower", "media_snr", "host_snr", "serdes_snr", "transmission")
STATUS_FIELDS = ("RxLOL", "TxLOL", "TxLOS", "RxLOS")

STAGE1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evidence_summary": {"type": "string"},
        "reasoning": {"type": "string"},
        "hypotheses": {
            "type": "object",
            "properties": {
                label: {
                    "type": "object",
                    "properties": {
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                    },
                    "required": ["confidence", "evidence"],
                    "additionalProperties": False,
                }
                for label in LABELS
            },
            "required": list(LABELS),
            "additionalProperties": False,
        },
        "ranked_diagnoses": {
            "type": "array",
            "items": {"type": "string", "enum": list(LABELS)},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["evidence_summary", "reasoning", "hypotheses", "ranked_diagnoses"],
    "additionalProperties": False,
}

STAGE2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnosis_root_cause": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_summary": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["diagnosis_root_cause", "confidence", "evidence_summary", "reasoning"],
    "additionalProperties": False,
}

V2_STAGE1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evidence_summary": {"type": "string"},
        "reasoning": {"type": "string"},
        "class_scores": {
            "type": "object",
            "properties": {label: {"type": "number"} for label in LABELS},
            "required": list(LABELS),
            "additionalProperties": False,
        },
        "ranked_diagnoses": {"type": "array", "items": {"type": "string", "enum": list(LABELS)}},
    },
    "required": ["evidence_summary", "reasoning", "class_scores", "ranked_diagnoses"],
    "additionalProperties": False,
}

V2_STAGE2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnosis_root_cause": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "number"},
        "evidence_summary": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["diagnosis_root_cause", "confidence", "evidence_summary", "reasoning"],
    "additionalProperties": False,
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _numeric_leaves(value: Any) -> list[float]:
    result: list[float] = []
    if isinstance(value, dict):
        for child in value.values():
            result.extend(_numeric_leaves(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_numeric_leaves(child))
    else:
        number = _number(value)
        if number is not None:
            result.append(number)
    return result


def _status(value: Any) -> float | None:
    number = _number(value)
    if number is not None:
        return number
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "on", "up", "alarm", "异常", "是"}:
            return 1.0
        if value in {"0", "false", "no", "off", "down", "normal", "正常", "否"}:
            return 0.0
    return None


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "delta": None, "slope": None}
    a = np.asarray(values, dtype=float)
    slope = float(np.polyfit(np.arange(len(a), dtype=float), a, 1)[0]) if len(a) > 1 else 0.0
    return {
        "n": int(len(a)),
        "mean": round(float(np.mean(a)), 6),
        "std": round(float(np.std(a)), 6),
        "min": round(float(np.min(a)), 6),
        "max": round(float(np.max(a)), 6),
        "delta": round(float(a[-1] - a[0]), 6),
        "slope": round(slope, 6),
    }


def _values(data: dict[str, Any], metric: str, key: str) -> list[float]:
    outer = data.get(metric)
    return _numeric_leaves(outer.get(key)) if isinstance(outer, dict) else []


def summarize_observations(data: dict[str, Any]) -> dict[str, Any]:
    """Compress observable telemetry; intentionally drops the explicit label."""
    summary: dict[str, Any] = {
        "alarm_name": data.get("alarm_name"),
        "alarm_time": data.get("alarm_time"),
        "region": data.get("region"),
        "link_location": data.get("link_location"),
        "numeric_series": {},
        "status_series": {},
    }
    for metric in SERIES:
        keys = SIDES if metric != "transmission" else ("l1-l2", "l2-l1", "l3-l4", "l4-l3")
        summary["numeric_series"][metric] = {
            key: _stats(_values(data, metric, key)) for key in keys
        }
    for metric in STATUS_FIELDS:
        outer = data.get(metric)
        by_side: dict[str, Any] = {}
        for side in SIDES:
            raw = outer.get(side) if isinstance(outer, dict) else None
            values: list[float] = []
            if isinstance(raw, dict):
                for item in raw.values():
                    parsed = _status(item)
                    if parsed is not None:
                        values.append(parsed)
            else:
                parsed = _status(raw)
                if parsed is not None:
                    values.append(parsed)
            by_side[side] = _stats(values)
        summary["status_series"][metric] = by_side
    # This relationship summary is derived only from observations, not labels.
    summary["side_comparisons"] = {}
    for metric in SERIES:
        sides = summary["numeric_series"][metric]
        left = sides.get("l1", sides.get("l1-l2", {})).get("mean")
        right = sides.get("l2", sides.get("l2-l1", {})).get("mean")
        if left is not None and right is not None:
            summary["side_comparisons"][metric] = round(float(left) - float(right), 6)
    return summary


def load_cases(data_root: Path) -> list[dict[str, Any]]:
    roots = [data_root / "data1", data_root / "data2"]
    if not any(root.is_dir() for root in roots):
        roots = [data_root]
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for label in LABELS:
            directory = root / label
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                case_id = path.stem
                if case_id in seen:
                    raise ValueError(f"duplicate case_id: {case_id}")
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError(f"case is not a JSON object: {path}")
                seen.add(case_id)
                cases.append({"case_id": case_id, "true_label": label, "data": data})
    cases.sort(key=lambda item: item["case_id"])
    if not cases:
        raise ValueError(f"no cases found below {data_root}")
    return cases


def _fallback_ranking(summary: dict[str, Any]) -> list[str]:
    """Observation-only fallback used only when a model response is unusable."""
    scores = {label: 0.0 for label in LABELS}
    if summary.get("alarm_name") == "接口CRC":
        scores["fiber"] += 0.25
    if summary.get("link_location") == "参数面":
        scores["fiber"] += 0.10
    differences = [abs(float(v)) for v in summary.get("side_comparisons", {}).values() if v is not None]
    asymmetry = float(np.mean(differences)) if differences else 0.0
    scores["local"] += asymmetry
    scores["remote"] += max(0.0, 0.5 - asymmetry)
    return sorted(LABELS, key=lambda label: (-scores[label], label))


def _json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"):text.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _json_stop_criteria(tokenizer: Any, prompt_tokens: int):
    """Stop greedy generation as soon as one complete JSON object is present."""
    import torch
    from transformers import StoppingCriteria

    class StopAtObject(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            stop = []
            for row in input_ids:
                text = tokenizer.decode(row[prompt_tokens:].detach().cpu().tolist(), skip_special_tokens=True)
                stop.append(_json_object(text) is not None)
            return torch.tensor(stop, dtype=torch.bool, device=input_ids.device)

    return StopAtObject()


def _valid_stage1(value: Any, fallback: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    hypotheses = value.get("hypotheses") if isinstance(value.get("hypotheses"), dict) else {}
    ranking = [item for item in value.get("ranked_diagnoses", []) if item in LABELS] if isinstance(value.get("ranked_diagnoses"), list) else []
    ranking = list(dict.fromkeys(ranking)) + [item for item in fallback if item not in ranking]
    return {
        "evidence_summary": str(value.get("evidence_summary", "model response unavailable; use observable summary")),
        "reasoning": str(value.get("reasoning", "deterministic parsing fallback")),
        "hypotheses": {
            label: {
                "confidence": float(hypotheses.get(label, {}).get("confidence", 0.0)) if isinstance(hypotheses.get(label), dict) else 0.0,
                "evidence": [str(item) for item in hypotheses.get(label, {}).get("evidence", [])[:4]] if isinstance(hypotheses.get(label), dict) and isinstance(hypotheses.get(label, {}).get("evidence", []), list) else [],
            }
            for label in LABELS
        },
        "ranked_diagnoses": ranking[:3],
    }


def _valid_stage2(value: Any, fallback: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    diagnosis = value.get("diagnosis_root_cause")
    if diagnosis not in LABELS:
        diagnosis = fallback
    return {
        "diagnosis_root_cause": diagnosis,
        "confidence": float(value.get("confidence", 0.0)) if isinstance(value.get("confidence", 0.0), (int, float)) else 0.0,
        "evidence_summary": str(value.get("evidence_summary", "deterministic parsing fallback")),
        "reasoning": str(value.get("reasoning", "deterministic parsing fallback")),
    }


class Local32BBackend:
    def __init__(self, model_path: Path, gpu_ids: str, seed: int, max_input_tokens: int, max_output_tokens: int, max_num_seqs: int) -> None:
        self.model_path = model_path
        self.gpu_ids = [item.strip() for item in gpu_ids.split(",") if item.strip()]
        self.seed = seed
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_num_seqs = max_num_seqs
        self.engine: Any = None
        self.tokenizer: Any = None
        self.call_count = 0
        self.retry_count = 0
        self.load_seconds: float | None = None

    def load(self) -> None:
        if self.engine is not None:
            return
        if self.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_ids)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        cache_root = Path(__file__).resolve().parents[2] / "bian-diagnosis-baseline" / "outputs" / ".cache"
        # Keep vLLM's small hardware probe cache inside this repository.
        os.environ.setdefault("VLLM_CACHE_ROOT", str(cache_root / "vllm"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        from vllm import LLM
        started = time.perf_counter()
        self.engine = LLM(
            model=str(self.model_path),
            tokenizer=str(self.model_path),
            trust_remote_code=False,
            tensor_parallel_size=max(1, len(self.gpu_ids)),
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=self.max_input_tokens + self.max_output_tokens,
            max_num_seqs=self.max_num_seqs,
            enforce_eager=True,
            # With this host topology vLLM's P2P probe disables its custom
            # all-reduce; NCCL over shared memory remains reliable.
            disable_custom_all_reduce=False,
            seed=self.seed,
            enable_prefix_caching=True,
        )
        self.tokenizer = self.engine.get_tokenizer()
        self.load_seconds = time.perf_counter() - started

    def _render(self, content: str) -> str:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered + ("</think>\n" if rendered.rstrip().endswith("<think>") else "<think>\n</think>\n")

    def generate_json(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.load()
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        pending = list(range(len(requests)))
        results: list[dict[str, Any] | None] = [None] * len(requests)
        errors: dict[int, str] = {}
        for attempt in range(2):
            if not pending:
                break
            prompts: list[str] = []
            params: list[Any] = []
            for index in pending:
                request = requests[index]
                content = request["prompt"]
                if index in errors:
                    content += "\n\nReturn one corrected JSON object matching the requested schema."
                prompt = self._render(content)
                token_count = len(self.tokenizer.encode(prompt))
                if token_count > self.max_input_tokens:
                    raise RuntimeError(f"prompt exceeds max_input_tokens: {token_count}")
                prompts.append(prompt)
                params.append(SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    seed=self.seed,
                    max_tokens=self.max_output_tokens,
                    guided_decoding=GuidedDecodingParams(json=request["schema"]),
                ))
            outputs = self.engine.generate(prompts, params, use_tqdm=False)
            next_pending: list[int] = []
            for position, index in enumerate(pending):
                self.call_count += 1
                text = outputs[position].outputs[0].text if outputs[position].outputs else ""
                parsed = _json_object(text)
                if parsed is None:
                    errors[index] = "JSON parse failure"
                    next_pending.append(index)
                else:
                    results[index] = {"value": parsed, "raw_preview": text[-1200:], "attempt": attempt + 1, "error": None}
            self.retry_count += len(next_pending)
            pending = next_pending
        for index in pending:
            results[index] = {"value": None, "raw_preview": "", "attempt": 2, "error": errors.get(index, "generation failure")}
        return [item for item in results if item is not None]


def score_frame(frame: pd.DataFrame) -> str:
    y_true = frame["true_label"].tolist()
    y_pred = frame["diagnosis_root_cause"].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    report = classification_report(y_true, y_pred, labels=list(LABELS), target_names=list(LABELS), digits=6, zero_division=0)
    lines = [
        "BiAn two-stage 32B scores",
        f"Accuracy: {accuracy_score(y_true, y_pred):.6f}",
        f"Macro Precision: {precision_score(y_true, y_pred, labels=list(LABELS), average='macro', zero_division=0):.6f}",
        f"Macro Recall: {recall_score(y_true, y_pred, labels=list(LABELS), average='macro', zero_division=0):.6f}",
        f"Macro F1: {f1_score(y_true, y_pred, labels=list(LABELS), average='macro', zero_division=0):.6f}",
        "",
        report.rstrip(),
        "",
        "Confusion Matrix (rows=true, columns=predicted; order=remote,local,fiber):",
        "              remote  local  fiber",
    ]
    lines.extend(f"{label:>12}  " + "  ".join(f"{int(value):>6}" for value in row) for label, row in zip(LABELS, cm))
    return "\n".join(lines) + "\n"


def _stage1_prompt(summary: dict[str, Any]) -> str:
    return (
        "You are Stage 1 of a hierarchical network-failure diagnosis. Analyze only the observable case summary below. "
        "The task is exactly one root-cause diagnosis among remote, local, fiber. Compare the three hypotheses using "
        "metric changes, side/location patterns, temporal slope/delta, status evidence, and relationships between metrics. "
        "Do not invent measurements. Return only the requested JSON object.\n\n"
        "OUTPUT SCHEMA: evidence_summary:string; reasoning:string; hypotheses:{remote:{confidence:0..1,evidence:[string]}, "
        "local:{confidence:0..1,evidence:[string]}, fiber:{confidence:0..1,evidence:[string]}}; "
        "ranked_diagnoses:[remote,local,fiber].\n\nOBSERVABLE_CASE_SUMMARY:\n" +
        json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def _stage2_prompt(summary: dict[str, Any], stage1: dict[str, Any]) -> str:
    return (
        "You are Stage 2 of a hierarchical network-failure diagnosis. Reconcile the Stage 1 analysis with the "
        "observable telemetry summary and make the final case-level diagnosis. Choose exactly one of remote, local, "
        "fiber. Weigh side asymmetry, common-mode versus directional effects, temporal behavior, status transitions, "
        "and metric relationships. Do not use any hidden label or filename. Return only the requested JSON object.\n\n"
        "OUTPUT SCHEMA: diagnosis_root_cause must be remote, local, or fiber; confidence:0..1; "
        "evidence_summary:string; reasoning:string.\n\nSTAGE1_ANALYSIS:\n" +
        json.dumps(stage1, ensure_ascii=False, separators=(",", ":")) +
        "\n\nOBSERVABLE_CASE_SUMMARY:\n" +
        json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def _stage1_prompt_v2(summary: dict[str, Any]) -> str:
    return (
        "Stage 1: diagnose one network case as remote, local, or fiber using only the observable summary. "
        "Compare location/side asymmetry, temporal delta/slope, optical power/SNR/transmission coupling, and status alarms. "
        "Remote means evidence centered beyond the local endpoint; local means endpoint-local behavior; fiber means path/media evidence. "
        "Do not invent data. Keep evidence_summary and reasoning concise (at most 40 words each). Return JSON only with "
        "evidence_summary, reasoning, class_scores {remote,local,fiber}, and ranked_diagnoses containing each class once.\n"
        "OBSERVATIONS:\n" + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def _stage2_prompt_v2(summary: dict[str, Any], stage1: dict[str, Any]) -> str:
    return (
        "Stage 2: independently verify the Stage 1 ranking against the observable summary, then choose exactly remote, local, or fiber. "
        "Prioritize consistent cross-metric, side/location, and temporal evidence; do not use filenames or hidden labels. "
        "Keep evidence_summary and reasoning concise (at most 40 words each). Return JSON only with diagnosis_root_cause, "
        "confidence, evidence_summary, reasoning.\nSTAGE1:\n" +
        json.dumps(stage1, ensure_ascii=False, separators=(",", ":")) +
        "\nOBSERVATIONS:\n" + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )


def _valid_stage1_v2(value: Any, fallback: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _valid_stage1(value, fallback)
    raw_scores = value.get("class_scores", {}) if isinstance(value.get("class_scores"), dict) else {}
    scores = {label: float(raw_scores.get(label, 0.0)) for label in LABELS}
    ranking = [item for item in value.get("ranked_diagnoses", []) if item in LABELS] if isinstance(value.get("ranked_diagnoses"), list) else []
    ranking = list(dict.fromkeys(ranking))
    ranking.extend(label for label in sorted(LABELS, key=lambda item: (-scores[item], item)) if label not in ranking)
    return {
        "evidence_summary": str(value.get("evidence_summary", "")),
        "reasoning": str(value.get("reasoning", "")),
        "hypotheses": {label: {"confidence": scores[label], "evidence": []} for label in LABELS},
        "ranked_diagnoses": ranking[:3] if ranking else fallback,
    }


def run(args: argparse.Namespace) -> int:
    cases = load_cases(args.data_root)
    if not args.model_path.is_dir():
        raise ValueError(f"model path does not exist: {args.model_path}")
    backend = Local32BBackend(args.model_path, args.gpu_ids, args.seed, args.max_input_tokens, args.max_output_tokens, args.max_num_seqs)
    is_v2 = args.prompt_version == "v2"
    stage1_prompt = _stage1_prompt_v2 if is_v2 else _stage1_prompt
    stage2_prompt = _stage2_prompt_v2 if is_v2 else _stage2_prompt
    stage1_schema = V2_STAGE1_SCHEMA if is_v2 else STAGE1_SCHEMA
    stage2_schema = V2_STAGE2_SCHEMA if is_v2 else STAGE2_SCHEMA
    stage1_validator = _valid_stage1_v2 if is_v2 else _valid_stage1
    if args.smoke_test:
        smoke_case = cases[0]
        summary = summarize_observations(smoke_case["data"])
        stage1_raw = backend.generate_json([{"prompt": stage1_prompt(summary), "schema": stage1_schema}])[0]
        stage1 = stage1_validator(stage1_raw["value"], _fallback_ranking(summary))
        stage2_raw = backend.generate_json([{"prompt": stage2_prompt(summary, stage1), "schema": stage2_schema}])[0]
        stage2 = _valid_stage2(stage2_raw["value"], stage1["ranked_diagnoses"][0])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "smoke_test.json").write_text(json.dumps({"model": "DeepSeek-R1-Distill-Qwen-32B", "stage1": stage1, "stage2": stage2, "calls": backend.call_count}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"32B smoke passed: {backend.call_count} calls, load={backend.load_seconds:.2f}s")
        return 0

    summaries = [summarize_observations(case["data"]) for case in cases]
    stage1_requests = [{"prompt": stage1_prompt(summary), "schema": stage1_schema} for summary in summaries]
    stage1_raw = backend.generate_json(stage1_requests)
    stage1_values = [stage1_validator(item["value"], _fallback_ranking(summary)) for item, summary in zip(stage1_raw, summaries)]
    stage2_requests = [{"prompt": stage2_prompt(summary, stage1), "schema": stage2_schema} for summary, stage1 in zip(summaries, stage1_values)]
    stage2_raw = backend.generate_json(stage2_requests)
    final_values = [_valid_stage2(item["value"], stage1["ranked_diagnoses"][0]) for item, stage1 in zip(stage2_raw, stage1_values)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The audit contains predictions and model responses only. Ground truth is
    # attached below, after inference has completely finished.
    audit_cases = []
    for case, s1, s2, r1, r2 in zip(cases, stage1_values, final_values, stage1_raw, stage2_raw):
        audit_cases.append({
            "case_id": case["case_id"],
            "stage1": s1,
            "stage2": s2,
            "stage1_raw_preview": r1["raw_preview"],
            "stage2_raw_preview": r2["raw_preview"],
            "stage1_error": r1["error"],
            "stage2_error": r2["error"],
        })
    audit = {
        "model": "DeepSeek-R1-Distill-Qwen-32B",
        "prompt_version": args.prompt_version,
        "stages": {"stage1_cases": len(stage1_values), "stage2_cases": len(final_values), "model_calls": backend.call_count, "retries": backend.retry_count},
        "cases": audit_cases,
    }
    (args.output_dir / "inference_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frame = pd.DataFrame({
        "case_id": [case["case_id"] for case in cases],
        "diagnosis_root_cause": [item["diagnosis_root_cause"] for item in final_values],
        "true_label": [case["true_label"] for case in cases],
    })
    if frame["case_id"].duplicated().any() or set(frame["diagnosis_root_cause"]) - set(LABELS):
        raise AssertionError("invalid BiAn result frame")
    frame.to_csv(args.output_dir / "bian_results.csv", index=False)
    (args.output_dir / "scores.txt").write_text(score_frame(frame), encoding="utf-8")
    (args.output_dir / "experiment_metadata.json").write_text(json.dumps({"model": "DeepSeek-R1-Distill-Qwen-32B", "prompt_version": args.prompt_version, "load_seconds": backend.load_seconds, "n_cases": len(cases), "stage1_cases": len(stage1_values), "stage2_cases": len(final_values), "model_calls": backend.call_count, "retries": backend.retry_count}, indent=2) + "\n", encoding="utf-8")
    print(f"BiAn complete: {len(frame)} cases; model calls={backend.call_count}")
    print((args.output_dir / "scores.txt").read_text(encoding="utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--gpu-ids", default="4,5", help="physical GPU ids, e.g. 4,5")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--prompt-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--smoke-test", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
