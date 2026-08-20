"""Small local-vLLM backend and label-free parsing helpers for BiAn v3."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


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
            pass
    return None


class Local32BBackend:
    """Greedy guided-JSON inference using a local tensor-parallel 32B model."""

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
                if len(self.tokenizer.encode(prompt)) > self.max_input_tokens:
                    raise RuntimeError("prompt exceeds max_input_tokens")
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
