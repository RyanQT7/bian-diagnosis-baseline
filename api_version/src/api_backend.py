"""OpenAI Responses API backend with bounded retries and usage accounting."""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any, Callable


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating a short surrounding explanation."""
    text = (text or "").strip()
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _integer(value: Any) -> int | None:
    try:
        if value is None:
            return None
        number = int(value)
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


class UsageLedger:
    """Accumulate usage returned by the API without storing credentials."""

    def __init__(self) -> None:
        self.stages: dict[str, dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "usage_missing": 0}
        )
        self.retries = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.failed_calls = 0

    @classmethod
    def from_dict(cls, value: Any) -> "UsageLedger":
        ledger = cls()
        if not isinstance(value, dict):
            return ledger
        for stage, stats in value.get("stages", {}).items():
            if isinstance(stats, dict):
                ledger.stages[stage].update({key: int(stats.get(key, 0) or 0) for key in ledger.stages[stage]})
        if isinstance(value.get("retries"), dict):
            ledger.retries.update({key: int(value["retries"].get(key, 0) or 0) for key in ledger.retries})
        ledger.failed_calls = int(value.get("failed_calls", 0) or 0)
        return ledger

    def record(self, stage: str, usage: Any, *, retry: bool = False, failed: bool = False) -> None:
        stats = self.stages[stage]
        stats["calls"] += 1
        if failed:
            self.failed_calls += 1
        input_tokens = _integer(_field(usage, "input_tokens"))
        output_tokens = _integer(_field(usage, "output_tokens"))
        total_tokens = _integer(_field(usage, "total_tokens"))
        if input_tokens is None or output_tokens is None:
            stats["usage_missing"] += 1
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        total_tokens = total_tokens if total_tokens is not None else input_tokens + output_tokens
        stats["input_tokens"] += input_tokens
        stats["output_tokens"] += output_tokens
        stats["total_tokens"] += total_tokens
        if retry:
            self.retries["calls"] += 1
            self.retries["input_tokens"] += input_tokens
            self.retries["output_tokens"] += output_tokens
            self.retries["total_tokens"] += total_tokens

    def as_dict(self, completed_cases: int | None = None) -> dict[str, Any]:
        stages = {stage: dict(stats) for stage, stats in sorted(self.stages.items())}
        total = {key: 0 for key in ("calls", "input_tokens", "output_tokens", "total_tokens")}
        for stats in stages.values():
            for key in total:
                total[key] += stats[key]
        result: dict[str, Any] = {
            "stages": stages,
            "retries": dict(self.retries),
            "overall": {**total, "failed_calls": self.failed_calls},
            "usage_source": "API response usage fields; missing values are reported, not inferred",
        }
        if completed_cases is not None and completed_cases > 0:
            result["average_per_case"] = {
                key: total[key] / completed_cases for key in ("input_tokens", "output_tokens", "total_tokens")
            }
        return result


class OpenAIResponsesBackend:
    """One Responses API request per stage, with at most ``max_retries`` retries."""

    def __init__(
        self,
        model: str,
        max_output_tokens: int,
        max_retries: int,
        reasoning_effort: str | None,
        ledger: UsageLedger,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_retries = min(max(0, max_retries), 2)
        self.reasoning_effort = reasoning_effort
        self.ledger = ledger
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _create(self, prompt: str, schema: dict[str, Any]) -> Any:
        format_name = str(schema.get("title", "bian_diagnosis")).lower().replace(" ", "_")[:64]
        schema_body = {key: value for key, value in schema.items() if key != "title"}
        payload: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": format_name,
                    "strict": True,
                    "schema": schema_body,
                }
            },
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        try:
            return self.client.responses.create(**payload)
        except Exception:
            # Some non-reasoning models reject the optional reasoning object. Retry
            # that request without the optional parameter; all normal retries remain bounded.
            if self.reasoning_effort and "reasoning" in payload:
                payload.pop("reasoning")
                return self.client.responses.create(**payload)
            raise

    @staticmethod
    def _response_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text
        output = getattr(response, "output", None)
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                content = _field(item, "content", [])
                if isinstance(content, list):
                    for part in content:
                        part_text = _field(part, "text")
                        if isinstance(part_text, str):
                            parts.append(part_text)
            return "\n".join(parts)
        return ""

    @staticmethod
    def _safe_error(error: Exception) -> str:
        text = str(error).replace(os.environ.get("OPENAI_API_KEY", ""), "[redacted]")
        return text[:500] or error.__class__.__name__

    def request_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        stage: str,
        validator: Callable[[dict[str, Any]], bool],
    ) -> dict[str, Any]:
        errors: list[str] = []
        for attempt in range(self.max_retries + 1):
            request_prompt = prompt
            if attempt:
                request_prompt += "\nReturn one corrected JSON object matching the schema. Do not add prose."
            try:
                response = self._create(request_prompt, schema)
                self.ledger.record(stage, getattr(response, "usage", None), retry=attempt > 0)
                raw_text = self._response_text(response)
                value = parse_json_object(raw_text)
                if value is not None and validator(value):
                    return {"value": value, "attempt": attempt + 1, "error": None}
                errors.append("JSON parse or schema-value validation failure")
            except Exception as error:  # API errors are recorded and retried a bounded number of times.
                self.ledger.record(stage, None, retry=attempt > 0, failed=True)
                errors.append(self._safe_error(error))
        return {"value": None, "attempt": self.max_retries + 1, "error": "; ".join(errors[-2:])}
