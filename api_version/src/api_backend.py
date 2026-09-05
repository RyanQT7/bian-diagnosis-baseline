"""OpenAI-compatible Chat Completions backend with bounded retries and accounting."""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _balanced_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def parse_json_object(text: str, *, prefer_last: bool = False) -> dict[str, Any] | None:
    """Parse JSON while tolerating a fence or short surrounding explanation."""
    text = (text or "").strip()
    unfenced = text
    fence = chr(96) * 3
    if unfenced.startswith(fence):
        lines = unfenced.splitlines()
        if lines and lines[0].lstrip().startswith(fence):
            lines = lines[1:]
        if lines and lines[-1].strip() == fence:
            lines = lines[:-1]
        unfenced = "\n".join(lines).strip()
    candidates = [text, unfenced]
    objects = _balanced_objects(unfenced)
    candidates.extend(reversed(objects) if prefer_last else objects)
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
        if value is None or isinstance(value, bool):
            return None
        number = int(value)
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


def _usage_values(usage: Any) -> dict[str, int | bool]:
    input_tokens = _integer(_field(usage, "input_tokens"))
    if input_tokens is None:
        input_tokens = _integer(_field(usage, "prompt_tokens"))
    output_tokens = _integer(_field(usage, "output_tokens"))
    if output_tokens is None:
        output_tokens = _integer(_field(usage, "completion_tokens"))
    total_tokens = _integer(_field(usage, "total_tokens"))
    unknown = usage is None or input_tokens is None or output_tokens is None
    input_value = input_tokens or 0
    output_value = output_tokens or 0
    total_value = total_tokens if total_tokens is not None else input_value + output_value
    input_details = _field(usage, "input_tokens_details") or _field(usage, "prompt_tokens_details")
    output_details = _field(usage, "output_tokens_details") or _field(usage, "completion_tokens_details")
    cached_tokens = _integer(_field(input_details, "cached_tokens")) or 0
    reasoning_tokens = _integer(_field(output_details, "reasoning_tokens")) or 0
    return {
        "input_tokens": input_value,
        "output_tokens": output_value,
        "total_tokens": total_value,
        "cached_input_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "unknown": unknown,
    }


def _exception_chain(error: Exception) -> list[BaseException]:
    chain: list[BaseException] = []
    pending: BaseException | None = error
    seen: set[int] = set()
    while pending is not None and id(pending) not in seen and len(chain) < 8:
        chain.append(pending)
        seen.add(id(pending))
        pending = pending.__cause__ or pending.__context__
    return chain


def _http_status(error: Exception) -> int | None:
    for item in _exception_chain(error):
        status = _integer(getattr(item, "status_code", None))
        if status is None:
            status = _integer(_field(getattr(item, "response", None), "status_code"))
        if status is not None:
            return status
    return None


def classify_exception(error: Exception) -> str:
    """Classify transport/provider errors without depending on SDK internals."""
    status = _http_status(error)
    if status == 429:
        return "HTTP_429"
    if status is not None and 500 <= status <= 599:
        return "HTTP_5XX"
    if status is not None and 400 <= status <= 499:
        return "HTTP_4XX"
    names = " ".join(type(item).__name__.lower() for item in _exception_chain(error))
    if "pooltimeout" in names or "pool_timeout" in names:
        return "POOL_TIMEOUT"
    if "connecttimeout" in names or "connect_timeout" in names:
        return "CONNECT_TIMEOUT"
    if "writetimeout" in names or "write_timeout" in names:
        return "WRITE_TIMEOUT"
    if "readtimeout" in names or "read_timeout" in names:
        return "READ_TIMEOUT"
    if "apitimeout" in names or "timeout" in names:
        return "TOTAL_TIMEOUT"
    if "connectionerror" in names or "connection" in names:
        return "CONNECT_ERROR"
    return "OTHER"


def _retryable(category: str) -> bool:
    return category in {
        "CONNECT_TIMEOUT", "READ_TIMEOUT", "WRITE_TIMEOUT", "POOL_TIMEOUT",
        "TOTAL_TIMEOUT", "CONNECT_ERROR", "HTTP_429", "HTTP_5XX",
        "UPSTREAM_ERROR", "EMPTY_RESPONSE", "TRUNCATED_RESPONSE",
    }


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "value"):
            result = _text_value(value.get(key))
            if result:
                return result
    return ""


class RequestStartLimiter:
    """Serialize request starts with an optional minimum interval."""

    def __init__(self, interval_seconds: float = 0.0, jitter_seconds: float = 0.0) -> None:
        if interval_seconds < 0 or jitter_seconds < 0:
            raise ValueError("request pacing values must be non-negative")
        self.interval_seconds = float(interval_seconds)
        self.jitter_seconds = float(jitter_seconds)
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._random = random.Random(42)

    def wait(self) -> None:
        if self.interval_seconds <= 0:
            return
        # Holding the lock through the sleep makes the release immediately
        # precede the request start for the current worker.  Later workers
        # reserve their start only after that release.
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            if delay:
                time.sleep(delay)
            gap = self.interval_seconds
            if self.jitter_seconds:
                gap += self._random.uniform(0.0, self.jitter_seconds)
            self._next_allowed = time.monotonic() + gap


class UsageLedger:
    """Accumulate provider usage and distinguish unknown usage from zero."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stages: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "unknown_usage_requests": 0,
                "usage_missing": 0,
            }
        )
        self.retries = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "unknown_usage_requests": 0,
        }
        self.failed_calls = 0

    @classmethod
    def from_dict(cls, value: Any) -> "UsageLedger":
        ledger = cls()
        if not isinstance(value, dict):
            return ledger
        for stage, stats in value.get("stages", {}).items():
            if isinstance(stats, dict):
                for key in ledger.stages[stage]:
                    parsed = _integer(stats.get(key, 0))
                    ledger.stages[stage][key] = parsed or 0
        if isinstance(value.get("retries"), dict):
            for key in ledger.retries:
                parsed = _integer(value["retries"].get(key, 0))
                ledger.retries[key] = parsed or 0
        overall = value.get("overall", {})
        ledger.failed_calls = _integer(
            value.get("failed_calls", overall.get("failed_calls", 0) if isinstance(overall, dict) else 0)
        ) or 0
        return ledger

    def record(self, stage: str, usage: Any, *, retry: bool = False, failed: bool = False) -> None:
        with self._lock:
            stats = self.stages[stage]
            stats["calls"] += 1
            if failed:
                self.failed_calls += 1
            values = _usage_values(usage)
            if values["unknown"]:
                stats["unknown_usage_requests"] += 1
                stats["usage_missing"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"):
                stats[key] += int(values[key])
            if retry:
                self.retries["calls"] += 1
                for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"):
                    self.retries[key] += int(values[key])
                if values["unknown"]:
                    self.retries["unknown_usage_requests"] += 1

    def as_dict(self, completed_cases: int | None = None) -> dict[str, Any]:
        with self._lock:
            stages = {stage: dict(stats) for stage, stats in sorted(self.stages.items())}
            retries = dict(self.retries)
            failed_calls = self.failed_calls
        aggregate_keys = (
            "calls", "input_tokens", "output_tokens", "total_tokens",
            "cached_input_tokens", "reasoning_tokens", "unknown_usage_requests", "usage_missing",
        )
        total = {key: 0 for key in aggregate_keys}
        for stats in stages.values():
            for key in aggregate_keys:
                total[key] += stats.get(key, 0)
        result: dict[str, Any] = {
            "stages": stages,
            "retries": retries,
            "overall": {**total, "failed_calls": failed_calls},
            "usage_source": "API response usage fields; timeout/transport usage is unknown, not zero-consumed",
        }
        if completed_cases is not None and completed_cases > 0:
            result["average_per_case"] = {
                key: total[key] / completed_cases
                for key in ("input_tokens", "output_tokens", "total_tokens")
            }
        return result

    def merge(self, other: "UsageLedger") -> None:
        """Merge one case-local ledger into this run ledger."""
        snapshot = other.as_dict()
        with self._lock:
            for stage, stats in snapshot.get("stages", {}).items():
                target = self.stages[stage]
                for key in target:
                    target[key] += int(stats.get(key, 0))
            for key in self.retries:
                self.retries[key] += int(snapshot.get("retries", {}).get(key, 0))
            self.failed_calls += int(snapshot.get("overall", {}).get("failed_calls", 0))


class OpenAIResponsesBackend:
    """OpenAI-compatible Chat Completions backend with one reusable client."""

    def __init__(
        self,
        model: str,
        max_output_tokens: int,
        max_retries: int,
        reasoning_effort: str | None,
        thinking_mode: str | None,
        ledger: UsageLedger,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 20.0,
        read_timeout_seconds: float | None = None,
        write_timeout_seconds: float = 60.0,
        pool_timeout_seconds: float = 30.0,
        request_log_path: Path | None = None,
        request_start_interval_seconds: float = 0.0,
        request_start_jitter_seconds: float = 0.0,
        structured_output_mode: str = "json-object",
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_transport_retries = min(max(0, max_retries), 1)
        self.max_format_retries = min(max(0, max_retries), 1)
        self.reasoning_effort = reasoning_effort
        if thinking_mode not in {None, "default", "disabled"}:
            raise ValueError("thinking_mode must be default, disabled, or None")
        self.thinking_mode = None if thinking_mode in {None, "default"} else thinking_mode
        self.ledger = ledger
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds if read_timeout_seconds is not None else timeout_seconds
        self.write_timeout_seconds = write_timeout_seconds
        self.pool_timeout_seconds = pool_timeout_seconds
        if structured_output_mode not in {"prompt-json", "json-object"}:
            raise ValueError("structured_output_mode must be prompt-json or json-object")
        self.structured_output_mode = structured_output_mode
        self.request_log_path = request_log_path
        self._request_start_limiter = RequestStartLimiter(
            request_start_interval_seconds, request_start_jitter_seconds
        )
        self._client: Any = None
        self._client_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._random = random.Random(42)

    @property
    def client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    if not os.environ.get("OPENAI_API_KEY"):
                        raise RuntimeError("OPENAI_API_KEY is not set")
                    from openai import OpenAI

                    kwargs: dict[str, Any] = {
                        "api_key": os.environ["OPENAI_API_KEY"],
                        "max_retries": 0,
                    }
                    selected_base_url = self.base_url or os.environ.get("CHATANYWHERE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                    if selected_base_url:
                        kwargs["base_url"] = selected_base_url.rstrip("/")
                    try:
                        import httpx
                        timeout = httpx.Timeout(
                            connect=self.connect_timeout_seconds,
                            read=self.read_timeout_seconds,
                            write=self.write_timeout_seconds,
                            pool=self.pool_timeout_seconds,
                        )
                        kwargs["timeout"] = timeout
                        # The API subprocess must not inherit the Codex
                        # session's proxy.  Keep this explicit in addition
                        # to the direct launcher so embedded callers are
                        # direct as well.
                        kwargs["http_client"] = httpx.Client(
                            timeout=timeout, trust_env=False, follow_redirects=True
                        )
                    except ImportError:
                        kwargs["timeout"] = self.read_timeout_seconds
                    # ChatAnywhere/DeepSeek compatibility: no optional reasoning_effort
                    # or provider-specific JSON-schema parameter is sent.
                    self._client = OpenAI(**kwargs)
        return self._client

    def _create(
        self,
        prompt: str,
        schema: dict[str, Any],
        max_output_tokens: int | None = None,
    ) -> Any:
        del schema
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens if max_output_tokens is not None else self.max_output_tokens,
        }
        if self.structured_output_mode == "json-object":
            payload["response_format"] = {"type": "json_object"}
        if self.thinking_mode == "disabled":
            # ChatAnywhere/DeepSeek-specific extra body; omitted in default mode.
            payload["extra_body"] = {"thinking": {"type": "disabled"}}
        return self.client.chat.completions.create(**payload)

    @staticmethod
    def _response_text(response: Any) -> tuple[str, bool]:
        if isinstance(response, str):
            return response, False
        choices = _field(response, "choices", [])
        if isinstance(choices, list) and choices:
            message = _field(choices[0], "message", {})
            content = _text_value(_field(message, "content", ""))
            if content.strip():
                return content, False
            reasoning = _text_value(
                _field(message, "reasoning_content", _field(message, "reasoning", ""))
            )
            if reasoning.strip():
                return reasoning, True
            choice_text = _text_value(_field(choices[0], "text", ""))
            if choice_text.strip():
                return choice_text, False
        return "", False

    @staticmethod
    def _reasoning_metadata(response: Any) -> dict[str, bool]:
        """Record provider-visible reasoning fields without persisting their text."""
        choices = _field(response, "choices", [])
        if not isinstance(choices, list) or not choices:
            return {"reasoning_content_present": False, "reasoning_content_nonempty": False}
        message = _field(choices[0], "message", {})
        marker = _field(message, "reasoning_content", None)
        if marker is None:
            marker = _field(message, "reasoning", None)
        return {
            "reasoning_content_present": marker is not None,
            "reasoning_content_nonempty": bool(_text_value(marker).strip()),
        }

    @staticmethod
    def _finish_reason(response: Any) -> str | None:
        choices = _field(response, "choices", [])
        if isinstance(choices, list) and choices:
            value = _field(choices[0], "finish_reason")
            return str(value) if value is not None else None
        return None

    @staticmethod
    def _safe_error(error: Exception) -> str:
        text = str(error)
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            text = text.replace(key, "[redacted]")
        return text[:500] or error.__class__.__name__

    def _append_request_log(self, payload: dict[str, Any]) -> None:
        if self.request_log_path is None:
            return
        with self._log_lock:
            self.request_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.request_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _backoff(self, retry_number: int) -> None:
        base = 2.0 if retry_number <= 1 else 5.0
        time.sleep(base + self._random.random())

    def request_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        stage: str,
        validator: Callable[[dict[str, Any]], bool],
        *,
        case_id: str | None = None,
        max_output_tokens: int | None = None,
        ledger: UsageLedger | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        last_raw_text = ""
        last_category: str | None = None
        last_finish_reason: str | None = None
        transport_retries = 0
        format_retries = 0
        attempt = 0
        while True:
            attempt += 1
            self._request_start_limiter.wait()
            request_start = time.monotonic()
            start_time = datetime.now(timezone.utc).isoformat()
            retry_kind = "initial" if attempt == 1 else (
                "format" if format_retries and transport_retries == 0 else "transport_or_format"
            )
            request_prompt = prompt
            if format_retries:
                request_prompt += (
                    "\nReturn ONLY one valid JSON object matching the requested fields. "
                    "Do not include markdown, code fences, or explanation."
                )
            try:
                response = self._create(request_prompt, schema, max_output_tokens)
                latency = time.monotonic() - request_start
                usage = _field(response, "usage")
                (ledger or self.ledger).record(stage, usage, retry=attempt > 1)
                raw_text, from_reasoning = self._response_text(response)
                reasoning_metadata = self._reasoning_metadata(response)
                last_raw_text = raw_text
                finish_reason = self._finish_reason(response)
                last_finish_reason = finish_reason
                value = parse_json_object(raw_text, prefer_last=from_reasoning)
                category: str | None = None
                if finish_reason == "length":
                    # A generation cap is deterministic for this request; do
                    # not spend another call retrying the same capped payload.
                    category = "OUTPUT_LIMIT_EXCEEDED"
                elif not raw_text.strip():
                    category = "EMPTY_RESPONSE"
                elif value is None:
                    category = "JSON_PARSE_ERROR"
                elif not validator(value):
                    if stage == "stage2" and "diagnosis_root_cause" in value:
                        category = "INVALID_CANDIDATE"
                    else:
                        category = "SCHEMA_VALIDATION_ERROR"
                values = _usage_values(usage)
                self._append_request_log({
                    "case_id": case_id,
                    "stage": stage,
                    "attempt": attempt,
                    "retry_kind": retry_kind,
                    "start_time": start_time,
                    "latency_seconds": round(latency, 3),
                    "status": "SUCCESS" if category is None else "INVALID_OUTPUT",
                    "error_category": category,
                    "exception_type": None,
                    "http_status": 200,
                    "finish_reason": finish_reason,
                    "input_tokens": values["input_tokens"],
                    "output_tokens": values["output_tokens"],
                    "total_tokens": values["total_tokens"],
                    "cached_input_tokens": values["cached_input_tokens"],
                    "reasoning_tokens": values["reasoning_tokens"],
                    **reasoning_metadata,
                    "usage_unknown": values["unknown"],
                })
                if category is None:
                    return {
                        "value": value,
                        "attempt": attempt,
                        "error": None,
                        "raw_text": raw_text,
                        "error_category": None,
                        "finish_reason": finish_reason,
                    }
                last_category = category
                errors.append(category)
                if category in {
                    "EMPTY_RESPONSE", "TRUNCATED_RESPONSE", "JSON_PARSE_ERROR",
                    "SCHEMA_VALIDATION_ERROR", "INVALID_CANDIDATE",
                } and format_retries < self.max_format_retries:
                    format_retries += 1
                    continue
                return {
                    "value": None,
                    "attempt": attempt,
                    "error": "; ".join(errors[-2:]),
                    "raw_text": last_raw_text,
                    "error_category": last_category,
                    "finish_reason": last_finish_reason,
                }
            except Exception as error:
                latency = time.monotonic() - request_start
                category = classify_exception(error)
                last_category = category
                errors.append(f"{category}: {self._safe_error(error)}")
                (ledger or self.ledger).record(stage, None, retry=attempt > 1, failed=True)
                self._append_request_log({
                    "case_id": case_id,
                    "stage": stage,
                    "attempt": attempt,
                    "retry_kind": retry_kind,
                    "start_time": start_time,
                    "latency_seconds": round(latency, 3),
                    "status": "FAILED",
                    "error_category": category,
                    "exception_type": error.__class__.__name__,
                    "http_status": _http_status(error),
                    "finish_reason": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 0,
                    "reasoning_content_present": False,
                    "reasoning_content_nonempty": False,
                    "usage_unknown": True,
                    "error": self._safe_error(error),
                })
                if _retryable(category) and transport_retries < self.max_transport_retries:
                    transport_retries += 1
                    self._backoff(transport_retries)
                    continue
                return {
                    "value": None,
                    "attempt": attempt,
                    "error": "; ".join(errors[-2:]),
                    "raw_text": last_raw_text,
                    "error_category": last_category,
                    "finish_reason": last_finish_reason,
                }
