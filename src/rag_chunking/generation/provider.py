"""Provider boundary, deterministic fake, and bounded OpenRouter adapter."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import GenerationConfig, GenerationProviderError
from .prompt import ChatMessage


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    text: str
    output_tokens: int | None = None
    input_tokens: int | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise GenerationProviderError("provider returned empty or non-text answer content", retryable=True)
        for name in ("output_tokens", "input_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise GenerationProviderError(f"provider returned invalid {name}", retryable=False)
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise GenerationProviderError("provider returned invalid finish_reason", retryable=False)


class GenerationProvider(Protocol):
    calls: int
    retries: int

    def complete(self, messages: tuple[ChatMessage, ...], config: GenerationConfig) -> ProviderResponse: ...


class DeterministicFakeGenerationProvider:
    """No-network provider with reproducible answers and optional transient failures."""

    def __init__(self, *, response_text: str | None = None, fail_times: int = 0) -> None:
        if fail_times < 0:
            raise ValueError("fail_times must be non-negative")
        self.response_text = response_text
        self.fail_times = fail_times
        self.calls = 0
        self.retries = 0

    def complete(self, messages: tuple[ChatMessage, ...], config: GenerationConfig) -> ProviderResponse:
        for attempt in range(config.max_retries + 1):
            self.calls += 1
            if self.calls > self.fail_times:
                break
            if attempt >= config.max_retries:
                raise GenerationProviderError(
                    "deterministic fake transient failure", retryable=True, attempts=attempt + 1
                )
            self.retries += 1
        else:  # pragma: no cover
            raise RuntimeError("fake retry loop ended unexpectedly")
        canonical = json.dumps(
            [message.to_dict() for message in messages], ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        )
        text = self.response_text
        if text is None:
            text = f"Deterministic answer {hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"
        return ProviderResponse(text=text, output_tokens=len(text.split()), finish_reason="stop")


class OpenRouterGenerationProvider:
    """OpenRouter OpenAI-compatible chat-completions transport."""

    _TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None,
        *, app_name: str | None = None,
        diagnostics_output: Path | None = None,
        raw_diagnostics_output: Path | None = None,
    ) -> None:
        configured = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        self._api_key = configured.strip() if configured else None
        self._base_url = (
            base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL
        ).strip().rstrip("/")
        self._app_name = app_name or os.environ.get("OPENROUTER_APP_NAME")
        self._diagnostics_output = diagnostics_output
        self._raw_diagnostics_output = raw_diagnostics_output
        self._diagnostics: list[dict[str, Any]] = []
        self._state_lock = threading.Lock()
        self._diagnostic_context = threading.local()
        if diagnostics_output is not None and diagnostics_output.exists():
            for line_number, line in enumerate(
                diagnostics_output.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"invalid provider diagnostic at {diagnostics_output}:{line_number}"
                    )
                self._diagnostics.append(value)
        self.calls = 0
        self.retries = 0

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    @property
    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        with self._state_lock:
            return tuple(self._diagnostics)

    def set_diagnostic_context(self, query_id: str, prompt_fingerprint: str) -> None:
        """Attach safe request identity without changing generation semantics."""
        self._diagnostic_context.query_id = query_id
        self._diagnostic_context.prompt_fingerprint = prompt_fingerprint

    def _increment_calls(self) -> None:
        with self._state_lock:
            self.calls += 1

    def _increment_retries(self) -> None:
        with self._state_lock:
            self.retries += 1

    def _record_diagnostic(
        self, *, attempt: int, event: str, config: GenerationConfig,
        envelope: Any = None, http_status: int | None = None,
        headers: Any = None, error: dict[str, Any] | None = None,
    ) -> None:
        choice: dict[str, Any] = {}
        message: dict[str, Any] = {}
        choices = envelope.get("choices") if isinstance(envelope, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            candidate = choice.get("message")
            if isinstance(candidate, dict):
                message = candidate
        content = message.get("content") if "content" in message else None
        usage = envelope.get("usage") if isinstance(envelope, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        completion_details = usage.get("completion_tokens_details")
        completion_details = completion_details if isinstance(completion_details, dict) else {}
        prompt_details = usage.get("prompt_tokens_details")
        prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
        response_headers = headers if headers is not None else {}
        get_header = getattr(response_headers, "get", lambda key: None)
        provider_error = envelope.get("error") if isinstance(envelope, dict) else None
        if isinstance(provider_error, dict):
            metadata = provider_error.get("metadata")
            provider_error = {
                "code": provider_error.get("code"),
                "type": provider_error.get("type"),
                "metadata_keys": sorted(metadata) if isinstance(metadata, dict) else [],
            }
        else:
            provider_error = None
        query_id = getattr(self._diagnostic_context, "query_id", None)
        prompt_fingerprint = getattr(self._diagnostic_context, "prompt_fingerprint", None)
        record = {
            "diagnostic_schema_version": "generation_provider_diagnostic_v1",
            "query_id": query_id,
            "prompt_fingerprint": prompt_fingerprint,
            "attempt": attempt,
            "event": event,
            "endpoint": f"{self._base_url}/chat/completions",
            "request_model": config.model,
            "request_temperature": config.temperature,
            "request_max_tokens": config.max_output_tokens,
            "request_reasoning_effort": config.reasoning_effort,
            "http_status": http_status,
            "request_id": (
                get_header("x-request-id") or get_header("X-Request-ID")
                or (envelope.get("id") if isinstance(envelope, dict) else None)
            ),
            "response_model": envelope.get("model") if isinstance(envelope, dict) else None,
            "response_provider": envelope.get("provider") if isinstance(envelope, dict) else None,
            "openrouter_metadata_present": bool(
                isinstance(envelope, dict) and "openrouter_metadata" in envelope
            ),
            "choice_count": len(choices) if isinstance(choices, list) else None,
            "finish_reason": choice.get("finish_reason"),
            "native_finish_reason": choice.get("native_finish_reason"),
            "message_content_type": (
                "null" if content is None else "string" if isinstance(content, str)
                else type(content).__name__
            ),
            "visible_content_length": len(content) if isinstance(content, str) else None,
            "content_is_null": "content" in message and content is None,
            "content_is_empty_string": isinstance(content, str) and not content,
            "reasoning_present": "reasoning" in message and message.get("reasoning") is not None,
            "reasoning_details_present": (
                "reasoning_details" in message and message.get("reasoning_details") is not None
            ),
            "refusal_present": "refusal" in message and message.get("refusal") is not None,
            "tool_calls_present": "tool_calls" in message and bool(message.get("tool_calls")),
            "annotations_present": "annotations" in message and bool(message.get("annotations")),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens", usage.get("reasoning_tokens")),
            "cached_tokens": prompt_details.get("cached_tokens", usage.get("cached_tokens")),
            "provider_error": provider_error,
            "transport_error": error,
        }
        with self._state_lock:
            self._diagnostics.append(record)
            if self._diagnostics_output is not None:
                self._diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
                serialized = "".join(
                    json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                    for item in self._diagnostics
                )
                self._diagnostics_output.write_text(serialized, encoding="utf-8")
            if self._raw_diagnostics_output is not None and envelope is not None:
                self._raw_diagnostics_output.mkdir(parents=True, exist_ok=True)
                safe_id = "".join(
                    character if character.isalnum() or character in "-_" else "_"
                    for character in (query_id or "unknown")
                )
                raw_path = self._raw_diagnostics_output / f"{safe_id}.attempt-{attempt}.json"
                raw_path.write_text(
                    json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    def complete(self, messages: tuple[ChatMessage, ...], config: GenerationConfig) -> ProviderResponse:
        if config.provider != "openrouter":
            raise ValueError("OpenRouterGenerationProvider requires provider='openrouter'")
        if not self._api_key:
            raise GenerationProviderError("OPENROUTER_API_KEY is required for live generation", retryable=False)
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.seed is not None:
            payload["seed"] = config.seed
        if config.reasoning_effort is not None:
            payload["reasoning"] = {"effort": config.reasoning_effort}
        if config.stop_sequences:
            payload["stop"] = list(config.stop_sequences)
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if self._app_name:
            headers["X-OpenRouter-Title"] = self._app_name

        last_error: GenerationProviderError | None = None
        for attempt in range(config.max_retries + 1):
            request = urllib.request.Request(
                f"{self._base_url}/chat/completions", data=encoded, headers=headers, method="POST",
            )
            self._increment_calls()
            try:
                with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                    http_status = getattr(response, "status", None) or getattr(response, "getcode", lambda: 200)()
                    response_headers = getattr(response, "headers", None)
                try:
                    envelope = json.loads(raw)
                except json.JSONDecodeError as error:
                    self._record_diagnostic(
                        attempt=attempt + 1, event="malformed_json", config=config,
                        http_status=http_status, headers=response_headers,
                        error={"type": "JSONDecodeError", "retryable": False},
                    )
                    raise GenerationProviderError(
                        "OpenRouter returned malformed response JSON", retryable=False, attempts=attempt + 1,
                    ) from error
                self._record_diagnostic(
                    attempt=attempt + 1, event="response", config=config,
                    envelope=envelope, http_status=http_status, headers=response_headers,
                )
                try:
                    parsed = self._extract(envelope)
                except GenerationProviderError as error:
                    last_error = GenerationProviderError(
                        str(error), retryable=error.retryable, attempts=attempt + 1,
                    )
                    if not error.retryable or attempt >= config.max_retries:
                        raise last_error from error
                    self._increment_retries()
                    time.sleep(config.retry_backoff_seconds * (2**attempt))
                    continue
                return parsed
            except urllib.error.HTTPError as error:
                retryable = error.code in self._TRANSIENT_STATUS_CODES
                try:
                    error_raw = error.read().decode("utf-8")
                    error_envelope = json.loads(error_raw)
                except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
                    error_envelope = None
                self._record_diagnostic(
                    attempt=attempt + 1, event="http_error", config=config,
                    envelope=error_envelope, http_status=error.code, headers=error.headers,
                    error={"type": "HTTPError", "retryable": retryable},
                )
                last_error = GenerationProviderError(
                    f"OpenRouter generation request failed with HTTP {error.code}",
                    retryable=retryable, attempts=attempt + 1, status_code=error.code,
                )
                if not retryable or attempt >= config.max_retries:
                    raise last_error from error
                self._increment_retries()
                retry_after = error.headers.get("Retry-After") if error.headers else None
                try:
                    delay = float(retry_after) if retry_after is not None else config.retry_backoff_seconds * (2**attempt)
                except ValueError:
                    delay = config.retry_backoff_seconds * (2**attempt)
                time.sleep(max(0.0, min(delay, 60.0)))
            except (urllib.error.URLError, TimeoutError) as error:
                self._record_diagnostic(
                    attempt=attempt + 1, event="transport_error", config=config,
                    error={"type": type(error).__name__, "retryable": True},
                )
                last_error = GenerationProviderError(
                    "OpenRouter generation request failed due to timeout or connection error",
                    retryable=True, attempts=attempt + 1,
                )
                if attempt >= config.max_retries:
                    raise last_error from error
                self._increment_retries()
                time.sleep(config.retry_backoff_seconds * (2**attempt))
        raise last_error or RuntimeError("OpenRouter retry loop ended unexpectedly")

    @staticmethod
    def _extract(envelope: Any) -> ProviderResponse:
        if not isinstance(envelope, dict):
            raise GenerationProviderError("OpenRouter returned malformed response envelope", retryable=False)
        try:
            choice = envelope["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise GenerationProviderError("OpenRouter response is missing answer content", retryable=False) from error
        if not isinstance(content, str) or not content.strip():
            raise GenerationProviderError("OpenRouter returned empty or non-text answer content", retryable=True)
        usage = envelope.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        output_tokens = usage.get("completion_tokens")
        input_tokens = usage.get("prompt_tokens")
        if type(output_tokens) is not int:
            output_tokens = None
        if type(input_tokens) is not int:
            input_tokens = None
        finish_reason = choice.get("finish_reason")
        return ProviderResponse(
            content, output_tokens=output_tokens, input_tokens=input_tokens,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        )
