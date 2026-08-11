"""Provider boundary, deterministic fake, and bounded OpenRouter adapter."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    ) -> None:
        configured = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        self._api_key = configured.strip() if configured else None
        self._base_url = (
            base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL
        ).strip().rstrip("/")
        self._app_name = app_name or os.environ.get("OPENROUTER_APP_NAME")
        self.calls = 0
        self.retries = 0

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

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
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if self._app_name:
            headers["X-OpenRouter-Title"] = self._app_name

        last_error: GenerationProviderError | None = None
        for attempt in range(config.max_retries + 1):
            request = urllib.request.Request(
                f"{self._base_url}/chat/completions", data=encoded, headers=headers, method="POST",
            )
            self.calls += 1
            try:
                with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                try:
                    envelope = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise GenerationProviderError(
                        "OpenRouter returned malformed response JSON", retryable=False, attempts=attempt + 1,
                    ) from error
                try:
                    parsed = self._extract(envelope)
                except GenerationProviderError as error:
                    last_error = GenerationProviderError(
                        str(error), retryable=error.retryable, attempts=attempt + 1,
                    )
                    if not error.retryable or attempt >= config.max_retries:
                        raise last_error from error
                    self.retries += 1
                    time.sleep(config.retry_backoff_seconds * (2**attempt))
                    continue
                return parsed
            except urllib.error.HTTPError as error:
                retryable = error.code in self._TRANSIENT_STATUS_CODES
                last_error = GenerationProviderError(
                    f"OpenRouter generation request failed with HTTP {error.code}",
                    retryable=retryable, attempts=attempt + 1, status_code=error.code,
                )
                if not retryable or attempt >= config.max_retries:
                    raise last_error from error
                self.retries += 1
                retry_after = error.headers.get("Retry-After") if error.headers else None
                try:
                    delay = float(retry_after) if retry_after is not None else config.retry_backoff_seconds * (2**attempt)
                except ValueError:
                    delay = config.retry_backoff_seconds * (2**attempt)
                time.sleep(max(0.0, min(delay, 60.0)))
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = GenerationProviderError(
                    "OpenRouter generation request failed due to timeout or connection error",
                    retryable=True, attempts=attempt + 1,
                )
                if attempt >= config.max_retries:
                    raise last_error from error
                self.retries += 1
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
