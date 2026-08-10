"""Boundary-planner abstraction and OpenRouter chat-completions adapter."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .prompt_schema import planner_json_schema
from .prompt_safety import validate_outbound_payload


DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
STRUCTURED_OUTPUT_POLICY = "json_schema_then_prompt_json_v1"


@dataclass(frozen=True, slots=True)
class PlannerModelConfig:
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    temperature: float = 0.0
    max_response_tokens: int = 4096
    seed: int | None = None
    timeout_seconds: float = 60.0
    structured_output_policy: str = STRUCTURED_OUTPUT_POLICY

    def __post_init__(self) -> None:
        if not self.provider or not self.model or not self.base_url:
            raise ValueError("provider, model, and base_url must be non-empty")
        if self.max_response_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("response-token and timeout limits must be positive")

    def identity(self) -> dict[str, object]:
        """Return cache/manifest-safe request identity; credentials never enter it."""

        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url.rstrip("/"),
            "temperature": self.temperature,
            "max_response_tokens": self.max_response_tokens,
            "seed": self.seed,
            "structured_output_policy": self.structured_output_policy,
        }


@dataclass(frozen=True, slots=True)
class PlannerResponse:
    text: str
    response_mode: str
    requested_model: str
    resolved_model: str | None = None
    capability_fallback_used: bool = False
    response_metadata: dict[str, object] = field(default_factory=dict)
    operational_metadata: dict[str, int | float] = field(default_factory=dict)

    def cache_metadata(self) -> dict[str, object]:
        return {
            "response_mode": self.response_mode,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "capability_fallback_used": self.capability_fallback_used,
            **self.response_metadata,
        }


class BoundaryPlanner(Protocol):
    def plan(
        self, system_prompt: str, user_prompt: str, config: PlannerModelConfig
    ) -> str | PlannerResponse:
        """Return JSON response text, optionally with non-secret transport metadata."""


class OpenRouterBoundaryPlanner:
    """OpenRouter adapter using its OpenAI-compatible chat-completions endpoint."""

    _FALLBACK_STATUS_CODES = frozenset({400, 404, 422})
    _TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        app_name: str | None = None,
        max_transport_retries: int = 3,
        backoff_seconds: float = 0.5,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
        self._app_name = app_name or os.environ.get("OPENROUTER_APP_NAME")
        if max_transport_retries < 0 or backoff_seconds < 0:
            raise ValueError("transport retry settings must be non-negative")
        self._max_transport_retries = max_transport_retries
        self._backoff_seconds = backoff_seconds

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def plan(
        self, system_prompt: str, user_prompt: str, config: PlannerModelConfig
    ) -> PlannerResponse:
        if config.provider != DEFAULT_PROVIDER:
            raise ValueError(f"OpenRouterBoundaryPlanner cannot serve provider {config.provider!r}")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is required for live prompt-based chunking")
        if config.seed is not None:
            raise ValueError("seed must be null for the OpenRouter planner configuration")
        validate_outbound_payload(
            system_prompt + "\n" + user_prompt, configured_secret=self._api_key
        )

        base_payload: dict[str, object] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_response_tokens,
        }
        structured_payload = {
            **base_payload,
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": planner_json_schema(),
            },
        }
        telemetry: dict[str, int | float] = {
            "transport_calls": 0,
            "transport_retries": 0,
            "http_429_responses": 0,
            "http_5xx_responses": 0,
            "network_errors": 0,
        }
        started = time.monotonic()
        try:
            body = self._post(structured_payload, config.timeout_seconds, telemetry)
            response = self._extract(body, config.model, "json_schema", False)
        except _ResponseFormatUnsupported:
            fallback_payload = dict(base_payload)
            fallback_payload["messages"] = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt + "\nReturn exactly one JSON object and no prose or code fences.",
                },
            ]
            body = self._post(
                fallback_payload,
                config.timeout_seconds,
                telemetry,
                allow_format_fallback=False,
            )
            response = self._extract(body, config.model, "prompt_json", True)
        telemetry["latency_seconds"] = time.monotonic() - started
        return PlannerResponse(
            text=response.text,
            response_mode=response.response_mode,
            requested_model=response.requested_model,
            resolved_model=response.resolved_model,
            capability_fallback_used=response.capability_fallback_used,
            response_metadata=response.response_metadata,
            operational_metadata=telemetry,
        )

    def _post(
        self,
        payload: dict[str, object],
        timeout: float,
        telemetry: dict[str, int | float],
        *,
        allow_format_fallback: bool = True,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._app_name:
            headers["X-OpenRouter-Title"] = self._app_name
        for attempt in range(self._max_transport_retries + 1):
            request = urllib.request.Request(
                f"{self._base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            telemetry["transport_calls"] += 1
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                if allow_format_fallback and error.code in self._FALLBACK_STATUS_CODES:
                    raise _ResponseFormatUnsupported from error
                if error.code == 429:
                    telemetry["http_429_responses"] += 1
                if 500 <= error.code <= 599:
                    telemetry["http_5xx_responses"] += 1
                if error.code not in self._TRANSIENT_STATUS_CODES or attempt >= self._max_transport_retries:
                    raise PlannerTransportError(
                        f"OpenRouter planner request failed with HTTP {error.code}", telemetry
                    ) from error
                telemetry["transport_retries"] += 1
                retry_after = error.headers.get("Retry-After") if error.headers else None
                try:
                    delay = float(retry_after) if retry_after is not None else self._backoff_seconds * (2**attempt)
                except ValueError:
                    delay = self._backoff_seconds * (2**attempt)
                time.sleep(max(0.0, min(delay, 60.0)))
            except urllib.error.URLError as error:
                telemetry["network_errors"] += 1
                if attempt >= self._max_transport_retries:
                    raise PlannerTransportError(
                        "OpenRouter planner request failed due to a connection error", telemetry
                    ) from error
                telemetry["transport_retries"] += 1
                time.sleep(self._backoff_seconds * (2**attempt))
            except json.JSONDecodeError as error:
                raise RuntimeError("OpenRouter planner returned malformed response JSON") from error
        else:  # pragma: no cover - loop exits by success or exception
            raise RuntimeError("OpenRouter transport retry loop ended unexpectedly")
        if not isinstance(body, dict):
            raise RuntimeError("OpenRouter planner returned an invalid response envelope")
        return body

    @staticmethod
    def _extract(
        body: dict[str, Any], requested_model: str, response_mode: str, fallback_used: bool
    ) -> PlannerResponse:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("OpenRouter response did not contain message content") from error
        if not isinstance(content, str):
            raise RuntimeError("OpenRouter response message content was not text")
        resolved_model = body.get("model") if isinstance(body.get("model"), str) else None
        metadata: dict[str, object] = {}
        if isinstance(body.get("id"), str):
            metadata["generation_id"] = body["id"]
        return PlannerResponse(
            text=content,
            response_mode=response_mode,
            requested_model=requested_model,
            resolved_model=resolved_model,
            capability_fallback_used=fallback_used,
            response_metadata=metadata,
        )


class _ResponseFormatUnsupported(RuntimeError):
    """The routed model/provider rejected required JSON-Schema parameters."""


class PlannerTransportError(RuntimeError):
    """A finite transport retry sequence failed, with non-secret telemetry."""

    def __init__(self, message: str, telemetry: dict[str, int | float]):
        super().__init__(message)
        self.telemetry = dict(telemetry)
