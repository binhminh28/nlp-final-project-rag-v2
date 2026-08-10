"""Embedding provider boundary, deterministic fake, and OpenRouter adapter."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from .models import EmbeddingConfig, validate_vector


class EmbeddingProvider(Protocol):
    config: EmbeddingConfig
    calls: int
    retries: int
    input_tokens: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def validate_provider_output(
    vectors: list[list[float]], requested_count: int, dimension: int
) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != requested_count:
        actual = len(vectors) if isinstance(vectors, list) else "malformed"
        raise ValueError(f"embedding response count {actual} != requested {requested_count}")
    for vector in vectors:
        validate_vector(vector, dimension)
    return vectors


class DeterministicFakeEmbeddingProvider:
    """Offline provider used by every default test and dry run."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.calls = 0
        self.retries = 0
        self.input_tokens = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            values = [((seed[i % len(seed)] / 255.0) * 2.0) - 1.0 for i in range(self.config.dimension)]
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append([value / norm for value in values])
        return vectors


class OpenRouterEmbeddingProvider:
    _TRANSIENT = frozenset({429, 500, 502, 503, 504, 529})

    def __init__(
        self, config: EmbeddingConfig, api_key: str | None = None,
        base_url: str | None = None, timeout_seconds: float = 60.0,
        max_retries: int = 3, backoff_seconds: float = 0.5,
    ) -> None:
        if config.provider != "openrouter":
            raise ValueError("OpenRouter provider requires provider='openrouter'")
        if timeout_seconds <= 0 or max_retries < 0 or backoff_seconds < 0:
            raise ValueError("invalid transport settings")
        self.config = config
        configured_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._api_key = configured_key.strip() if configured_key else None
        self._base_url = (
            base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        ).strip().rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self.calls = 0
        self.retries = 0
        self.input_tokens = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is required for live embeddings")
        payload: dict[str, Any] = {
            "model": self.config.model, "input": texts,
            "dimensions": self.config.dimension, "encoding_format": "float",
        }
        if self.config.input_type:
            payload["input_type"] = self.config.input_type
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        for attempt in range(self._max_retries + 1):
            self.calls += 1
            request = urllib.request.Request(
                f"{self._base_url}/embeddings", data=body, method="POST",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                if error.code not in self._TRANSIENT or attempt >= self._max_retries:
                    raise RuntimeError(f"OpenRouter embedding request failed with HTTP {error.code}") from error
                self.retries += 1
                retry_after = error.headers.get("Retry-After") if error.headers else None
                try:
                    delay = float(retry_after) if retry_after is not None else self._backoff * (2**attempt)
                except ValueError:
                    delay = self._backoff * (2**attempt)
                time.sleep(max(0.0, min(delay, 60.0)))
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt >= self._max_retries:
                    raise RuntimeError("OpenRouter embedding request failed due to a connection error") from error
                self.retries += 1
                time.sleep(self._backoff * (2**attempt))
            except json.JSONDecodeError as error:
                raise RuntimeError("OpenRouter returned malformed JSON") from error
        if not isinstance(envelope, dict) or not isinstance(envelope.get("data"), list):
            raise ValueError("OpenRouter returned a malformed embedding envelope")
        data = envelope["data"]
        try:
            ordered = sorted(data, key=lambda item: item["index"])
            if [item["index"] for item in ordered] != list(range(len(texts))):
                raise ValueError("embedding response indexes are missing or duplicated")
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, TypeError) as error:
            raise ValueError("OpenRouter returned malformed embedding items") from error
        usage = envelope.get("usage")
        if isinstance(usage, dict) and type(usage.get("prompt_tokens")) is int:
            self.input_tokens += usage["prompt_tokens"]
        return validate_provider_output(vectors, len(texts), self.config.dimension)
