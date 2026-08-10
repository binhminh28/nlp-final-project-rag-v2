import io
import json
import urllib.error

import pytest

from rag_chunking.embedding.models import EmbeddingConfig
from rag_chunking.embedding.provider import OpenRouterEmbeddingProvider, validate_provider_output


def _config() -> EmbeddingConfig:
    return EmbeddingConfig(provider="openrouter", model="openai/text-embedding-3-small", dimension=3)


class _Response:
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self):
        return json.dumps(self.body).encode()


def test_provider_retries_429_and_preserves_response_order(monkeypatch) -> None:
    responses = [
        urllib.error.HTTPError("url", 429, "limited", {"Retry-After": "0"}, io.BytesIO()),
        _Response({"data": [
            {"index": 1, "embedding": [0.0, 1.0, 0.0]},
            {"index": 0, "embedding": [1.0, 0.0, 0.0]},
        ], "usage": {"prompt_tokens": 2}}),
    ]
    def urlopen(*args, **kwargs):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    provider = OpenRouterEmbeddingProvider(_config(), api_key="secret", backoff_seconds=0)
    assert provider.embed_texts(["a", "b"])[0] == [1.0, 0.0, 0.0]
    assert provider.calls == 2 and provider.retries == 1 and provider.input_tokens == 2


def test_provider_strips_environment_whitespace(monkeypatch) -> None:
    captured = {}

    def urlopen(request, **kwargs):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        return _Response({"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]})

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret\r")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1\r")
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    provider = OpenRouterEmbeddingProvider(_config())
    provider.embed_texts(["a"])
    assert captured == {
        "url": "https://openrouter.ai/api/v1/embeddings",
        "authorization": "Bearer secret",
    }


@pytest.mark.parametrize("vectors, message", [
    ([[1.0, 2.0, 3.0]], "response count"),
    ([[1.0, 2.0], [1.0, 2.0]], "dimension"),
    ([[1.0, 2.0, float("nan")], [1.0, 2.0, 3.0]], "non-finite"),
])
def test_provider_output_rejects_malformed_vectors(vectors, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_provider_output(vectors, 2, 3)


def test_permanent_http_error_is_not_retried(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("url", 400, "bad", {}, io.BytesIO())
    monkeypatch.setattr("urllib.request.urlopen", fail)
    provider = OpenRouterEmbeddingProvider(_config(), api_key="secret", backoff_seconds=0)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        provider.embed_texts(["a"])
    assert provider.calls == 1 and provider.retries == 0


@pytest.mark.parametrize("error", [
    urllib.error.HTTPError("url", 503, "busy", {}, io.BytesIO()),
    TimeoutError("timeout"),
])
def test_transient_5xx_and_timeout_are_retried(monkeypatch, error) -> None:
    responses = [error, _Response({"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]})]
    def urlopen(*args, **kwargs):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    provider = OpenRouterEmbeddingProvider(_config(), api_key="secret", backoff_seconds=0)
    assert provider.embed_texts(["a"])[0] == [1.0, 0.0, 0.0]
    assert provider.retries == 1
