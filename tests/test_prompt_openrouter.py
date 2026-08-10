from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from rag_chunking.chunking.prompt_based import PromptBasedChunker
from rag_chunking.chunking.prompt_cache import request_digest
from rag_chunking.chunking.prompt_client import (
    DEFAULT_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_PROVIDER,
    OpenRouterBoundaryPlanner,
    PlannerModelConfig,
)
from rag_chunking.chunking.prompt_config import load_project_dotenv
from rag_chunking.chunking.prompt_statistics import prompt_corpus_statistics
from rag_chunking.chunking.prompt_safety import OutboundPayloadSafetyError, validate_outbound_payload
from rag_chunking.chunking.prompt_writer import write_prompt_based_artifacts
from rag_chunking.data.models import DocumentBlock, NormalizedDocument


class _Response:
    def __init__(self, value: dict[str, object]):
        self.data = json.dumps(value).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.data


class _FakePlanner:
    def plan(self, system_prompt: str, user_prompt: str, config: PlannerModelConfig) -> str:
        value = json.loads(user_prompt)
        return json.dumps({"groups": [{
            "start_block_index": value["batch_start_block_index"],
            "end_block_index": value["batch_end_block_index"],
            "reason": "fixture",
        }]})


def _document() -> NormalizedDocument:
    return NormalizedDocument(
        doc_id="angular:openrouter.md",
        source="angular",
        relative_path="openrouter.md",
        filename="openrouter.md",
        source_sha256="fixture-hash",
        blocks=[DocumentBlock(type="paragraph", text="Small fixture.")],
    )


def test_openrouter_configuration_defaults() -> None:
    config = PlannerModelConfig()
    assert config.provider == DEFAULT_PROVIDER == "openrouter"
    assert config.model == DEFAULT_MODEL == "deepseek/deepseek-v4-flash-0731"
    assert config.base_url == DEFAULT_OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert config.seed is None


def test_project_dotenv_loads_without_overwriting_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=file-secret\nPROMPT_PLANNER_MODEL=file-model\n", encoding="utf-8"
    )
    monkeypatch.setenv("PROMPT_PLANNER_MODEL", "process-model")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert load_project_dotenv(tmp_path) == tmp_path / ".env"
    assert __import__("os").environ["OPENROUTER_API_KEY"] == "file-secret"
    assert __import__("os").environ["PROMPT_PLANNER_MODEL"] == "process-model"


def test_missing_openrouter_key_fails_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    planner = OpenRouterBoundaryPlanner(api_key="")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
        planner.plan("system", "{}", PlannerModelConfig())


@pytest.mark.parametrize("unsafe", [
    "OPENROUTER_API_KEY=sk-or-v1-abcdefghijklmnopqrstuvwxyz123456",
    "-----BEGIN PRIVATE KEY-----",
    r"C:\Users\person\secret.txt",
    "/home/person/project/secret.txt",
])
def test_outbound_safety_rejects_high_confidence_secrets_and_paths(unsafe: str) -> None:
    with pytest.raises(OutboundPayloadSafetyError):
        validate_outbound_payload(unsafe, configured_secret="configured-secret-value")


def test_outbound_safety_rejects_exact_configured_secret() -> None:
    with pytest.raises(OutboundPayloadSafetyError, match="configured API key"):
        validate_outbound_payload("prefix configured-secret-value suffix", configured_secret="configured-secret-value")


def test_openrouter_structured_request_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    def fake_urlopen(request: object, timeout: float) -> _Response:
        requests.append(request)
        return _Response({
            "id": "generation-fixture",
            "model": DEFAULT_MODEL,
            "choices": [{"message": {"content": '{"groups":[]}'}}],
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    response = OpenRouterBoundaryPlanner(api_key="secret-fixture").plan(
        "system", "user", PlannerModelConfig()
    )
    payload = json.loads(requests[0].data.decode("utf-8"))
    assert requests[0].full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["provider"]["require_parameters"] is True
    assert payload["max_tokens"] == 4096
    assert "seed" not in payload
    assert response.response_mode == "json_schema"
    assert response.capability_fallback_used is False


def test_openrouter_json_fallback_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    def fake_urlopen(request: object, timeout: float) -> _Response:
        requests.append(request)
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, 400, "unsupported", {}, None)
        return _Response({"choices": [{"message": {"content": '{"groups":[]}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    response = OpenRouterBoundaryPlanner(api_key="secret-fixture").plan(
        "system", "user", PlannerModelConfig()
    )
    fallback_payload = json.loads(requests[1].data.decode("utf-8"))
    assert "response_format" not in fallback_payload
    assert fallback_payload["messages"][1]["content"].endswith("no prose or code fences.")
    assert response.response_mode == "prompt_json"
    assert response.capability_fallback_used is True


def test_openrouter_retries_429_with_separate_transport_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request: object, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "limited", {"Retry-After": "0"}, None)
        return _Response({"choices": [{"message": {"content": '{"groups":[]}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    response = OpenRouterBoundaryPlanner(
        api_key="secret-fixture", max_transport_retries=1, backoff_seconds=0
    ).plan("system", "user", PlannerModelConfig())
    assert response.operational_metadata["transport_calls"] == 2
    assert response.operational_metadata["transport_retries"] == 1
    assert response.operational_metadata["http_429_responses"] == 1


def test_provider_identity_separates_cache_and_secret_never_serializes(tmp_path: Path) -> None:
    secret = "never-serialize-this-key"
    document = _document()
    openrouter_config = PlannerModelConfig()
    old_config = PlannerModelConfig(
        provider="openai", model="gpt-4.1-mini", base_url="https://api.openai.com/v1"
    )
    assert openrouter_config.identity() != old_config.identity()
    assert request_digest({"model": openrouter_config.identity()}) != request_digest(
        {"model": old_config.identity()}
    )

    chunker = PromptBasedChunker(_FakePlanner(), tmp_path / "cache", model_config=openrouter_config)
    chunks = chunker.chunk(document)
    stats = prompt_corpus_statistics([document], chunks, chunker.metrics)
    write_prompt_based_artifacts(
        chunks, [document], tmp_path / "output", chunker.config, openrouter_config,
        chunker.tokenizer, stats, "input.jsonl",
    )
    OpenRouterBoundaryPlanner(api_key=secret)  # Client construction alone must not serialize credentials.
    generated = [*list((tmp_path / "cache").glob("*.json")), *list((tmp_path / "output").glob("*"))]
    assert generated
    assert all(secret not in path.read_text(encoding="utf-8") for path in generated if path.is_file())
    assert secret not in json.dumps(openrouter_config.identity())
