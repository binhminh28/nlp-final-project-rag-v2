from __future__ import annotations

import json
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest

from rag_chunking.chunking.tokenizer import TiktokenTokenizer
from rag_chunking.context import ContextBuildInput, ContextBuilder, ContextConfig
from rag_chunking.embedding.artifacts import artifact_sha256, serialize_embedding_records, write_embedding_artifacts
from rag_chunking.embedding.index import build_local_index
from rag_chunking.embedding.models import EmbeddingConfig, EmbeddingRecord, canonical_fingerprint, content_sha256
from rag_chunking.embedding.provider import DeterministicFakeEmbeddingProvider
from rag_chunking.generation import (
    ANSWER_SYSTEM_PROMPT,
    AnswerPromptBuilder,
    AnswerResult,
    DeterministicFakeGenerationProvider,
    GenerationCache,
    GenerationCacheError,
    GenerationConfig,
    GenerationInput,
    GenerationIntegrityError,
    GenerationInputOverflowError,
    GenerationProviderError,
    GenerationService,
    OpenRouterGenerationProvider,
    ProviderResponse,
    run_generation,
)
from rag_chunking.generation.cache import generation_cache_key
from rag_chunking.retrieval import (
    SAME_TOP_K, RetrievalHit, RetrievalProtocolConfig, RetrievalRequest,
    RetrievalService, apply_retrieval_protocol,
)


TOKENIZER = TiktokenTokenizer()
STRATEGIES = ("fixed_size", "structure_aware", "prompt_based")


def context(
    text: str = "authoritative evidence", *, strategy: str = "fixed_size",
    query_id: str = "q-1", score: float = 0.99, metadata: dict | None = None,
):
    hit = RetrievalHit(
        rank=3, score=score, chunk_id=f"{strategy}-chunk", doc_id="doc-1",
        source="angular", relative_path="guide.md", strategy=strategy, text=text,
        metadata=metadata or {}, token_count=len(TOKENIZER.encode(text)),
        character_count=len(text), chunk_config_fingerprint=f"chunks-{strategy}",
    )
    build_input = ContextBuildInput(
        query_id=query_id, question="upstream question", strategy=strategy,
        selected_hits=(hit,), retrieval_config_fingerprint="retrieval-fp",
        protocol_config_fingerprint="protocol-fp",
        embedding_config_fingerprint="embedding-fp", index_fingerprint=f"index-{strategy}",
        dataset_fingerprint="dataset-fp",
    )
    return ContextBuilder(ContextConfig()).build(build_input)


def config(**changes):
    return GenerationConfig(provider="fake", model="fake-v1", **changes)


def v2_config(**changes):
    values = {
        "provider": "fake", "model": "fake-v1",
        "schema_version": "generation_config_v2",
        "completion_integrity_policy": "require_stop",
        "response_handling_contract": "nonempty_text_require_stop_v2",
        "prepared_context_token_budget": 4096,
    }
    values.update(changes)
    return GenerationConfig(**values)


def generation_input(cfg=None, *, question="  Exact question?\n", result=None, query_id="q-1"):
    cfg = cfg or config()
    result = result or context(query_id=query_id)
    return GenerationInput.create(query_id, question, result, cfg)


def test_config_fingerprint_is_canonical_and_all_semantic_fields_invalidate():
    base = config()
    assert base.fingerprint == canonical_fingerprint(base.identity())
    for changed in (
        replace(base, model="fake-v2"), replace(base, temperature=0.2),
        replace(base, top_p=0.9), replace(base, max_output_tokens=256),
        replace(base, seed=7), replace(base, timeout_seconds=30),
        replace(base, max_retries=1), replace(base, context_window_tokens=16000),
        replace(base, prompt_template_version="answer_prompt_v2"),
        replace(base, system_prompt_version="answer_system_v2", system_prompt="Different instruction."),
    ):
        assert changed.fingerprint != base.fingerprint


def test_frozen_prompt_exact_format_whitespace_and_determinism():
    cfg = config()
    item = generation_input(cfg, question="  Exact question?\n", result=context(" line one\nline two "))
    first = AnswerPromptBuilder(cfg).build(item)
    second = AnswerPromptBuilder(cfg).build(item)
    assert first == second
    assert first.messages[0].content == ANSWER_SYSTEM_PROMPT
    assert first.messages[1].content == (
        "Question:\n  Exact question?\n\n\nContext:\n"
        "[CONTEXT 1]\n line one\nline two \n\nAnswer:"
    )
    assert first.prompt_fingerprint == second.prompt_fingerprint


def test_prompt_fingerprint_binds_question_context_order_and_version():
    cfg = config()
    baseline = AnswerPromptBuilder(cfg).build(generation_input(cfg))
    changed_question = AnswerPromptBuilder(cfg).build(generation_input(cfg, question="Different?"))
    changed_context = AnswerPromptBuilder(cfg).build(generation_input(cfg, result=context("different")))
    v2 = replace(cfg, prompt_template_version="answer_prompt_v2")
    changed_version = AnswerPromptBuilder(v2).build(generation_input(v2))
    assert len({
        baseline.prompt_fingerprint, changed_question.prompt_fingerprint,
        changed_context.prompt_fingerprint, changed_version.prompt_fingerprint,
    }) == 4


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_fairness_gold_and_metadata_isolation(strategy: str):
    cfg = config()
    result = context(
        "same evidence", strategy=strategy, score=0.1,
        metadata={"gold_answer": "LEAKED ANSWER", "gold_evidence": "LEAKED EVIDENCE", "score": 99},
    )
    prompt = AnswerPromptBuilder(cfg).build(generation_input(cfg, result=result))
    serialized = json.dumps(prompt.provider_messages())
    assert strategy not in serialized
    assert "LEAKED" not in serialized
    assert "0.1" not in serialized and "99" not in serialized
    assert "[CONTEXT 1]\nsame evidence" in prompt.messages[1].content


def test_strategy_is_provenance_only_and_same_context_has_same_prompt():
    cfg = config()
    prompts = [
        AnswerPromptBuilder(cfg).build(generation_input(cfg, result=context("same evidence", strategy=name)))
        for name in STRATEGIES
    ]
    assert len({item.prompt_fingerprint for item in prompts}) == 1
    assert len({item.messages for item in prompts}) == 1


def test_context_order_changes_prompt_fingerprint():
    def multi(texts):
        hits = tuple(RetrievalHit(
            rank=i, score=1 / i, chunk_id=f"c-{i}-{text}", doc_id="doc", source="angular",
            relative_path="x.md", strategy="fixed_size", text=text, metadata={},
            token_count=len(TOKENIZER.encode(text)), character_count=len(text),
            chunk_config_fingerprint="chunks",
        ) for i, text in enumerate(texts, 1))
        envelope = ContextBuildInput(
            "q-1", "q", "fixed_size", hits, "retrieval", "protocol", "embedding", "index", "dataset"
        )
        return ContextBuilder(ContextConfig()).build(envelope)
    cfg = config()
    left = AnswerPromptBuilder(cfg).build(generation_input(cfg, result=multi(("alpha", "beta"))))
    right = AnswerPromptBuilder(cfg).build(generation_input(cfg, result=multi(("beta", "alpha"))))
    assert left.prompt_fingerprint != right.prompt_fingerprint


def test_token_accounting_includes_format_and_chat_overhead():
    cfg = config()
    prompt = AnswerPromptBuilder(cfg).build(generation_input(cfg))
    usage = prompt.input_tokens
    assert usage.context_tokens == len(TOKENIZER.encode(context().rendered_context))
    assert usage.question_tokens == len(TOKENIZER.encode("  Exact question?\n"))
    assert usage.system_instruction_tokens > 0
    assert usage.user_formatting_tokens > 0
    assert usage.chat_framing_tokens > 0
    assert usage.total_input_tokens > usage.context_tokens + usage.question_tokens


class NeverCalledProvider:
    calls = 0
    retries = 0

    def complete(self, messages, cfg):
        self.calls += 1
        raise AssertionError("must not be called")


def test_overflow_is_typed_and_detected_before_provider_or_cache(tmp_path: Path):
    probe_cfg = config(context_window_tokens=1000, max_output_tokens=10)
    item = generation_input(probe_cfg)
    total = AnswerPromptBuilder(probe_cfg).build(item).input_tokens.total_input_tokens
    cfg = config(context_window_tokens=total + 9, max_output_tokens=10)
    item = generation_input(cfg)
    provider = NeverCalledProvider()
    service = GenerationService(cfg, provider, cache=GenerationCache(tmp_path / "cache"))
    with pytest.raises(GenerationInputOverflowError) as caught:
        service.generate(item)
    assert caught.value.context_fingerprint == item.context.context_fingerprint
    assert caught.value.input_tokens == total
    assert provider.calls == 0
    assert not (tmp_path / "cache").exists()


def test_fake_provider_is_deterministic_and_retries_are_bounded():
    cfg = config(max_retries=2, retry_backoff_seconds=0)
    item = generation_input(cfg)
    first = GenerationService(cfg, DeterministicFakeGenerationProvider()).generate(item)
    second = GenerationService(cfg, DeterministicFakeGenerationProvider()).generate(item)
    assert first.answer_text == second.answer_text
    flaky = DeterministicFakeGenerationProvider(fail_times=2)
    assert GenerationService(cfg, flaky).generate(item).answer_text
    assert flaky.calls == 3 and flaky.retries == 2
    exhausted = DeterministicFakeGenerationProvider(fail_times=3)
    with pytest.raises(GenerationProviderError) as caught:
        GenerationService(cfg, exhausted).generate(item)
    assert caught.value.attempts == 3


def test_empty_fake_response_is_rejected():
    cfg = config()
    with pytest.raises(GenerationProviderError, match="empty"):
        GenerationService(cfg, DeterministicFakeGenerationProvider(response_text=" \n")).generate(
            generation_input(cfg)
        )


class Response:
    def __init__(self, value):
        self.data = json.dumps(value).encode()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None
    def read(self):
        return self.data


def test_openrouter_request_usage_and_success(monkeypatch):
    requests = []
    def urlopen(request, timeout):
        requests.append((request, timeout))
        return Response({
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        })
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    cfg = GenerationConfig(provider="openrouter", model="model", top_p=0.8, seed=4)
    result = GenerationService(cfg, OpenRouterGenerationProvider(api_key="secret")).generate(
        generation_input(cfg)
    )
    payload = json.loads(requests[0][0].data)
    assert payload["messages"] == AnswerPromptBuilder(cfg).build(generation_input(cfg)).provider_messages()
    assert payload["top_p"] == 0.8 and payload["seed"] == 4
    assert result.input_tokens.provider_reported_input_tokens == 12
    assert result.output_tokens == 3 and result.finish_reason == "stop"


@pytest.mark.parametrize("code,retryable,calls", [(429, True, 2), (503, True, 2), (400, False, 1)])
def test_openrouter_http_retry_policy(monkeypatch, code, retryable, calls):
    count = 0
    def urlopen(request, timeout):
        nonlocal count
        count += 1
        if count == 1:
            raise urllib.error.HTTPError(request.full_url, code, "failure", {}, None)
        return Response({"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    cfg = GenerationConfig(
        provider="openrouter", model="model", max_retries=1, retry_backoff_seconds=0
    )
    provider = OpenRouterGenerationProvider(api_key="secret")
    if retryable:
        assert GenerationService(cfg, provider).generate(generation_input(cfg)).answer_text == "ok"
    else:
        with pytest.raises(GenerationProviderError) as caught:
            GenerationService(cfg, provider).generate(generation_input(cfg))
        assert caught.value.retryable is False
    assert count == calls


def test_openrouter_timeout_empty_and_malformed_handling(monkeypatch):
    cfg = GenerationConfig(
        provider="openrouter", model="model", max_retries=1, retry_backoff_seconds=0
    )
    count = 0
    def timeout_then_ok(request, timeout):
        nonlocal count
        count += 1
        if count == 1:
            raise TimeoutError("timeout")
        return Response({"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr("urllib.request.urlopen", timeout_then_ok)
    assert GenerationService(cfg, OpenRouterGenerationProvider(api_key="secret")).generate(
        generation_input(cfg)
    ).answer_text == "ok"

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response({"choices": []}))
    with pytest.raises(GenerationProviderError, match="missing") as malformed:
        GenerationService(cfg, OpenRouterGenerationProvider(api_key="secret")).generate(generation_input(cfg))
    assert malformed.value.retryable is False

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response({"choices": [{"message": {"content": None}}]}))
    provider = OpenRouterGenerationProvider(api_key="secret")
    with pytest.raises(GenerationProviderError, match="empty") as empty:
        GenerationService(cfg, provider).generate(generation_input(cfg))
    assert empty.value.attempts == 2 and provider.calls == 2


def test_cache_reuse_invalidation_corruption_and_lineage(tmp_path: Path):
    cache = GenerationCache(tmp_path / "cache")
    cfg = config()
    provider = DeterministicFakeGenerationProvider()
    service = GenerationService(cfg, provider, cache=cache)
    original = service.generate(generation_input(cfg))
    repeated = service.generate(generation_input(cfg))
    assert repeated == original and provider.calls == 1 and service.cache_hits == 1

    for item in (
        generation_input(cfg, question="changed"),
        generation_input(cfg, result=context("changed context")),
    ):
        service.generate(item)
    assert provider.calls == 3
    for changed_cfg in (
        replace(cfg, model="other"), replace(cfg, temperature=0.2),
        replace(cfg, max_output_tokens=256),
    ):
        GenerationService(changed_cfg, provider, cache=cache).generate(generation_input(changed_cfg))
    assert provider.calls == 6
    path = tmp_path / "cache" / f"{generation_cache_key(original.prompt_fingerprint, cfg.fingerprint)}.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(GenerationCacheError):
        GenerationService(cfg, provider, cache=cache).generate(generation_input(cfg))


def test_answer_result_round_trip_and_fingerprints_survive():
    cfg = config()
    result = GenerationService(cfg, DeterministicFakeGenerationProvider()).generate(generation_input(cfg))
    restored = AnswerResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored == result
    assert restored.query_id == "q-1"
    assert restored.context_fingerprint == context().context_fingerprint
    bad = restored.to_dict()
    bad["answer_text"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint"):
        AnswerResult.from_dict(bad)


def test_artifacts_manifest_last_failures_and_resume_via_cache(tmp_path: Path):
    cfg = config(max_retries=0)
    cache = GenerationCache(tmp_path / "cache")
    items = [
        generation_input(cfg, query_id="q-1", result=context("one", query_id="q-1")),
        generation_input(cfg, query_id="q-2", result=context("two", query_id="q-2")),
    ]
    failing = DeterministicFakeGenerationProvider(fail_times=1)
    partial = run_generation(items, GenerationService(cfg, failing, cache=cache), tmp_path / "run")
    assert partial.complete is False
    assert not (tmp_path / "run" / "manifest.json").exists()
    assert [item["query_id"] for item in partial.failures] == ["q-1"]
    assert (tmp_path / "run" / "answers.jsonl").exists()

    provider = DeterministicFakeGenerationProvider()
    complete = run_generation(items, GenerationService(cfg, provider, cache=cache), tmp_path / "run")
    assert complete.complete is True
    assert provider.calls == 1  # q-2 reused from the partial run cache
    assert json.loads((tmp_path / "run" / "manifest.json").read_text())["complete"] is True
    calls = provider.calls
    reused = run_generation(items, GenerationService(cfg, provider, cache=cache), tmp_path / "run")
    assert reused.complete and provider.calls == calls


def test_interrupted_run_never_commits(tmp_path: Path):
    class Interrupting:
        calls = 0
        retries = 0
        def complete(self, messages, cfg):
            self.calls += 1
            raise KeyboardInterrupt()
    cfg = config()
    with pytest.raises(KeyboardInterrupt):
        run_generation(
            [generation_input(cfg)], GenerationService(cfg, Interrupting()), tmp_path / "run"
        )
    assert not (tmp_path / "run" / "manifest.json").exists()


def test_offline_three_strategy_context_to_answer_integration(tmp_path: Path):
    cfg = config()
    service = GenerationService(
        cfg, DeterministicFakeGenerationProvider(), cache=GenerationCache(tmp_path / "cache")
    )
    inputs = [
        generation_input(cfg, result=context(f"evidence variant {index}", strategy=strategy))
        for index, strategy in enumerate(STRATEGIES)
    ]
    first = [service.generate(item) for item in inputs]
    calls = service.provider.calls
    second = [service.generate(item) for item in inputs]
    assert all(item.status == "success" for item in first)
    assert len({item.generation_config_fingerprint for item in first}) == 1
    assert len({item.prompt_fingerprint for item in first}) == 3
    assert [item.answer_text for item in first] == [item.answer_text for item in second]
    assert service.provider.calls == calls


def test_offline_retrieval_protocol_context_generation_all_strategies(tmp_path: Path):
    embedding = EmbeddingConfig(provider="fake", model="fake-embedding-v1", dimension=4)
    index_directories = {}
    for strategy in STRATEGIES:
        text = f"authoritative {strategy} evidence"
        record = EmbeddingRecord(
            chunk_id=f"angular:guide.md::{strategy}::000000", doc_id="angular:guide.md",
            strategy=strategy, chunk_config_fingerprint=f"chunks-{strategy}", text=text,
            metadata={"gold_answer": "must remain upstream-only"},
            embedding=[1.0, 0.0, 0.0, 0.0], embedding_provider=embedding.provider,
            embedding_model=embedding.model, embedding_dimension=embedding.dimension,
            embedding_config_fingerprint=embedding.fingerprint, source="angular",
            relative_path="guide.md", chunk_index=0,
            token_count=len(TOKENIZER.encode(text)), text_sha256=content_sha256(text),
        )
        embedding_dir = tmp_path / "embeddings" / strategy
        index_dir = tmp_path / "indexes" / strategy
        serialized = serialize_embedding_records([record])
        manifest = {
            "schema_version": "embedding_record_v1", "complete": True, "corpus": "angular",
            "chunk_strategy": strategy, "chunk_config_fingerprint": f"chunks-{strategy}",
            "chunk_manifest_sha256": f"fixture-{strategy}", "chunk_artifact": "fixture",
            "documents": 1, "chunk_count": 1, "embedding_provider": embedding.provider,
            "embedding_model": embedding.model, "embedding_dimension": embedding.dimension,
            "embedding_configuration": embedding.identity(),
            "embedding_config_fingerprint": embedding.fingerprint,
            "embedding_artifact_fingerprint": artifact_sha256(serialized),
        }
        write_embedding_artifacts(embedding_dir, [record], manifest, {"embedded_chunks": 1})
        build_local_index(embedding_dir, index_dir)
        index_directories[strategy] = index_dir

    retrieval = RetrievalService(
        corpus="angular", index_directories=index_directories, embedding_config=embedding,
        provider=DeterministicFakeEmbeddingProvider(embedding),
        query_cache_directory=tmp_path / "query-cache", repository_root=tmp_path,
    )
    vector = [1.0, 0.0, 0.0, 0.0]
    protocol = RetrievalProtocolConfig(SAME_TOP_K, top_k=1, candidate_k=1)
    generation_cfg = config()
    generation = GenerationService(
        generation_cfg, DeterministicFakeGenerationProvider(),
        cache=GenerationCache(tmp_path / "generation-cache"),
    )
    answers = []
    for strategy in STRATEGIES:
        retrieved = retrieval.retrieve(
            RetrievalRequest("exact shared question", strategy, 1), query_vector=vector
        )
        selection = apply_retrieval_protocol(retrieved.hits, protocol)
        handoff = ContextBuildInput.from_retrieval(
            query_id="q-shared", result=retrieved, selection=selection,
            protocol_config_fingerprint=protocol.fingerprint,
            dataset_fingerprint="dataset",
        )
        authoritative = ContextBuilder(ContextConfig()).build(handoff)
        answer = generation.generate(GenerationInput.create(
            "q-shared", "exact shared question", authoritative, generation_cfg
        ))
        answers.append(answer)
        assert "gold_answer" not in AnswerPromptBuilder(generation_cfg).build(
            GenerationInput.create("q-shared", "exact shared question", authoritative, generation_cfg)
        ).messages[1].content
    assert all(answer.status == "success" for answer in answers)
    assert len({answer.generation_config_fingerprint for answer in answers}) == 1
    assert len({answer.prompt_fingerprint for answer in answers}) == 3
    calls = generation.provider.calls
    for answer, strategy in zip(answers, STRATEGIES):
        retrieved = retrieval.retrieve(
            RetrievalRequest("exact shared question", strategy, 1), query_vector=vector
        )
        selection = apply_retrieval_protocol(retrieved.hits, protocol)
        authoritative = ContextBuilder(ContextConfig()).build(ContextBuildInput.from_retrieval(
            query_id="q-shared", result=retrieved, selection=selection,
            protocol_config_fingerprint=protocol.fingerprint, dataset_fingerprint="dataset",
        ))
        repeated = generation.generate(GenerationInput.create(
            "q-shared", "exact shared question", authoritative, generation_cfg
        ))
        assert repeated.answer_text == answer.answer_text
    assert generation.provider.calls == calls


def test_v1_fingerprint_is_preserved_and_numeric_semantics_are_normalized():
    integer = GenerationConfig(
        provider="openrouter", model="openai/gpt-5-mini", temperature=0,
        max_output_tokens=512,
    )
    floating = replace(integer, temperature=0.0)
    assert integer.fingerprint == floating.fingerprint
    assert integer.fingerprint == "1045c3382003284e5fc02ebe1b86834d45423dc08313e773c4409acf4bad6cb6"


def test_v2_semantics_change_fingerprint_and_separate_cache(tmp_path: Path):
    v1 = config(max_output_tokens=1024)
    v2 = v2_config(max_output_tokens=1024, reasoning_effort="low")
    assert v1.fingerprint != v2.fingerprint
    cache = GenerationCache(tmp_path / "cache")
    provider = DeterministicFakeGenerationProvider()
    GenerationService(v1, provider, cache=cache).generate(generation_input(v1))
    GenerationService(v2, provider, cache=cache).generate(generation_input(v2))
    GenerationService(v2, provider, cache=cache).generate(generation_input(v2))
    assert provider.calls == 2


def test_v2_rejects_length_and_does_not_cache(tmp_path: Path):
    class LengthProvider:
        calls = 0
        retries = 0

        def complete(self, messages, cfg):
            self.calls += 1
            return ProviderResponse("partial answer", output_tokens=1024, finish_reason="length")

    cfg = v2_config(max_output_tokens=1024)
    provider = LengthProvider()
    cache = GenerationCache(tmp_path / "cache")
    with pytest.raises(GenerationIntegrityError, match="did not complete normally"):
        GenerationService(cfg, provider, cache=cache).generate(generation_input(cfg))
    assert provider.calls == 1
    assert not (tmp_path / "cache").exists()


def test_v2_rejects_mismatched_prepared_context_budget():
    cfg = v2_config(prepared_context_token_budget=2048)
    with pytest.raises(ValueError, match="prepared context token budget"):
        GenerationService(cfg, DeterministicFakeGenerationProvider()).generate(
            generation_input(cfg)
        )


def test_openrouter_reasoning_payload_and_safe_diagnostics(tmp_path: Path, monkeypatch):
    requests = []
    envelope = {
        "id": "request-fixture", "model": "openai/gpt-5-mini", "provider": "OpenAI",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": "complete answer", "reasoning": "fixture reasoning",
                "reasoning_details": [{"type": "reasoning.text", "text": "fixture"}],
            },
        }],
        "usage": {
            "prompt_tokens": 50, "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 12},
            "prompt_tokens_details": {"cached_tokens": 7},
        },
    }

    def urlopen(request, timeout):
        requests.append(request)
        return Response(envelope)
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    cfg = GenerationConfig(
        provider="openrouter", model="openai/gpt-5-mini", max_output_tokens=1024,
        schema_version="generation_config_v2", reasoning_effort="low",
        completion_integrity_policy="require_stop",
        response_handling_contract="nonempty_text_require_stop_v2",
        prepared_context_token_budget=4096,
    )
    diagnostic_path = tmp_path / "diagnostics.jsonl"
    raw_path = tmp_path / "raw"
    provider = OpenRouterGenerationProvider(
        api_key="never-serialize-this-secret", diagnostics_output=diagnostic_path,
        raw_diagnostics_output=raw_path,
    )
    result = GenerationService(cfg, provider).generate(generation_input(cfg))
    payload = json.loads(requests[0].data)
    assert payload["max_tokens"] == 1024
    assert payload["reasoning"] == {"effort": "low"}
    assert "max_completion_tokens" not in payload and "max_output_tokens" not in payload
    assert result.finish_reason == "stop"
    diagnostic = json.loads(diagnostic_path.read_text().strip())
    assert diagnostic["query_id"] == "q-1"
    assert diagnostic["reasoning_present"] is True
    assert diagnostic["reasoning_details_present"] is True
    assert diagnostic["reasoning_tokens"] == 12
    assert diagnostic["cached_tokens"] == 7
    generated = diagnostic_path.read_text() + next(raw_path.iterdir()).read_text()
    assert "never-serialize-this-secret" not in generated
    assert "Question:" not in generated


@pytest.mark.parametrize("content,content_type,is_null,is_empty", [
    (None, "null", True, False),
    ("", "string", False, True),
    ([{"type": "text", "text": "answer"}], "list", False, False),
])
def test_openrouter_diagnoses_invalid_content_shapes(
    tmp_path: Path, monkeypatch, content, content_type, is_null, is_empty,
):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: Response({
            "choices": [{"finish_reason": "length", "message": {"content": content}}],
            "usage": {"completion_tokens": 512},
        }),
    )
    cfg = GenerationConfig(
        provider="openrouter", model="model", max_retries=0, retry_backoff_seconds=0,
    )
    provider = OpenRouterGenerationProvider(
        api_key="secret", diagnostics_output=tmp_path / "diagnostics.jsonl",
    )
    with pytest.raises(GenerationProviderError, match="empty or non-text"):
        GenerationService(cfg, provider).generate(generation_input(cfg))
    diagnostic = provider.diagnostics[0]
    assert diagnostic["message_content_type"] == content_type
    assert diagnostic["content_is_null"] is is_null
    assert diagnostic["content_is_empty_string"] is is_empty
    assert diagnostic["finish_reason"] == "length"


def test_canonical_gpt5mini_v2_config_has_expected_fingerprint():
    path = Path(__file__).parents[1] / "configs" / "generation_gpt5mini_v2.json"
    cfg = GenerationConfig(**json.loads(path.read_text(encoding="utf-8")))
    assert cfg.fingerprint == "c4f4768ec9b80361dfd0a1e252f74ff348aa4e4c953bcca02761ba345f38b301"
    assert cfg.reasoning_effort == "low"
    assert cfg.completion_integrity_policy == "require_stop"
