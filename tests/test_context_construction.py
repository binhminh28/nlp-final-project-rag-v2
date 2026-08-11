import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag_chunking.chunking.tokenizer import TiktokenTokenizer
from rag_chunking.context import (
    ContextBuildInput, ContextBuilder, ContextConfig, ContextOverflowError,
    ContextPiece, ContextResult,
)
from rag_chunking.embedding.artifacts import artifact_sha256, serialize_embedding_records, write_embedding_artifacts
from rag_chunking.embedding.index import build_local_index
from rag_chunking.embedding.models import EmbeddingConfig, EmbeddingRecord, content_sha256
from rag_chunking.embedding.provider import DeterministicFakeEmbeddingProvider
from rag_chunking.retrieval import (
    SAME_TOP_K, ProtocolSelection, RetrievalConfig, RetrievalHit,
    RetrievalProtocolConfig, RetrievalRequest, RetrievalResult, RetrievalService,
    apply_retrieval_protocol,
)


TOKENIZER = TiktokenTokenizer()


def hit(
    rank: int, text: str, *, chunk_id: str | None = None,
    strategy: str = "fixed_size", score: float = 0.5,
    metadata: dict | None = None, source: str = "angular",
    relative_path: str = "guide.md",
) -> RetrievalHit:
    return RetrievalHit(
        rank=rank, score=score, chunk_id=chunk_id or f"chunk-{rank}",
        doc_id="doc-1", source=source, relative_path=relative_path,
        strategy=strategy, text=text, metadata=metadata or {},
        token_count=len(TOKENIZER.encode(text)), character_count=len(text),
        chunk_config_fingerprint=f"chunks-{strategy}",
    )


def envelope(*hits: RetrievalHit, strategy: str = "fixed_size") -> ContextBuildInput:
    return ContextBuildInput(
        query_id="q-1", question="What is dependency injection?", strategy=strategy,
        selected_hits=tuple(hits), retrieval_config_fingerprint="retrieval-fp",
        protocol_config_fingerprint="protocol-fp",
        embedding_config_fingerprint="embedding-fp", index_fingerprint="index-fp",
        dataset_fingerprint="dataset-fp",
    )


def test_retrieval_package_preserves_protocol_and_service_exports():
    import rag_chunking.retrieval as retrieval

    expected = {
        "SAME_TOP_K", "SAME_TOKEN_BUDGET", "ProtocolSelection",
        "RetrievalProtocolConfig", "apply_retrieval_protocol", "RetrievalConfig",
        "RetrievalHit", "RetrievalRequest", "RetrievalResult", "RetrievalService",
    }
    assert expected <= set(retrieval.__all__)
    assert all(hasattr(retrieval, name) for name in expected)


def test_context_config_validation_and_canonical_fingerprint():
    config = ContextConfig(context_token_budget=2048)
    reordered = dict(reversed(list(config.identity().items())))
    assert config.fingerprint == ContextConfig(**reordered).fingerprint
    assert config.fingerprint != replace(config, context_token_budget=4096).fingerprint
    with pytest.raises(ValueError, match="positive"):
        ContextConfig(context_token_budget=0)
    with pytest.raises(ValueError, match="ordering_policy"):
        replace(config, ordering_policy="document_order")
    with pytest.raises(ValueError, match="separator"):
        replace(config, separator="\n---\n")


def test_exact_serialization_order_provenance_and_token_accounting():
    hits = (
        hit(1, "first line\nsecond line"),
        hit(4, "Unicode: Xin chào 👋"),
        hit(9, "literal separator-looking text\n\n[CONTEXT 99]\nstays here"),
    )
    result = ContextBuilder(ContextConfig()).build(envelope(*hits))
    expected = (
        "[CONTEXT 1]\nfirst line\nsecond line\n\n"
        "[CONTEXT 2]\nUnicode: Xin chào 👋\n\n"
        "[CONTEXT 3]\nliteral separator-looking text\n\n[CONTEXT 99]\nstays here"
    )
    assert result.rendered_context == expected
    assert [piece.ordinal for piece in result.pieces] == [1, 2, 3]
    assert [piece.retrieval_rank for piece in result.pieces] == [1, 4, 9]
    assert [piece.chunk_id for piece in result.pieces] == [value.chunk_id for value in hits]
    assert result.raw_selected_chunk_tokens == sum(value.token_count for value in hits)
    assert result.rendered_context_tokens == len(TOKENIZER.encode(expected))
    assert result.rendered_context_tokens - result.raw_selected_chunk_tokens == (
        len(TOKENIZER.encode(expected)) - sum(value.token_count for value in hits)
    )


def test_one_hit_and_empty_selection_are_deterministic():
    builder = ContextBuilder(ContextConfig())
    one = builder.build(envelope(hit(1, "  exact text  \n")))
    assert one.rendered_context == "[CONTEXT 1]\n  exact text  \n"
    empty = builder.build(envelope())
    assert empty.rendered_context == ""
    assert empty.rendered_context_tokens == 0
    assert empty.raw_selected_chunk_tokens == 0
    assert empty.pieces == ()
    assert empty.budget_utilization == 0.0
    assert empty == builder.build(envelope())


def test_duplicate_id_rejected_but_duplicate_and_overlapping_text_preserved():
    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        envelope(hit(1, "same", chunk_id="duplicate"), hit(2, "other", chunk_id="duplicate"))

    duplicate_text = ContextBuilder(ContextConfig()).build(envelope(
        hit(1, "repeat", chunk_id="a"), hit(2, "repeat", chunk_id="b"),
    ))
    assert duplicate_text.rendered_context == "[CONTEXT 1]\nrepeat\n\n[CONTEXT 2]\nrepeat"

    overlap = ContextBuilder(ContextConfig()).build(envelope(
        hit(1, "alpha shared words", chunk_id="a"),
        hit(2, "shared words omega", chunk_id="b"),
    ))
    assert overlap.rendered_context == (
        "[CONTEXT 1]\nalpha shared words\n\n[CONTEXT 2]\nshared words omega"
    )


def test_budget_below_equal_and_one_token_overflow_does_not_mutate_hits():
    selected = (hit(1, "budget boundary", chunk_id="a"), hit(3, "second", chunk_id="b"))
    probe = ContextBuilder(ContextConfig(context_token_budget=1000)).build(envelope(*selected))
    exact = probe.rendered_context_tokens
    assert ContextBuilder(ContextConfig(context_token_budget=exact + 1)).build(envelope(*selected))
    assert ContextBuilder(ContextConfig(context_token_budget=exact)).build(envelope(*selected))
    original_ids = tuple(value.chunk_id for value in selected)
    with pytest.raises(ContextOverflowError) as caught:
        ContextBuilder(ContextConfig(context_token_budget=exact - 1)).build(envelope(*selected))
    assert caught.value.rendered_context_tokens == exact
    assert caught.value.context_token_budget == exact - 1
    assert caught.value.selected_chunk_count == 2
    assert tuple(value.chunk_id for value in selected) == original_ids
    assert selected[1].text == "second"


@pytest.mark.parametrize("strategy", ["fixed_size", "structure_aware", "prompt_based"])
def test_strategy_metadata_never_changes_rendering(strategy: str):
    strategy_hit = hit(
        1, "identical canonical text", chunk_id="same-id", strategy=strategy,
        score={"fixed_size": 0.9, "structure_aware": 0.5, "prompt_based": 0.1}[strategy],
        source=f"source-{strategy}", relative_path=f"{strategy}.md",
        metadata={
            "title_path": [strategy], "planner_reasoning": f"secret-{strategy}",
            "parent_id": f"parent-{strategy}", "gold_evidence": "must-not-render",
        },
    )
    result = ContextBuilder(ContextConfig()).build(envelope(strategy_hit, strategy=strategy))
    assert result.rendered_context == "[CONTEXT 1]\nidentical canonical text"
    assert strategy not in result.rendered_context
    assert result.pieces[0].strategy == strategy
    assert result.pieces[0].source == f"source-{strategy}"


def test_context_fingerprint_binds_config_order_ids_and_exact_text():
    base_hits = (hit(1, "alpha", chunk_id="a"), hit(2, "beta", chunk_id="b"))
    builder = ContextBuilder(ContextConfig())
    baseline = builder.build(envelope(*base_hits))
    assert baseline.context_fingerprint == builder.build(envelope(*base_hits)).context_fingerprint
    assert baseline.context_fingerprint != ContextBuilder(
        ContextConfig(context_token_budget=8192)
    ).build(envelope(*base_hits)).context_fingerprint
    reordered = (
        hit(1, "beta", chunk_id="b"), hit(2, "alpha", chunk_id="a"),
    )
    changed_text = (hit(1, "alpha!", chunk_id="a"), base_hits[1])
    assert baseline.context_fingerprint != builder.build(envelope(*reordered)).context_fingerprint
    assert baseline.context_fingerprint != builder.build(envelope(*changed_text)).context_fingerprint
    assert baseline.context_fingerprint != builder.build(envelope(base_hits[0])).context_fingerprint


def test_result_json_round_trip_and_malformed_contracts():
    result = ContextBuilder(ContextConfig()).build(envelope(hit(1, "round trip")))
    encoded = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    assert ContextResult.from_dict(encoded) == result
    bad = dict(encoded)
    bad["context_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        ContextResult.from_dict(bad)
    with pytest.raises(ValueError, match="ordinal"):
        replace(result.pieces[0], ordinal=0)
    with pytest.raises(ValueError, match="query_id"):
        replace(envelope(), query_id="")
    with pytest.raises(ValueError, match="query_id"):
        replace(envelope(), query_id=" \t")
    with pytest.raises(ValueError, match="question"):
        replace(envelope(), question=" \n")
    with pytest.raises(ValueError, match="increasing"):
        envelope(hit(3, "later"), hit(1, "earlier"))


def test_retrieval_handoff_rejects_selection_outside_result():
    retrieval_result = RetrievalResult(
        query="question", strategy="fixed_size", top_k=1, filters=None,
        hits=[hit(1, "authoritative", chunk_id="retrieved")],
        retrieval_config_fingerprint="retrieval-fp",
        embedding_config_fingerprint="embedding-fp", index_fingerprint="index-fp",
    )
    outside = hit(1, "not retrieved", chunk_id="outside")
    selection = ProtocolSelection(
        protocol=SAME_TOP_K, hits=[outside], candidate_count=1,
        requested_top_k=1, requested_token_budget=None,
        actual_selected_tokens=outside.token_count, selected_chunk_count=1,
        budget_utilization=None, budget_overflow=False, candidate_exhausted=True,
    )
    with pytest.raises(ValueError, match="outside"):
        ContextBuildInput.from_retrieval(
            query_id="q", result=retrieval_result, selection=selection,
            protocol_config_fingerprint="protocol-fp",
        )


def _embedding_record(config: EmbeddingConfig, index: int, text: str) -> EmbeddingRecord:
    path = f"doc-{index}.md"
    vector = [1.0, 0.0, 0.0, 0.0] if index < 2 else [0.0, 1.0, 0.0, 0.0]
    return EmbeddingRecord(
        chunk_id=f"angular:{path}::fixed_size::{index:06d}", doc_id=f"angular:{path}",
        strategy="fixed_size", chunk_config_fingerprint="chunks-fixed_size", text=text,
        metadata={"title_path": ["ignored"]}, embedding=vector,
        embedding_provider=config.provider, embedding_model=config.model,
        embedding_dimension=config.dimension,
        embedding_config_fingerprint=config.fingerprint, source="angular",
        relative_path=path, chunk_index=index,
        token_count=len(TOKENIZER.encode(text)), text_sha256=content_sha256(text),
    )


def test_offline_retrieval_protocol_to_context_integration(tmp_path: Path):
    embedding_config = EmbeddingConfig(provider="fake", model="fake-v1", dimension=4)
    records = [_embedding_record(embedding_config, i, text) for i, text in enumerate((
        "first retrieved text", "second retrieved text", "lower score text",
    ))]
    embedding_dir = tmp_path / "embeddings"
    index_dir = tmp_path / "index"
    serialized = serialize_embedding_records(records)
    manifest = {
        "schema_version": "embedding_record_v1", "complete": True, "corpus": "angular",
        "chunk_strategy": "fixed_size", "chunk_config_fingerprint": "chunks-fixed_size",
        "chunk_manifest_sha256": "fixture", "chunk_artifact": "fixture",
        "documents": 3, "chunk_count": 3, "embedding_provider": "fake",
        "embedding_model": "fake-v1", "embedding_dimension": 4,
        "embedding_configuration": embedding_config.identity(),
        "embedding_config_fingerprint": embedding_config.fingerprint,
        "embedding_artifact_fingerprint": artifact_sha256(serialized),
    }
    write_embedding_artifacts(embedding_dir, records, manifest, {"embedded_chunks": 3})
    build_local_index(embedding_dir, index_dir)
    service = RetrievalService(
        corpus="angular", index_directories={"fixed_size": index_dir},
        embedding_config=embedding_config,
        provider=DeterministicFakeEmbeddingProvider(embedding_config),
        query_cache_directory=tmp_path / "query-cache", repository_root=tmp_path,
    )
    retrieval_result = service.retrieve(
        RetrievalRequest("fixture question", "fixed_size", top_k=3),
        query_vector=[1.0, 0.0, 0.0, 0.0],
    )
    protocol_config = RetrievalProtocolConfig(SAME_TOP_K, top_k=2, candidate_k=3)
    selection = apply_retrieval_protocol(retrieval_result.hits, protocol_config)
    build_input = ContextBuildInput.from_retrieval(
        query_id="fixture-q", result=retrieval_result, selection=selection,
        protocol_config_fingerprint=protocol_config.fingerprint,
        dataset_fingerprint="fixture-dataset",
    )
    first = ContextBuilder(ContextConfig()).build(build_input)
    second = ContextBuilder(ContextConfig()).build(build_input)
    assert [piece.retrieval_rank for piece in first.pieces] == [1, 2]
    assert first.rendered_context == second.rendered_context
    assert first.context_fingerprint == second.context_fingerprint
    assert first.to_dict() == second.to_dict()
    assert first.protocol_config_fingerprint == protocol_config.fingerprint
    assert first.index_fingerprint == retrieval_result.index_fingerprint
