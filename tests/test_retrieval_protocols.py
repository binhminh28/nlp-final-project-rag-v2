from dataclasses import replace

import pytest

from rag_chunking.retrieval.models import RetrievalHit
from rag_chunking.retrieval.protocols import (
    SAME_TOKEN_BUDGET, SAME_TOP_K, RetrievalProtocolConfig, apply_retrieval_protocol,
)


def hit(rank: int, tokens: int, chunk_id: str | None = None) -> RetrievalHit:
    text = "x" * tokens
    return RetrievalHit(
        rank=rank, score=1.0 - rank / 100, chunk_id=chunk_id or f"chunk-{rank}",
        doc_id="doc", source="source", relative_path="doc.md", strategy="fixed_size",
        text=text, metadata={}, token_count=tokens, character_count=len(text),
        chunk_config_fingerprint="chunks",
    )


def test_same_top_k_exact_and_candidate_exhaustion():
    candidates = [hit(rank, 10) for rank in range(1, 5)]
    selected = apply_retrieval_protocol(candidates, RetrievalProtocolConfig(SAME_TOP_K, top_k=3, candidate_k=4))
    assert [value.rank for value in selected.hits] == [1, 2, 3]
    assert selected.actual_selected_tokens == 30
    short = apply_retrieval_protocol(candidates[:2], RetrievalProtocolConfig(SAME_TOP_K, top_k=3, candidate_k=4))
    assert len(short.hits) == 2 and short.candidate_exhausted


def test_token_budget_skips_nonfitting_preserves_order_and_has_no_duplicates():
    candidates = [hit(1, 60), hit(2, 50), hit(3, 40), hit(4, 10)]
    config = RetrievalProtocolConfig(SAME_TOKEN_BUDGET, candidate_k=4, token_budget=100)
    selected = apply_retrieval_protocol(candidates, config)
    assert [value.rank for value in selected.hits] == [1, 3]
    assert selected.actual_selected_tokens == 100
    assert selected.budget_utilization == 1.0 and not selected.budget_overflow


def test_token_budget_oversized_first_and_candidate_exhaustion():
    config = RetrievalProtocolConfig(SAME_TOKEN_BUDGET, candidate_k=5, token_budget=50)
    selected = apply_retrieval_protocol([hit(1, 51), hit(2, 1)], config)
    assert [value.rank for value in selected.hits] == [1]
    assert selected.actual_selected_tokens == 51 and selected.budget_overflow
    assert selected.candidate_exhausted


def test_protocol_fingerprint_is_canonical_and_semantic():
    first = RetrievalProtocolConfig(SAME_TOKEN_BUDGET, candidate_k=20, token_budget=2048)
    same = RetrievalProtocolConfig(**dict(reversed(list(first.identity().items()))))
    changed = replace(first, token_budget=4096)
    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != changed.fingerprint
    top_k = RetrievalProtocolConfig(SAME_TOP_K, top_k=5, candidate_k=50)
    # Candidate pool and token policy cannot affect conventional top-k output.
    assert top_k.fingerprint == replace(top_k, candidate_k=100, tokenizer="unused").fingerprint


def test_protocol_rejects_duplicates_and_invalid_configuration():
    with pytest.raises(ValueError, match="duplicate"):
        apply_retrieval_protocol([hit(1, 1, "same"), hit(2, 1, "same")], RetrievalProtocolConfig(SAME_TOP_K))
    with pytest.raises(ValueError, match="token_budget"):
        RetrievalProtocolConfig(SAME_TOKEN_BUDGET)
    with pytest.raises(ValueError, match="at least top_k"):
        RetrievalProtocolConfig(SAME_TOP_K, top_k=10, candidate_k=5)
