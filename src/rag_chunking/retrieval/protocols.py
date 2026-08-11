"""Deterministic post-ranking policies for fair retrieval experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from rag_chunking.embedding.models import canonical_fingerprint

from .models import RetrievalHit


PROTOCOL_SCHEMA_VERSION = "retrieval_budget_protocol_v1"
SAME_TOP_K = "same_top_k"
SAME_TOKEN_BUDGET = "same_token_budget"
BudgetMode = Literal["same_top_k", "same_token_budget"]


@dataclass(frozen=True, slots=True)
class RetrievalProtocolConfig:
    """Output-affecting settings applied after canonical dense ranking."""

    mode: BudgetMode
    top_k: int = 5
    candidate_k: int = 50
    token_budget: int | None = None
    tokenizer: str = "tiktoken:cl100k_base"
    budget_policy: str = "whole_chunks_skip_nonfitting_oversized_first_v1"
    tie_breaking_rule: str = "score_desc_chunk_id_asc"
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.mode not in (SAME_TOP_K, SAME_TOKEN_BUDGET):
            raise ValueError(f"unsupported retrieval protocol {self.mode!r}")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if type(self.candidate_k) is not int or self.candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")
        if self.mode == SAME_TOP_K and self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        if self.mode == SAME_TOKEN_BUDGET:
            if type(self.token_budget) is not int or self.token_budget <= 0:
                raise ValueError("token_budget must be a positive integer for same_token_budget")
        elif self.token_budget is not None:
            raise ValueError("token_budget is only valid for same_token_budget")
        if not self.tokenizer or not self.budget_policy or not self.tie_breaking_rule:
            raise ValueError("protocol identities must be non-empty")

    def identity(self) -> dict[str, object]:
        """Return only settings that can affect this protocol's output."""

        common: dict[str, object] = {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "tie_breaking_rule": self.tie_breaking_rule,
        }
        if self.mode == SAME_TOP_K:
            return {**common, "top_k": self.top_k}
        return {
            **common,
            "candidate_k": self.candidate_k,
            "token_budget": self.token_budget,
            "tokenizer": self.tokenizer,
            "budget_policy": self.budget_policy,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.identity())


@dataclass(frozen=True, slots=True)
class ProtocolSelection:
    protocol: BudgetMode
    hits: list[RetrievalHit]
    candidate_count: int
    requested_top_k: int | None
    requested_token_budget: int | None
    actual_selected_tokens: int
    selected_chunk_count: int
    budget_utilization: float | None
    budget_overflow: bool
    candidate_exhausted: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_retrieval_protocol(
    ranked_candidates: list[RetrievalHit], config: RetrievalProtocolConfig,
) -> ProtocolSelection:
    """Select unique whole chunks while preserving their dense ranking order.

    Token accounting includes chunk text only through the canonical persisted
    ``token_count``. Metadata and separators are excluded and text is never
    truncated. If rank one alone is oversized it is selected and flagged. A
    later non-fitting chunk is skipped while the scan continues.
    """

    candidates = ranked_candidates[: config.candidate_k]
    if len({hit.chunk_id for hit in candidates}) != len(candidates):
        raise ValueError("ranked candidates contain duplicate chunk IDs")
    if any(left.rank >= right.rank for left, right in zip(candidates, candidates[1:])):
        raise ValueError("ranked candidates must preserve strictly increasing dense ranks")

    overflow = False
    if config.mode == SAME_TOP_K:
        selected = candidates[: config.top_k]
        requested_top_k: int | None = config.top_k
        requested_budget: int | None = None
        utilization: float | None = None
    else:
        assert config.token_budget is not None
        selected = []
        total = 0
        for hit in candidates:
            if not selected and hit.token_count > config.token_budget:
                selected.append(hit)
                total = hit.token_count
                overflow = True
                break
            if total + hit.token_count <= config.token_budget:
                selected.append(hit)
                total += hit.token_count
        requested_top_k = None
        requested_budget = config.token_budget
        utilization = total / config.token_budget

    actual_tokens = sum(hit.token_count for hit in selected)
    return ProtocolSelection(
        protocol=config.mode,
        hits=selected,
        candidate_count=len(candidates),
        requested_top_k=requested_top_k,
        requested_token_budget=requested_budget,
        actual_selected_tokens=actual_tokens,
        selected_chunk_count=len(selected),
        budget_utilization=utilization,
        budget_overflow=overflow,
        candidate_exhausted=len(candidates) < config.candidate_k,
    )
