"""Immutable contracts for deterministic retrieval-to-context handoff."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from rag_chunking.chunking.tokenizer import TiktokenTokenizer
from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.retrieval.models import KNOWN_STRATEGIES, RetrievalHit, RetrievalResult
from rag_chunking.retrieval.protocols import ProtocolSelection


CONTEXT_CONFIG_SCHEMA_VERSION = "context_config_v1"
CONTEXT_RESULT_SCHEMA_VERSION = "context_result_v1"
CONTEXT_FORMAT_VERSION = "context_format_v1"
CANONICAL_TOKENIZER = "tiktoken:cl100k_base"


def _require_nonempty(value: str | None, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Versioned semantic settings shared by every chunking strategy."""

    context_token_budget: int = 4096
    schema_version: str = CONTEXT_CONFIG_SCHEMA_VERSION
    format_version: str = CONTEXT_FORMAT_VERSION
    tokenizer: str = CANONICAL_TOKENIZER
    ordering_policy: str = "retrieval_rank"
    duplicate_chunk_policy: str = "reject"
    duplicate_text_policy: str = "preserve"
    overflow_policy: str = "fail"
    label_format: str = "[CONTEXT {ordinal}]"
    separator: str = "\n\n"
    separator_version: str = "blank_line_v1"
    metadata_inclusion_policy: str = "neutral_ordinal_labels_only"

    def __post_init__(self) -> None:
        if self.schema_version != CONTEXT_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported context config schema {self.schema_version!r}")
        if self.format_version != CONTEXT_FORMAT_VERSION:
            raise ValueError(f"unsupported context format {self.format_version!r}")
        if self.tokenizer != CANONICAL_TOKENIZER:
            raise ValueError(f"unsupported context tokenizer {self.tokenizer!r}")
        expected = {
            "ordering_policy": "retrieval_rank",
            "duplicate_chunk_policy": "reject",
            "duplicate_text_policy": "preserve",
            "overflow_policy": "fail",
            "separator_version": "blank_line_v1",
            "metadata_inclusion_policy": "neutral_ordinal_labels_only",
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise ValueError(f"unsupported {name} {getattr(self, name)!r}")
        if type(self.context_token_budget) is not int or self.context_token_budget <= 0:
            raise ValueError("context_token_budget must be a positive integer")
        if self.label_format != "[CONTEXT {ordinal}]":
            raise ValueError("unsupported label_format")
        if self.separator != "\n\n":
            raise ValueError("unsupported separator")

    def identity(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.identity())


@dataclass(frozen=True, slots=True)
class ContextBuildInput:
    """Minimal reproducibility envelope around authoritative selected hits."""

    query_id: str
    question: str
    strategy: str
    selected_hits: tuple[RetrievalHit, ...]
    retrieval_config_fingerprint: str
    protocol_config_fingerprint: str
    embedding_config_fingerprint: str
    index_fingerprint: str
    dataset_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "query_id", "question", "retrieval_config_fingerprint",
            "protocol_config_fingerprint", "embedding_config_fingerprint",
            "index_fingerprint",
        ):
            _require_nonempty(getattr(self, name), name)
        if not self.question.strip():
            raise ValueError("question must not be blank")
        _require_nonempty(self.dataset_fingerprint, "dataset_fingerprint", optional=True)
        if self.strategy not in KNOWN_STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if not isinstance(self.selected_hits, tuple):
            object.__setattr__(self, "selected_hits", tuple(self.selected_hits))
        if not all(isinstance(hit, RetrievalHit) for hit in self.selected_hits):
            raise ValueError("selected_hits must contain RetrievalHit values")
        if any(hit.strategy != self.strategy for hit in self.selected_hits):
            raise ValueError("selected hit strategy does not match input strategy")
        chunk_ids = [hit.chunk_id for hit in self.selected_hits]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("selected hits contain duplicate chunk IDs")
        ranks = [hit.rank for hit in self.selected_hits]
        if any(left >= right for left, right in zip(ranks, ranks[1:])):
            raise ValueError("selected hits must preserve strictly increasing retrieval ranks")

    @classmethod
    def from_retrieval(
        cls, *, query_id: str, result: RetrievalResult,
        selection: ProtocolSelection, protocol_config_fingerprint: str,
        dataset_fingerprint: str | None = None,
    ) -> "ContextBuildInput":
        """Create a handoff without reloading or re-retrieving selected chunks."""

        if selection.selected_chunk_count != len(selection.hits):
            raise ValueError("protocol selection count does not match its hits")
        if selection.actual_selected_tokens != sum(hit.token_count for hit in selection.hits):
            raise ValueError("protocol selection token count does not match its hits")
        result_hits = {hit.chunk_id: hit for hit in result.hits}
        if any(result_hits.get(hit.chunk_id) != hit for hit in selection.hits):
            raise ValueError("protocol selection contains a hit outside the retrieval result")
        return cls(
            query_id=query_id, question=result.query, strategy=result.strategy,
            selected_hits=tuple(selection.hits),
            retrieval_config_fingerprint=result.retrieval_config_fingerprint,
            protocol_config_fingerprint=protocol_config_fingerprint,
            embedding_config_fingerprint=result.embedding_config_fingerprint,
            index_fingerprint=result.index_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected_hits"] = [hit.to_dict() for hit in self.selected_hits]
        return value


@dataclass(frozen=True, slots=True)
class ContextPiece:
    ordinal: int
    retrieval_rank: int
    chunk_id: str
    doc_id: str
    source: str
    relative_path: str
    score: float
    raw_chunk_token_count: int
    rendered_block_token_count: int
    strategy: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ValueError("ordinal must be a positive integer")
        if type(self.retrieval_rank) is not int or self.retrieval_rank <= 0:
            raise ValueError("retrieval_rank must be a positive integer")
        for name in ("chunk_id", "doc_id", "source", "relative_path"):
            _require_nonempty(getattr(self, name), name)
        if self.strategy not in KNOWN_STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if type(self.score) not in (int, float) or not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if type(self.raw_chunk_token_count) is not int or self.raw_chunk_token_count < 0:
            raise ValueError("raw_chunk_token_count must be non-negative")
        if type(self.rendered_block_token_count) is not int or self.rendered_block_token_count < 0:
            raise ValueError("rendered_block_token_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContextPiece":
        if not isinstance(value, dict):
            raise ValueError("context piece must be an object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown context piece fields: {unknown}")
        return cls(**value)


def context_identity(
    context_config_fingerprint: str, pieces: tuple[ContextPiece, ...], rendered_context: str,
) -> dict[str, Any]:
    """Semantic identity excludes diagnostic scores and runtime state."""

    return {
        "context_config_fingerprint": context_config_fingerprint,
        "ordered_provenance": [
            {
                "ordinal": piece.ordinal,
                "retrieval_rank": piece.retrieval_rank,
                "chunk_id": piece.chunk_id,
                "doc_id": piece.doc_id,
            }
            for piece in pieces
        ],
        "rendered_context": rendered_context,
    }


@dataclass(frozen=True, slots=True)
class ContextResult:
    query_id: str
    strategy: str
    context_config_fingerprint: str
    retrieval_config_fingerprint: str
    protocol_config_fingerprint: str
    embedding_config_fingerprint: str
    index_fingerprint: str
    dataset_fingerprint: str | None
    pieces: tuple[ContextPiece, ...]
    rendered_context: str
    raw_selected_chunk_tokens: int
    rendered_context_tokens: int
    context_token_budget: int
    budget_utilization: float
    context_fingerprint: str
    tokenizer: str = CANONICAL_TOKENIZER
    status: str = "valid"
    schema_version: str = CONTEXT_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTEXT_RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported context result schema {self.schema_version!r}")
        if self.status != "valid":
            raise ValueError("a ContextResult must have valid status")
        for name in (
            "query_id", "context_config_fingerprint", "retrieval_config_fingerprint",
            "protocol_config_fingerprint", "embedding_config_fingerprint",
            "index_fingerprint", "context_fingerprint",
        ):
            _require_nonempty(getattr(self, name), name)
        _require_nonempty(self.dataset_fingerprint, "dataset_fingerprint", optional=True)
        if self.strategy not in KNOWN_STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if self.tokenizer != CANONICAL_TOKENIZER:
            raise ValueError(f"unsupported context tokenizer {self.tokenizer!r}")
        if not isinstance(self.pieces, tuple):
            object.__setattr__(self, "pieces", tuple(self.pieces))
        if [piece.ordinal for piece in self.pieces] != list(range(1, len(self.pieces) + 1)):
            raise ValueError("piece ordinals must be contiguous from 1")
        ranks = [piece.retrieval_rank for piece in self.pieces]
        if any(left >= right for left, right in zip(ranks, ranks[1:])):
            raise ValueError("pieces must preserve strictly increasing retrieval ranks")
        if any(piece.strategy != self.strategy for piece in self.pieces):
            raise ValueError("piece strategy does not match result strategy")
        if len({piece.chunk_id for piece in self.pieces}) != len(self.pieces):
            raise ValueError("pieces contain duplicate chunk IDs")
        if type(self.context_token_budget) is not int or self.context_token_budget <= 0:
            raise ValueError("context_token_budget must be positive")
        for name in ("raw_selected_chunk_tokens", "rendered_context_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.raw_selected_chunk_tokens != sum(piece.raw_chunk_token_count for piece in self.pieces):
            raise ValueError("raw selected token total does not match pieces")
        actual_tokens = len(TiktokenTokenizer().encode(self.rendered_context))
        if self.rendered_context_tokens != actual_tokens:
            raise ValueError("rendered context token count is not reproducible")
        if not self.pieces and self.rendered_context:
            raise ValueError("empty pieces require an empty rendered context")
        if self.pieces and not self.rendered_context:
            raise ValueError("non-empty pieces require rendered context")
        if self.rendered_context_tokens > self.context_token_budget:
            raise ValueError("successful context exceeds its token budget")
        expected_utilization = self.rendered_context_tokens / self.context_token_budget
        if type(self.budget_utilization) is not float or self.budget_utilization != expected_utilization:
            raise ValueError("budget_utilization does not match rendered tokens and budget")
        expected_fingerprint = canonical_fingerprint(
            context_identity(self.context_config_fingerprint, self.pieces, self.rendered_context)
        )
        if self.context_fingerprint != expected_fingerprint:
            raise ValueError("context fingerprint does not match contents")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["pieces"] = [piece.to_dict() for piece in self.pieces]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContextResult":
        if not isinstance(value, dict):
            raise ValueError("context result must be an object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown context result fields: {unknown}")
        data = dict(value)
        raw_pieces = data.get("pieces")
        if not isinstance(raw_pieces, (list, tuple)):
            raise ValueError("pieces must be an array")
        data["pieces"] = tuple(ContextPiece.from_dict(item) for item in raw_pieces)
        return cls(**data)
