"""Backend-independent retrieval contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from rag_chunking.chunking.models import validate_json_value
from rag_chunking.embedding.models import canonical_fingerprint


RETRIEVAL_SCHEMA_VERSION = "dense_retrieval_v1"
KNOWN_STRATEGIES = frozenset({"fixed_size", "structure_aware", "prompt_based"})
SUPPORTED_FILTERS = frozenset({"strategy", "doc_id", "source"})


def normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    similarity_metric: str = "cosine"
    schema_version: str = RETRIEVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RETRIEVAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported retrieval schema {self.schema_version!r}")
        if self.similarity_metric != "cosine":
            raise ValueError("only cosine similarity is supported")

    def identity(self) -> dict[str, str]:
        return {"schema_version": self.schema_version, "similarity_metric": self.similarity_metric}

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.identity())


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    strategy: str
    top_k: int = 5
    filters: dict[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", normalize_query(self.query))
        if self.strategy not in KNOWN_STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.filters is not None:
            if not isinstance(self.filters, dict):
                raise ValueError("filters must be an object")
            unknown = sorted(set(self.filters) - SUPPORTED_FILTERS)
            if unknown:
                raise ValueError(f"unsupported retrieval filters: {unknown}")
            if any(not isinstance(value, str) or not value for value in self.filters.values()):
                raise ValueError("filter values must be non-empty strings")
            normalized = dict(sorted(self.filters.items()))
            if "strategy" in normalized and normalized["strategy"] != self.strategy:
                raise ValueError("strategy filter conflicts with requested strategy")
            object.__setattr__(self, "filters", normalized)


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    source: str
    relative_path: str
    strategy: str
    text: str
    metadata: dict[str, Any]
    token_count: int
    character_count: int
    chunk_config_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if type(self.score) not in (int, float) or not math.isfinite(self.score):
            raise ValueError("score must be finite")
        for name in ("chunk_id", "doc_id", "source", "relative_path", "strategy", "text", "chunk_config_fingerprint"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.strategy not in KNOWN_STRATEGIES:
            raise ValueError("hit has unknown strategy")
        if self.token_count <= 0 or self.character_count != len(self.text):
            raise ValueError("hit length metadata is invalid")
        validate_json_value(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    strategy: str
    top_k: int
    filters: dict[str, str] | None
    hits: list[RetrievalHit] = field(default_factory=list)
    query_embedding_cache_hit: bool = False
    retrieval_config_fingerprint: str = ""
    embedding_config_fingerprint: str = ""
    index_fingerprint: str = ""

    def __post_init__(self) -> None:
        if [hit.rank for hit in self.hits] != list(range(1, len(self.hits) + 1)):
            raise ValueError("result ranks must be contiguous from 1")
        if any(hit.strategy != self.strategy for hit in self.hits):
            raise ValueError("result contains a hit from another strategy")
        if len({hit.chunk_id for hit in self.hits}) != len(self.hits):
            raise ValueError("result contains duplicate chunk IDs")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value
