"""Canonical embedding records and experiment identity contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from rag_chunking.chunking.models import Chunk, validate_json_value


EMBEDDING_SCHEMA_VERSION = "embedding_record_v1"
INDEX_SCHEMA_VERSION = "local_cosine_index_v1"
CACHE_SCHEMA_VERSION = "embedding_cache_v1"


def canonical_fingerprint(value: dict[str, Any]) -> str:
    validate_json_value(value)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_vector(vector: list[float], dimension: int) -> None:
    if not isinstance(vector, list) or not vector:
        raise ValueError("embedding must be a non-empty list")
    if len(vector) != dimension:
        raise ValueError(f"embedding dimension {len(vector)} != expected {dimension}")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
        raise ValueError("embedding contains a non-finite or non-numeric value")


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimension: int
    max_batch_items: int = 64
    max_batch_tokens: int = 8192
    max_input_tokens: int = 8192
    tokenizer: str = "cl100k_base"
    input_type: str | None = None
    encoding_format: str = "float"

    def __post_init__(self) -> None:
        if not self.provider or not self.model or not self.tokenizer:
            raise ValueError("provider, model, and tokenizer must be non-empty")
        for name in ("dimension", "max_batch_items", "max_batch_tokens", "max_input_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.encoding_format != "float":
            raise ValueError("only canonical float embeddings are supported")

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": EMBEDDING_SCHEMA_VERSION,
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "tokenizer": self.tokenizer,
            "max_input_tokens": self.max_input_tokens,
            "input_type": self.input_type,
            "encoding_format": self.encoding_format,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.identity())


@dataclass(slots=True)
class EmbeddingRecord:
    chunk_id: str
    doc_id: str
    strategy: str
    chunk_config_fingerprint: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    embedding_config_fingerprint: str
    source: str
    relative_path: str
    chunk_index: int
    token_count: int
    text_sha256: str
    schema_version: str = EMBEDDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "chunk_id", "doc_id", "strategy", "chunk_config_fingerprint", "text",
            "embedding_provider", "embedding_model", "embedding_config_fingerprint",
            "source", "relative_path", "text_sha256", "schema_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != EMBEDDING_SCHEMA_VERSION:
            raise ValueError(f"unsupported embedding schema {self.schema_version!r}")
        if self.chunk_index < 0 or self.token_count <= 0:
            raise ValueError("chunk_index/token_count are invalid")
        if content_sha256(self.text) != self.text_sha256:
            raise ValueError("text_sha256 does not match text")
        validate_json_value(self.metadata, "metadata")
        validate_vector(self.embedding, self.embedding_dimension)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EmbeddingRecord":
        if not isinstance(value, dict):
            raise ValueError("embedding record must be an object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown embedding fields: {unknown}")
        return cls(**value)

    @classmethod
    def from_chunk(
        cls, chunk: Chunk, vector: list[float], config: EmbeddingConfig,
        chunk_config_fingerprint: str,
    ) -> "EmbeddingRecord":
        return cls(
            chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, strategy=chunk.strategy,
            chunk_config_fingerprint=chunk_config_fingerprint, text=chunk.text,
            metadata=dict(chunk.metadata), embedding=vector,
            embedding_provider=config.provider, embedding_model=config.model,
            embedding_dimension=config.dimension,
            embedding_config_fingerprint=config.fingerprint, source=chunk.source,
            relative_path=chunk.relative_path, chunk_index=chunk.chunk_index,
            token_count=chunk.token_count, text_sha256=content_sha256(chunk.text),
        )


def index_identity(
    *, corpus: str, strategy: str, chunk_config_fingerprint: str,
    embedding_config_fingerprint: str, embedding_artifact_fingerprint: str,
) -> dict[str, str]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "backend": "local_cosine_jsonl",
        "corpus": corpus,
        "strategy": strategy,
        "chunk_config_fingerprint": chunk_config_fingerprint,
        "embedding_config_fingerprint": embedding_config_fingerprint,
        "embedding_artifact_fingerprint": embedding_artifact_fingerprint,
    }
