"""Content-addressed query embedding cache, separate from document embeddings."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rag_chunking.chunking.models import validate_json_value
from rag_chunking.embedding.models import EmbeddingConfig, canonical_fingerprint, content_sha256, validate_vector


QUERY_CACHE_SCHEMA_VERSION = "query_embedding_cache_v1"


class QueryEmbeddingCache:
    def __init__(self, directory: Path, config: EmbeddingConfig):
        self.directory = directory
        self.config = config

    def key(self, normalized_query: str) -> str:
        return canonical_fingerprint({
            "cache_schema_version": QUERY_CACHE_SCHEMA_VERSION,
            "normalized_query": normalized_query,
            "embedding_config_fingerprint": self.config.fingerprint,
            "input_type": "query",
        })

    def get(self, normalized_query: str) -> list[float] | None:
        digest = self.key(normalized_query)
        path = self.directory / self.config.fingerprint / f"{digest}.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "cache_schema_version": QUERY_CACHE_SCHEMA_VERSION,
                "cache_key": digest,
                "query_sha256": content_sha256(normalized_query),
                "embedding_config_fingerprint": self.config.fingerprint,
                "input_type": "query",
                "dimension": self.config.dimension,
            }
            if not isinstance(value, dict):
                raise ValueError("entry is not an object")
            for name, item in expected.items():
                if value.get(name) != item:
                    raise ValueError(f"{name} identity mismatch")
            vector = value.get("embedding")
            validate_vector(vector, self.config.dimension)
            return vector
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"Corrupt query embedding cache entry {path}: {error}") from error

    def put(self, normalized_query: str, vector: list[float]) -> None:
        validate_vector(vector, self.config.dimension)
        digest = self.key(normalized_query)
        directory = self.directory / self.config.fingerprint
        value: dict[str, Any] = {
            "cache_schema_version": QUERY_CACHE_SCHEMA_VERSION,
            "cache_key": digest,
            "query_sha256": content_sha256(normalized_query),
            "embedding_config_fingerprint": self.config.fingerprint,
            "input_type": "query",
            "dimension": self.config.dimension,
            "embedding": vector,
        }
        validate_json_value(value)
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / f"{digest}.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
