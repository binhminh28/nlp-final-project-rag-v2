"""Content-addressed, validated, atomic embedding cache."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rag_chunking.chunking.models import validate_json_value

from .models import CACHE_SCHEMA_VERSION, EmbeddingConfig, canonical_fingerprint, content_sha256, validate_vector


class EmbeddingCache:
    def __init__(self, directory: Path, config: EmbeddingConfig):
        self.directory = directory
        self.config = config

    def key(self, text: str) -> str:
        return canonical_fingerprint({
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "text_sha256": content_sha256(text),
            "embedding_configuration": self.config.identity(),
        })

    def get(self, text: str) -> list[float] | None:
        digest = self.key(text)
        path = self.directory / f"{digest}.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("entry is not an object")
            expected = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "cache_key": digest,
                "text_sha256": content_sha256(text),
                "embedding_config_fingerprint": self.config.fingerprint,
                "provider": self.config.provider,
                "model": self.config.model,
                "dimension": self.config.dimension,
            }
            for name, item in expected.items():
                if value.get(name) != item:
                    raise ValueError(f"{name} identity mismatch")
            vector = value.get("embedding")
            validate_vector(vector, self.config.dimension)
            return vector
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"Corrupt embedding cache entry {path}: {error}") from error

    def put(self, text: str, vector: list[float]) -> None:
        validate_vector(vector, self.config.dimension)
        digest = self.key(text)
        value: dict[str, Any] = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": digest,
            "text_sha256": content_sha256(text),
            "embedding_config_fingerprint": self.config.fingerprint,
            "provider": self.config.provider,
            "model": self.config.model,
            "dimension": self.config.dimension,
            "embedding": vector,
        }
        validate_json_value(value)
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        self.directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / f"{digest}.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
