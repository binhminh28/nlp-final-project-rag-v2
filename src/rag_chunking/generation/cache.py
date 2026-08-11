"""Atomic content-addressed cache for validated answer results."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rag_chunking.embedding.models import canonical_fingerprint

from .models import AnswerResult


GENERATION_CACHE_SCHEMA_VERSION = "generation_cache_v1"


class GenerationCacheError(ValueError):
    pass


def generation_cache_key(prompt_fingerprint: str, generation_config_fingerprint: str) -> str:
    return canonical_fingerprint({
        "cache_schema_version": GENERATION_CACHE_SCHEMA_VERSION,
        "prompt_fingerprint": prompt_fingerprint,
        "generation_config_fingerprint": generation_config_fingerprint,
    })


class GenerationCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(self, prompt_fingerprint: str, generation_config_fingerprint: str) -> AnswerResult | None:
        key = generation_cache_key(prompt_fingerprint, generation_config_fingerprint)
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or value.get("cache_schema_version") != GENERATION_CACHE_SCHEMA_VERSION
                or value.get("cache_key") != key
                or value.get("prompt_fingerprint") != prompt_fingerprint
                or value.get("generation_config_fingerprint") != generation_config_fingerprint
                or not isinstance(value.get("result"), dict)
            ):
                raise ValueError("identity/schema validation failed")
            result = AnswerResult.from_dict(value["result"])
            if result.prompt_fingerprint != prompt_fingerprint or result.generation_config_fingerprint != generation_config_fingerprint:
                raise ValueError("cached result identity mismatch")
            return result
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise GenerationCacheError(f"Corrupt generation cache entry {path}: {error}") from error

    def put(self, result: AnswerResult) -> None:
        key = generation_cache_key(result.prompt_fingerprint, result.generation_config_fingerprint)
        self.directory.mkdir(parents=True, exist_ok=True)
        value: dict[str, Any] = {
            "cache_schema_version": GENERATION_CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "prompt_fingerprint": result.prompt_fingerprint,
            "generation_config_fingerprint": result.generation_config_fingerprint,
            "result": result.to_dict(),
        }
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        handle, temp_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.directory / f"{key}.json")
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
