"""Content-addressed, persistent cache for validated planner responses."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


CACHE_VERSION = "prompt_response_cache_v2"


class CachedPlannerResponse:
    def __init__(self, response: str, metadata: dict[str, Any]):
        self.response = response
        self.metadata = metadata


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_digest(request: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()


class PromptResponseCache:
    def __init__(self, directory: Path):
        self.directory = directory

    def get(self, request: dict[str, Any]) -> CachedPlannerResponse | None:
        digest = request_digest(request)
        path = self.directory / f"{digest}.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Corrupt prompt cache entry {path}: {error}") from error
        if (
            not isinstance(value, dict)
            or value.get("cache_version") != CACHE_VERSION
            or value.get("request_sha256") != digest
            or value.get("request") != request
            or not isinstance(value.get("response"), str)
            or not isinstance(value.get("response_metadata"), dict)
        ):
            raise ValueError(f"Prompt cache entry failed identity/schema validation: {path}")
        return CachedPlannerResponse(value["response"], value["response_metadata"])

    def put(
        self, request: dict[str, Any], response: str, response_metadata: dict[str, Any] | None = None
    ) -> None:
        digest = request_digest(request)
        self.directory.mkdir(parents=True, exist_ok=True)
        value = {
            "cache_version": CACHE_VERSION,
            "request_sha256": digest,
            "request": request,
            "response": response,
            "response_metadata": response_metadata or {},
        }
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        handle, temp_name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.directory / f"{digest}.json")
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
