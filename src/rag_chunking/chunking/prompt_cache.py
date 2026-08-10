"""Content-addressed, persistent cache for validated planner responses."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import validate_json_value


CACHE_VERSION = "prompt_response_cache_v2"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


class CachedPlannerResponse:
    def __init__(self, response: str, metadata: dict[str, Any]):
        self.response = response
        self.metadata = metadata


def canonical_json(value: object) -> str:
    validate_json_value(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


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
            value = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
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
        validate_json_value(request, "request")
        validate_json_value(response_metadata or {}, "response_metadata")
        digest = request_digest(request)
        self.directory.mkdir(parents=True, exist_ok=True)
        value = {
            "cache_version": CACHE_VERSION,
            "request_sha256": digest,
            "request": request,
            "response": response,
            "response_metadata": response_metadata or {},
        }
        data = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
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
