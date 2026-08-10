"""Deterministic JSONL, manifest, and statistics output for chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_chunking.data.models import NORMALIZED_SCHEMA_VERSION

from .fixed_size import FixedSizeChunkingConfig
from .models import Chunk, validate_json_value
from .tokenizer import TiktokenTokenizer


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def serialize_json(value: dict[str, Any]) -> str:
    validate_json_value(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"


def serialize_chunks_jsonl(chunks: list[Chunk]) -> str:
    return "".join(
        json.dumps(
            chunk.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for chunk in chunks
    )


def _write_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def write_fixed_size_artifacts(
    chunks: list[Chunk],
    output_dir: Path,
    config: FixedSizeChunkingConfig,
    tokenizer: TiktokenTokenizer,
    statistics: dict[str, Any],
    source_input: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "source_schema_version": NORMALIZED_SCHEMA_VERSION,
        "strategy": "fixed_size",
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "stride": config.stride,
        "tokenizer": tokenizer.name,
        "boundary_policy": "utf8_safe_minimal_backoff",
        "source_input": Path(source_input).as_posix(),
        "documents": statistics["documents"],
        "chunks": statistics["chunks"],
    }
    # Materialize all content before touching final paths. A schema/JSON error
    # therefore cannot leave a new partial artifact set or overwrite a valid one.
    serialized = {
        "chunks.jsonl": serialize_chunks_jsonl(chunks),
        "manifest.json": serialize_json(manifest),
        "stats.json": serialize_json(statistics),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in serialized.items():
        _write_text(output_dir / name, value)


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(
                    Chunk.from_dict(
                        json.loads(
                            line,
                            parse_constant=_reject_json_constant,
                            object_pairs_hook=_unique_json_object,
                        )
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid chunk JSONL at {path}:{line_number}: {error}") from error
    return chunks
