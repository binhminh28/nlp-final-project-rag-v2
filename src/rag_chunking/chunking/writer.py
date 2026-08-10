"""Deterministic JSONL, manifest, and statistics output for chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_chunking.data.models import NORMALIZED_SCHEMA_VERSION

from .fixed_size import FixedSizeChunkingConfig
from .models import Chunk
from .tokenizer import TiktokenTokenizer


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def write_fixed_size_artifacts(
    chunks: list[Chunk],
    output_dir: Path,
    config: FixedSizeChunkingConfig,
    tokenizer: TiktokenTokenizer,
    statistics: dict[str, Any],
    source_input: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(
                json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            stream.write("\n")
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
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "stats.json", statistics)


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(Chunk.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid chunk JSONL at {path}:{line_number}: {error}") from error
    return chunks
