"""Deterministic JSONL, manifest, and statistics output for chunks."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
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
        stream.flush()
        os.fsync(stream.fileno())


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    """Hash only the canonical, output-affecting strategy configuration."""

    validate_json_value(configuration, "configuration")
    encoded = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_artifact_inputs(
    chunks: list[Chunk], statistics: dict[str, Any], strategy: str
) -> None:
    """Reject stale counts, mixed schemas, duplicate IDs, and unstable ordering."""

    if statistics.get("chunks") != len(chunks):
        raise ValueError(
            "artifact statistics disagree for chunks: "
            f"{statistics.get('chunks')!r} != {len(chunks)}"
        )
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("artifacts contain duplicate chunk IDs")
    next_index: dict[str, int] = {}
    completed_documents: set[str] = set()
    current_document: str | None = None
    for chunk in chunks:
        chunk.validate()
        if chunk.strategy != strategy:
            raise ValueError(f"{strategy} artifacts contain a {chunk.strategy!r} chunk")
        if chunk.doc_id != current_document:
            if chunk.doc_id in completed_documents:
                raise ValueError("artifact chunks are not grouped in deterministic document order")
            if current_document is not None:
                completed_documents.add(current_document)
            current_document = chunk.doc_id
        expected = next_index.get(chunk.doc_id, 0)
        if chunk.chunk_index != expected:
            raise ValueError(
                f"artifacts for {chunk.doc_id!r} have non-contiguous chunk indexes"
            )
        next_index[chunk.doc_id] = expected + 1


def write_artifact_set(output_dir: Path, serialized: dict[str, str]) -> None:
    """Stage and rollback the three-file artifact set; publish manifest last."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_stage in output_dir.glob(".artifact-stage-*"):
        if stale_stage.is_dir():
            shutil.rmtree(stale_stage)
    stage = Path(tempfile.mkdtemp(prefix=".artifact-stage-", dir=output_dir))
    installed: list[str] = []
    backups: list[str] = []
    publish_order = [name for name in ("chunks.jsonl", "stats.json", "manifest.json") if name in serialized]
    try:
        for name, value in serialized.items():
            path = stage / name
            _write_text(path, value)
        for name in publish_order:
            final = output_dir / name
            backup = stage / f"{name}.previous"
            if final.exists():
                os.replace(final, backup)
                backups.append(name)
        for name in publish_order:
            os.replace(stage / name, output_dir / name)
            installed.append(name)
    except BaseException:
        for name in reversed(installed):
            final = output_dir / name
            if final.exists():
                final.unlink()
        for name in reversed(backups):
            backup = stage / f"{name}.previous"
            if backup.exists():
                os.replace(backup, output_dir / name)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def write_fixed_size_artifacts(
    chunks: list[Chunk],
    output_dir: Path,
    config: FixedSizeChunkingConfig,
    tokenizer: TiktokenTokenizer,
    statistics: dict[str, Any],
    source_input: str,
) -> None:
    validate_artifact_inputs(chunks, statistics, "fixed_size")
    configuration = {
        "strategy": "fixed_size",
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "tokenizer": tokenizer.name,
        "boundary_policy": "utf8_and_token_roundtrip_safe_minimal_backoff_v1",
    }
    manifest = {
        "schema_version": 1,
        "source_schema_version": NORMALIZED_SCHEMA_VERSION,
        "strategy": "fixed_size",
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "stride": config.stride,
        "tokenizer": tokenizer.name,
        "boundary_policy": configuration["boundary_policy"],
        "configuration": configuration,
        "config_fingerprint": configuration_fingerprint(configuration),
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
    write_artifact_set(output_dir, serialized)


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
