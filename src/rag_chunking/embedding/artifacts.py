"""Canonical readers and transactional embedding/index artifact publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag_chunking.chunking.writer import serialize_json, write_artifact_set

from .models import EmbeddingRecord


def serialize_embedding_records(records: list[EmbeddingRecord]) -> str:
    return "".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )


def read_embedding_records(path: Path) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(EmbeddingRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid embedding JSONL at {path}:{line_number}: {error}") from error
    return records


def artifact_sha256(serialized_records: str) -> str:
    return hashlib.sha256(serialized_records.encode("utf-8")).hexdigest()


def write_embedding_artifacts(
    output_dir: Path, records: list[EmbeddingRecord], manifest: dict[str, Any], stats: dict[str, Any]
) -> None:
    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("embedding_config_fingerprint") != manifest.get("embedding_config_fingerprint"):
            raise ValueError("refusing to overwrite embedding artifact with a different embedding identity")
        if existing.get("chunk_manifest_sha256") != manifest.get("chunk_manifest_sha256"):
            raise ValueError("refusing to overwrite embedding artifact built from a different chunk manifest")
    if len(records) != manifest.get("chunk_count") or len(records) != stats.get("embedded_chunks"):
        raise ValueError("complete embedding artifact counts disagree")
    if len({record.chunk_id for record in records}) != len(records):
        raise ValueError("embedding artifact contains duplicate chunk IDs")
    data = serialize_embedding_records(records)
    expected_hash = artifact_sha256(data)
    if manifest.get("embedding_artifact_fingerprint") != expected_hash:
        raise ValueError("embedding artifact fingerprint does not match serialized records")
    write_artifact_set(output_dir, {
        "embeddings.jsonl": data,
        "stats.json": serialize_json(stats),
        "manifest.json": serialize_json(manifest),
    })


def write_index_artifacts(
    output_dir: Path, index_jsonl: str, manifest: dict[str, Any], stats: dict[str, Any]
) -> None:
    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("index_fingerprint") != manifest.get("index_fingerprint"):
            raise ValueError("refusing to overwrite a different vector index identity")
    write_artifact_set(output_dir, {
        "index.jsonl": index_jsonl,
        "stats.json": serialize_json(stats),
        "manifest.json": serialize_json(manifest),
    })
