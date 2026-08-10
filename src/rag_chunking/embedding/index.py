"""Small persistent cosine index with canonical payloads and filters."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .artifacts import artifact_sha256, read_embedding_records, serialize_embedding_records, write_index_artifacts
from .models import INDEX_SCHEMA_VERSION, EmbeddingRecord, canonical_fingerprint, index_identity, validate_vector


@dataclass(frozen=True, slots=True)
class SearchHit:
    index_id: str
    chunk_id: str
    score: float
    payload: dict[str, Any]


class VectorIndex(Protocol):
    def search(self, vector: list[float], k: int = 5, filters: dict[str, str] | None = None) -> list[SearchHit]: ...


def _index_id(record: EmbeddingRecord) -> str:
    return canonical_fingerprint({
        "chunk_id": record.chunk_id,
        "chunk_config_fingerprint": record.chunk_config_fingerprint,
        "embedding_config_fingerprint": record.embedding_config_fingerprint,
    })


def _payload(record: EmbeddingRecord) -> dict[str, Any]:
    return {
        "chunk_id": record.chunk_id, "doc_id": record.doc_id,
        "strategy": record.strategy, "chunk_config_fingerprint": record.chunk_config_fingerprint,
        "embedding_config_fingerprint": record.embedding_config_fingerprint,
        "source": record.source, "relative_path": record.relative_path,
        "chunk_index": record.chunk_index,
    }


def build_local_index(embedding_dir: Path, output_dir: Path) -> dict[str, Any]:
    embedding_manifest = json.loads((embedding_dir / "manifest.json").read_text(encoding="utf-8"))
    if not embedding_manifest.get("complete"):
        raise ValueError("embedding manifest is absent or incomplete")
    records = read_embedding_records(embedding_dir / "embeddings.jsonl")
    expected = embedding_manifest.get("chunk_count")
    if len(records) != expected:
        raise ValueError(f"embedding manifest expects {expected} records, found {len(records)}")
    if len({record.chunk_id for record in records}) != len(records):
        raise ValueError("duplicate logical embedding records")
    dimension = embedding_manifest["embedding_dimension"]
    if artifact_sha256(serialize_embedding_records(records)) != embedding_manifest.get("embedding_artifact_fingerprint"):
        raise ValueError("embedding artifact fingerprint mismatch")
    for record in records:
        if record.strategy != embedding_manifest["chunk_strategy"]:
            raise ValueError("embedding record strategy does not match manifest")
        if record.chunk_config_fingerprint != embedding_manifest["chunk_config_fingerprint"]:
            raise ValueError("embedding record chunk identity does not match manifest")
        if record.embedding_config_fingerprint != embedding_manifest["embedding_config_fingerprint"]:
            raise ValueError("embedding record configuration does not match manifest")
    entries: list[dict[str, Any]] = []
    for record in records:
        validate_vector(record.embedding, dimension)
        entries.append({"index_id": _index_id(record), "vector": record.embedding, "payload": _payload(record)})
    if len({entry["index_id"] for entry in entries}) != len(entries):
        raise ValueError("duplicate vector index IDs")
    serialized = "".join(
        json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for entry in entries
    )
    identity = index_identity(
        corpus=embedding_manifest["corpus"], strategy=embedding_manifest["chunk_strategy"],
        chunk_config_fingerprint=embedding_manifest["chunk_config_fingerprint"],
        embedding_config_fingerprint=embedding_manifest["embedding_config_fingerprint"],
        embedding_artifact_fingerprint=embedding_manifest["embedding_artifact_fingerprint"],
    )
    manifest = {
        **identity, "index_fingerprint": canonical_fingerprint(identity),
        "collection_name": f"{embedding_manifest['corpus']}-{embedding_manifest['chunk_strategy']}-{canonical_fingerprint(identity)[:12]}",
        "vector_count": len(entries), "dimension": dimension,
        "embedding_model": embedding_manifest["embedding_model"],
        "embedding_artifact": embedding_dir.as_posix(),
    }
    stats = {
        "expected_vectors": expected, "indexed_vectors": len(entries),
        "missing_records": 0, "duplicate_records": 0, "invalid_dimensions": 0,
        "unresolved_chunk_ids": 0,
    }
    write_index_artifacts(output_dir, serialized, manifest, stats)
    return manifest


class LocalCosineIndex:
    def __init__(self, directory: Path):
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.dimension = self.manifest["dimension"]
        self.entries: list[dict[str, Any]] = []
        with (directory / "index.jsonl").open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    entry = json.loads(line)
                    validate_vector(entry["vector"], self.dimension)
                    if not isinstance(entry["payload"], dict) or not entry["payload"].get("chunk_id"):
                        raise ValueError("invalid payload")
                    self.entries.append(entry)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid index at {directory}:{line_number}: {error}") from error
        if len(self.entries) != self.manifest["vector_count"]:
            raise ValueError("index manifest count mismatch")

    def search(self, vector: list[float], k: int = 5, filters: dict[str, str] | None = None) -> list[SearchHit]:
        validate_vector(vector, self.dimension)
        if k <= 0:
            raise ValueError("k must be positive")
        query_norm = math.sqrt(sum(value * value for value in vector))
        if query_norm == 0:
            raise ValueError("query vector must have non-zero norm")
        hits: list[SearchHit] = []
        for entry in self.entries:
            payload = entry["payload"]
            if filters and any(payload.get(key) != value for key, value in filters.items()):
                continue
            candidate = entry["vector"]
            norm = math.sqrt(sum(value * value for value in candidate))
            score = sum(left * right for left, right in zip(vector, candidate, strict=True)) / (query_norm * norm) if norm else -1.0
            hits.append(SearchHit(entry["index_id"], payload["chunk_id"], score, payload))
        return sorted(hits, key=lambda hit: (-hit.score, hit.index_id))[:k]
