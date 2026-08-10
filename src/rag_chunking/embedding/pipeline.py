"""Generic Unified Chunk Format -> cached embedding artifact pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.tokenizer import TiktokenTokenizer
from rag_chunking.chunking.writer import read_chunks_jsonl, serialize_json

from .artifacts import artifact_sha256, serialize_embedding_records, write_embedding_artifacts
from .cache import EmbeddingCache
from .models import EMBEDDING_SCHEMA_VERSION, EmbeddingConfig, EmbeddingRecord
from .provider import EmbeddingProvider, validate_provider_output


@dataclass(slots=True)
class EmbeddingRunResult:
    records: list[EmbeddingRecord]
    manifest: dict[str, Any]
    stats: dict[str, Any]


def _load_chunk_artifact(directory: Path) -> tuple[list[Chunk], dict[str, Any], str]:
    manifest_path = directory / "manifest.json"
    chunks_path = directory / "chunks.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("chunk manifest must be an object")
    chunks = read_chunks_jsonl(chunks_path)
    required = ("strategy", "config_fingerprint", "chunks", "documents")
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(f"chunk manifest is missing {missing}")
    if manifest["chunks"] != len(chunks):
        raise ValueError("chunk manifest count disagrees with chunks.jsonl")
    if any(chunk.strategy != manifest["strategy"] for chunk in chunks):
        raise ValueError("chunk artifact mixes strategies")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("chunk artifact contains duplicate chunk IDs")
    return chunks, manifest, hashlib.sha256(manifest_bytes).hexdigest()


def plan_batches(
    chunks: list[Chunk], config: EmbeddingConfig, count_tokens: Callable[[str], int]
) -> list[list[Chunk]]:
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_tokens = 0
    for chunk in chunks:
        tokens = count_tokens(chunk.text)
        if tokens > config.max_input_tokens:
            raise ValueError(
                f"chunk {chunk.chunk_id!r} has {tokens} embedding tokens; limit is {config.max_input_tokens}; text was not truncated"
            )
        if current and (len(current) >= config.max_batch_items or current_tokens + tokens > config.max_batch_tokens):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(chunk)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def run_embedding_pipeline(
    chunk_artifact_dir: Path, output_dir: Path, cache_dir: Path,
    provider: EmbeddingProvider, *, corpus: str, limit: int | None = None,
) -> EmbeddingRunResult:
    chunks, chunk_manifest, chunk_manifest_sha256 = _load_chunk_artifact(chunk_artifact_dir)
    source_chunk_count = len(chunks)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        chunks = chunks[:limit]
    config = provider.config
    tokenizer = TiktokenTokenizer(config.tokenizer.removeprefix("tiktoken:"))
    batches = plan_batches(chunks, config, lambda text: len(tokenizer.encode(text)))
    cache = EmbeddingCache(cache_dir, config)
    vectors_by_key: dict[str, list[float]] = {}
    cache_hits = 0
    cache_misses = 0
    batch_sizes: list[int] = []

    try:
        for batch in batches:
            missing_texts: list[str] = []
            missing_keys: list[str] = []
            missing_seen: set[str] = set()
            for chunk in batch:
                key = cache.key(chunk.text)
                if key in vectors_by_key:
                    cache_hits += 1
                    continue
                vector = cache.get(chunk.text)
                if vector is not None:
                    vectors_by_key[key] = vector
                    cache_hits += 1
                else:
                    if key not in missing_seen:
                        cache_misses += 1
                        missing_seen.add(key)
                        missing_keys.append(key)
                        missing_texts.append(chunk.text)
            if missing_texts:
                vectors = validate_provider_output(
                    provider.embed_texts(missing_texts), len(missing_texts), config.dimension
                )
                batch_sizes.append(len(missing_texts))
                for text, key, vector in zip(missing_texts, missing_keys, vectors, strict=True):
                    cache.put(text, vector)
                    vectors_by_key[key] = vector
    except BaseException as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "complete": False, "error_type": type(error).__name__, "error": str(error),
            "cached_embeddings": len(vectors_by_key),
            "failed_chunk_ids": [chunk.chunk_id for chunk in batch if cache.key(chunk.text) not in vectors_by_key],
        }
        (output_dir / "failure.json").write_text(serialize_json(failure), encoding="utf-8", newline="\n")
        raise

    records = [
        EmbeddingRecord.from_chunk(
            chunk, vectors_by_key[cache.key(chunk.text)], config, chunk_manifest["config_fingerprint"]
        )
        for chunk in chunks
    ]
    data = serialize_embedding_records(records)
    artifact_fingerprint = artifact_sha256(data)
    stats = {
        "total_chunks": len(chunks), "embedded_chunks": len(records),
        "cache_hits": cache_hits, "cache_misses": cache_misses,
        "model_calls": provider.calls, "failed_chunks": 0,
        "embedding_dimension": config.dimension, "batch_count": len(batch_sizes),
        "min_batch_size": min(batch_sizes, default=0), "max_batch_size": max(batch_sizes, default=0),
        "avg_batch_size": (sum(batch_sizes) / len(batch_sizes)) if batch_sizes else 0.0,
        "total_input_tokens": provider.input_tokens, "retry_count": provider.retries,
    }
    manifest = {
        "schema_version": EMBEDDING_SCHEMA_VERSION, "complete": True, "corpus": corpus,
        "build_scope": "sample" if limit is not None else "full",
        "source_chunk_count": source_chunk_count,
        "chunk_strategy": chunk_manifest["strategy"],
        "chunk_config_fingerprint": chunk_manifest["config_fingerprint"],
        "chunk_manifest_sha256": chunk_manifest_sha256,
        "chunk_artifact": chunk_artifact_dir.as_posix(),
        "documents": chunk_manifest["documents"], "chunk_count": len(chunks),
        "embedding_provider": config.provider, "embedding_model": config.model,
        "embedding_dimension": config.dimension, "embedding_configuration": config.identity(),
        "embedding_config_fingerprint": config.fingerprint,
        "embedding_artifact_fingerprint": artifact_fingerprint,
    }
    failure_path = output_dir / "failure.json"
    write_embedding_artifacts(output_dir, records, manifest, stats)
    if failure_path.exists():
        failure_path.unlink()
    return EmbeddingRunResult(records, manifest, stats)
