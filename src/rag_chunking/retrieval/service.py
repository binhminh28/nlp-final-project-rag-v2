"""Manifest-validated retrieval orchestration over the local cosine backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_chunking.embedding.artifacts import artifact_sha256, read_embedding_records, serialize_embedding_records
from rag_chunking.embedding.index import LocalCosineIndex
from rag_chunking.embedding.models import EmbeddingConfig, EmbeddingRecord, INDEX_SCHEMA_VERSION, canonical_fingerprint, index_identity, validate_vector
from rag_chunking.embedding.provider import EmbeddingProvider

from .cache import QueryEmbeddingCache
from .models import RetrievalConfig, RetrievalHit, RetrievalRequest, RetrievalResult


def _resolve_artifact(reference: str, repository_root: Path) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else repository_root / path


class RetrievalService:
    def __init__(
        self, *, corpus: str, index_directories: dict[str, Path],
        embedding_config: EmbeddingConfig, provider: EmbeddingProvider,
        query_cache_directory: Path, retrieval_config: RetrievalConfig | None = None,
        repository_root: Path | None = None,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must be non-empty")
        self.corpus = corpus
        self.embedding_config = embedding_config
        self.provider = provider
        if provider.config.fingerprint != embedding_config.fingerprint:
            raise ValueError("provider and retrieval embedding configurations differ")
        self.config = retrieval_config or RetrievalConfig()
        self.cache = QueryEmbeddingCache(query_cache_directory, embedding_config)
        self.repository_root = (repository_root or Path.cwd()).resolve()
        self.indexes: dict[str, LocalCosineIndex] = {}
        self.manifests: dict[str, dict[str, Any]] = {}
        self.records: dict[str, dict[str, EmbeddingRecord]] = {}
        for strategy, directory in sorted(index_directories.items()):
            self._load_strategy(strategy, directory)

    def _load_strategy(self, strategy: str, directory: Path) -> None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"missing index manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {"schema_version", "backend", "corpus", "strategy", "chunk_config_fingerprint", "embedding_config_fingerprint", "embedding_artifact_fingerprint", "index_fingerprint", "vector_count", "dimension", "embedding_artifact"}
        missing = sorted(required - set(manifest)) if isinstance(manifest, dict) else sorted(required)
        if missing:
            raise ValueError(f"index manifest is missing fields: {missing}")
        if manifest["schema_version"] != INDEX_SCHEMA_VERSION or manifest["backend"] != "local_cosine_jsonl":
            raise ValueError("unsupported index schema/backend")
        if manifest["corpus"] != self.corpus or manifest["strategy"] != strategy:
            raise ValueError("index corpus/strategy mismatch")
        if manifest["embedding_config_fingerprint"] != self.embedding_config.fingerprint:
            raise ValueError("index embedding fingerprint mismatch")
        if manifest["dimension"] != self.embedding_config.dimension:
            raise ValueError("index embedding dimension mismatch")
        identity = index_identity(
            corpus=manifest["corpus"], strategy=manifest["strategy"],
            chunk_config_fingerprint=manifest["chunk_config_fingerprint"],
            embedding_config_fingerprint=manifest["embedding_config_fingerprint"],
            embedding_artifact_fingerprint=manifest["embedding_artifact_fingerprint"],
        )
        if canonical_fingerprint(identity) != manifest["index_fingerprint"]:
            raise ValueError("index fingerprint mismatch")
        embedding_dir = _resolve_artifact(manifest["embedding_artifact"], self.repository_root)
        embedding_manifest = json.loads((embedding_dir / "manifest.json").read_text(encoding="utf-8"))
        if not embedding_manifest.get("complete"):
            raise ValueError("embedding artifact is incomplete")
        for left, right in (("corpus", "corpus"), ("chunk_strategy", "strategy"), ("embedding_config_fingerprint", "embedding_config_fingerprint"), ("embedding_artifact_fingerprint", "embedding_artifact_fingerprint")):
            if embedding_manifest.get(left) != manifest.get(right):
                raise ValueError(f"index/embedding lineage mismatch for {left}")
        records_list = read_embedding_records(embedding_dir / "embeddings.jsonl")
        if not records_list or len(records_list) != manifest["vector_count"]:
            raise ValueError("empty index or vector count mismatch")
        if artifact_sha256(serialize_embedding_records(records_list)) != manifest["embedding_artifact_fingerprint"]:
            raise ValueError("embedding artifact fingerprint mismatch")
        records = {record.chunk_id: record for record in records_list}
        if len(records) != len(records_list):
            raise ValueError("duplicate chunk IDs in embedding artifact")
        index = LocalCosineIndex(directory)
        indexed_ids = [entry["payload"].get("chunk_id") for entry in index.entries]
        if len(indexed_ids) != len(set(indexed_ids)) or set(indexed_ids) != set(records):
            raise ValueError("index contains duplicate or unresolved chunk IDs")
        self.indexes[strategy] = index
        self.manifests[strategy] = manifest
        self.records[strategy] = records

    def embed_query(self, query: str) -> tuple[list[float], bool]:
        vector = self.cache.get(query)
        if vector is not None:
            return vector, True
        vectors = self.provider.embed_texts([query])
        if len(vectors) != 1:
            raise ValueError("query provider returned an unexpected vector count")
        vector = vectors[0]
        validate_vector(vector, self.embedding_config.dimension)
        self.cache.put(query, vector)
        return vector, False

    def retrieve(self, request: RetrievalRequest, *, query_vector: list[float] | None = None, cache_hit: bool | None = None) -> RetrievalResult:
        if request.strategy not in self.indexes:
            raise ValueError(f"strategy {request.strategy!r} has no loaded index")
        if query_vector is None:
            query_vector, actual_cache_hit = self.embed_query(request.query)
        else:
            validate_vector(query_vector, self.embedding_config.dimension)
            actual_cache_hit = bool(cache_hit)
        raw_hits = self.indexes[request.strategy].search(query_vector, request.top_k, request.filters)
        hits: list[RetrievalHit] = []
        for rank, raw in enumerate(raw_hits, 1):
            record = self.records[request.strategy].get(raw.chunk_id)
            if record is None:
                raise ValueError(f"unresolved chunk ID {raw.chunk_id!r}")
            if record.strategy != request.strategy:
                raise ValueError("resolved chunk strategy mismatch")
            hits.append(RetrievalHit(
                rank=rank, score=raw.score, chunk_id=record.chunk_id, doc_id=record.doc_id,
                source=record.source, relative_path=record.relative_path, strategy=record.strategy,
                text=record.text, metadata=dict(record.metadata), token_count=record.token_count,
                character_count=len(record.text), chunk_config_fingerprint=record.chunk_config_fingerprint,
            ))
        manifest = self.manifests[request.strategy]
        return RetrievalResult(
            query=request.query, strategy=request.strategy, top_k=request.top_k,
            filters=request.filters, hits=hits, query_embedding_cache_hit=actual_cache_hit,
            retrieval_config_fingerprint=self.config.fingerprint,
            embedding_config_fingerprint=self.embedding_config.fingerprint,
            index_fingerprint=manifest["index_fingerprint"],
        )
