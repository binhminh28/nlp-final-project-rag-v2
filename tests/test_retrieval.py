import json
from pathlib import Path

import pytest

from rag_chunking.embedding.artifacts import artifact_sha256, serialize_embedding_records, write_embedding_artifacts
from rag_chunking.embedding.index import build_local_index
from rag_chunking.embedding.models import EmbeddingConfig, EmbeddingRecord, content_sha256
from rag_chunking.embedding.provider import DeterministicFakeEmbeddingProvider
from rag_chunking.retrieval.cache import QueryEmbeddingCache
from rag_chunking.retrieval.models import RetrievalConfig, RetrievalRequest, normalize_query
from rag_chunking.retrieval.service import RetrievalService


def config(model="fake-v1", dimension=4):
    return EmbeddingConfig(provider="fake", model=model, dimension=dimension)


def record(cfg, index, text, vector, strategy="fixed_size", path=None):
    path = path or f"doc-{index}.md"
    return EmbeddingRecord(
        chunk_id=f"angular:{path}::{strategy}::{index:06d}", doc_id=f"angular:{path}",
        strategy=strategy, chunk_config_fingerprint=f"chunks-{strategy}", text=text,
        metadata={"kind": "test"}, embedding=vector, embedding_provider=cfg.provider,
        embedding_model=cfg.model, embedding_dimension=cfg.dimension,
        embedding_config_fingerprint=cfg.fingerprint, source="angular", relative_path=path,
        chunk_index=index, token_count=2, text_sha256=content_sha256(text),
    )


def make_index(root, cfg, strategy="fixed_size", records=None):
    records = records or [record(cfg, 0, "alpha", [1.0, 0.0, 0.0, 0.0], strategy)]
    embedding_dir = root / "embeddings" / strategy
    index_dir = root / "indexes" / strategy
    data = serialize_embedding_records(records)
    manifest = {
        "schema_version": "embedding_record_v1", "complete": True, "corpus": "angular",
        "chunk_strategy": strategy, "chunk_config_fingerprint": f"chunks-{strategy}",
        "chunk_manifest_sha256": "hash", "chunk_artifact": "unused", "documents": len(records),
        "chunk_count": len(records), "embedding_provider": cfg.provider, "embedding_model": cfg.model,
        "embedding_dimension": cfg.dimension, "embedding_configuration": cfg.identity(),
        "embedding_config_fingerprint": cfg.fingerprint, "embedding_artifact_fingerprint": artifact_sha256(data),
    }
    write_embedding_artifacts(embedding_dir, records, manifest, {"embedded_chunks": len(records)})
    built = build_local_index(embedding_dir, index_dir)
    return index_dir, built


def test_query_normalization_and_request_validation():
    assert normalize_query("  Xin chào, thế giới!  ") == "Xin chào, thế giới!"
    assert normalize_query("inject(Foo<T>)") == "inject(Foo<T>)"
    assert RetrievalRequest(" query ", "fixed_size", 2).query == "query"
    for query in ("", " \n\t"):
        with pytest.raises(ValueError, match="empty"):
            RetrievalRequest(query, "fixed_size")
    with pytest.raises(ValueError, match="unknown strategy"):
        RetrievalRequest("q", "other")
    with pytest.raises(ValueError, match="positive"):
        RetrievalRequest("q", "fixed_size", 0)
    with pytest.raises(ValueError, match="unsupported"):
        RetrievalRequest("q", "fixed_size", filters={"unknown": "x"})
    with pytest.raises(ValueError, match="conflicts"):
        RetrievalRequest("q", "fixed_size", filters={"strategy": "prompt_based"})


def test_retrieval_config_fingerprint_is_stable():
    assert RetrievalConfig().fingerprint == RetrievalConfig().fingerprint


def test_query_cache_identity_validation_and_config_isolation(tmp_path: Path):
    first = QueryEmbeddingCache(tmp_path, config())
    vector = [1.0, 0.0, 0.0, 0.0]
    first.put("query", vector)
    assert first.get("query") == vector
    assert first.get("different") is None
    assert QueryEmbeddingCache(tmp_path, config(model="v2")).get("query") is None
    path = tmp_path / config().fingerprint / f"{first.key('query')}.json"
    value = json.loads(path.read_text())
    value["embedding"] = [1.0]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Corrupt query"):
        first.get("query")
    value["embedding"] = [float("nan")] * 4
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Corrupt query"):
        first.get("query")
    path.write_text("broken")
    with pytest.raises(ValueError, match="Corrupt query"):
        first.get("query")


def test_service_maps_hits_filters_caches_and_tie_breaks(tmp_path: Path):
    cfg = config()
    records = [
        record(cfg, 1, "same one", [1.0, 0.0, 0.0, 0.0], path="b.md"),
        record(cfg, 0, "same zero", [1.0, 0.0, 0.0, 0.0], path="a.md"),
        record(cfg, 2, "other", [0.0, 1.0, 0.0, 0.0], path="c.md"),
    ]
    index_dir, _ = make_index(tmp_path, cfg, records=records)
    provider = DeterministicFakeEmbeddingProvider(cfg)
    service = RetrievalService(
        corpus="angular", index_directories={"fixed_size": index_dir}, embedding_config=cfg,
        provider=provider, query_cache_directory=tmp_path / "cache", repository_root=tmp_path,
    )
    vector = [1.0, 0.0, 0.0, 0.0]
    result = service.retrieve(RetrievalRequest("  query  ", "fixed_size", 2), query_vector=vector)
    assert [hit.chunk_id for hit in result.hits] == sorted(hit.chunk_id for hit in result.hits)
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert result.hits[0].text and result.hits[0].metadata == {"kind": "test"}
    filtered = service.retrieve(RetrievalRequest("query", "fixed_size", filters={"doc_id": "angular:c.md"}), query_vector=vector)
    assert [hit.relative_path for hit in filtered.hits] == ["c.md"]
    assert service.retrieve(RetrievalRequest("query", "fixed_size", filters={"source": "missing"}), query_vector=vector).hits == []
    first = service.retrieve(RetrievalRequest("cached", "fixed_size"))
    second = service.retrieve(RetrievalRequest(" cached ", "fixed_size"))
    assert first.query_embedding_cache_hit is False and second.query_embedding_cache_hit is True
    assert provider.calls == 1


def test_service_rejects_manifest_mismatch_and_unresolved_ids(tmp_path: Path):
    cfg = config()
    index_dir, _ = make_index(tmp_path, cfg)
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["corpus"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="corpus/strategy"):
        RetrievalService(corpus="angular", index_directories={"fixed_size": index_dir}, embedding_config=cfg, provider=DeterministicFakeEmbeddingProvider(cfg), query_cache_directory=tmp_path / "cache", repository_root=tmp_path)
