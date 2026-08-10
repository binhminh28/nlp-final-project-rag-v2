import json
from pathlib import Path

import pytest

from rag_chunking.embedding.artifacts import artifact_sha256, serialize_embedding_records, write_embedding_artifacts
from rag_chunking.embedding.index import LocalCosineIndex, build_local_index
from rag_chunking.embedding.models import EmbeddingConfig, EmbeddingRecord


def _record(index: int, vector: list[float]) -> EmbeddingRecord:
    config = EmbeddingConfig(provider="fake", model="v1", dimension=3)
    text = f"text {index}"
    return EmbeddingRecord(
        chunk_id=f"chunk-{index}", doc_id=f"doc-{index % 2}", strategy="fixed_size",
        chunk_config_fingerprint="chunk-config", text=text, metadata={}, embedding=vector,
        embedding_provider=config.provider, embedding_model=config.model,
        embedding_dimension=3, embedding_config_fingerprint=config.fingerprint,
        source="angular", relative_path=f"{index}.md", chunk_index=index,
        token_count=2,
        text_sha256=__import__("hashlib").sha256(text.encode()).hexdigest(),
    )


def _embedding_artifact(path: Path, records: list[EmbeddingRecord]) -> None:
    data = serialize_embedding_records(records)
    manifest = {
        "schema_version": "embedding_record_v1", "complete": True, "corpus": "angular",
        "chunk_strategy": "fixed_size", "chunk_config_fingerprint": "chunk-config",
        "chunk_manifest_sha256": "manifest-hash", "chunk_artifact": "chunks",
        "documents": 2, "chunk_count": len(records), "embedding_provider": "fake",
        "embedding_model": "v1", "embedding_dimension": 3,
        "embedding_configuration": {},
        "embedding_config_fingerprint": records[0].embedding_config_fingerprint,
        "embedding_artifact_fingerprint": artifact_sha256(data),
    }
    write_embedding_artifacts(path, records, manifest, {"embedded_chunks": len(records)})


def test_index_integrity_similarity_and_metadata_filters(tmp_path: Path) -> None:
    embedding_dir, index_dir = tmp_path / "embeddings", tmp_path / "index"
    records = [_record(0, [1.0, 0.0, 0.0]), _record(1, [0.0, 1.0, 0.0])]
    _embedding_artifact(embedding_dir, records)
    manifest = build_local_index(embedding_dir, index_dir)
    index = LocalCosineIndex(index_dir)
    assert manifest["vector_count"] == 2
    assert index.search([1.0, 0.0, 0.0])[0].chunk_id == "chunk-0"
    assert index.search([1.0, 0.0, 0.0], filters={"doc_id": "doc-1"})[0].chunk_id == "chunk-1"
    assert index.search([1.0, 0.0, 0.0], filters={"strategy": "missing"}) == []


def test_index_rejects_duplicate_logical_records(tmp_path: Path) -> None:
    embedding_dir = tmp_path / "embeddings"
    records = [_record(0, [1.0, 0.0, 0.0]), _record(0, [1.0, 0.0, 0.0])]
    data = serialize_embedding_records(records)
    embedding_dir.mkdir()
    (embedding_dir / "embeddings.jsonl").write_text(data, encoding="utf-8")
    (embedding_dir / "manifest.json").write_text(json.dumps({
        "complete": True, "chunk_count": 2, "embedding_dimension": 3,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate logical"):
        build_local_index(embedding_dir, tmp_path / "index")


def test_index_rejects_partial_embedding_build(tmp_path: Path) -> None:
    source = tmp_path / "partial"
    source.mkdir()
    (source / "manifest.json").write_text('{"complete":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        build_local_index(source, tmp_path / "index")
