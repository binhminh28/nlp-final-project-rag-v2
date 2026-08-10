import json
from pathlib import Path

import pytest

from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.writer import configuration_fingerprint, serialize_chunks_jsonl, serialize_json
from rag_chunking.embedding.cache import EmbeddingCache
from rag_chunking.embedding.models import EmbeddingConfig, canonical_fingerprint
from rag_chunking.embedding.pipeline import plan_batches, run_embedding_pipeline
from rag_chunking.embedding.provider import DeterministicFakeEmbeddingProvider


def _chunk(index: int, text: str = "technical text") -> Chunk:
    return Chunk(
        chunk_id=f"angular:a.md::fixed::{index:06d}", strategy="fixed_size",
        doc_id="angular:a.md", source="angular", relative_path="a.md",
        chunk_index=index, text=f"{text} {index}", token_start=index,
        token_end=index + 2, token_count=2, chunk_size=512, chunk_overlap=64,
        tokenizer="tiktoken:cl100k_base",
    )


def _artifact(directory: Path, chunks: list[Chunk]) -> None:
    directory.mkdir()
    configuration = {"strategy": "fixed_size", "size": 512}
    (directory / "chunks.jsonl").write_text(serialize_chunks_jsonl(chunks), encoding="utf-8")
    (directory / "manifest.json").write_text(serialize_json({
        "strategy": "fixed_size", "config_fingerprint": configuration_fingerprint(configuration),
        "chunks": len(chunks), "documents": 1,
    }), encoding="utf-8")


def _config(**changes) -> EmbeddingConfig:
    values = dict(provider="fake", model="deterministic-v1", dimension=8,
                  max_batch_items=2, max_batch_tokens=100, max_input_tokens=100)
    values.update(changes)
    return EmbeddingConfig(**values)


def test_fingerprint_is_key_order_independent_and_output_sensitive() -> None:
    assert canonical_fingerprint({"a": 1, "b": 2}) == canonical_fingerprint({"b": 2, "a": 1})
    assert _config().fingerprint != _config(model="deterministic-v2").fingerprint


def test_pipeline_is_generic_cached_resumable_and_byte_deterministic(tmp_path: Path) -> None:
    source, output, cache = tmp_path / "chunks", tmp_path / "embeddings", tmp_path / "cache"
    _artifact(source, [_chunk(0), _chunk(1), _chunk(2)])
    first_provider = DeterministicFakeEmbeddingProvider(_config())
    first = run_embedding_pipeline(source, output, cache, first_provider, corpus="angular")
    first_bytes = {name: (output / name).read_bytes() for name in ("embeddings.jsonl", "manifest.json")}
    assert first.stats["cache_misses"] == 3
    assert first.stats["model_calls"] == 2
    second_provider = DeterministicFakeEmbeddingProvider(_config())
    second = run_embedding_pipeline(source, output, cache, second_provider, corpus="angular")
    assert second.stats["cache_hits"] == 3
    assert second.stats["cache_misses"] == second.stats["model_calls"] == 0
    assert {name: (output / name).read_bytes() for name in first_bytes} == first_bytes


def test_stale_manifest_identity_cannot_be_overwritten(tmp_path: Path) -> None:
    source, output, cache = tmp_path / "chunks", tmp_path / "embeddings", tmp_path / "cache"
    _artifact(source, [_chunk(0)])
    run_embedding_pipeline(source, output, cache, DeterministicFakeEmbeddingProvider(_config()), corpus="angular")
    with pytest.raises(ValueError, match="different embedding identity"):
        run_embedding_pipeline(
            source, output, cache,
            DeterministicFakeEmbeddingProvider(_config(model="deterministic-v2")), corpus="angular",
        )


def test_explicit_sample_is_never_labeled_as_full(tmp_path: Path) -> None:
    source = tmp_path / "chunks"
    _artifact(source, [_chunk(0), _chunk(1)])
    result = run_embedding_pipeline(
        source, tmp_path / "output", tmp_path / "cache",
        DeterministicFakeEmbeddingProvider(_config()), corpus="angular", limit=1,
    )
    assert result.manifest["build_scope"] == "sample"
    assert result.manifest["source_chunk_count"] == 2
    assert result.manifest["chunk_count"] == 1


def test_changed_text_or_config_is_a_cache_miss(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache", _config())
    vector = [0.0] * 8
    cache.put("same", vector)
    assert cache.get("same") == vector
    assert cache.get("changed") is None
    assert EmbeddingCache(tmp_path / "cache", _config(model="v2")).get("same") is None


def test_corrupt_cache_fails_visibly(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, _config())
    path = tmp_path / f"{cache.key('text')}.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt embedding cache"):
        cache.get("text")


def test_failure_keeps_cached_successes_and_never_publishes_manifest(tmp_path: Path) -> None:
    class FailingProvider(DeterministicFakeEmbeddingProvider):
        def embed_texts(self, texts):
            if self.calls == 1:
                raise TimeoutError("timeout")
            return super().embed_texts(texts)

    source, output, cache = tmp_path / "chunks", tmp_path / "output", tmp_path / "cache"
    _artifact(source, [_chunk(0), _chunk(1), _chunk(2)])
    provider = FailingProvider(_config(max_batch_items=1))
    with pytest.raises(TimeoutError):
        run_embedding_pipeline(source, output, cache, provider, corpus="angular")
    assert len(list(cache.glob("*.json"))) == 1
    assert not (output / "manifest.json").exists()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["complete"] is False and failure["failed_chunk_ids"]


def test_batch_planner_enforces_both_limits_without_truncation() -> None:
    chunks = [_chunk(0, "one"), _chunk(1, "two"), _chunk(2, "three")]
    assert [len(batch) for batch in plan_batches(chunks, _config(max_batch_items=2), lambda _: 3)] == [2, 1]
    assert [len(batch) for batch in plan_batches(chunks, _config(max_batch_tokens=5), lambda _: 3)] == [1, 1, 1]
    with pytest.raises(ValueError, match="was not truncated"):
        plan_batches(chunks, _config(max_input_tokens=2), lambda _: 3)
