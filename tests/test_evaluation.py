import json
from pathlib import Path

import pytest

from rag_chunking.embedding.provider import DeterministicFakeEmbeddingProvider
from rag_chunking.evaluation.dataset import load_evaluation_dataset
from rag_chunking.evaluation.metrics import aggregate, evaluate_ranking
from rag_chunking.evaluation.runner import run_retrieval_benchmark
from rag_chunking.retrieval.service import RetrievalService
from test_retrieval import config, make_index, record


def write_dataset(path: Path, records):
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def test_metrics_are_hand_computable():
    metric = evaluate_ranking(["x", "y", "z", "b"], {"y", "b"})
    assert metric == {
        "first_relevant_rank": 2, "hit_at_1": 0, "hit_at_3": 1, "hit_at_5": 1,
        "hit_at_10": 1, "recall_at_5": 1.0, "recall_at_10": 1.0,
        "reciprocal_rank": 0.5,
    }
    missed = evaluate_ranking(["x"], {"a"})
    assert missed["first_relevant_rank"] is None and missed["reciprocal_rank"] == 0
    values = aggregate([{**metric}, {**missed}])
    assert values["query_count"] == 2 and values["hit_at_1"] == 0 and values["mrr"] == 0.25


@pytest.mark.parametrize("records,match", [
    ([{"query_id": "a", "query": "q", "category": "conceptual", "relevant_sources": ["a.md"]}, {"query_id": "a", "query": "q2", "category": "conceptual", "relevant_sources": ["a.md"]}], "duplicate query_id"),
    ([{"query_id": "a", "query": " ", "category": "conceptual", "relevant_sources": ["a.md"]}], "empty"),
    ([{"query_id": "a", "query": "q", "category": "wrong", "relevant_sources": ["a.md"]}], "unknown category"),
    ([{"query_id": "a", "query": "q", "category": "conceptual", "relevant_sources": []}], "at least one"),
    ([{"query_id": "a", "query": "q", "category": "conceptual", "relevant_sources": ["missing.md"]}], "absent"),
    ([{"query_id": "a", "query": "q", "category": "conceptual", "relevant_sources": ["a.md", "a.md"]}], "duplicate relevance"),
])
def test_dataset_validator_rejects_invalid_records(tmp_path: Path, records, match):
    path = tmp_path / "data.jsonl"
    write_dataset(path, records)
    with pytest.raises(ValueError, match=match):
        load_evaluation_dataset(path, {"a.md"})


def test_dataset_fingerprint_ignores_record_and_key_order(tmp_path: Path):
    left = [
        {"query_id": "b", "query": "second", "category": "how_to", "relevant_sources": ["b.md"]},
        {"query_id": "a", "query": "first", "category": "conceptual", "relevant_sources": ["a.md"]},
    ]
    right = [dict(reversed(list(item.items()))) for item in reversed(left)]
    one, two = tmp_path / "one", tmp_path / "two"
    write_dataset(one, left); write_dataset(two, right)
    assert load_evaluation_dataset(one, {"a.md", "b.md"}).fingerprint == load_evaluation_dataset(two, {"a.md", "b.md"}).fingerprint


def test_offline_query_to_benchmark_is_deterministic(tmp_path: Path):
    cfg = config()
    index_dirs = {}
    for strategy in ("fixed_size", "structure_aware", "prompt_based"):
        records = [
            record(cfg, 0, "exact query", [1.0, 0.0, 0.0, 0.0], strategy, "a.md"),
            record(cfg, 1, "other", [0.0, 1.0, 0.0, 0.0], strategy, "b.md"),
        ]
        index_dirs[strategy], _ = make_index(tmp_path, cfg, strategy, records)
    dataset_path = tmp_path / "dataset.jsonl"
    write_dataset(dataset_path, [{"query_id": "q1", "query": "exact query", "category": "conceptual", "relevant_sources": ["a.md"]}])
    dataset = load_evaluation_dataset(dataset_path, {"a.md", "b.md"})
    service = RetrievalService(corpus="angular", index_directories=index_dirs, embedding_config=cfg, provider=DeterministicFakeEmbeddingProvider(cfg), query_cache_directory=tmp_path / "cache", repository_root=tmp_path)
    first = run_retrieval_benchmark(service, dataset, tmp_path / "results", strategies=list(index_dirs))
    deterministic = {name: (first.output_directory / name).read_bytes() for name in ("per_query.jsonl", "aggregate.json", "comparison.json", "failures.json", "manifest.json", "baseline_report.md")}
    second_provider = DeterministicFakeEmbeddingProvider(cfg)
    second_service = RetrievalService(corpus="angular", index_directories=index_dirs, embedding_config=cfg, provider=second_provider, query_cache_directory=tmp_path / "cache", repository_root=tmp_path)
    second = run_retrieval_benchmark(second_service, dataset, tmp_path / "results", strategies=list(index_dirs))
    assert second.stats["query_embedding_cache_hits"] == 1
    assert second.stats["provider_calls"] == 0
    assert deterministic == {name: (second.output_directory / name).read_bytes() for name in deterministic}
    assert all(second.aggregates[strategy]["overall"]["hit_at_1"] == 1 for strategy in index_dirs)
