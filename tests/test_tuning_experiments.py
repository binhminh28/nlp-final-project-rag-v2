import json
from pathlib import Path

from rag_chunking.embedding.models import EmbeddingConfig
from rag_chunking.evaluation.dataset import EvaluationDataset, EvaluationQuery
from rag_chunking.retrieval.models import RetrievalConfig
from rag_chunking.tuning.config import ExperimentConfig
from rag_chunking.tuning.dense import publish_dense_depth_experiments
from rag_chunking.tuning.diversity import source_cap
from rag_chunking.tuning.fusion import reciprocal_rank_fusion
from rag_chunking.tuning.lexical import BM25Config, BM25Index, tokenize
from rag_chunking.tuning.statistics import paired_bootstrap
from rag_chunking.tuning.metrics import evaluate_depth_ranking


def experiment(**overrides) -> ExperimentConfig:
    values = {
        "experiment_id": "E1-depth-10", "experiment_name": "depth_10",
        "experiment_family": "E1", "dataset_fingerprint": "dataset-a",
        "retrieval_config_fingerprint": "retrieval-a",
        "embedding_config_fingerprint": "embedding-a",
        "index_fingerprints": {"fixed_size": "index-a"}, "candidate_depth": 10,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def test_experiment_fingerprint_is_canonical_and_output_only() -> None:
    left = experiment(index_fingerprints={"b": "2", "a": "1"})
    right = experiment(index_fingerprints={"a": "1", "b": "2"})
    assert left.fingerprint == right.fingerprint
    assert left.fingerprint != experiment(candidate_depth=20).fingerprint
    assert left.fingerprint != experiment(dataset_fingerprint="dataset-b").fingerprint
    # Runtime telemetry is not part of ExperimentConfig and cannot alter identity.
    runtime_a = {"latency": 1.0, "fingerprint": left.fingerprint}
    runtime_b = {"latency": 99.0, "fingerprint": left.fingerprint}
    assert runtime_a["fingerprint"] == runtime_b["fingerprint"]


def test_depth_metrics_and_prefix_behavior() -> None:
    ranking = ["x.md", "x.md", "gold.md", "other.md", "second.md"]
    five = evaluate_depth_ranking(ranking, {"gold.md", "second.md"}, 5)
    assert five["first_relevant_rank"] == 3
    assert five["hit_at_1"] == 0 and five["hit_at_3"] == 1
    assert five["recall_at_5"] == 1.0
    assert five["hit_at_10"] is None and five["recall_at_10"] is None
    assert five["unique_source_count"] == 4 and five["duplicate_source_chunks"] == 1
    ten_ranking = ranking + [f"tail-{index}.md" for index in range(5)]
    ten = evaluate_depth_ranking(ten_ranking, {"gold.md", "second.md"}, 10)
    assert ten_ranking[:5] == ranking
    assert ten["first_relevant_rank"] == five["first_relevant_rank"]
    assert ten["recall_at_5"] == five["recall_at_5"]


def test_materialized_depths_have_distinct_identity_and_stable_prefix(tmp_path: Path) -> None:
    records = [EvaluationQuery("q1", "query", "conceptual", ["gold.md"])]
    dataset = EvaluationDataset(records, "dataset-fingerprint")
    strategies = ["fixed_size", "structure_aware", "prompt_based"]
    hits = {}
    for strategy in strategies:
        values = []
        for rank in range(1, 51):
            source = "gold.md" if rank == 7 else f"source-{rank}.md"
            values.append({"rank": rank, "relative_path": source, "chunk_id": f"{strategy}-{rank}"})
        hits[("q1", strategy)] = values
    embedding = EmbeddingConfig(provider="fake", model="fake", dimension=4)
    outputs = publish_dense_depth_experiments(
        corpus="angular", dataset=dataset, strategies=strategies, all_hits=hits,
        index_fingerprints={name: f"index-{name}" for name in strategies},
        embedding_config=embedding, retrieval_config=RetrievalConfig(), depths=[5, 10, 20, 50],
        output_root=tmp_path, runtime_stats={"provider_calls": 0, "latency": 1.2},
    )
    assert len({path.name for path in outputs}) == 4
    rows = {}
    for output in outputs:
        manifest = json.loads((output / "manifest.json").read_text())
        row = json.loads((output / "per_query.jsonl").read_text().splitlines()[0])
        rows[manifest["candidate_depth"]] = row
        assert manifest["complete"] and manifest["experiment_fingerprint"] == output.name
        assert "latency" not in json.loads((output / "config.json").read_text())
    assert [hit["chunk_id"] for hit in rows[5]["hits"]] == [hit["chunk_id"] for hit in rows[50]["hits"][:5]]
    assert rows[5]["first_relevant_rank"] is None and rows[10]["first_relevant_rank"] == 7


def test_source_cap_is_stable_and_handles_edge_cases() -> None:
    hits = [
        {"rank": 1, "chunk_id": "a1", "relative_path": "a.md"},
        {"rank": 2, "chunk_id": "a2", "relative_path": "a.md"},
        {"rank": 3, "chunk_id": "b1", "relative_path": "b.md"},
        {"rank": 4, "chunk_id": "c1", "relative_path": "c.md"},
    ]
    assert [hit["chunk_id"] for hit in source_cap(hits, 1, 3)] == ["a1", "b1", "c1"]
    assert [hit["rank"] for hit in source_cap(hits, 1, 3)] == [1, 2, 3]
    assert [hit["chunk_id"] for hit in source_cap(hits, 2, 3)] == ["a1", "a2", "b1"]
    assert source_cap([], 1, 3) == []


def test_rrf_matches_hand_calculation_merges_duplicates_and_breaks_ties() -> None:
    first = [
        {"chunk_id": "a", "rank": 1, "score": 99.0},
        {"chunk_id": "b", "rank": 2, "score": 98.0},
    ]
    second = [
        {"chunk_id": "b", "rank": 1, "score": -3.0},
        {"chunk_id": "c", "rank": 2, "score": -4.0},
    ]
    fused = reciprocal_rank_fusion([first, second], rank_constant=0)
    assert [hit["chunk_id"] for hit in fused] == ["b", "a", "c"]
    assert fused[0]["score"] == 1.5
    assert fused[1]["score"] == 1.0 and fused[2]["score"] == 0.5
    assert [hit["rank"] for hit in fused] == [1, 2, 3]
    tied = reciprocal_rank_fusion([[{"chunk_id": "z"}], [{"chunk_id": "a"}]], rank_constant=0)
    assert [hit["chunk_id"] for hit in tied] == ["a", "z"]


def test_rrf_rejects_duplicate_candidate_within_one_ranking() -> None:
    hits = [{"chunk_id": "a"}, {"chunk_id": "a"}]
    try:
        reciprocal_rank_fusion([hits])
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate candidate was accepted")


def test_bm25_tokenization_code_unicode_and_stable_ties() -> None:
    assert tokenize("@Input() café NG0100") == ["@input", "(", ")", "café", "ng0100"]
    chunks = []
    from rag_chunking.chunking.models import Chunk
    for index, (chunk_id, text) in enumerate((("b", "generic"), ("a", "generic"), ("exact", "signal input api"))):
        chunks.append(Chunk(
            chunk_id=chunk_id, doc_id=f"d{index}", source="angular", relative_path=f"{chunk_id}.md",
            strategy="fixed_size", chunk_index=0, text=text, token_count=1,
            token_start=None, token_end=None, chunk_size=8, chunk_overlap=0, tokenizer="test", metadata={},
        ))
    index = BM25Index(chunks, BM25Config())
    assert index.search("signal", 1)[0]["chunk_id"] == "exact"
    assert [hit["chunk_id"] for hit in index.search("absent", 3)] == ["a", "b", "exact"]
    assert index.fingerprint == BM25Index(list(reversed(chunks)), BM25Config()).fingerprint


def test_paired_bootstrap_is_seeded_and_reports_paired_delta() -> None:
    first = paired_bootstrap([0.0, 1.0], [1.0, 1.0], samples=100)
    second = paired_bootstrap([0.0, 1.0], [1.0, 1.0], samples=100)
    assert first == second
    assert first["delta"] == 0.5
    assert first["delta_ci_low"] <= first["delta"] <= first["delta_ci_high"]


def test_offline_lexical_fusion_evaluation_pipeline() -> None:
    from rag_chunking.chunking.models import Chunk
    chunks = [Chunk(
        chunk_id=name, strategy="fixed_size", doc_id=name, source="angular",
        relative_path=f"{name}.md", chunk_index=0, text=text,
        token_start=None, token_end=None, token_count=1, chunk_size=8,
        chunk_overlap=0, tokenizer="test",
    ) for name, text in (("noise", "unrelated"), ("gold", "exact signal api"))]
    lexical = BM25Index(chunks).search("signal api", 2)
    dense = [dict(lexical[1], rank=1, score=0.9), dict(lexical[0], rank=2, score=0.8)]
    fused = reciprocal_rank_fusion([dense, lexical], rank_constant=60, limit=2)
    metrics = evaluate_depth_ranking([hit["relative_path"] for hit in fused], {"gold.md"}, 2)
    assert len({hit["chunk_id"] for hit in fused}) == 2
    assert metrics["hit_at_1"] == 1 and metrics["first_relevant_rank"] == 1
