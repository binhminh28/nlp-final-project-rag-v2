"""Shared deterministic publication for ranked retrieval experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.evaluation.dataset import CATEGORIES, EvaluationDataset

from .config import ExperimentConfig
from .metrics import TUNING_METRICS_VERSION, aggregate_depth, evaluate_depth_ranking


def publish_ranked_experiment(
    *, config: ExperimentConfig, corpus: str, dataset: EvaluationDataset, strategies: list[str],
    rankings: dict[tuple[str, str], list[dict[str, Any]]], output_depth: int,
    output_root: Path, reference_rows: dict[tuple[str, str], dict[str, Any]], stats: dict[str, Any],
) -> Path:
    rows = []
    for record in dataset.records:
        for strategy in strategies:
            hits = rankings[(record.query_id, strategy)][:output_depth]
            metrics = evaluate_depth_ranking([hit["relative_path"] for hit in hits], set(record.relevant_sources), output_depth)
            rows.append({"query_id": record.query_id, "query": record.query, "category": record.category,
                         "strategy": strategy, "relevant_targets": record.relevant_sources,
                         "candidate_depth": output_depth, "hits": hits, **metrics})
    aggregate = {strategy: aggregate_depth([row for row in rows if row["strategy"] == strategy], output_depth) for strategy in strategies}
    categories = {strategy: {category: aggregate_depth([row for row in rows if row["strategy"] == strategy and row["category"] == category], output_depth) for category in sorted(CATEGORIES)} for strategy in strategies}
    comparisons = []
    counts = {"improved": 0, "unchanged": 0, "degraded": 0}
    for row in rows:
        ref = reference_rows[(row["query_id"], row["strategy"])]
        old, new = ref.get("first_relevant_rank") or 10**9, row.get("first_relevant_rank") or 10**9
        outcome = "unchanged" if old == new else ("improved" if new < old else "degraded")
        counts[outcome] += 1
        comparisons.append({"query_id": row["query_id"], "strategy": row["strategy"], "outcome": outcome,
                            "reference_first_relevant_rank": ref.get("first_relevant_rank"),
                            "experiment_first_relevant_rank": row.get("first_relevant_rank")})
    output = output_root / config.fingerprint
    manifest = {"schema_version": config.schema_version, "complete": True, "experiment_id": config.experiment_id,
                "experiment_family": config.experiment_family, "experiment_fingerprint": config.fingerprint,
                "corpus": corpus, "dataset_fingerprint": dataset.fingerprint, "metrics_version": TUNING_METRICS_VERSION,
                "query_count": len(dataset.records), "strategy_count": len(strategies)}
    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    write_artifact_set(output, {"config.json": serialize_json(config.identity()), "per_query.jsonl": jsonl,
                                "aggregate.json": serialize_json(aggregate), "category_metrics.json": serialize_json(categories),
                                "comparison.json": serialize_json({**counts, "queries": comparisons}),
                                "stats.json": serialize_json(stats), "manifest.json": serialize_json(manifest)})
    return output
