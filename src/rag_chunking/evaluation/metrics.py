"""Small hand-auditable binary source-relevance metrics."""

from __future__ import annotations

from typing import Any


METRICS_VERSION = "binary_relative_path_metrics_v1"
CUTOFFS = (1, 3, 5, 10)


def evaluate_ranking(retrieved_sources: list[str], relevant_sources: set[str]) -> dict[str, int | float | None]:
    if not relevant_sources:
        raise ValueError("metrics require at least one relevance target")
    first_rank = next((rank for rank, source in enumerate(retrieved_sources, 1) if source in relevant_sources), None)
    result: dict[str, int | float | None] = {"first_relevant_rank": first_rank}
    for cutoff in CUTOFFS:
        top = retrieved_sources[:cutoff]
        result[f"hit_at_{cutoff}"] = int(any(source in relevant_sources for source in top))
    for cutoff in (5, 10):
        distinct_relevant = set(retrieved_sources[:cutoff]) & relevant_sources
        result[f"recall_at_{cutoff}"] = len(distinct_relevant) / len(relevant_sources)
    result["reciprocal_rank"] = 0.0 if first_rank is None else 1.0 / first_rank
    return result


def aggregate(records: list[dict[str, Any]]) -> dict[str, int | float]:
    if not records:
        return {"query_count": 0, **{name: 0.0 for name in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr", "recall_at_5", "recall_at_10")}}
    return {
        "query_count": len(records),
        "hit_at_1": sum(item["hit_at_1"] for item in records) / len(records),
        "hit_at_3": sum(item["hit_at_3"] for item in records) / len(records),
        "hit_at_5": sum(item["hit_at_5"] for item in records) / len(records),
        "hit_at_10": sum(item["hit_at_10"] for item in records) / len(records),
        "mrr": sum(item["reciprocal_rank"] for item in records) / len(records),
        "recall_at_5": sum(item["recall_at_5"] for item in records) / len(records),
        "recall_at_10": sum(item["recall_at_10"] for item in records) / len(records),
    }
