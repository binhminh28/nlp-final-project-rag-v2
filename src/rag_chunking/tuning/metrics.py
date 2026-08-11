"""Depth-aware source relevance and diversity diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any


TUNING_METRICS_VERSION = "binary_relative_path_tuning_metrics_v1"
HIT_CUTOFFS = (1, 3, 5, 10, 20)
RECALL_CUTOFFS = (5, 10, 20, 50)


def evaluate_depth_ranking(
    retrieved_sources: list[str], relevant_sources: set[str], candidate_depth: int,
) -> dict[str, int | float | None]:
    if not relevant_sources:
        raise ValueError("at least one relevance target is required")
    if candidate_depth <= 0 or len(retrieved_sources) > candidate_depth:
        raise ValueError("invalid candidate depth/ranking length")
    first = next((rank for rank, source in enumerate(retrieved_sources, 1) if source in relevant_sources), None)
    result: dict[str, int | float | None] = {
        "first_relevant_rank": first,
        "reciprocal_rank": 0.0 if first is None else 1.0 / first,
    }
    for cutoff in HIT_CUTOFFS:
        result[f"hit_at_{cutoff}"] = (
            int(any(source in relevant_sources for source in retrieved_sources[:cutoff]))
            if cutoff <= candidate_depth else None
        )
    for cutoff in RECALL_CUTOFFS:
        result[f"recall_at_{cutoff}"] = (
            len(set(retrieved_sources[:cutoff]) & relevant_sources) / len(relevant_sources)
            if cutoff <= candidate_depth else None
        )
    counts = Counter(retrieved_sources)
    result["unique_source_count"] = len(counts)
    result["same_source_concentration"] = max(counts.values(), default=0) / max(len(retrieved_sources), 1)
    result["duplicate_source_chunks"] = len(retrieved_sources) - len(counts)
    return result


def aggregate_depth(records: list[dict[str, Any]], candidate_depth: int) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = {"query_count": len(records)}
    names = [f"hit_at_{cutoff}" for cutoff in HIT_CUTOFFS]
    names += [f"recall_at_{cutoff}" for cutoff in RECALL_CUTOFFS]
    names += ["reciprocal_rank", "unique_source_count", "same_source_concentration", "duplicate_source_chunks"]
    for name in names:
        values = [record[name] for record in records if record.get(name) is not None]
        key = "mrr" if name == "reciprocal_rank" else (
            "mean_unique_sources" if name == "unique_source_count" else
            "mean_same_source_concentration" if name == "same_source_concentration" else
            "mean_duplicate_source_chunks" if name == "duplicate_source_chunks" else name
        )
        result[key] = sum(values) / len(values) if values else None
    result["candidate_depth"] = candidate_depth
    return result

