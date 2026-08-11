"""Small hand-auditable binary source-relevance metrics."""

from __future__ import annotations

from typing import Any


METRICS_VERSION = "binary_relative_path_metrics_v1"
EVIDENCE_METRICS_VERSION = "evidence_coverage_metrics_v1"
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


def evaluate_evidence_coverage(unit_coverages: list[float]) -> dict[str, int | float]:
    """Macro-ready evidence metrics for one query.

    Each value represents the fraction of one required evidence unit covered by
    selected chunks. Partial boundary coverage is retained for auditability;
    ``all_evidence_retrieved`` requires complete coverage of every unit.
    """

    if not unit_coverages:
        raise ValueError("evidence metrics require at least one evidence unit")
    if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in unit_coverages):
        raise ValueError("evidence unit coverage must be between zero and one")
    coverage = sum(unit_coverages) / len(unit_coverages)
    return {
        "evidence_unit_count": len(unit_coverages),
        "covered_evidence_units": sum(value >= 1.0 for value in unit_coverages),
        "evidence_coverage": coverage,
        "all_evidence_retrieved": int(all(value >= 1.0 for value in unit_coverages)),
    }


def aggregate_evidence(records: list[dict[str, Any]]) -> dict[str, int | float]:
    if not records:
        return {"query_count": 0, "evidence_coverage": 0.0, "all_evidence_retrieved_rate": 0.0}
    return {
        "query_count": len(records),
        "evidence_coverage": sum(item["evidence_coverage"] for item in records) / len(records),
        "all_evidence_retrieved_rate": sum(item["all_evidence_retrieved"] for item in records) / len(records),
    }
