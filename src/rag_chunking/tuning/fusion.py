"""Deterministic rank fusion primitives."""

from __future__ import annotations

from typing import Any


RRF_SCHEMA_VERSION = "reciprocal_rank_fusion_v1"


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]], *, rank_constant: int = 60,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse rankings by chunk identity using sum(1 / (k + rank))."""
    if type(rank_constant) is not int or rank_constant < 0:
        raise ValueError("RRF rank constant must be a non-negative integer")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("RRF limit must be positive")
    scores: dict[str, float] = {}
    representatives: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for position, hit in enumerate(ranking, 1):
            chunk_id = hit.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
                raise ValueError("each input ranking must contain unique valid chunk IDs")
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + position)
            representatives.setdefault(chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    if limit is not None:
        ordered = ordered[:limit]
    output = []
    for rank, chunk_id in enumerate(ordered, 1):
        value = dict(representatives[chunk_id])
        value["rank"] = rank
        value["score"] = scores[chunk_id]
        output.append(value)
    return output
