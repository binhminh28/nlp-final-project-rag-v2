"""Deterministic source-cap diversification ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_chunking.chunking.writer import serialize_json, write_artifact_set

from .config import ExperimentConfig
from .metrics import TUNING_METRICS_VERSION, aggregate_depth, evaluate_depth_ranking


def source_cap(hits: list[dict[str, Any]], max_per_source: int | None, result_depth: int) -> list[dict[str, Any]]:
    if max_per_source is not None and (type(max_per_source) is not int or max_per_source <= 0):
        raise ValueError("source cap must be a positive integer or null")
    if result_depth <= 0:
        raise ValueError("result depth must be positive")
    counts: dict[str, int] = {}
    selected = []
    for hit in hits:
        source = hit["relative_path"]
        if max_per_source is not None and counts.get(source, 0) >= max_per_source:
            continue
        counts[source] = counts.get(source, 0) + 1
        selected.append({**hit, "rank": len(selected) + 1})
        if len(selected) == result_depth:
            break
    return selected


def _jsonl(values: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for value in values)


def run_diversity_ablation(
    *, e1_depth50_directory: Path, output_root: Path, caps: list[int], result_depth: int,
) -> list[Path]:
    source_manifest = json.loads((e1_depth50_directory / "manifest.json").read_text())
    source_config = json.loads((e1_depth50_directory / "config.json").read_text())
    source_rows = [json.loads(line) for line in (e1_depth50_directory / "per_query.jsonl").read_text().splitlines()]
    if source_manifest.get("candidate_depth") < result_depth or source_manifest.get("experiment_family") != "E1":
        raise ValueError("diversity source must be a sufficiently deep E1 experiment")
    strategies = sorted({row["strategy"] for row in source_rows})
    outputs = []
    for cap in caps:
        config = ExperimentConfig(
            experiment_id=f"E2-source-cap-{cap}", experiment_name=f"dense_source_cap_{cap}",
            experiment_family="E2", dataset_fingerprint=source_manifest["dataset_fingerprint"],
            retrieval_config_fingerprint=source_manifest["retrieval_config_fingerprint"],
            embedding_config_fingerprint=source_manifest["embedding_config_fingerprint"],
            index_fingerprints=source_manifest["index_fingerprints"],
            candidate_depth=source_manifest["candidate_depth"],
            ranking_method="dense_cosine_source_cap",
            diversity={"policy": "max_chunks_per_relative_path_v1", "max_chunks_per_source": cap, "result_depth": result_depth},
        )
        rows = []
        for source_row in source_rows:
            hits = source_cap(source_row["hits"], cap, result_depth)
            metrics = evaluate_depth_ranking(
                [hit["relative_path"] for hit in hits], set(source_row["relevant_targets"]), result_depth,
            )
            rows.append({
                "query_id": source_row["query_id"], "query": source_row["query"],
                "category": source_row["category"], "strategy": source_row["strategy"],
                "relevant_targets": source_row["relevant_targets"], "candidate_depth": source_manifest["candidate_depth"],
                "result_depth": result_depth, "hits": hits, **metrics,
            })
        aggregates = {}
        categories = {}
        regressions = []
        reference_by_key = {(row["query_id"], row["strategy"]): row for row in source_rows}
        for strategy in strategies:
            selected = [row for row in rows if row["strategy"] == strategy]
            aggregates[strategy] = aggregate_depth(selected, result_depth)
            categories[strategy] = {
                category: aggregate_depth([row for row in selected if row["category"] == category], result_depth)
                for category in sorted({row["category"] for row in selected})
            }
        for row in rows:
            reference = reference_by_key[(row["query_id"], row["strategy"])]
            ref_rank = reference["first_relevant_rank"] if reference["first_relevant_rank"] and reference["first_relevant_rank"] <= result_depth else None
            new_rank = row["first_relevant_rank"]
            ref_value, new_value = ref_rank or 10**9, new_rank or 10**9
            outcome = "unchanged" if ref_value == new_value else ("improved" if new_value < ref_value else "degraded")
            if outcome != "unchanged":
                regressions.append({"query_id": row["query_id"], "strategy": row["strategy"], "outcome": outcome, "reference_rank": ref_rank, "experiment_rank": new_rank})
        manifest = {
            "schema_version": config.schema_version, "complete": True,
            "experiment_fingerprint": config.fingerprint, "experiment_id": config.experiment_id,
            "experiment_family": "E2", "dataset_fingerprint": config.dataset_fingerprint,
            "source_experiment_fingerprint": source_manifest["experiment_fingerprint"],
            "candidate_depth": config.candidate_depth, "result_depth": result_depth,
            "metrics_version": TUNING_METRICS_VERSION, "strategy_count": len(strategies),
            "query_count": len({row["query_id"] for row in rows}),
        }
        output = output_root / config.fingerprint
        write_artifact_set(output, {
            "config.json": serialize_json(config.identity()), "per_query.jsonl": _jsonl(rows),
            "aggregate.json": serialize_json(aggregates), "category_metrics.json": serialize_json(categories),
            "regressions.jsonl": _jsonl(regressions),
            "stats.json": serialize_json({"provider_calls": 0, "source_cap": cap}),
            "manifest.json": serialize_json(manifest),
        })
        outputs.append(output)
    return outputs
