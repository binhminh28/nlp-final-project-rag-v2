"""Fair multi-strategy retrieval benchmark and deterministic publication."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.retrieval.models import RetrievalRequest
from rag_chunking.retrieval.service import RetrievalService

from .dataset import CATEGORIES, EVALUATION_DATASET_SCHEMA_VERSION, EvaluationDataset
from .metrics import METRICS_VERSION, aggregate, evaluate_ranking


BENCHMARK_SCHEMA_VERSION = "retrieval_benchmark_v1"


@dataclass(slots=True)
class BenchmarkRunResult:
    benchmark_fingerprint: str
    output_directory: Path
    per_query: list[dict[str, Any]]
    aggregates: dict[str, Any]
    comparison: dict[str, Any]
    failures: dict[str, Any]
    stats: dict[str, Any]


def _jsonl(values: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for value in values)


def _pairwise(per_query: list[dict[str, Any]], strategies: list[str]) -> dict[str, Any]:
    by_query = {(item["query_id"], item["strategy"]): item for item in per_query}
    query_ids = sorted({item["query_id"] for item in per_query})
    result: dict[str, Any] = {}
    for offset, left in enumerate(strategies):
        for right in strategies[offset + 1:]:
            counts = {"a_wins": 0, "ties": 0, "b_wins": 0}
            details = []
            for query_id in query_ids:
                left_rank = by_query[(query_id, left)]["first_relevant_rank"]
                right_rank = by_query[(query_id, right)]["first_relevant_rank"]
                left_value = left_rank if left_rank is not None else 10**9
                right_value = right_rank if right_rank is not None else 10**9
                outcome = "ties" if left_value == right_value else ("a_wins" if left_value < right_value else "b_wins")
                counts[outcome] += 1
                details.append({"query_id": query_id, "outcome": outcome, "a_rank": left_rank, "b_rank": right_rank})
            result[f"{left}_vs_{right}"] = {"strategy_a": left, "strategy_b": right, **counts, "queries": details}
    return result


def _failure_artifacts(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    failures: dict[str, Any] = {}
    for cutoff in (1, 5, 10):
        failures[f"miss_at_{cutoff}"] = [item for item in per_query if not item[f"hit_at_{cutoff}"]]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in per_query:
        grouped.setdefault(item["query_id"], []).append(item)
    failures["strategy_disagreements"] = [
        {"query_id": query_id, "outcomes": {item["strategy"]: item["first_relevant_rank"] for item in sorted(items, key=lambda x: x["strategy"])}}
        for query_id, items in sorted(grouped.items())
        if len({item["first_relevant_rank"] for item in items}) > 1
    ]
    return failures


def run_retrieval_benchmark(
    service: RetrievalService, dataset: EvaluationDataset, output_root: Path,
    *, strategies: list[str], depth: int = 10, baseline_name: str = "baseline_v1",
) -> BenchmarkRunResult:
    if depth < 10:
        raise ValueError("evaluation depth must be at least 10")
    if len(strategies) != len(set(strategies)) or not strategies:
        raise ValueError("strategies must be non-empty and unique")
    if set(strategies) - set(service.indexes):
        raise ValueError("one or more evaluation strategies are not loaded")
    identity = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "dataset_schema_version": EVALUATION_DATASET_SCHEMA_VERSION,
        "dataset_fingerprint": dataset.fingerprint,
        "corpus": service.corpus,
        "retrieval_config_fingerprint": service.config.fingerprint,
        "embedding_config_fingerprint": service.embedding_config.fingerprint,
        "index_fingerprints": {name: service.manifests[name]["index_fingerprint"] for name in sorted(strategies)},
        "metrics_version": METRICS_VERSION,
        "ground_truth_level": "relative_path",
        "evaluation_depth": depth,
    }
    fingerprint = canonical_fingerprint(identity)
    output_directory = output_root / baseline_name / fingerprint
    started = time.monotonic()
    calls_before = service.provider.calls
    token_before = service.provider.input_tokens
    cache_hits = cache_misses = retrieval_calls = 0
    per_query: list[dict[str, Any]] = []
    try:
        for record in dataset.records:
            vector, cache_hit = service.embed_query(record.query)
            cache_hits += int(cache_hit)
            cache_misses += int(not cache_hit)
            for strategy in strategies:
                result = service.retrieve(
                    RetrievalRequest(record.query, strategy, depth),
                    query_vector=vector, cache_hit=cache_hit,
                )
                retrieval_calls += 1
                metrics = evaluate_ranking([hit.relative_path for hit in result.hits], set(record.relevant_sources))
                per_query.append({
                    "query_id": record.query_id, "query": record.query, "category": record.category,
                    "strategy": strategy, "retrieval_config_fingerprint": result.retrieval_config_fingerprint,
                    "embedding_config_fingerprint": result.embedding_config_fingerprint,
                    "index_fingerprint": result.index_fingerprint, "top_k": depth,
                    "relevant_targets": record.relevant_sources,
                    "hits": [hit.to_dict() for hit in result.hits], **metrics,
                })
        per_query.sort(key=lambda item: (item["query_id"], item["strategy"]))
        aggregates: dict[str, Any] = {}
        for strategy in strategies:
            strategy_records = [item for item in per_query if item["strategy"] == strategy]
            aggregates[strategy] = {
                "overall": aggregate(strategy_records),
                "categories": {
                    category: aggregate([item for item in strategy_records if item["category"] == category])
                    for category in sorted(CATEGORIES) if any(item["category"] == category for item in strategy_records)
                },
            }
        comparison = _pairwise(per_query, strategies)
        failures = _failure_artifacts(per_query)
        manifest = {**identity, "benchmark_fingerprint": fingerprint, "baseline_name": baseline_name, "complete": True, "query_count": len(dataset.records), "strategy_count": len(strategies)}
        stats = {
            "query_count": len(dataset.records), "strategy_count": len(strategies),
            "query_embedding_cache_hits": cache_hits, "query_embedding_cache_misses": cache_misses,
            "provider_calls": service.provider.calls - calls_before,
            "query_embedding_tokens": service.provider.input_tokens - token_before,
            "retrieval_calls": retrieval_calls, "total_runtime_seconds": time.monotonic() - started,
        }
        report = _render_report(manifest, aggregates, comparison, failures)
        write_artifact_set(output_directory, {
            "per_query.jsonl": _jsonl(per_query), "aggregate.json": serialize_json(aggregates),
            "comparison.json": serialize_json(comparison), "failures.json": serialize_json(failures),
            "baseline_report.md": report, "stats.json": serialize_json(stats),
            "manifest.json": serialize_json(manifest),
        })
        failure_path = output_directory / "failure.json"
        if failure_path.exists():
            failure_path.unlink()
    except BaseException as error:
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "failure.json").write_text(serialize_json({"complete": False, "error_type": type(error).__name__, "error": str(error)}), encoding="utf-8")
        raise
    return BenchmarkRunResult(fingerprint, output_directory, per_query, aggregates, comparison, failures, stats)


def _render_report(manifest: dict[str, Any], aggregates: dict[str, Any], comparison: dict[str, Any], failures: dict[str, Any]) -> str:
    lines = ["# Dense Retrieval Baseline", "", f"Benchmark fingerprint: `{manifest['benchmark_fingerprint']}`", "", "## Overall metrics", "", "| Strategy | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Recall@5 | Recall@10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for strategy, value in aggregates.items():
        metric = value["overall"]
        lines.append(f"| {strategy} | {metric['hit_at_1']:.4f} | {metric['hit_at_3']:.4f} | {metric['hit_at_5']:.4f} | {metric['hit_at_10']:.4f} | {metric['mrr']:.4f} | {metric['recall_at_5']:.4f} | {metric['recall_at_10']:.4f} |")
    lines.extend(["", "## Pairwise first-relevant-rank comparison", ""])
    for name, value in comparison.items():
        lines.append(f"- {name}: {value['a_wins']} / {value['ties']} / {value['b_wins']} (A wins / ties / B wins)")
    lines.extend(["", "## Failure counts", "", f"- Miss@1: {len(failures['miss_at_1'])}", f"- Miss@5: {len(failures['miss_at_5'])}", f"- Miss@10: {len(failures['miss_at_10'])}", ""])
    return "\n".join(lines)
