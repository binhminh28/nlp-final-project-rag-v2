"""Memory-bounded dense candidate-depth experiments and diagnostics."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.embedding.models import EmbeddingConfig
from rag_chunking.embedding.provider import EmbeddingProvider
from rag_chunking.evaluation.dataset import CATEGORIES, EvaluationDataset
from rag_chunking.retrieval.cache import QueryEmbeddingCache
from rag_chunking.retrieval.models import RetrievalConfig, RetrievalRequest
from rag_chunking.retrieval.service import RetrievalService

from .config import ExperimentConfig
from .metrics import TUNING_METRICS_VERSION, aggregate_depth, evaluate_depth_ranking


def _jsonl(values: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for value in values)


def retrieve_dense_candidates(
    *, corpus: str, dataset: EvaluationDataset, strategies: list[str],
    index_directories: dict[str, Path], embedding_config: EmbeddingConfig,
    provider_factory: Any, query_cache_directory: Path, depth: int,
    repository_root: Path,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, str], dict[str, Any]]:
    cache = QueryEmbeddingCache(query_cache_directory, embedding_config)
    vectors = {}
    for record in dataset.records:
        vector = cache.get(record.query)
        if vector is None:
            raise ValueError(f"candidate experiment requires cached query embedding for {record.query_id}")
        vectors[record.query_id] = vector
    all_hits: dict[tuple[str, str], list[dict[str, Any]]] = {}
    index_fingerprints: dict[str, str] = {}
    timings: dict[str, float] = {}
    for strategy in strategies:
        started = time.monotonic()
        provider: EmbeddingProvider = provider_factory()
        service = RetrievalService(
            corpus=corpus, index_directories={strategy: index_directories[strategy]},
            embedding_config=embedding_config, provider=provider,
            query_cache_directory=query_cache_directory, repository_root=repository_root,
        )
        index_fingerprints[strategy] = service.manifests[strategy]["index_fingerprint"]
        for record in dataset.records:
            result = service.retrieve(
                RetrievalRequest(record.query, strategy, depth),
                query_vector=vectors[record.query_id], cache_hit=True,
            )
            all_hits[(record.query_id, strategy)] = [hit.to_dict() for hit in result.hits]
        if provider.calls:
            raise ValueError("cached dense experiment unexpectedly called embedding provider")
        timings[strategy] = time.monotonic() - started
        del service, provider
        gc.collect()
    return all_hits, index_fingerprints, {
        "query_embedding_cache_hits": len(dataset.records), "query_embedding_cache_misses": 0,
        "provider_calls": 0, "strategy_runtime_seconds": timings,
    }


def _pairwise(rows: list[dict[str, Any]], strategies: list[str]) -> dict[str, Any]:
    by = {(row["query_id"], row["strategy"]): row for row in rows}
    result: dict[str, Any] = {}
    for offset, left in enumerate(strategies):
        for right in strategies[offset + 1:]:
            counts = {"a_wins": 0, "ties": 0, "b_wins": 0}
            details = []
            for query_id in sorted({row["query_id"] for row in rows}):
                a = by[(query_id, left)]["first_relevant_rank"]
                b = by[(query_id, right)]["first_relevant_rank"]
                av, bv = (a or 10**9), (b or 10**9)
                outcome = "ties" if av == bv else ("a_wins" if av < bv else "b_wins")
                counts[outcome] += 1
                details.append({"query_id": query_id, "a_rank": a, "b_rank": b, "outcome": outcome})
            result[f"{left}_vs_{right}"] = {"strategy_a": left, "strategy_b": right, **counts, "queries": details}
    return result


def _overlap(rows: list[dict[str, Any]], strategies: list[str], depth: int) -> dict[str, Any]:
    by = {(row["query_id"], row["strategy"]): row for row in rows}
    details = []
    aggregate: dict[str, list[float]] = {}
    for query_id in sorted({row["query_id"] for row in rows}):
        relevant = set(next(row["relevant_targets"] for row in rows if row["query_id"] == query_id))
        sets = {strategy: {hit["relative_path"] for hit in by[(query_id, strategy)]["hits"]} for strategy in strategies}
        item: dict[str, Any] = {"query_id": query_id, "strategy_exclusive_relevant": {}}
        for offset, left in enumerate(strategies):
            others = set().union(*(sets[name] for name in strategies if name != left))
            item["strategy_exclusive_relevant"][left] = sorted((sets[left] - others) & relevant)
            for right in strategies[offset + 1:]:
                union = sets[left] | sets[right]
                jaccard = len(sets[left] & sets[right]) / len(union) if union else 1.0
                name = f"{left}_vs_{right}"
                item[name] = {"source_intersection": len(sets[left] & sets[right]), "source_union": len(union), "jaccard": jaccard}
                aggregate.setdefault(name, []).append(jaccard)
        details.append(item)
    return {
        "candidate_depth": depth,
        "mean_pairwise_source_jaccard": {name: sum(values) / len(values) for name, values in sorted(aggregate.items())},
        "exclusive_relevant_counts": {
            strategy: sum(bool(item["strategy_exclusive_relevant"][strategy]) for item in details)
            for strategy in strategies
        },
        "queries": details,
    }


def publish_dense_depth_experiments(
    *, corpus: str, dataset: EvaluationDataset, strategies: list[str],
    all_hits: dict[tuple[str, str], list[dict[str, Any]]], index_fingerprints: dict[str, str],
    embedding_config: EmbeddingConfig, retrieval_config: RetrievalConfig,
    depths: list[int], output_root: Path, runtime_stats: dict[str, Any],
) -> list[Path]:
    if sorted(set(depths)) != sorted(depths) or max(depths) > min(len(value) for value in all_hits.values()):
        raise ValueError("depths must be unique and available in candidate results")
    output_paths = []
    records = {record.query_id: record for record in dataset.records}
    for depth in depths:
        config = ExperimentConfig(
            experiment_id=f"E1-depth-{depth}", experiment_name=f"dense_candidate_depth_{depth}",
            experiment_family="E1", dataset_fingerprint=dataset.fingerprint,
            retrieval_config_fingerprint=retrieval_config.fingerprint,
            embedding_config_fingerprint=embedding_config.fingerprint,
            index_fingerprints=dict(sorted(index_fingerprints.items())), candidate_depth=depth,
        )
        rows = []
        for query_id in sorted(records):
            record = records[query_id]
            for strategy in strategies:
                hits = all_hits[(query_id, strategy)][:depth]
                metrics = evaluate_depth_ranking([hit["relative_path"] for hit in hits], set(record.relevant_sources), depth)
                rows.append({
                    "query_id": query_id, "query": record.query, "category": record.category,
                    "strategy": strategy, "relevant_targets": record.relevant_sources,
                    "candidate_depth": depth, "hits": hits, **metrics,
                })
        aggregates = {}
        categories = {}
        for strategy in strategies:
            selected = [row for row in rows if row["strategy"] == strategy]
            aggregates[strategy] = aggregate_depth(selected, depth)
            categories[strategy] = {
                category: aggregate_depth([row for row in selected if row["category"] == category], depth)
                for category in sorted(CATEGORIES)
            }
        comparison = _pairwise(rows, strategies)
        overlap = _overlap(rows, strategies, depth)
        manifest = {
            "schema_version": config.schema_version, "complete": True,
            "experiment_fingerprint": config.fingerprint, "experiment_id": config.experiment_id,
            "experiment_family": config.experiment_family, "corpus": corpus,
            "dataset_fingerprint": dataset.fingerprint, "metrics_version": TUNING_METRICS_VERSION,
            "candidate_depth": depth, "strategy_count": len(strategies), "query_count": len(records),
            "index_fingerprints": dict(sorted(index_fingerprints.items())),
            "embedding_config_fingerprint": embedding_config.fingerprint,
            "retrieval_config_fingerprint": retrieval_config.fingerprint,
        }
        output = output_root / config.fingerprint
        write_artifact_set(output, {
            "config.json": serialize_json(config.identity()), "per_query.jsonl": _jsonl(rows),
            "aggregate.json": serialize_json(aggregates), "category_metrics.json": serialize_json(categories),
            "comparison.json": serialize_json(comparison), "candidate_overlap.json": serialize_json(overlap),
            "stats.json": serialize_json({**runtime_stats, "materialized_depth": depth}),
            "manifest.json": serialize_json(manifest),
        })
        output_paths.append(output)
    return output_paths
