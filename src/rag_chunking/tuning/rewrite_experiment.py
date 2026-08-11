"""Isolated rewrite-only and original-plus-rewrite retrieval experiments."""

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
from .fusion import RRF_SCHEMA_VERSION, reciprocal_rank_fusion
from .metrics import TUNING_METRICS_VERSION, aggregate_depth, evaluate_depth_ranking


def load_rewrite_records(path: Path, dataset: EvaluationDataset) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            query_id = value.get("query_id")
            if query_id in values:
                raise ValueError(f"duplicate rewrite query ID: {query_id}")
            values[query_id] = value
    expected = {record.query_id: record.query for record in dataset.records}
    if set(values) != set(expected):
        raise ValueError("rewrite records do not cover the canonical dataset exactly")
    for query_id, query in expected.items():
        if values[query_id].get("original_query") != query:
            raise ValueError(f"rewrite original query mismatch for {query_id}")
        if not values[query_id].get("rewritten_query"):
            raise ValueError(f"empty effective rewrite for {query_id}")
    return values


def prepare_rewrite_embeddings(
    rewrites: dict[str, dict[str, str]], cache: QueryEmbeddingCache,
    provider: EmbeddingProvider, *, populate: bool, batch_size: int = 32,
    limit: int | None = None,
) -> dict[str, Any]:
    unique = sorted({value["rewritten_query"] for value in rewrites.values()})
    missing = [query for query in unique if cache.get(query) is None]
    estimated_tokens = sum(max(1, len(query.split())) for query in missing)
    if missing and not populate:
        raise ValueError(
            f"{len(missing)} rewritten query embeddings are missing; estimated whitespace tokens={estimated_tokens}"
        )
    selected = missing[:limit] if limit is not None else missing
    started = time.monotonic()
    for offset in range(0, len(selected), batch_size):
        texts = selected[offset:offset + batch_size]
        vectors = provider.embed_texts(texts)
        for text, vector in zip(texts, vectors, strict=True):
            cache.put(text, vector)
    return {
        "unique_rewrite_queries": len(unique), "embedding_cache_hits": len(unique) - len(missing),
        "embedding_cache_misses": len(missing), "embeddings_populated": len(selected),
        "embeddings_remaining": len(missing) - len(selected), "embedding_provider_calls": provider.calls,
        "embedding_provider_retries": provider.retries, "embedding_input_tokens": provider.input_tokens,
        "estimated_missing_whitespace_tokens": estimated_tokens,
        "embedding_population_seconds": time.monotonic() - started,
    }


def run_rewrite_retrieval(
    *, corpus: str, dataset: EvaluationDataset, rewrites: dict[str, dict[str, str]],
    strategies: list[str], index_directories: dict[str, Path], embedding_config: EmbeddingConfig,
    provider_factory: Any, query_cache_directory: Path, repository_root: Path,
    retrieval_depth: int = 20, output_depth: int = 10, rrf_constant: int = 60,
    original_candidates: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, dict[tuple[str, str], list[dict[str, Any]]]], dict[str, str], dict[str, Any]]:
    cache = QueryEmbeddingCache(query_cache_directory, embedding_config)
    vectors = {}
    for record in dataset.records:
        original = cache.get(record.query)
        rewritten = cache.get(rewrites[record.query_id]["rewritten_query"])
        if original is None or rewritten is None:
            raise ValueError(f"missing cached query vector for {record.query_id}")
        vectors[record.query_id] = (original, rewritten)
    variants = {"Q1": {}, "Q2": {}}
    fingerprints: dict[str, str] = {}
    timings: dict[str, float] = {}
    for strategy in strategies:
        started = time.monotonic()
        provider = provider_factory()
        service = RetrievalService(
            corpus=corpus, index_directories={strategy: index_directories[strategy]},
            embedding_config=embedding_config, provider=provider,
            query_cache_directory=query_cache_directory, repository_root=repository_root,
        )
        fingerprints[strategy] = service.manifests[strategy]["index_fingerprint"]
        for record in dataset.records:
            original_vector, rewritten_vector = vectors[record.query_id]
            rewritten_query = rewrites[record.query_id]["rewritten_query"]
            original_values = None if original_candidates is None else original_candidates.get((record.query_id, strategy))
            if original_values is None:
                original_values = [hit.to_dict() for hit in service.retrieve(
                    RetrievalRequest(record.query, strategy, retrieval_depth),
                    query_vector=original_vector, cache_hit=True,
                ).hits]
            if len(original_values) < retrieval_depth:
                raise ValueError(f"reference candidates too shallow for {record.query_id}/{strategy}")
            rewrite_hits = service.retrieve(
                RetrievalRequest(rewritten_query, strategy, retrieval_depth),
                query_vector=rewritten_vector, cache_hit=True,
            ).hits
            rewrite_values = [hit.to_dict() for hit in rewrite_hits]
            variants["Q1"][(record.query_id, strategy)] = rewrite_values[:output_depth]
            variants["Q2"][(record.query_id, strategy)] = reciprocal_rank_fusion(
                [original_values[:retrieval_depth], rewrite_values],
                rank_constant=rrf_constant, limit=output_depth,
            )
        if provider.calls:
            raise ValueError("rewrite retrieval unexpectedly called embedding provider")
        timings[strategy] = time.monotonic() - started
        del service, provider
        gc.collect()
    return variants, fingerprints, {
        "retrieval_depth_per_query_variant": retrieval_depth, "output_depth": output_depth,
        "rrf_rank_constant": rrf_constant, "provider_calls": 0,
        "strategy_runtime_seconds": timings,
    }


def _comparison(rows: list[dict[str, Any]], reference_rows: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    details = []
    counts = {"improved": 0, "unchanged": 0, "degraded": 0}
    for row in rows:
        reference = reference_rows[(row["query_id"], row["strategy"])]
        old = reference.get("first_relevant_rank") or 10**9
        new = row.get("first_relevant_rank") or 10**9
        outcome = "unchanged" if new == old else ("improved" if new < old else "degraded")
        counts[outcome] += 1
        details.append({
            "query_id": row["query_id"], "strategy": row["strategy"], "outcome": outcome,
            "reference_first_relevant_rank": reference.get("first_relevant_rank"),
            "experiment_first_relevant_rank": row.get("first_relevant_rank"),
        })
    return {**counts, "queries": details}


def publish_rewrite_experiments(
    *, corpus: str, dataset: EvaluationDataset, rewrites: dict[str, dict[str, str]],
    variants: dict[str, dict[tuple[str, str], list[dict[str, Any]]]], strategies: list[str],
    index_fingerprints: dict[str, str], embedding_config: EmbeddingConfig,
    retrieval_config: RetrievalConfig, rewrite_config_fingerprint: str,
    reference_per_query: Path, output_root: Path, runtime_stats: dict[str, Any],
    output_depth: int = 10, rrf_constant: int = 60,
) -> list[Path]:
    reference_rows = {}
    with reference_per_query.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            reference_rows[(value["query_id"], value["strategy"])] = value
    outputs = []
    records = {record.query_id: record for record in dataset.records}
    for variant in ("Q1", "Q2"):
        fusion = None if variant == "Q1" else {"method": "rrf", "schema_version": RRF_SCHEMA_VERSION, "rank_constant": rrf_constant}
        config = ExperimentConfig(
            experiment_id=f"E4-{variant.lower()}", experiment_name={"Q1": "rewrite_only", "Q2": "original_plus_rewrite_rrf"}[variant],
            experiment_family="E4", dataset_fingerprint=dataset.fingerprint,
            retrieval_config_fingerprint=retrieval_config.fingerprint,
            embedding_config_fingerprint=embedding_config.fingerprint,
            index_fingerprints=dict(sorted(index_fingerprints.items())), candidate_depth=output_depth,
            ranking_method="dense_cosine" if variant == "Q1" else "reciprocal_rank_fusion",
            query_transform={"variant": variant, "rewrite_config_fingerprint": rewrite_config_fingerprint, "original_query_preserved": variant == "Q2"},
            fusion_config=fusion,
        )
        rows = []
        for query_id in sorted(records):
            record = records[query_id]
            for strategy in strategies:
                hits = variants[variant][(query_id, strategy)]
                metrics = evaluate_depth_ranking([hit["relative_path"] for hit in hits], set(record.relevant_sources), output_depth)
                rows.append({
                    "query_id": query_id, "query": record.query,
                    "rewritten_query": rewrites[query_id]["rewritten_query"],
                    "rewrite_status": rewrites[query_id]["status"], "category": record.category,
                    "strategy": strategy, "relevant_targets": record.relevant_sources,
                    "candidate_depth": output_depth, "hits": hits, **metrics,
                })
        aggregate = {strategy: aggregate_depth([row for row in rows if row["strategy"] == strategy], output_depth) for strategy in strategies}
        categories = {strategy: {category: aggregate_depth([row for row in rows if row["strategy"] == strategy and row["category"] == category], output_depth) for category in sorted(CATEGORIES)} for strategy in strategies}
        comparison = _comparison(rows, reference_rows)
        output = output_root / config.fingerprint
        manifest = {
            "schema_version": config.schema_version, "complete": True,
            "experiment_id": config.experiment_id, "experiment_family": "E4",
            "experiment_fingerprint": config.fingerprint, "corpus": corpus,
            "dataset_fingerprint": dataset.fingerprint, "metrics_version": TUNING_METRICS_VERSION,
            "query_count": len(records), "strategy_count": len(strategies),
            "rewrite_config_fingerprint": rewrite_config_fingerprint,
            "index_fingerprints": dict(sorted(index_fingerprints.items())),
        }
        serialized_rows = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
        write_artifact_set(output, {
            "config.json": serialize_json(config.identity()), "per_query.jsonl": serialized_rows,
            "aggregate.json": serialize_json(aggregate), "category_metrics.json": serialize_json(categories),
            "comparison.json": serialize_json(comparison), "stats.json": serialize_json(runtime_stats),
            "manifest.json": serialize_json(manifest),
        })
        outputs.append(output)
    return outputs
