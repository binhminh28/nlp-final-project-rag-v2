"""Evidence-aware evaluation under explicit, comparable retrieval budgets."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.data.models import NormalizedDocument
from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.retrieval.models import RetrievalRequest
from rag_chunking.retrieval.protocols import RetrievalProtocolConfig, apply_retrieval_protocol
from rag_chunking.retrieval.service import RetrievalService

from .evidence import EvidenceMapping, map_evidence_to_chunks, retrieved_evidence_coverage
from .metrics import EVIDENCE_METRICS_VERSION, METRICS_VERSION, aggregate, aggregate_evidence, evaluate_evidence_coverage, evaluate_ranking
from .qa_dataset import QADataset


EVIDENCE_BENCHMARK_SCHEMA_VERSION = "evidence_retrieval_benchmark_v1"


@dataclass(slots=True)
class EvidenceBenchmarkResult:
    benchmark_fingerprint: str
    output_directory: Path
    per_query: list[dict[str, Any]]
    aggregates: dict[str, Any]
    stats: dict[str, Any]


def _jsonl(values: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for value in values)


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)])


def _budget_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    tokens = [int(item["actual_selected_tokens"]) for item in records]
    chunks = [int(item["selected_chunk_count"]) for item in records]
    utilization = [item["budget_utilization"] for item in records if item["budget_utilization"] is not None]
    return {
        "mean_retrieved_tokens": statistics.fmean(tokens) if tokens else 0.0,
        "median_retrieved_tokens": statistics.median(tokens) if tokens else 0.0,
        "p95_retrieved_tokens": _p95(tokens),
        "mean_selected_chunk_count": statistics.fmean(chunks) if chunks else 0.0,
        "mean_budget_utilization": statistics.fmean(utilization) if utilization else 0.0,
    }


def _mapping_relevant_ids(mappings: list[EvidenceMapping]) -> set[str]:
    return {chunk_id for mapping in mappings for chunk_id in mapping.matched_chunk_ids}


def _record_doc_ids(record) -> set[str]:
    return {item.doc_id for item in record.evidence} if record.evidence else {record.doc_id}


def _map_record_evidence(record, documents, chunks, strategy) -> list[EvidenceMapping]:
    if not record.evidence:
        return map_evidence_to_chunks(
            record, documents[record.doc_id], chunks, strategy,
        )
    mappings: list[EvidenceMapping] = []
    for evidence in record.evidence:
        projection = replace(
            record, doc_id=evidence.doc_id,
            evidence_sentences=list(evidence.evidence_sentences),
            evidence_sections=[], evidence=[],
        )
        projected = map_evidence_to_chunks(
            projection, documents[evidence.doc_id], chunks, strategy,
        )
        mappings.extend(
            replace(mapping, evidence_id=f"{evidence.evidence_id}:sentence:{index}")
            for index, mapping in enumerate(projected)
        )
    return mappings


def run_evidence_retrieval_benchmark(
    service: RetrievalService,
    dataset: QADataset,
    documents: dict[str, NormalizedDocument],
    chunks_by_strategy: dict[str, list[Chunk]],
    output_root: Path,
    *,
    strategies: list[str],
    protocols: list[RetrievalProtocolConfig],
    corpus_fingerprint: str,
    chunk_artifact_fingerprints: dict[str, str],
    run_name: str = "evidence_dev_v1",
) -> EvidenceBenchmarkResult:
    if not strategies or len(strategies) != len(set(strategies)):
        raise ValueError("strategies must be non-empty and unique")
    if not protocols or len({protocol.mode for protocol in protocols}) != len(protocols):
        raise ValueError("protocol modes must be non-empty and unique")
    if set(strategies) - set(service.indexes) or set(strategies) - set(chunks_by_strategy):
        raise ValueError("all strategies require a loaded index and chunk artifact")
    declared_doc_ids = {doc_id for record in dataset.records for doc_id in _record_doc_ids(record)}
    if declared_doc_ids - set(documents):
        raise ValueError("one or more QA documents are not loaded")
    candidate_depth = max(protocol.candidate_k for protocol in protocols)
    identity = {
        "schema_version": EVIDENCE_BENCHMARK_SCHEMA_VERSION,
        "qa_schema_version": dataset.schema_version,
        "dataset_fingerprint": dataset.fingerprint,
        "corpus": service.corpus,
        "corpus_fingerprint": corpus_fingerprint,
        "chunk_config_fingerprints": {
            name: service.manifests[name]["chunk_config_fingerprint"] for name in sorted(strategies)
        },
        "chunk_artifact_fingerprints": {name: chunk_artifact_fingerprints[name] for name in sorted(strategies)},
        "embedding_config_fingerprint": service.embedding_config.fingerprint,
        "embedding_artifact_fingerprints": {
            name: service.manifests[name]["embedding_artifact_fingerprint"] for name in sorted(strategies)
        },
        "index_fingerprints": {name: service.manifests[name]["index_fingerprint"] for name in sorted(strategies)},
        "base_retrieval_config_fingerprint": service.config.fingerprint,
        "protocols": [protocol.identity() for protocol in sorted(protocols, key=lambda value: value.mode)],
        "metrics_versions": [METRICS_VERSION, EVIDENCE_METRICS_VERSION],
        "historical_ground_truth_level": "relative_path",
        "evidence_ground_truth_level": "source_evidence_mapped_to_chunk_id",
        "candidate_depth": candidate_depth,
        "tie_breaking_rule": "score_desc_chunk_id_asc",
    }
    fingerprint = canonical_fingerprint(identity)
    output_directory = output_root / run_name / fingerprint
    started = time.monotonic()
    per_query: list[dict[str, Any]] = []
    mappings_cache: dict[tuple[str, str], list[EvidenceMapping]] = {}
    cache_hits = cache_misses = retrieval_calls = 0
    try:
        for record in dataset.records:
            vector, cache_hit = service.embed_query(record.question)
            cache_hits += int(cache_hit)
            cache_misses += int(not cache_hit)
            for strategy in strategies:
                mappings = _map_record_evidence(
                    record, documents, chunks_by_strategy[strategy], strategy,
                )
                mappings_cache[(record.id, strategy)] = mappings
                if not mappings:
                    raise ValueError(f"QA record {record.id!r} declares no evidence units")
                candidates = service.retrieve(
                    RetrievalRequest(record.question, strategy, candidate_depth),
                    query_vector=vector, cache_hit=cache_hit,
                )
                retrieval_calls += 1
                for protocol in protocols:
                    selection = apply_retrieval_protocol(candidates.hits, protocol)
                    retrieved_ids = [hit.chunk_id for hit in selection.hits]
                    relevant_ids = _mapping_relevant_ids(mappings)
                    # Preserve the historical metrics' source-path semantics.
                    # Evidence/chunk relevance is reported separately below.
                    evidence_doc_ids = _record_doc_ids(record)
                    relevant_sources = {documents[doc_id].relative_path for doc_id in evidence_doc_ids}
                    ranking = evaluate_ranking(
                        [hit.relative_path for hit in selection.hits], relevant_sources,
                    )
                    unit_coverages = [retrieved_evidence_coverage(mapping, set(retrieved_ids)) for mapping in mappings]
                    evidence_metrics = evaluate_evidence_coverage(unit_coverages)
                    covered = [mapping.evidence_id for mapping, value in zip(mappings, unit_coverages, strict=True) if value >= 1.0]
                    uncovered = [mapping.evidence_id for mapping, value in zip(mappings, unit_coverages, strict=True) if value < 1.0]
                    per_query.append({
                        "query_id": record.id, "question": record.question,
                        "question_type": record.question_type, "difficulty": record.difficulty,
                        "doc_id": record.doc_id, "evidence_doc_ids": sorted(evidence_doc_ids),
                        "strategy": strategy, "protocol": protocol.mode,
                        "retrieval_config_fingerprint": service.config.fingerprint,
                        "protocol_config_fingerprint": protocol.fingerprint,
                        "embedding_config_fingerprint": service.embedding_config.fingerprint,
                        "index_fingerprint": candidates.index_fingerprint,
                        "candidate_k": protocol.candidate_k,
                        "candidate_chunk_ids": [hit.chunk_id for hit in candidates.hits[: protocol.candidate_k]],
                        "hits": [hit.to_dict() for hit in selection.hits],
                        "retrieved_chunk_ids": retrieved_ids,
                        "evidence_units": [mapping.evidence_id for mapping in mappings],
                        "evidence_mappings": [mapping.to_dict() for mapping in mappings],
                        "evidence_unit_coverages": unit_coverages,
                        "covered_evidence": covered, "uncovered_evidence": uncovered,
                        "relevant_targets": sorted(relevant_sources),
                        "relevant_chunk_ids": sorted(relevant_ids),
                        **{key: value for key, value in selection.to_dict().items() if key != "hits"},
                        **ranking, **evidence_metrics,
                    })
        per_query.sort(key=lambda item: (item["query_id"], item["strategy"], item["protocol"]))
        aggregates: dict[str, Any] = {}
        for protocol in sorted({item["protocol"] for item in per_query}):
            aggregates[protocol] = {}
            for strategy in strategies:
                records = [item for item in per_query if item["protocol"] == protocol and item["strategy"] == strategy]
                aggregates[protocol][strategy] = {
                    "retrieval": aggregate(records),
                    "evidence": aggregate_evidence(records),
                    "budget": _budget_summary(records),
                }
        manifest = {
            **identity, "benchmark_fingerprint": fingerprint, "run_name": run_name,
            "complete": True, "query_count": len(dataset.records),
            "strategy_count": len(strategies), "protocol_count": len(protocols),
        }
        stats = {
            "query_count": len(dataset.records), "strategy_count": len(strategies),
            "protocol_count": len(protocols), "query_embedding_cache_hits": cache_hits,
            "query_embedding_cache_misses": cache_misses, "retrieval_calls": retrieval_calls,
            "total_runtime_seconds": time.monotonic() - started,
        }
        write_artifact_set(output_directory, {
            "per_query.jsonl": _jsonl(per_query),
            "aggregate.json": serialize_json(aggregates),
            "stats.json": serialize_json(stats),
            "manifest.json": serialize_json(manifest),
        })
        failure_path = output_directory / "failure.json"
        if failure_path.exists():
            failure_path.unlink()
    except BaseException as error:
        output_directory.mkdir(parents=True, exist_ok=True)
        # A failure marker is diagnostic only; absence of complete manifest is
        # the validity rule and an existing committed manifest is never removed.
        failure = serialize_json({"complete": False, "error_type": type(error).__name__, "error": str(error)})
        (output_directory / "failure.json").write_text(failure, encoding="utf-8")
        raise
    return EvidenceBenchmarkResult(fingerprint, output_directory, per_query, aggregates, stats)
