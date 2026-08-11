"""Gold-free preparation of retrieval/protocol/context generation inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.context import ContextBuildInput, ContextBuilder, ContextConfig, ContextResult
from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.retrieval import (
    RetrievalProtocolConfig, RetrievalRequest, RetrievalService,
    apply_retrieval_protocol,
)
from rag_chunking.retrieval.models import KNOWN_STRATEGIES


BENCHMARK_INPUT_SCHEMA_VERSION = "answer_benchmark_inputs_v1"
CANONICAL_STRATEGIES = ("fixed_size", "structure_aware", "prompt_based")


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    query_id: str
    question: str

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id or any(ch.isspace() for ch in self.query_id):
            raise ValueError("benchmark query_id must be non-empty and contain no whitespace")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("benchmark question must be non-empty")


@dataclass(frozen=True, slots=True)
class PreparedBenchmarkInputs:
    output_directory: Path
    preparation_fingerprint: str
    contexts_by_strategy: dict[str, tuple[ContextResult, ...]]
    manifest: dict[str, Any]
    reused: bool


def validate_prepared_benchmark_inputs(
    output_directory: Path, *, dataset_fingerprint: str,
    expected_queries: tuple[BenchmarkQuery, ...],
) -> PreparedBenchmarkInputs:
    """Validate committed preparation lineage for the evaluation compatibility gate."""

    manifest_path = output_directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("prepared benchmark inputs are not committed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != BENCHMARK_INPUT_SCHEMA_VERSION:
        raise ValueError("prepared benchmark input schema is incompatible")
    if manifest.get("complete") is not True or manifest.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("prepared benchmark input dataset/commit lineage mismatch")
    expected = [
        {"query_id": item.query_id, "question": item.question} for item in expected_queries
    ]
    if manifest.get("queries") != expected or manifest.get("strategies") != list(CANONICAL_STRATEGIES):
        raise ValueError("prepared benchmark input query/strategy coverage mismatch")
    identity_keys = (
        "schema_version", "dataset_fingerprint", "corpus", "corpus_fingerprint",
        "queries", "strategies", "retrieval_config_fingerprint",
        "protocol_configuration", "protocol_config_fingerprint",
        "context_configuration", "context_config_fingerprint",
        "embedding_config_fingerprint", "index_fingerprints",
    )
    identity = {key: manifest[key] for key in identity_keys}
    if manifest.get("preparation_fingerprint") != canonical_fingerprint(identity):
        raise ValueError("prepared benchmark fingerprint does not match manifest contents")
    result = _load_reused(output_directory, manifest, CANONICAL_STRATEGIES)
    for strategy, contexts in result.contexts_by_strategy.items():
        if any(
            context.dataset_fingerprint != dataset_fingerprint
            or context.context_config_fingerprint != manifest["context_config_fingerprint"]
            or context.protocol_config_fingerprint != manifest["protocol_config_fingerprint"]
            or context.retrieval_config_fingerprint != manifest["retrieval_config_fingerprint"]
            or context.embedding_config_fingerprint != manifest["embedding_config_fingerprint"]
            or context.index_fingerprint != manifest["index_fingerprints"][strategy]
            for context in contexts
        ):
            raise ValueError("prepared ContextResult lineage does not match manifest")
    return result


def validate_generation_requests_against_preparation(
    prepared: PreparedBenchmarkInputs, generation_directories: dict[str, Path],
) -> None:
    if tuple(sorted(generation_directories)) != tuple(sorted(CANONICAL_STRATEGIES)):
        raise ValueError("prepared comparison requires all three generation strategies")
    questions = {item["query_id"]: item["question"] for item in prepared.manifest["queries"]}
    for strategy in CANONICAL_STRATEGIES:
        path = generation_directories[strategy] / "manifest.json"
        if not path.is_file():
            raise ValueError(f"generation run is not committed for {strategy}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = [
            {
                "query_id": context.query_id,
                "question": questions[context.query_id],
                "context_fingerprint": context.context_fingerprint,
            }
            for context in prepared.contexts_by_strategy[strategy]
        ]
        if manifest.get("complete") is not True or manifest.get("requests") != expected:
            raise ValueError(f"generation requests do not match prepared contexts for {strategy}")


def project_benchmark_queries(records: list[object]) -> tuple[BenchmarkQuery, ...]:
    """Project only retrieval-legal fields from QA records."""

    queries = tuple(sorted(
        (BenchmarkQuery(getattr(record, "id"), getattr(record, "question")) for record in records),
        key=lambda item: item.query_id,
    ))
    if not queries or len({item.query_id for item in queries}) != len(queries):
        raise ValueError("benchmark queries must be non-empty and unique")
    return queries


def _identity(
    service: RetrievalService, queries: tuple[BenchmarkQuery, ...],
    dataset_fingerprint: str, corpus_fingerprint: str,
    protocol: RetrievalProtocolConfig, context_config: ContextConfig,
    strategies: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": BENCHMARK_INPUT_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "corpus": service.corpus,
        "corpus_fingerprint": corpus_fingerprint,
        "queries": [
            {"query_id": item.query_id, "question": item.question} for item in queries
        ],
        "strategies": list(strategies),
        "retrieval_config_fingerprint": service.config.fingerprint,
        "protocol_configuration": protocol.identity(),
        "protocol_config_fingerprint": protocol.fingerprint,
        "context_configuration": context_config.identity(),
        "context_config_fingerprint": context_config.fingerprint,
        "embedding_config_fingerprint": service.embedding_config.fingerprint,
        "index_fingerprints": {
            strategy: service.manifests[strategy]["index_fingerprint"]
            for strategy in strategies
        },
    }


def _jsonl(values: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for value in values
    )


def _load_reused(
    output_directory: Path, manifest: dict[str, Any], strategies: tuple[str, ...],
) -> PreparedBenchmarkInputs:
    contexts: dict[str, tuple[ContextResult, ...]] = {}
    expected_ids = [item["query_id"] for item in manifest["queries"]]
    for strategy in strategies:
        path = output_directory / f"{strategy}.generation_inputs.jsonl"
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != {"query_id", "question", "context"}:
                raise ValueError("committed benchmark input record has an invalid schema")
            context = ContextResult.from_dict(value["context"])
            if context.strategy != strategy or context.query_id != value["query_id"]:
                raise ValueError("committed benchmark input context lineage mismatch")
            values.append(context)
        if [item.query_id for item in values] != expected_ids:
            raise ValueError("committed benchmark input query coverage mismatch")
        contexts[strategy] = tuple(values)
    return PreparedBenchmarkInputs(
        output_directory, manifest["preparation_fingerprint"], contexts, manifest, True,
    )


def prepare_answer_benchmark_inputs(
    service: RetrievalService, queries: tuple[BenchmarkQuery, ...],
    *, dataset_fingerprint: str, corpus_fingerprint: str,
    protocol: RetrievalProtocolConfig, context_config: ContextConfig,
    output_directory: Path,
    strategies: tuple[str, ...] = CANONICAL_STRATEGIES,
) -> PreparedBenchmarkInputs:
    """Retrieve and build contexts once; never receives gold-bearing QA records."""

    if strategies != CANONICAL_STRATEGIES:
        raise ValueError("canonical answer preparation requires all three strategies in fixed order")
    if set(strategies) - KNOWN_STRATEGIES or set(strategies) - set(service.indexes):
        raise ValueError("all canonical strategies require loaded indexes")
    if not dataset_fingerprint or not corpus_fingerprint:
        raise ValueError("dataset and corpus fingerprints are required")
    identity = _identity(
        service, queries, dataset_fingerprint, corpus_fingerprint,
        protocol, context_config, strategies,
    )
    preparation_fingerprint = canonical_fingerprint(identity)
    existing_path = output_directory / "manifest.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("preparation_fingerprint") != preparation_fingerprint:
            raise ValueError("refusing to overwrite committed benchmark inputs with a different identity")
        if existing.get("complete") is not True or any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("committed benchmark input manifest failed identity validation")
        return _load_reused(output_directory, existing, strategies)

    builder = ContextBuilder(context_config)
    contexts: dict[str, list[ContextResult]] = {strategy: [] for strategy in strategies}
    cache_hits = cache_misses = retrieval_calls = 0
    for query in queries:
        vector, cache_hit = service.embed_query(query.question)
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
        for strategy in strategies:
            result = service.retrieve(
                RetrievalRequest(query.question, strategy, protocol.candidate_k),
                query_vector=vector, cache_hit=cache_hit,
            )
            retrieval_calls += 1
            selection = apply_retrieval_protocol(result.hits, protocol)
            handoff = ContextBuildInput.from_retrieval(
                query_id=query.query_id, result=result, selection=selection,
                protocol_config_fingerprint=protocol.fingerprint,
                dataset_fingerprint=dataset_fingerprint,
            )
            contexts[strategy].append(builder.build(handoff))
    serialized: dict[str, str] = {}
    by_question = {item.query_id: item.question for item in queries}
    for strategy in strategies:
        serialized[f"{strategy}.generation_inputs.jsonl"] = _jsonl([
            {
                "query_id": context.query_id,
                "question": by_question[context.query_id],
                "context": context.to_dict(),
            }
            for context in contexts[strategy]
        ])
    stats = {
        "query_count": len(queries), "strategy_count": len(strategies),
        "retrieval_calls": retrieval_calls,
        "query_embedding_cache_hits": cache_hits,
        "query_embedding_cache_misses": cache_misses,
        "provider_calls": service.provider.calls,
    }
    manifest = {
        **identity, "preparation_fingerprint": preparation_fingerprint,
        "query_count": len(queries), "strategy_count": len(strategies),
        "artifacts": [f"{strategy}.generation_inputs.jsonl" for strategy in strategies],
        "complete": True,
    }
    serialized["stats.json"] = serialize_json(stats)
    serialized["manifest.json"] = serialize_json(manifest)
    write_artifact_set(output_directory, serialized)
    return PreparedBenchmarkInputs(
        output_directory, preparation_fingerprint,
        {strategy: tuple(values) for strategy, values in contexts.items()},
        manifest, False,
    )
