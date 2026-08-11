"""Strategy-neutral offline answer evaluation and manifest-last publication."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.retrieval.models import KNOWN_STRATEGIES

from .answer_artifacts import CommittedGenerationRun, load_committed_generation_run
from .answer_metrics import evaluate_references, normalize_answer
from .answer_models import (
    AGGREGATE_EVALUATION_SCHEMA_VERSION, ANSWER_EVALUATION_RUN_SCHEMA_VERSION,
    PER_QUERY_EVALUATION_SCHEMA_VERSION, AggregateEvaluationResult,
    EvaluationConfig, EvaluationRunInput, PerQueryEvaluationResult,
    per_query_fingerprint,
)
from .qa_dataset import QADataset, QARecord


@dataclass(frozen=True, slots=True)
class AnswerBenchmarkResult:
    output_directory: Path
    benchmark_fingerprint: str
    per_query: tuple[PerQueryEvaluationResult, ...]
    aggregates: dict[str, AggregateEvaluationResult]
    paired: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} must contain an object")
        values.append(value)
    return values


def validate_answer_benchmark_artifacts(output_directory: Path) -> dict[str, Any]:
    """Validate a committed evaluation artifact set and all semantic hashes."""

    manifest_path = output_directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("evaluation artifacts are not committed")
    try:
        manifest = _read_json_object(manifest_path)
        if manifest.get("schema_version") != ANSWER_EVALUATION_RUN_SCHEMA_VERSION:
            raise ValueError("incompatible answer evaluation manifest schema")
        if manifest.get("complete") is not True:
            raise ValueError("answer evaluation manifest is not complete")
        evaluations = tuple(
            PerQueryEvaluationResult.from_dict(item)
            for item in _read_jsonl_objects(output_directory / "evaluations.jsonl")
        )
        paired = _read_jsonl_objects(output_directory / "paired.jsonl")
        summary = _read_json_object(output_directory / "summary.json")
        stats = _read_json_object(output_directory / "stats.json")
        config = EvaluationConfig(**manifest.get("evaluation_config", {}))
        if config.fingerprint != manifest.get("evaluation_config_fingerprint"):
            raise ValueError("evaluation config fingerprint does not match manifest")
        if summary.get("schema_version") != "answer_evaluation_summary_v1":
            raise ValueError("incompatible answer evaluation summary schema")
        aggregates = {
            strategy: AggregateEvaluationResult.from_dict(value)
            for strategy, value in summary.get("strategies", {}).items()
        }
        if [item.evaluation_fingerprint for item in evaluations] != manifest.get("per_query_evaluation_fingerprints"):
            raise ValueError("evaluation result fingerprints do not match manifest")
        if {name: item.aggregate_fingerprint for name, item in sorted(aggregates.items())} != manifest.get("aggregate_fingerprints"):
            raise ValueError("aggregate fingerprints do not match manifest")
        if [item.get("paired_fingerprint") for item in paired] != manifest.get("paired_fingerprints"):
            raise ValueError("paired fingerprints do not match manifest")
        for item in paired:
            identity = {key: item[key] for key in ("query_id", "category", "gold_answers", "strategies")}
            if item.get("paired_fingerprint") != canonical_fingerprint(identity):
                raise ValueError("paired result fingerprint does not match contents")
        semantic_identity = {
            key: manifest[key]
            for key in (
                "schema_version", "dataset_fingerprint", "evaluation_config",
                "evaluation_config_fingerprint", "generation_run_identities",
                "evaluation_inputs", "strategies",
                "per_query_evaluation_fingerprints", "aggregate_fingerprints", "paired_fingerprints",
            )
        }
        if manifest.get("benchmark_fingerprint") != canonical_fingerprint(semantic_identity):
            raise ValueError("answer benchmark fingerprint does not match manifest identity")
        if stats.get("per_query_result_count") != len(evaluations):
            raise ValueError("evaluation stats count does not match results")
        if manifest.get("strategy_count") != len(aggregates):
            raise ValueError("evaluation strategy count does not match summary")
        if len(evaluations) != manifest.get("query_count") * manifest.get("strategy_count"):
            raise ValueError("evaluation query/strategy coverage is incomplete")
        if len(paired) != manifest.get("query_count"):
            raise ValueError("paired query coverage is incomplete")
        if any(item.dataset_fingerprint != manifest.get("dataset_fingerprint") for item in evaluations):
            raise ValueError("evaluation dataset lineage does not match manifest")
        if any(item.evaluation_config_fingerprint != config.fingerprint for item in evaluations):
            raise ValueError("per-query evaluation config lineage does not match manifest")
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("committed answer evaluation artifact failed validation") from error
    return manifest


def _metric_names(config: EvaluationConfig) -> tuple[str, ...]:
    names: list[str] = []
    for name in config.enabled_metrics:
        if name == "token_f1":
            names.extend(("token_precision", "token_recall"))
        names.append(name)
    return tuple(names)


def _evaluate_record(
    record: QARecord, run: CommittedGenerationRun, config: EvaluationConfig,
    dataset_fingerprint: str,
) -> PerQueryEvaluationResult:
    answers = {item.query_id: item for item in run.answers}
    failures = {item.query_id: item for item in run.failures}
    answer = answers.get(record.id)
    failure = failures.get(record.id)
    references = (record.answer,)
    normalized_references = tuple(normalize_answer(value) for value in references)
    metric_names = _metric_names(config)
    if answer is not None:
        calculated, indexes = evaluate_references(answer.answer_text, references, config.enabled_metrics)
        metrics: dict[str, float | None] = {name: calculated[name] for name in metric_names}
        state = {
            "generated_answer": answer.answer_text,
            "normalized_generated_answer": normalize_answer(answer.answer_text),
            "generation_status": "success", "evaluation_status": "evaluated",
            "generation_error_type": None,
            "generation_result_fingerprint": answer.result_fingerprint,
            "prompt_fingerprint": answer.prompt_fingerprint,
            "context_fingerprint": answer.context_fingerprint,
        }
    elif failure is not None:
        metrics = {name: None for name in metric_names}
        indexes = {name: None for name in metric_names}
        state = {
            "generated_answer": None, "normalized_generated_answer": None,
            "generation_status": "failed", "evaluation_status": "generation_failed",
            "generation_error_type": failure.error_type,
            "generation_result_fingerprint": None, "prompt_fingerprint": None,
            "context_fingerprint": failure.context_fingerprint,
        }
    else:
        metrics = {name: None for name in metric_names}
        indexes = {name: None for name in metric_names}
        state = {
            "generated_answer": None, "normalized_generated_answer": None,
            "generation_status": "missing", "evaluation_status": "missing_generation_result",
            "generation_error_type": None, "generation_result_fingerprint": None,
            "prompt_fingerprint": None, "context_fingerprint": None,
        }
    semantic = {
        "schema_version": PER_QUERY_EVALUATION_SCHEMA_VERSION,
        "query_id": record.id, "strategy": run.strategy, "category": record.question_type,
        "question": record.question, "gold_answers": list(references),
        "generated_answer": state["generated_answer"],
        "normalized_gold_answers": list(normalized_references),
        "normalized_generated_answer": state["normalized_generated_answer"],
        "generation_status": state["generation_status"],
        "evaluation_status": state["evaluation_status"], "metrics": metrics,
        "best_reference_indexes": indexes,
        "generation_error_type": state["generation_error_type"],
        "generation_result_fingerprint": state["generation_result_fingerprint"],
        "prompt_fingerprint": state["prompt_fingerprint"],
        "context_fingerprint": state["context_fingerprint"],
        "generation_config_fingerprint": run.generation_config_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "evaluation_config_fingerprint": config.fingerprint,
    }
    return PerQueryEvaluationResult(
        query_id=record.id, strategy=run.strategy, category=record.question_type,
        question=record.question, gold_answers=references,
        normalized_gold_answers=normalized_references,
        generated_answer=state["generated_answer"],
        normalized_generated_answer=state["normalized_generated_answer"],
        generation_status=state["generation_status"],
        evaluation_status=state["evaluation_status"], metrics=metrics,
        best_reference_indexes=indexes,
        generation_error_type=state["generation_error_type"],
        generation_result_fingerprint=state["generation_result_fingerprint"],
        prompt_fingerprint=state["prompt_fingerprint"],
        context_fingerprint=state["context_fingerprint"],
        generation_config_fingerprint=run.generation_config_fingerprint,
        dataset_fingerprint=dataset_fingerprint,
        evaluation_config_fingerprint=config.fingerprint,
        evaluation_fingerprint=per_query_fingerprint(semantic),
    )


def evaluate_generation_run(
    dataset: QADataset, run: CommittedGenerationRun, config: EvaluationConfig,
) -> tuple[tuple[PerQueryEvaluationResult, ...], AggregateEvaluationResult]:
    results = tuple(
        _evaluate_record(record, run, config, dataset.fingerprint)
        for record in dataset.records
    )
    return results, _aggregate(results, run, config, dataset.fingerprint)


def _summary(records: tuple[PerQueryEvaluationResult, ...], metric_names: tuple[str, ...]) -> dict[str, Any]:
    expected = len(records)
    evaluated = sum(item.evaluation_status == "evaluated" for item in records)
    successes = sum(item.generation_status == "success" for item in records)
    failures = sum(item.generation_status == "failed" for item in records)
    missing = sum(item.generation_status == "missing" for item in records)
    metric_counts = {
        name: sum(item.metrics.get(name) is not None for item in records)
        for name in metric_names
    }
    metric_means = {
        name: (
            sum(float(item.metrics[name]) for item in records if item.metrics.get(name) is not None) / metric_counts[name]
            if metric_counts[name] else None
        )
        for name in metric_names
    }
    return {
        "total_expected_queries": expected,
        "successfully_generated_queries": successes,
        "generation_failures": failures,
        "evaluated_queries": evaluated,
        "missing_queries": missing,
        "generation_success_rate": successes / expected if expected else 0.0,
        "evaluation_coverage": evaluated / expected if expected else 0.0,
        "metric_means": metric_means,
        "metric_counts": metric_counts,
        "success_aware_metric_means": {
            name: sum(float(item.metrics.get(name) or 0.0) for item in records) / expected if expected else 0.0
            for name in metric_names
        },
        "success_aware_metric_counts": {name: expected for name in metric_names},
    }


def _aggregate(
    records: tuple[PerQueryEvaluationResult, ...], run: CommittedGenerationRun,
    config: EvaluationConfig, dataset_fingerprint: str,
) -> AggregateEvaluationResult:
    metric_names = _metric_names(config)
    overall = _summary(records, metric_names)
    categories = {
        category: _summary(tuple(item for item in records if item.category == category), metric_names)
        for category in sorted({item.category for item in records})
    }
    semantic = {
        "schema_version": AGGREGATE_EVALUATION_SCHEMA_VERSION, "strategy": run.strategy,
        **overall, "per_category": categories,
        "evaluation_config_fingerprint": config.fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "generation_run_identity": run.committed_identity,
        "per_query_evaluation_fingerprints": [item.evaluation_fingerprint for item in records],
    }
    return AggregateEvaluationResult(
        strategy=run.strategy, **overall, per_category=categories,
        evaluation_config_fingerprint=config.fingerprint,
        dataset_fingerprint=dataset_fingerprint,
        generation_run_identity=run.committed_identity,
        per_query_evaluation_fingerprints=tuple(item.evaluation_fingerprint for item in records),
        aggregate_fingerprint=canonical_fingerprint(semantic),
    )


def build_paired_results(
    dataset: QADataset, results: dict[str, tuple[PerQueryEvaluationResult, ...]],
) -> tuple[dict[str, Any], ...]:
    indexed = {
        strategy: {item.query_id: item for item in values}
        for strategy, values in results.items()
    }
    paired = []
    for record in dataset.records:
        strategies = {}
        for strategy in sorted(results):
            item = indexed[strategy][record.id]
            strategies[strategy] = {
                "generation_status": item.generation_status,
                "evaluation_status": item.evaluation_status,
                "metrics": item.metrics,
                "generation_result_fingerprint": item.generation_result_fingerprint,
                "evaluation_fingerprint": item.evaluation_fingerprint,
                "evidence_diagnostics": None,
            }
        identity = {
            "query_id": record.id, "category": record.question_type,
            "gold_answers": [record.answer], "strategies": strategies,
        }
        paired.append({**identity, "paired_fingerprint": canonical_fingerprint(identity)})
    return tuple(paired)


def _jsonl(values: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for value in values
    )


def run_answer_benchmark(
    dataset: QADataset, generation_directories: dict[str, Path],
    output_directory: Path, *, config: EvaluationConfig | None = None,
) -> AnswerBenchmarkResult:
    config = config or EvaluationConfig()
    if not generation_directories or set(generation_directories) - KNOWN_STRATEGIES:
        raise ValueError("generation directories must use known, unique strategies")
    runs = {
        strategy: load_committed_generation_run(path, dataset, strategy=strategy)
        for strategy, path in sorted(generation_directories.items())
    }
    generation_configs = {run.generation_config_fingerprint for run in runs.values()}
    if len(generation_configs) != 1:
        raise ValueError("strategies use incompatible generation configurations")
    if len({run.committed_identity for run in runs.values()}) != len(runs):
        raise ValueError("strategies must identify distinct committed generation runs")
    for field in (
        "context_config_fingerprint", "retrieval_config_fingerprint",
        "protocol_config_fingerprint", "embedding_config_fingerprint",
    ):
        values = {
            getattr(answer, field)
            for run in runs.values() for answer in run.answers
        }
        if len(values) > 1:
            raise ValueError(f"strategies use incompatible {field}")
    evaluation_inputs = {
        strategy: EvaluationRunInput(
            dataset_fingerprint=dataset.fingerprint,
            generation_run_identity=run.committed_identity,
            generation_run_fingerprint=run.run_fingerprint,
            generation_config_fingerprint=run.generation_config_fingerprint,
            evaluation_config_fingerprint=config.fingerprint,
            strategy=strategy,
            expected_query_ids=tuple(record.id for record in dataset.records),
        )
        for strategy, run in runs.items()
    }
    by_strategy: dict[str, tuple[PerQueryEvaluationResult, ...]] = {}
    aggregates: dict[str, AggregateEvaluationResult] = {}
    for strategy, run in runs.items():
        by_strategy[strategy], aggregates[strategy] = evaluate_generation_run(dataset, run, config)
    per_query = tuple(
        sorted(
            (item for values in by_strategy.values() for item in values),
            key=lambda item: (item.query_id, item.strategy),
        )
    )
    paired = build_paired_results(dataset, by_strategy)
    identity = {
        "schema_version": ANSWER_EVALUATION_RUN_SCHEMA_VERSION,
        "dataset_fingerprint": dataset.fingerprint,
        "evaluation_config": config.identity(),
        "evaluation_config_fingerprint": config.fingerprint,
        "generation_run_identities": {
            strategy: run.committed_identity for strategy, run in sorted(runs.items())
        },
        "evaluation_inputs": {
            strategy: item.identity() for strategy, item in sorted(evaluation_inputs.items())
        },
        "strategies": sorted(runs),
        "per_query_evaluation_fingerprints": [item.evaluation_fingerprint for item in per_query],
        "aggregate_fingerprints": {
            strategy: item.aggregate_fingerprint for strategy, item in sorted(aggregates.items())
        },
        "paired_fingerprints": [item["paired_fingerprint"] for item in paired],
    }
    benchmark_fingerprint = canonical_fingerprint(identity)
    manifest = {
        **identity, "benchmark_fingerprint": benchmark_fingerprint,
        "query_count": len(dataset.records), "strategy_count": len(runs), "complete": True,
        "artifacts": ["evaluations.jsonl", "summary.json", "paired.jsonl", "stats.json"],
    }
    stats = {
        "query_count": len(dataset.records), "strategy_count": len(runs),
        "per_query_result_count": len(per_query),
        "evaluated_count": sum(item.evaluation_status == "evaluated" for item in per_query),
        "generation_failure_count": sum(item.evaluation_status == "generation_failed" for item in per_query),
        "missing_result_count": sum(item.evaluation_status == "missing_generation_result" for item in per_query),
    }
    existing = output_directory / "manifest.json"
    if existing.exists():
        try:
            stored = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing evaluation manifest is invalid") from error
        if stored.get("benchmark_fingerprint") != benchmark_fingerprint:
            raise ValueError("refusing to overwrite a committed evaluation with a different identity")
    write_artifact_set(output_directory, {
        "evaluations.jsonl": _jsonl([item.to_dict() for item in per_query]),
        "summary.json": serialize_json({
            "schema_version": "answer_evaluation_summary_v1",
            "strategies": {name: item.to_dict() for name, item in sorted(aggregates.items())},
        }),
        "paired.jsonl": _jsonl(list(paired)),
        "stats.json": serialize_json(stats),
        "manifest.json": serialize_json(manifest),
    })
    validate_answer_benchmark_artifacts(output_directory)
    return AnswerBenchmarkResult(
        output_directory, benchmark_fingerprint, per_query, aggregates, paired, manifest,
    )
