"""Versioned immutable contracts for offline deterministic answer evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from rag_chunking.embedding.models import canonical_fingerprint


EVALUATION_CONFIG_SCHEMA_VERSION = "answer_evaluation_config_v1"
PER_QUERY_EVALUATION_SCHEMA_VERSION = "answer_evaluation_result_v1"
AGGREGATE_EVALUATION_SCHEMA_VERSION = "answer_evaluation_aggregate_v1"
ANSWER_NORMALIZATION_VERSION = "answer_normalization_v1"
TOKENIZATION_POLICY = "unicode_word_or_symbol_v1"
ANSWER_EVALUATION_RUN_SCHEMA_VERSION = "answer_evaluation_run_v1"
EVALUATION_INPUT_SCHEMA_VERSION = "answer_evaluation_input_v1"

SUPPORTED_METRICS = (
    "normalized_exact_match",
    "token_f1",
    "normalized_containment",
)
GENERATION_STATUSES = frozenset({"success", "failed", "missing"})
EVALUATION_STATUSES = frozenset({
    "evaluated", "generation_failed", "missing_generation_result",
})


def _nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Every setting that can affect answer scores or their denominators."""

    normalization_version: str = ANSWER_NORMALIZATION_VERSION
    tokenization_policy: str = TOKENIZATION_POLICY
    enabled_metrics: tuple[str, ...] = SUPPORTED_METRICS
    multiple_reference_policy: str = "best_per_metric_then_lowest_index_v1"
    missing_answer_policy: str = "exclude_quality_zero_end_to_end_v1"
    failed_generation_policy: str = "exclude_quality_zero_end_to_end_v1"
    aggregate_policy: str = "quality_and_success_aware_v1"
    category_aggregation_policy: str = "question_type_exact_v1"
    evidence_diagnostic_policy: str = "deferred_without_context_provenance_v1"
    schema_version: str = EVALUATION_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        expected = {
            "normalization_version": ANSWER_NORMALIZATION_VERSION,
            "tokenization_policy": TOKENIZATION_POLICY,
            "multiple_reference_policy": "best_per_metric_then_lowest_index_v1",
            "missing_answer_policy": "exclude_quality_zero_end_to_end_v1",
            "failed_generation_policy": "exclude_quality_zero_end_to_end_v1",
            "aggregate_policy": "quality_and_success_aware_v1",
            "category_aggregation_policy": "question_type_exact_v1",
            "evidence_diagnostic_policy": "deferred_without_context_provenance_v1",
            "schema_version": EVALUATION_CONFIG_SCHEMA_VERSION,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise ValueError(f"unsupported {name} {getattr(self, name)!r}")
        if not isinstance(self.enabled_metrics, tuple):
            object.__setattr__(self, "enabled_metrics", tuple(self.enabled_metrics))
        if not self.enabled_metrics or len(set(self.enabled_metrics)) != len(self.enabled_metrics):
            raise ValueError("enabled_metrics must be non-empty and unique")
        if set(self.enabled_metrics) - set(SUPPORTED_METRICS):
            raise ValueError("enabled_metrics contains an unsupported metric")
        object.__setattr__(
            self, "enabled_metrics",
            tuple(name for name in SUPPORTED_METRICS if name in self.enabled_metrics),
        )

    def identity(self) -> dict[str, Any]:
        value = asdict(self)
        value["enabled_metrics"] = list(self.enabled_metrics)
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.identity())


@dataclass(frozen=True, slots=True)
class GenerationFailure:
    query_id: str
    error_type: str
    error: str
    context_fingerprint: str
    generation_config_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "query_id", "error_type", "error", "context_fingerprint",
            "generation_config_fingerprint",
        ):
            _nonempty(getattr(self, name), name)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationFailure":
        if not isinstance(value, dict):
            raise ValueError("generation failure must be an object")
        required = {item.name for item in fields(cls)}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"missing generation failure fields: {missing}")
        # Retry diagnostics are allowed but are not evaluation semantics.
        allowed = required | {"retryable", "attempts", "status_code"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown generation failure fields: {unknown}")
        return cls(**{name: value[name] for name in required})


@dataclass(frozen=True, slots=True)
class EvaluationRunInput:
    dataset_fingerprint: str
    generation_run_identity: str
    generation_run_fingerprint: str
    generation_config_fingerprint: str
    evaluation_config_fingerprint: str
    strategy: str
    expected_query_ids: tuple[str, ...]
    schema_version: str = EVALUATION_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "dataset_fingerprint", "generation_run_identity",
            "generation_run_fingerprint", "generation_config_fingerprint",
            "evaluation_config_fingerprint", "strategy",
        ):
            _nonempty(getattr(self, name), name)
        if self.schema_version != EVALUATION_INPUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation input schema {self.schema_version!r}")
        if not isinstance(self.expected_query_ids, tuple):
            object.__setattr__(self, "expected_query_ids", tuple(self.expected_query_ids))
        if not self.expected_query_ids or len(set(self.expected_query_ids)) != len(self.expected_query_ids):
            raise ValueError("expected query IDs must be non-empty and unique")
        if self.expected_query_ids != tuple(sorted(self.expected_query_ids)):
            raise ValueError("expected query IDs must be in canonical order")

    def identity(self) -> dict[str, Any]:
        value = asdict(self)
        value["expected_query_ids"] = list(self.expected_query_ids)
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.identity())


@dataclass(frozen=True, slots=True)
class PerQueryEvaluationResult:
    query_id: str
    strategy: str
    category: str
    question: str
    gold_answers: tuple[str, ...]
    generated_answer: str | None
    normalized_gold_answers: tuple[str, ...]
    normalized_generated_answer: str | None
    generation_status: str
    evaluation_status: str
    metrics: dict[str, float | None]
    best_reference_indexes: dict[str, int | None]
    generation_error_type: str | None
    generation_result_fingerprint: str | None
    prompt_fingerprint: str | None
    context_fingerprint: str | None
    generation_config_fingerprint: str
    dataset_fingerprint: str
    evaluation_config_fingerprint: str
    evaluation_fingerprint: str
    schema_version: str = PER_QUERY_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "query_id", "strategy", "category", "question",
            "generation_config_fingerprint", "dataset_fingerprint",
            "evaluation_config_fingerprint", "evaluation_fingerprint",
        ):
            _nonempty(getattr(self, name), name)
        if self.schema_version != PER_QUERY_EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported per-query evaluation schema {self.schema_version!r}")
        if self.generation_status not in GENERATION_STATUSES:
            raise ValueError("unknown generation status")
        if self.evaluation_status not in EVALUATION_STATUSES:
            raise ValueError("unknown evaluation status")
        if not self.gold_answers or any(not isinstance(value, str) for value in self.gold_answers):
            raise ValueError("gold_answers must contain at least one string")
        if len(self.normalized_gold_answers) != len(self.gold_answers):
            raise ValueError("normalized gold diagnostics do not match references")
        if self.generation_status == "success" and self.evaluation_status != "evaluated":
            raise ValueError("successful generation must be evaluated")
        if self.generation_status != "success" and self.evaluation_status == "evaluated":
            raise ValueError("non-successful generation cannot be evaluated")
        if self.evaluation_status == "evaluated" and any(value is None for value in self.metrics.values()):
            raise ValueError("evaluated metrics must be numeric")
        if self.evaluation_status != "evaluated" and any(value is not None for value in self.metrics.values()):
            raise ValueError("unevaluated metrics must be null")
        expected = per_query_fingerprint(self.semantic_identity())
        if self.evaluation_fingerprint != expected:
            raise ValueError("evaluation fingerprint does not match per-query contents")

    def semantic_identity(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "strategy": self.strategy,
            "category": self.category,
            "question": self.question,
            "gold_answers": list(self.gold_answers),
            "generated_answer": self.generated_answer,
            "normalized_gold_answers": list(self.normalized_gold_answers),
            "normalized_generated_answer": self.normalized_generated_answer,
            "generation_status": self.generation_status,
            "evaluation_status": self.evaluation_status,
            "metrics": self.metrics,
            "best_reference_indexes": self.best_reference_indexes,
            "generation_error_type": self.generation_error_type,
            "generation_result_fingerprint": self.generation_result_fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "context_fingerprint": self.context_fingerprint,
            "generation_config_fingerprint": self.generation_config_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "evaluation_config_fingerprint": self.evaluation_config_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["gold_answers"] = list(self.gold_answers)
        value["normalized_gold_answers"] = list(self.normalized_gold_answers)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PerQueryEvaluationResult":
        if not isinstance(value, dict):
            raise ValueError("per-query evaluation must be an object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown per-query evaluation fields: {unknown}")
        data = dict(value)
        data["gold_answers"] = tuple(data.get("gold_answers", ()))
        data["normalized_gold_answers"] = tuple(data.get("normalized_gold_answers", ()))
        return cls(**data)


def per_query_fingerprint(identity: dict[str, Any]) -> str:
    value = dict(identity)
    value.pop("evaluation_fingerprint", None)
    return canonical_fingerprint(value)


@dataclass(frozen=True, slots=True)
class AggregateEvaluationResult:
    strategy: str
    total_expected_queries: int
    successfully_generated_queries: int
    generation_failures: int
    evaluated_queries: int
    missing_queries: int
    generation_success_rate: float
    evaluation_coverage: float
    metric_means: dict[str, float | None]
    metric_counts: dict[str, int]
    success_aware_metric_means: dict[str, float]
    success_aware_metric_counts: dict[str, int]
    per_category: dict[str, dict[str, Any]]
    evaluation_config_fingerprint: str
    dataset_fingerprint: str
    generation_run_identity: str
    per_query_evaluation_fingerprints: tuple[str, ...]
    aggregate_fingerprint: str
    schema_version: str = AGGREGATE_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "strategy", "evaluation_config_fingerprint", "dataset_fingerprint",
            "generation_run_identity", "aggregate_fingerprint",
        ):
            _nonempty(getattr(self, name), name)
        if self.schema_version != AGGREGATE_EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported aggregate schema {self.schema_version!r}")
        for name in (
            "total_expected_queries", "successfully_generated_queries",
            "generation_failures", "evaluated_queries", "missing_queries",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if len(self.per_query_evaluation_fingerprints) != self.total_expected_queries:
            raise ValueError("aggregate must bind every expected per-query result")
        expected = canonical_fingerprint(self.semantic_identity())
        if self.aggregate_fingerprint != expected:
            raise ValueError("aggregate fingerprint does not match contents")

    def semantic_identity(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("aggregate_fingerprint")
        value["per_query_evaluation_fingerprints"] = list(
            self.per_query_evaluation_fingerprints
        )
        return value

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["per_query_evaluation_fingerprints"] = list(self.per_query_evaluation_fingerprints)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AggregateEvaluationResult":
        if not isinstance(value, dict):
            raise ValueError("aggregate evaluation must be an object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown aggregate fields: {unknown}")
        data = dict(value)
        data["per_query_evaluation_fingerprints"] = tuple(
            data.get("per_query_evaluation_fingerprints", ())
        )
        return cls(**data)
