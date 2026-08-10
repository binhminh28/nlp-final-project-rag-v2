"""Strict, strategy-neutral retrieval evaluation dataset loading."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.retrieval.models import normalize_query


EVALUATION_DATASET_SCHEMA_VERSION = "retrieval_evaluation_dataset_v1"
CATEGORIES = frozenset({
    "conceptual", "how_to", "api_lookup", "configuration", "code_related",
    "terminology", "paraphrase", "cross_document",
})


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query_id: str
    query: str
    category: str
    relevant_sources: list[str]
    notes: str | None = None
    difficulty: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string")
        if any(character.isspace() for character in self.query_id):
            raise ValueError("query_id must not contain whitespace")
        object.__setattr__(self, "query", normalize_query(self.query))
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")
        if not isinstance(self.relevant_sources, list) or not self.relevant_sources:
            raise ValueError("at least one relevant source is required")
        if any(not isinstance(item, str) or not item or item.startswith("/") or "\\" in item for item in self.relevant_sources):
            raise ValueError("relevant sources must be non-empty portable relative paths")
        if len(self.relevant_sources) != len(set(self.relevant_sources)):
            raise ValueError("duplicate relevance targets")
        object.__setattr__(self, "relevant_sources", sorted(self.relevant_sources))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationQuery":
        if not isinstance(value, dict):
            raise ValueError("evaluation record must be an object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown evaluation fields: {unknown}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    records: list[EvaluationQuery]
    fingerprint: str
    schema_version: str = EVALUATION_DATASET_SCHEMA_VERSION


def load_evaluation_dataset(path: Path, available_sources: set[str]) -> EvaluationDataset:
    records: list[EvaluationQuery] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
                    record = EvaluationQuery.from_dict(raw)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid evaluation JSONL at {path}:{line_number}: {error}") from error
                records.append(record)
    except OSError:
        raise
    if not records:
        raise ValueError("evaluation dataset is empty")
    ids = [record.query_id for record in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ValueError(f"duplicate query_id values: {duplicates}")
    missing = sorted({source for record in records for source in record.relevant_sources if source not in available_sources})
    if missing:
        raise ValueError(f"relevance targets are absent from corpus: {missing}")
    ordered = sorted(records, key=lambda item: item.query_id)
    identity = {
        "schema_version": EVALUATION_DATASET_SCHEMA_VERSION,
        "ground_truth_level": "relative_path",
        "records": [record.to_dict() for record in ordered],
    }
    return EvaluationDataset(ordered, canonical_fingerprint(identity))
