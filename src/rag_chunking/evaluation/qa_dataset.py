"""Adapter and validator for the externally owned evidence QA contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag_chunking.data.models import NormalizedDocument
from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.retrieval.models import normalize_query


QA_DATASET_SCHEMA_VERSION = "evidence_qa_dataset_v1"


@dataclass(frozen=True, slots=True)
class EvidenceSpec:
    """A string evidence item, optionally carrying exact source provenance."""

    text: str
    block_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("evidence text must be non-empty")
        if self.block_index is not None and (type(self.block_index) is not int or self.block_index < 0):
            raise ValueError("evidence block_index must be non-negative")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("evidence char_start and char_end must be supplied together")
        if self.char_start is not None:
            if self.block_index is None:
                raise ValueError("character provenance requires block_index")
            if type(self.char_start) is not int or type(self.char_end) is not int:
                raise ValueError("evidence character offsets must be integers")
            if not 0 <= self.char_start < self.char_end:
                raise ValueError("evidence character span must be non-empty")

    @classmethod
    def from_value(cls, value: object) -> "EvidenceSpec":
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, dict):
            raise ValueError("evidence items must be strings or provenance objects")
        unknown = sorted(set(value) - {"text", "block_index", "char_start", "char_end"})
        if unknown:
            raise ValueError(f"unknown evidence fields: {unknown}")
        return cls(**value)

    def to_external_value(self) -> str | dict[str, object]:
        if self.block_index is None and self.char_start is None:
            return self.text
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class QARecord:
    id: str
    doc_id: str
    question: str
    answer: str
    evidence_sentences: list[EvidenceSpec]
    evidence_sections: list[str]
    question_type: str
    difficulty: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or any(ch.isspace() for ch in self.id):
            raise ValueError("id must be non-empty and contain no whitespace")
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError("doc_id must be non-empty")
        object.__setattr__(self, "question", normalize_query(self.question))
        if not isinstance(self.answer, str):
            raise ValueError("answer must be a string")
        if not isinstance(self.evidence_sentences, list):
            raise ValueError("evidence_sentences must be a list")
        if not isinstance(self.evidence_sections, list) or any(
            not isinstance(value, str) or not value.strip() for value in self.evidence_sections
        ):
            raise ValueError("evidence_sections must be a list of non-empty strings")
        if not isinstance(self.question_type, str) or not self.question_type.strip():
            raise ValueError("question_type must be non-empty")
        if not isinstance(self.difficulty, str) or not self.difficulty.strip():
            raise ValueError("difficulty must be non-empty")
        if self.notes is not None and not isinstance(self.notes, str):
            raise ValueError("notes must be null or a string")

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, require_answer: bool = True) -> "QARecord":
        if not isinstance(value, dict):
            raise ValueError("QA record must be an object")
        required = {"id", "doc_id", "question", "evidence_sentences", "evidence_sections", "question_type", "difficulty"}
        if require_answer:
            required.add("answer")
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"missing QA fields: {missing}")
        allowed = required | {"answer", "notes"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown QA fields: {unknown}")
        data = dict(value)
        data.setdefault("answer", "")
        if require_answer and (not isinstance(data["answer"], str) or not data["answer"].strip()):
            raise ValueError("answer must be non-empty")
        raw_evidence = data.get("evidence_sentences")
        if not isinstance(raw_evidence, list):
            raise ValueError("evidence_sentences must be a list")
        data["evidence_sentences"] = [EvidenceSpec.from_value(item) for item in raw_evidence]
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_sentences"] = [item.to_external_value() for item in self.evidence_sentences]
        return value


@dataclass(frozen=True, slots=True)
class QADataset:
    records: list[QARecord]
    fingerprint: str
    schema_version: str = QA_DATASET_SCHEMA_VERSION


@dataclass(slots=True)
class QASemanticValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("QA JSON must contain an array")
        return value
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid QA JSONL at {path}:{line_number}: {error}") from error
    return records


def load_qa_dataset(
    path: Path, available_doc_ids: set[str], *, require_answer: bool = True,
) -> QADataset:
    records: list[QARecord] = []
    for index, raw in enumerate(_read_records(path), 1):
        try:
            records.append(QARecord.from_dict(raw, require_answer=require_answer))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid QA record at {path}:{index}: {error}") from error
    if not records:
        raise ValueError("QA dataset is empty")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("QA dataset contains duplicate id values")
    missing = sorted({record.doc_id for record in records} - available_doc_ids)
    if missing:
        raise ValueError(f"QA doc_id values are absent from canonical corpus: {missing}")
    ordered = sorted(records, key=lambda record: record.id)
    identity = {
        "schema_version": QA_DATASET_SCHEMA_VERSION,
        "records": [record.to_dict() for record in ordered],
    }
    return QADataset(ordered, canonical_fingerprint(identity))


def validate_qa_semantics(
    dataset: QADataset, documents: dict[str, NormalizedDocument],
) -> QASemanticValidationReport:
    """Report source mismatches without mutating or repairing external data."""

    report = QASemanticValidationReport()
    for record in dataset.records:
        document = documents.get(record.doc_id)
        if document is None:
            report.errors.append(f"{record.id}: doc_id not loaded: {record.doc_id}")
            continue
        source = "\n\n".join(block.text for block in document.blocks)
        for index, evidence in enumerate(record.evidence_sentences):
            if evidence.text not in source and _normalize_text(evidence.text) not in _normalize_text(source):
                report.warnings.append(f"{record.id}: evidence sentence {index} not found in source")
        paths = _document_section_paths(document)
        for section in record.evidence_sections:
            if _normalize_text(section) not in paths:
                report.warnings.append(f"{record.id}: evidence section not found: {section}")
        if not record.evidence_sentences and not record.evidence_sections:
            report.warnings.append(f"{record.id}: answer has no declared evidence")
    return report


def _normalize_text(value: str) -> str:
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _document_section_paths(document: NormalizedDocument) -> set[str]:
    stack: list[str] = []
    paths: set[str] = set()
    for block in document.blocks:
        if block.type != "heading" or block.level is None:
            continue
        stack = stack[: max(0, block.level - 1)]
        stack.append(block.text)
        paths.add(_normalize_text(block.text))
        paths.add(_normalize_text(" > ".join(stack)))
    return paths
