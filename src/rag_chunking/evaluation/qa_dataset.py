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
TEAM_QA_DATASET_SCHEMA_VERSION = "team_evidence_qa_adapter_v1"


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
class GoldEvidenceItem:
    """One authored evidence item; document and section identity stay attached."""

    evidence_id: str
    doc_id: str
    section_path: list[str]
    evidence_sentences: list[EvidenceSpec]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError("evidence doc_id must be non-empty")
        if not isinstance(self.section_path, list) or not self.section_path or any(
            not isinstance(item, str) or not item.strip() for item in self.section_path
        ):
            raise ValueError("section_path must be a non-empty list of non-empty strings")
        if not isinstance(self.evidence_sentences, list) or not self.evidence_sentences:
            raise ValueError("evidence_sentences must be a non-empty list")

    @classmethod
    def from_dict(cls, value: object) -> "GoldEvidenceItem":
        if not isinstance(value, dict):
            raise ValueError("evidence items must be objects")
        required = {"evidence_id", "doc_id", "section_path", "evidence_sentences"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"missing evidence fields: {missing}")
        unknown = sorted(set(value) - required)
        if unknown:
            raise ValueError(f"unknown evidence fields: {unknown}")
        sentences = value["evidence_sentences"]
        if not isinstance(sentences, list):
            raise ValueError("evidence_sentences must be a list")
        return cls(
            evidence_id=value["evidence_id"], doc_id=value["doc_id"],
            section_path=value["section_path"],
            evidence_sentences=[EvidenceSpec.from_value(item) for item in sentences],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id, "doc_id": self.doc_id,
            "section_path": list(self.section_path),
            "evidence_sentences": [item.to_external_value() for item in self.evidence_sentences],
        }


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
    reasoning_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: list[GoldEvidenceItem] = field(default_factory=list)

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
        if self.reasoning_type is not None and (
            not isinstance(self.reasoning_type, str) or not self.reasoning_type.strip()
        ):
            raise ValueError("reasoning_type must be null or a non-empty string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")
        if not isinstance(self.evidence, list):
            raise ValueError("evidence must be a list")

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
        allowed = required | {"answer", "notes", "reasoning_type", "metadata", "evidence"}
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
        raw_items = data.get("evidence", [])
        if not isinstance(raw_items, list):
            raise ValueError("evidence must be a list")
        data["evidence"] = [GoldEvidenceItem.from_dict(item) for item in raw_items]
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_sentences"] = [item.to_external_value() for item in self.evidence_sentences]
        if self.evidence:
            value["evidence"] = [item.to_dict() for item in self.evidence]
        else:
            value.pop("evidence")
        if self.reasoning_type is None:
            value.pop("reasoning_type")
        if not self.metadata:
            value.pop("metadata")
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
        value = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_object,
        )
        if not isinstance(value, list):
            raise ValueError("QA JSON must contain an array")
        return value
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(
                line, parse_constant=_reject_constant, object_pairs_hook=_unique_object,
            ))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid QA JSONL at {path}:{line_number}: {error}") from error
    return records


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def qa_dataset_fingerprint(
    records: list[QARecord], *, schema_version: str = QA_DATASET_SCHEMA_VERSION,
) -> str:
    if schema_version not in {QA_DATASET_SCHEMA_VERSION, TEAM_QA_DATASET_SCHEMA_VERSION}:
        raise ValueError(f"unsupported QA dataset schema {schema_version!r}")
    ordered = sorted(records, key=lambda record: record.id)
    identity = {
        "schema_version": schema_version,
        "records": [record.to_dict() for record in ordered],
    }
    if schema_version == TEAM_QA_DATASET_SCHEMA_VERSION:
        identity["adapter_version"] = TEAM_QA_DATASET_SCHEMA_VERSION
    return canonical_fingerprint(identity)


def adapt_team_qa_record(value: dict[str, Any]) -> QARecord:
    """Adapt the immutable team schema without flattening structured evidence."""

    if not isinstance(value, dict):
        raise ValueError("QA record must be an object")
    required = {
        "question_id", "question", "reference_answer", "difficulty",
        "question_type", "reasoning_type", "evidence", "metadata",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"missing team QA fields: {missing}")
    unknown = sorted(set(value) - required)
    if unknown:
        raise ValueError(f"unknown team QA fields: {unknown}")
    if not isinstance(value["reference_answer"], str) or not value["reference_answer"].strip():
        raise ValueError("answer must be non-empty")
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must be a non-empty list")
    items = [GoldEvidenceItem.from_dict(item) for item in evidence]
    evidence_ids = [item.evidence_id for item in items]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("record contains duplicate evidence_id values")
    metadata = value["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    expected_metadata = {
        "evidence_scope", "num_evidence", "semantic_competition",
        "content_type", "answerable_from_evidence",
    }
    missing_metadata = sorted(expected_metadata - set(metadata))
    if missing_metadata:
        raise ValueError(f"missing metadata fields: {missing_metadata}")
    unknown_metadata = sorted(set(metadata) - expected_metadata)
    if unknown_metadata:
        raise ValueError(f"unknown metadata fields: {unknown_metadata}")
    if type(metadata["num_evidence"]) is not int or metadata["num_evidence"] != len(items):
        raise ValueError("metadata.num_evidence does not match evidence length")
    if not isinstance(metadata["evidence_scope"], str) or not metadata["evidence_scope"].strip():
        raise ValueError("metadata.evidence_scope must be non-empty")
    if not isinstance(metadata["content_type"], str) or not metadata["content_type"].strip():
        raise ValueError("metadata.content_type must be non-empty")
    for name in ("semantic_competition", "answerable_from_evidence"):
        if type(metadata[name]) is not bool:
            raise ValueError(f"metadata.{name} must be boolean")
    return QARecord(
        id=value["question_id"], doc_id=items[0].doc_id,
        question=value["question"], answer=value["reference_answer"],
        evidence_sentences=[], evidence_sections=[],
        question_type=value["question_type"], difficulty=value["difficulty"],
        reasoning_type=value["reasoning_type"], metadata=dict(metadata), evidence=items,
    )


def load_team_qa_dataset(path: Path) -> QADataset:
    records: list[QARecord] = []
    for index, raw in enumerate(_read_records(path), 1):
        try:
            records.append(adapt_team_qa_record(raw))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid team QA record at {path}:{index}: {error}") from error
    if not records:
        raise ValueError("QA dataset is empty")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("QA dataset contains duplicate question_id values")
    ordered = sorted(records, key=lambda record: record.id)
    return QADataset(
        ordered, qa_dataset_fingerprint(ordered, schema_version=TEAM_QA_DATASET_SCHEMA_VERSION),
        TEAM_QA_DATASET_SCHEMA_VERSION,
    )


def is_team_qa_dataset(path: Path) -> bool:
    """Identify the explicit team contract; never select by filename."""

    records = _read_records(path)
    return bool(records and isinstance(records[0], dict) and "question_id" in records[0])


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
    return QADataset(ordered, qa_dataset_fingerprint(ordered))


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


def validate_canonical_qa_dataset(
    path: Path, documents: dict[str, NormalizedDocument],
) -> tuple[QADataset, QASemanticValidationReport]:
    """Strict teammate-handoff validation without repairing QA content."""

    dataset = load_qa_dataset(path, set(documents))
    if dataset.schema_version != QA_DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported QA dataset schema {dataset.schema_version!r}")
    expected_fingerprint = qa_dataset_fingerprint(
        dataset.records, schema_version=dataset.schema_version,
    )
    if dataset.fingerprint != expected_fingerprint:
        raise ValueError("QA dataset fingerprint does not match semantic contents")
    report = validate_qa_semantics(dataset, documents)
    report.errors.extend(_strict_provenance_errors(dataset, documents))
    return dataset, report


def _strict_provenance_errors(
    dataset: QADataset, documents: dict[str, NormalizedDocument],
) -> list[str]:
    errors: list[str] = []
    for record in dataset.records:
        document = documents[record.doc_id]
        if not record.evidence_sentences and not record.evidence_sections:
            errors.append(f"{record.id}: at least one evidence sentence or section is required")
        source = "\n\n".join(block.text for block in document.blocks)
        for index, evidence in enumerate(record.evidence_sentences):
            label = f"{record.id}: evidence sentence {index}"
            if evidence.block_index is None:
                if evidence.text not in source and _normalize_text(evidence.text) not in _normalize_text(source):
                    errors.append(f"{label} is absent from the canonical document")
                continue
            if evidence.block_index >= len(document.blocks):
                errors.append(f"{label} block_index is outside the canonical document")
                continue
            block = document.blocks[evidence.block_index]
            if evidence.char_start is None:
                if evidence.text not in block.text and _normalize_text(evidence.text) not in _normalize_text(block.text):
                    errors.append(f"{label} is absent from its declared block")
                continue
            assert evidence.char_end is not None
            if evidence.char_end > len(block.text):
                errors.append(f"{label} character span exceeds its declared block")
            elif block.text[evidence.char_start:evidence.char_end] != evidence.text:
                errors.append(f"{label} text does not equal its declared character span")
        paths = _document_section_paths(document)
        for section in record.evidence_sections:
            if _normalize_text(section) not in paths:
                errors.append(f"{record.id}: evidence section is absent: {section}")
    return errors


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
