"""Offline, deterministic team-dataset compatibility audit and benchmark gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.data.models import NORMALIZED_SCHEMA_VERSION, NormalizedDocument
from rag_chunking.embedding.models import canonical_fingerprint

from .evidence import map_evidence_to_chunks
from .qa_dataset import (
    EvidenceSpec, GoldEvidenceItem, QADataset, QARecord,
    TEAM_QA_DATASET_SCHEMA_VERSION, load_team_qa_dataset,
)


COMPATIBILITY_SCHEMA_VERSION = "dataset_chunk_compatibility_v1"
GATE_POLICY_VERSION = "strict_all_required_evidence_all_strategies_v1"
EXPECTED_TEAM_FIELDS = frozenset({
    "question_id", "question", "reference_answer", "difficulty",
    "question_type", "reasoning_type", "evidence", "metadata",
})
EXPECTED_EVIDENCE_FIELDS = frozenset({
    "evidence_id", "doc_id", "section_path", "evidence_sentences",
})
EXPECTED_METADATA_FIELDS = frozenset({
    "evidence_scope", "num_evidence", "semantic_competition",
    "content_type", "answerable_from_evidence",
})


@dataclass(frozen=True, slots=True)
class DocumentResolution:
    status: str
    document: NormalizedDocument | None
    candidates: tuple[str, ...]
    method: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class SectionResolution:
    status: str
    block_start: int | None
    block_end: int | None
    candidates: tuple[str, ...]
    method: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class SentenceResolution:
    status: str
    block_index: int | None
    char_start: int | None
    char_end: int | None
    candidates: tuple[str, ...]
    method: str | None
    reason: str

    @property
    def resolved(self) -> bool:
        return self.status in {"exact_match", "normalized_exact_match"}


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    dataset: QADataset
    report: dict[str, Any]
    unresolved_cases: tuple[dict[str, Any], ...]
    mappings: tuple[dict[str, Any], ...]
    output_directory: Path | None = None

    @property
    def passed(self) -> bool:
        return self.report["gate_decision"] == "PASS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_text(value: str) -> str:
    """NFKC, case folding, and whitespace collapse only."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


_HEADING_ANCHOR = re.compile(r"\s*\{#[^}]+\}\s*$")


def _canonical_heading(value: str) -> str:
    # The parser intentionally retains inline Markdown in headings. Removing
    # delimiter-only syntax gives the rendered heading without semantic search.
    value = _HEADING_ANCHOR.sub("", value)
    value = value.replace("`", "").replace("**", "").replace("__", "")
    return _canonical_text(value)


def _normal_form_with_map(value: str) -> tuple[str, list[int]]:
    output: list[str] = []
    offsets: list[int] = []
    pending_space = False
    for index, character in enumerate(value):
        normalized = unicodedata.normalize("NFKC", character).casefold()
        for item in normalized:
            if item.isspace():
                pending_space = bool(output)
                continue
            if pending_space:
                output.append(" ")
                offsets.append(index)
                pending_space = False
            output.append(item)
            offsets.append(index)
    return "".join(output), offsets


_INLINE_LINK = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
_REFERENCE_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]+\]")
_LIST_PREFIX = re.compile(r"(?m)^[ \t]*(?:[-+*]|\d+[.)])[ \t]+")


def _rendered_markdown_form_with_map(value: str) -> tuple[str, list[int]]:
    """Canonical visible text for delimiter-only Markdown with source offsets."""

    keep = [True] * len(value)
    for pattern in (_INLINE_LINK, _REFERENCE_LINK):
        for match in pattern.finditer(value):
            label_start, label_end = match.span(1)
            for index in range(match.start(), label_start):
                keep[index] = False
            for index in range(label_end, match.end()):
                keep[index] = False
    for match in _LIST_PREFIX.finditer(value):
        for index in range(match.start(), match.end()):
            keep[index] = False
    for index, character in enumerate(value):
        if character in "`*_":
            keep[index] = False
    visible = "".join(character for index, character in enumerate(value) if keep[index])
    visible_to_source = [index for index, allowed in enumerate(keep) if allowed]
    normalized, visible_offsets = _normal_form_with_map(visible)
    return normalized, [visible_to_source[index] for index in visible_offsets]


def _all_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = 0
    while needle:
        start = text.find(needle, cursor)
        if start < 0:
            break
        result.append((start, start + len(needle)))
        cursor = start + max(1, len(needle))
    return result


def resolve_document(doc_id: str, documents: Iterable[NormalizedDocument]) -> DocumentResolution:
    """Resolve exact IDs, then one explicit corpus-namespace transformation."""

    values = list(documents)
    exact = [document for document in values if document.doc_id == doc_id]
    if len(exact) == 1:
        return DocumentResolution("resolved", exact[0], (exact[0].doc_id,), "exact_doc_id", "exact canonical doc_id")
    if len(exact) > 1:
        return DocumentResolution("ambiguous", None, tuple(sorted(item.doc_id for item in exact)), None, "duplicate canonical doc_id")
    if not isinstance(doc_id, str) or not doc_id or doc_id.startswith("/") or "\\" in doc_id:
        return DocumentResolution("invalid", None, (), None, "doc_id is not a portable canonical identifier")
    if ":" not in doc_id:
        transformed = [document for document in values if f"{document.source}:{doc_id}" == document.doc_id]
        if len(transformed) == 1:
            item = transformed[0]
            return DocumentResolution("resolved", item, (item.doc_id,), "add_corpus_namespace", "unique source namespace plus relative path")
        if len(transformed) > 1:
            return DocumentResolution("ambiguous", None, tuple(sorted(item.doc_id for item in transformed)), None, "namespace transformation has multiple candidates")
    return DocumentResolution("unresolved", None, (), None, "no exact canonical document identity")


def _sections(document: NormalizedDocument) -> list[tuple[int, int, tuple[str, ...]]]:
    headings: list[tuple[int, int, tuple[str, ...]]] = []
    stack: list[str] = []
    for index, block in enumerate(document.blocks):
        if block.type != "heading" or block.level is None:
            continue
        stack = stack[: max(0, block.level - 1)]
        stack.append(block.text)
        headings.append((index, len(document.blocks), tuple(stack)))
    result = []
    for position, (start, _, path) in enumerate(headings):
        level = len(path)
        end = len(document.blocks)
        for next_start, _, next_path in headings[position + 1:]:
            if len(next_path) <= level:
                end = next_start
                break
        result.append((start, end, path))
    return result


def resolve_section(document: NormalizedDocument, section_path: list[str]) -> SectionResolution:
    if not isinstance(section_path, list) or not section_path or any(
        not isinstance(item, str) or not item.strip() for item in section_path
    ):
        return SectionResolution("invalid", None, None, (), None, "section_path is malformed")
    sections = _sections(document)
    target = tuple(section_path)
    exact = [item for item in sections if item[2] == target]
    if len(exact) == 1:
        start, end, path = exact[0]
        return SectionResolution("resolved", start, end, (" > ".join(path),), "exact_path", "exact full heading path")
    if len(exact) > 1:
        return SectionResolution("ambiguous", None, None, tuple(" > ".join(item[2]) for item in exact), None, "heading path identifies multiple sections")
    suffix = [item for item in sections if item[2][-len(target):] == target]
    if len(suffix) == 1:
        start, end, path = suffix[0]
        return SectionResolution("resolved", start, end, (" > ".join(path),), "exact_suffix", "unique exact heading-path suffix")
    if len(suffix) > 1:
        return SectionResolution("ambiguous", None, None, tuple(" > ".join(item[2]) for item in suffix), None, "heading path suffix identifies multiple sections")
    normalized_target = tuple(_canonical_heading(item) for item in target)
    normalized = [item for item in sections if tuple(
        _canonical_heading(part) for part in item[2]
    ) == normalized_target]
    if len(normalized) == 1:
        start, end, path = normalized[0]
        return SectionResolution("resolved", start, end, (" > ".join(path),), "normalized_heading_path", "NFKC/case/whitespace and Markdown-delimiter normalization")
    if len(normalized) > 1:
        return SectionResolution("ambiguous", None, None, tuple(" > ".join(item[2]) for item in normalized), None, "normalized heading path identifies multiple sections")
    normalized_suffix = [item for item in sections if tuple(
        _canonical_heading(part) for part in item[2]
    )[-len(normalized_target):] == normalized_target]
    if len(normalized_suffix) == 1:
        start, end, path = normalized_suffix[0]
        return SectionResolution("resolved", start, end, (" > ".join(path),), "normalized_heading_suffix", "unique normalized heading-path suffix")
    if len(normalized_suffix) > 1:
        return SectionResolution("ambiguous", None, None, tuple(" > ".join(item[2]) for item in normalized_suffix), None, "normalized heading suffix identifies multiple sections")
    return SectionResolution("unresolved", None, None, (), None, "section heading path is absent from the canonical document")


def resolve_sentence(
    document: NormalizedDocument, text: str, section: SectionResolution,
) -> SentenceResolution:
    if not isinstance(text, str) or not text.strip():
        return SentenceResolution("invalid", None, None, None, (), None, "evidence sentence is empty")
    start = section.block_start if section.status == "resolved" else 0
    end = section.block_end if section.status == "resolved" else len(document.blocks)
    assert start is not None and end is not None
    exact: list[tuple[int, int, int]] = []
    for block_index in range(start, end):
        for char_start, char_end in _all_occurrences(document.blocks[block_index].text, text):
            exact.append((block_index, char_start, char_end))
    if len(exact) == 1:
        block, char_start, char_end = exact[0]
        return SentenceResolution("exact_match", block, char_start, char_end, (f"block:{block}:{char_start}-{char_end}",), "exact_block_text", "unique exact occurrence")
    if len(exact) > 1:
        candidates = tuple(f"block:{block}:{left}-{right}" for block, left, right in exact)
        return SentenceResolution("ambiguous", None, None, None, candidates, None, "multiple exact occurrences remain within the resolved scope")
    target, _ = _normal_form_with_map(text)
    normalized: list[tuple[int, int, int]] = []
    for block_index in range(start, end):
        block_text = document.blocks[block_index].text
        value, offsets = _normal_form_with_map(block_text)
        for left, right in _all_occurrences(value, target):
            if offsets and right > left:
                normalized.append((block_index, offsets[left], offsets[right - 1] + 1))
    if len(normalized) == 1:
        block, char_start, char_end = normalized[0]
        return SentenceResolution("normalized_exact_match", block, char_start, char_end, (f"block:{block}:{char_start}-{char_end}",), "nfkc_casefold_whitespace", "unique normalized occurrence")
    if len(normalized) > 1:
        candidates = tuple(f"block:{block}:{left}-{right}" for block, left, right in normalized)
        return SentenceResolution("ambiguous", None, None, None, candidates, None, "multiple normalized occurrences remain within the resolved scope")
    rendered_target, _ = _rendered_markdown_form_with_map(text)
    rendered: list[tuple[int, int, int]] = []
    for block_index in range(start, end):
        block_text = document.blocks[block_index].text
        value, offsets = _rendered_markdown_form_with_map(block_text)
        for left, right in _all_occurrences(value, rendered_target):
            if offsets and right > left:
                rendered.append((block_index, offsets[left], offsets[right - 1] + 1))
    if len(rendered) == 1:
        block, char_start, char_end = rendered[0]
        return SentenceResolution(
            "normalized_exact_match", block, char_start, char_end,
            (f"block:{block}:{char_start}-{char_end}",),
            "rendered_markdown_nfkc_casefold_whitespace",
            "unique deterministic rendered-Markdown occurrence",
        )
    if len(rendered) > 1:
        candidates = tuple(f"block:{block}:{left}-{right}" for block, left, right in rendered)
        return SentenceResolution(
            "ambiguous", None, None, None, candidates, None,
            "multiple rendered-Markdown occurrences remain within the resolved scope",
        )
    return SentenceResolution("unresolved", None, None, None, (), None, "evidence text is absent under deterministic exact normalization")


def _missing_sentence_root_cause(document: NormalizedDocument, text: str) -> str:
    """Classify diagnostics only; this never changes match or gate status."""

    target, _ = _rendered_markdown_form_with_map(text)
    best = max(
        (
            difflib.SequenceMatcher(
                None, target, _rendered_markdown_form_with_map(block.text)[0],
                autojunk=False,
            ).ratio()
            for block in document.blocks
        ),
        default=0.0,
    )
    if best >= 0.95:
        return "evidence_text_normalization_issue"
    return "corpus_version_mismatch"


def _read_raw_records(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"dataset record {line_number} must be an object")
        values.append(value)
    return values


def inspect_team_dataset(path: Path) -> dict[str, Any]:
    records = _read_raw_records(path)
    field_presence = {name: sum(name in item for item in records) for name in sorted(EXPECTED_TEAM_FIELDS)}
    unknown_fields = sorted({name for item in records for name in set(item) - EXPECTED_TEAM_FIELDS})
    evidence_counts: Counter[int] = Counter()
    malformed_evidence = 0
    metadata_mismatches = 0
    referenced_documents: set[str] = set()
    evidence_sentences = 0
    evidence_items = 0
    unknown_evidence_fields: set[str] = set()
    unknown_metadata_fields: set[str] = set()
    for item in records:
        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            malformed_evidence += 1
            continue
        evidence_counts[len(evidence)] += 1
        evidence_items += len(evidence)
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            unknown_metadata_fields.update(set(metadata) - EXPECTED_METADATA_FIELDS)
            metadata_mismatches += metadata.get("num_evidence") != len(evidence)
        else:
            metadata_mismatches += 1
        for entry in evidence:
            if not isinstance(entry, dict):
                malformed_evidence += 1
                continue
            unknown_evidence_fields.update(set(entry) - EXPECTED_EVIDENCE_FIELDS)
            if set(entry) != EXPECTED_EVIDENCE_FIELDS:
                malformed_evidence += 1
            if isinstance(entry.get("doc_id"), str):
                referenced_documents.add(entry["doc_id"])
            sentences = entry.get("evidence_sentences")
            if isinstance(sentences, list):
                evidence_sentences += len(sentences)
            else:
                malformed_evidence += 1
    ids = [item.get("question_id") for item in records]
    duplicates = sorted(name for name, count in Counter(ids).items() if count > 1)
    required_empty = {
        name: sum(name not in item or item.get(name) in (None, "", [], {}) for item in records)
        for name in sorted(EXPECTED_TEAM_FIELDS)
    }
    def distribution(getter) -> dict[str, int]:
        return dict(sorted(Counter(str(getter(item)) for item in records).items()))
    return {
        "record_count": len(records), "field_presence": field_presence,
        "duplicate_question_ids": duplicates, "empty_required_fields": required_empty,
        "distributions": {
            "difficulty": distribution(lambda item: item.get("difficulty")),
            "question_type": distribution(lambda item: item.get("question_type")),
            "reasoning_type": distribution(lambda item: item.get("reasoning_type")),
            "evidence_scope": distribution(lambda item: item.get("metadata", {}).get("evidence_scope")),
            "evidence_count": {str(key): value for key, value in sorted(evidence_counts.items())},
        },
        "evidence_item_count": evidence_items,
        "evidence_sentence_count": evidence_sentences,
        "unique_referenced_documents": len(referenced_documents),
        "malformed_evidence_objects": malformed_evidence,
        "metadata_num_evidence_mismatches": metadata_mismatches,
        "unknown_fields": unknown_fields,
        "unknown_evidence_fields": sorted(unknown_evidence_fields),
        "unknown_metadata_fields": sorted(unknown_metadata_fields),
    }


def _validate_chunks(
    strategy: str, chunks: list[Chunk], documents: list[NormalizedDocument],
    manifest: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if not chunks:
        errors.append("chunk artifact is empty")
        return errors
    if any(item.strategy != strategy for item in chunks):
        errors.append("chunk artifact contains another strategy")
    ids = [item.chunk_id for item in chunks]
    if len(ids) != len(set(ids)):
        errors.append("chunk artifact contains duplicate chunk IDs")
    if manifest is not None:
        if manifest.get("strategy") != strategy or manifest.get("chunks") != len(chunks):
            errors.append("chunk manifest strategy/count mismatch")
        if manifest.get("source_schema_version") != NORMALIZED_SCHEMA_VERSION:
            errors.append("chunk manifest source schema mismatch")
        expected_docs = {item.doc_id for item in documents}
        actual_docs = {item.doc_id for item in chunks}
        if manifest.get("documents") != len(expected_docs) or actual_docs != expected_docs:
            errors.append("chunk manifest/document coverage mismatch")
    return errors


def _jsonl(values: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for value in values
    )


def _breakdown(records: list[QARecord], compatible: set[str], attribute: str) -> dict[str, dict[str, int]]:
    groups: dict[str, list[QARecord]] = defaultdict(list)
    for record in records:
        value = record.metadata.get(attribute) if attribute == "evidence_scope" else getattr(record, attribute)
        groups[str(value)].append(record)
    return {
        name: {"total": len(items), "compatible": sum(item.id in compatible for item in items)}
        for name, items in sorted(groups.items())
    }


def _markdown_report(report: dict[str, Any]) -> str:
    mapping = report["chunk_mapping"]
    lines = [
        "# Dataset Compatibility Audit", "",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Fingerprint: `{report['dataset_fingerprint']}`",
        f"- Questions: {report['question_count']}",
        f"- Evidence items / sentences: {report['evidence_item_count']} / {report['evidence_sentence_count']}",
        f"- Schema-valid questions: {report['valid_question_count']}",
        f"- Compatible questions: {report['compatible_question_count']}",
        f"- Gate: **{report['gate_decision']}**", "",
        "## Provenance resolution", "",
        f"- Documents: {report['document_resolution']}",
        f"- Sections: {report['section_resolution']}",
        f"- Sentences: {report['evidence_sentence_resolution']}", "",
        "## Evidence-to-chunk mapping", "",
        "| Strategy | Compatible questions | Evidence mapped | Evidence unmapped |",
        "| --- | ---: | ---: | ---: |",
    ]
    for strategy, value in mapping.items():
        lines.append(f"| {strategy} | {value['compatible_questions']} | {value['mapped_evidence']} | {value['unmapped_evidence']} |")
    lines.extend(["", "## Gate reasons", ""])
    lines.extend(f"- {item}" for item in report["gate_reasons"])
    lines.extend(["", "## Failure root causes", ""])
    lines.extend(
        f"- {name}: {count}"
        for name, count in report["failure_root_cause_counts"].items()
    )
    lines.extend(["", "## Compatibility by evidence scope", ""])
    lines.extend(
        f"- {name}: {value['compatible']} / {value['total']}"
        for name, value in report["breakdown"]["evidence_scope"].items()
    )
    lines.extend(["", "This is an offline compatibility result, not a retrieval or answer benchmark.", ""])
    return "\n".join(lines)


def audit_dataset_compatibility(
    *, dataset_path: Path, documents: list[NormalizedDocument],
    chunks_by_strategy: dict[str, list[Chunk]],
    chunk_manifests: dict[str, dict[str, Any]] | None = None,
    raw_root: Path | None = None, output_directory: Path | None = None,
) -> CompatibilityResult:
    """Audit all required evidence and optionally publish manifest-last artifacts."""

    dataset = load_team_qa_dataset(dataset_path)
    raw_audit = inspect_team_dataset(dataset_path)
    strategies = list(chunks_by_strategy)
    if not strategies:
        raise ValueError("at least one chunk strategy is required")
    chunk_errors = {
        strategy: _validate_chunks(
            strategy, chunks_by_strategy[strategy], documents,
            (chunk_manifests or {}).get(strategy),
        )
        for strategy in strategies
    }
    doc_counts = Counter(item.doc_id for item in documents)
    corpus_integrity_errors = [f"duplicate canonical doc_id: {doc_id}" for doc_id, count in doc_counts.items() if count > 1]
    if raw_root is not None:
        for document in documents:
            raw_path = raw_root / document.relative_path
            if not raw_path.is_file():
                corpus_integrity_errors.append(f"missing raw source: {raw_path.as_posix()}")
            elif _sha256(raw_path) != document.source_sha256:
                corpus_integrity_errors.append(f"raw/processed source fingerprint mismatch: {document.doc_id}")

    document_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    sentence_counts: Counter[str] = Counter()
    unresolved: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    evidence_mapped: dict[str, set[tuple[str, str]]] = {name: set() for name in strategies}
    strategy_question_ok: dict[str, dict[str, bool]] = {
        name: {record.id: True for record in dataset.records} for name in strategies
    }
    question_provenance_ok = {record.id: True for record in dataset.records}
    warnings_by_question: dict[str, list[str]] = defaultdict(list)

    for record in dataset.records:
        for evidence in record.evidence:
            key = (record.id, evidence.evidence_id)
            doc_resolution = resolve_document(evidence.doc_id, documents)
            document_counts[doc_resolution.status] += 1
            if doc_resolution.status != "resolved":
                question_provenance_ok[record.id] = False
                for strategy in strategies:
                    strategy_question_ok[strategy][record.id] = False
                unresolved.append({
                    "question_id": record.id, "evidence_id": evidence.evidence_id,
                    "evidence_sentence_index": None, "original_doc_id": evidence.doc_id,
                    "canonical_doc_candidate": None, "section_path": evidence.section_path,
                    "failure_stage": "document_resolution", "failure_reason": doc_resolution.reason,
                    "strategy": None, "candidate_chunk_ids": [], "candidates": list(doc_resolution.candidates),
                    "root_cause": "doc_id_convention_mismatch" if doc_resolution.status == "unresolved" else "ambiguous_document_mapping",
                })
                continue
            document = doc_resolution.document
            assert document is not None
            if doc_resolution.method != "exact_doc_id":
                warnings_by_question[record.id].append(f"{evidence.evidence_id}: document resolved by {doc_resolution.method}")
            section = resolve_section(document, evidence.section_path)
            section_counts[section.status] += 1
            if section.status != "resolved":
                question_provenance_ok[record.id] = False
                unresolved.append({
                    "question_id": record.id, "evidence_id": evidence.evidence_id,
                    "evidence_sentence_index": None, "original_doc_id": evidence.doc_id,
                    "canonical_doc_candidate": document.doc_id, "section_path": evidence.section_path,
                    "failure_stage": "section_resolution", "failure_reason": section.reason,
                    "strategy": None, "candidate_chunk_ids": [], "candidates": list(section.candidates),
                    "root_cause": "section_path_mismatch" if section.status == "unresolved" else "ambiguous_section_mapping",
                })
            elif section.method == "normalized_heading_path":
                warnings_by_question[record.id].append(f"{evidence.evidence_id}: normalized section match")

            resolved_specs: list[tuple[int, EvidenceSpec]] = []
            sentence_details: list[dict[str, Any]] = []
            for sentence_index, sentence in enumerate(evidence.evidence_sentences):
                resolution = resolve_sentence(document, sentence.text, section)
                sentence_counts[resolution.status] += 1
                detail = {
                    "question_id": record.id, "evidence_id": evidence.evidence_id,
                    "evidence_sentence_index": sentence_index, "original_doc_id": evidence.doc_id,
                    "canonical_doc_id": document.doc_id, "section_path": evidence.section_path,
                    "sentence_status": resolution.status, "sentence_match_method": resolution.method,
                    "block_index": resolution.block_index, "char_start": resolution.char_start,
                    "char_end": resolution.char_end,
                }
                sentence_details.append(detail)
                if resolution.resolved:
                    assert resolution.block_index is not None
                    assert resolution.char_start is not None and resolution.char_end is not None
                    source_text = document.blocks[resolution.block_index].text[
                        resolution.char_start:resolution.char_end
                    ]
                    resolved_specs.append((sentence_index, EvidenceSpec(
                        source_text, resolution.block_index,
                        resolution.char_start, resolution.char_end,
                    )))
                    if resolution.status == "normalized_exact_match":
                        warnings_by_question[record.id].append(
                            f"{evidence.evidence_id} sentence {sentence_index}: normalized exact match"
                        )
                else:
                    question_provenance_ok[record.id] = False
                    unresolved.append({
                        "question_id": record.id, "evidence_id": evidence.evidence_id,
                        "evidence_sentence_index": sentence_index, "original_doc_id": evidence.doc_id,
                        "canonical_doc_candidate": document.doc_id, "section_path": evidence.section_path,
                        "failure_stage": "evidence_sentence_resolution", "failure_reason": resolution.reason,
                        "strategy": None, "candidate_chunk_ids": [], "candidates": list(resolution.candidates),
                        "root_cause": (
                            _missing_sentence_root_cause(document, sentence.text)
                            if resolution.status == "unresolved" else "ambiguous_evidence_text"
                        ),
                    })

            for strategy in strategies:
                mapped_all = len(resolved_specs) == len(evidence.evidence_sentences)
                strategy_rows: list[dict[str, Any]] = []
                if resolved_specs:
                    projection = QARecord(
                        id=record.id, doc_id=document.doc_id, question=record.question,
                        answer=record.answer,
                        evidence_sentences=[item for _, item in resolved_specs],
                        evidence_sections=[], question_type=record.question_type,
                        difficulty=record.difficulty,
                    )
                    evidence_mappings = map_evidence_to_chunks(
                        projection, document, chunks_by_strategy[strategy], strategy,
                    )
                    for (sentence_index, _), item in zip(
                        resolved_specs, evidence_mappings, strict=True,
                    ):
                        row = {
                            **sentence_details[sentence_index], "strategy": strategy,
                            "matched_chunk_ids": item.matched_chunk_ids,
                            "mapping_method": item.match_method, "coverage": item.coverage,
                        }
                        strategy_rows.append(row)
                        if item.coverage < 1.0 or not item.matched_chunk_ids:
                            mapped_all = False
                            unresolved.append({
                                "question_id": record.id, "evidence_id": evidence.evidence_id,
                                "evidence_sentence_index": sentence_index, "original_doc_id": evidence.doc_id,
                                "canonical_doc_candidate": document.doc_id, "section_path": evidence.section_path,
                                "failure_stage": "evidence_chunk_mapping", "failure_reason": "no complete provenance coverage in committed chunks",
                                "strategy": strategy, "candidate_chunk_ids": item.matched_chunk_ids,
                                "candidates": [], "root_cause": "chunk_coverage_issue",
                            })
                if mapped_all:
                    evidence_mapped[strategy].add(key)
                else:
                    strategy_question_ok[strategy][record.id] = False
                mappings.extend(strategy_rows)

    for strategy, errors in chunk_errors.items():
        if errors:
            for record in dataset.records:
                strategy_question_ok[strategy][record.id] = False
            for error in errors:
                unresolved.append({
                    "question_id": None, "evidence_id": None, "evidence_sentence_index": None,
                    "original_doc_id": None, "canonical_doc_candidate": None, "section_path": [],
                    "failure_stage": "chunk_artifact_integrity", "failure_reason": error,
                    "strategy": strategy, "candidate_chunk_ids": [], "candidates": [],
                    "root_cause": "corrupt_chunk_provenance",
                })

    compatible = {
        record.id for record in dataset.records
        if question_provenance_ok[record.id]
        and all(strategy_question_ok[strategy][record.id] for strategy in strategies)
    }
    evidence_total = sum(len(record.evidence) for record in dataset.records)
    gate_reasons: list[str] = []
    if raw_audit["malformed_evidence_objects"] or raw_audit["metadata_num_evidence_mismatches"]:
        gate_reasons.append("dataset schema or evidence metadata is invalid")
    if corpus_integrity_errors:
        gate_reasons.append("canonical raw/processed corpus integrity failed")
    if document_counts["unresolved"] or document_counts["ambiguous"] or document_counts["invalid"]:
        gate_reasons.append("one or more document references do not resolve uniquely")
    if section_counts["unresolved"] or section_counts["ambiguous"] or section_counts["invalid"]:
        gate_reasons.append("one or more section paths do not resolve uniquely")
    if sentence_counts["unresolved"] or sentence_counts["ambiguous"] or sentence_counts["invalid"]:
        gate_reasons.append("one or more evidence sentences do not resolve uniquely")
    if any(len(evidence_mapped[name]) != evidence_total for name in strategies):
        gate_reasons.append("one or more evidence items do not map to every chunk strategy")
    if any(chunk_errors.values()):
        gate_reasons.append("one or more chunk artifacts failed integrity validation")
    if not gate_reasons:
        gate_reasons.append("all strict dataset, provenance, and chunk coverage checks passed")

    strategy_compatible_sets = {
        strategy: {question_id for question_id, ok in statuses.items() if ok and question_provenance_ok[question_id]}
        for strategy, statuses in strategy_question_ok.items()
    }
    unresolved.sort(key=lambda item: (
        item["question_id"] or "", item["evidence_id"] or "",
        -1 if item["evidence_sentence_index"] is None else item["evidence_sentence_index"],
        item["failure_stage"], item["strategy"] or "",
    ))
    mappings.sort(key=lambda item: (
        item["question_id"], item["evidence_id"], item["evidence_sentence_index"], item["strategy"],
    ))
    report: dict[str, Any] = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "adapter_version": TEAM_QA_DATASET_SCHEMA_VERSION,
        "gate_policy_version": GATE_POLICY_VERSION,
        "dataset_path": dataset_path.as_posix(), "dataset_fingerprint": dataset.fingerprint,
        "dataset_schema_version": dataset.schema_version,
        "corpus_fingerprint": canonical_fingerprint({
            "schema_version": NORMALIZED_SCHEMA_VERSION,
            "documents": [item.to_dict() for item in sorted(documents, key=lambda value: value.doc_id)],
        }),
        "chunk_artifact_fingerprints": {
            strategy: canonical_fingerprint({
                "strategy": strategy,
                "chunks": [item.to_dict() for item in sorted(
                    chunks_by_strategy[strategy], key=lambda value: value.chunk_id,
                )],
                "manifest": (chunk_manifests or {}).get(strategy),
            })
            for strategy in strategies
        },
        "dataset_audit": raw_audit,
        "question_count": len(dataset.records), "valid_question_count": len(dataset.records),
        "invalid_question_count": 0,
        "compatible_question_count": len(compatible),
        "incompatible_question_count": len(dataset.records) - len(compatible),
        "unique_referenced_documents": raw_audit["unique_referenced_documents"],
        "evidence_item_count": evidence_total,
        "evidence_sentence_count": raw_audit["evidence_sentence_count"],
        "document_resolution": {
            "resolved": document_counts["resolved"], "unresolved": document_counts["unresolved"],
            "ambiguous": document_counts["ambiguous"], "invalid": document_counts["invalid"],
        },
        "section_resolution": {
            "resolved": section_counts["resolved"], "unresolved": section_counts["unresolved"],
            "ambiguous": section_counts["ambiguous"], "invalid": section_counts["invalid"],
        },
        "evidence_sentence_resolution": {
            "exact": sentence_counts["exact_match"],
            "normalized_exact": sentence_counts["normalized_exact_match"],
            "unresolved": sentence_counts["unresolved"],
            "ambiguous": sentence_counts["ambiguous"], "invalid": sentence_counts["invalid"],
        },
        "chunk_mapping": {
            strategy: {
                "total_evidence": evidence_total,
                "mapped_evidence": len(evidence_mapped[strategy]),
                "unmapped_evidence": evidence_total - len(evidence_mapped[strategy]),
                "compatible_questions": len(strategy_compatible_sets[strategy]),
                "artifact_errors": chunk_errors[strategy],
            }
            for strategy in strategies
        },
        "breakdown": {
            name: _breakdown(dataset.records, compatible, name)
            for name in ("difficulty", "question_type", "reasoning_type", "evidence_scope")
        },
        "question_status": {
            record.id: (
                "compatible_with_warning" if record.id in compatible and warnings_by_question[record.id]
                else "compatible" if record.id in compatible
                else "incompatible"
            )
            for record in dataset.records
        },
        "incompatible_question_ids": sorted(set(record.id for record in dataset.records) - compatible),
        "warnings": [
            {"question_id": question_id, "messages": sorted(set(messages))}
            for question_id, messages in sorted(warnings_by_question.items())
        ],
        "corpus_integrity_errors": sorted(corpus_integrity_errors),
        "unresolved_case_count": len(unresolved),
        "failure_root_cause_counts": dict(sorted(Counter(
            item["root_cause"] for item in unresolved
        ).items())),
        "unresolved_cases_fingerprint": canonical_fingerprint(unresolved),
        "evidence_chunk_mappings_fingerprint": canonical_fingerprint(mappings),
        "gate_decision": "PASS" if not gate_reasons[:-1] and gate_reasons == ["all strict dataset, provenance, and chunk coverage checks passed"] else "FAIL",
        "gate_reasons": gate_reasons,
    }
    report["compatibility_fingerprint"] = canonical_fingerprint({
        key: value for key, value in report.items() if key != "compatibility_fingerprint"
    })
    final_output = None
    if output_directory is not None:
        manifest = {
            "schema_version": COMPATIBILITY_SCHEMA_VERSION, "complete": True,
            "compatibility_fingerprint": report["compatibility_fingerprint"],
            "dataset_path": dataset_path.as_posix(), "dataset_fingerprint": dataset.fingerprint,
            "dataset_schema_version": dataset.schema_version,
            "adapter_version": TEAM_QA_DATASET_SCHEMA_VERSION,
            "gate_policy_version": GATE_POLICY_VERSION,
            "corpus_fingerprint": report["corpus_fingerprint"],
            "chunk_artifact_fingerprints": report["chunk_artifact_fingerprints"],
            "unresolved_cases_fingerprint": report["unresolved_cases_fingerprint"],
            "evidence_chunk_mappings_fingerprint": report["evidence_chunk_mappings_fingerprint"],
            "gate_decision": report["gate_decision"], "strategies": strategies,
            "question_count": len(dataset.records), "unresolved_case_count": len(unresolved),
        }
        write_artifact_set(output_directory, {
            "compatibility_report.json": serialize_json(report),
            "unresolved_cases.jsonl": _jsonl(unresolved),
            "evidence_chunk_mappings.jsonl": _jsonl(mappings),
            "DATASET_COMPATIBILITY_AUDIT.md": _markdown_report(report),
            "manifest.json": serialize_json(manifest),
        })
        final_output = output_directory
    return CompatibilityResult(dataset, report, tuple(unresolved), tuple(mappings), final_output)
