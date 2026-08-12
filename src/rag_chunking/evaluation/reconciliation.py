"""Deterministic, human-reviewed reconciliation for compatibility failures."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import difflib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.data.models import NORMALIZED_SCHEMA_VERSION, NormalizedDocument
from rag_chunking.embedding.models import canonical_fingerprint

from .compatibility import (
    COMPATIBILITY_SCHEMA_VERSION, _canonical_heading, _rendered_markdown_form_with_map,
    _sections,
)
from .qa_dataset import QADataset, QARecord, load_team_qa_dataset


RECONCILIATION_SCHEMA_VERSION = "dataset_corpus_reconciliation_v1"
RECONCILIATION_ALGORITHM_VERSION = "deterministic_lexical_candidates_v1"
CONFIDENCE_ORDER = {"high": 0, "medium": 1, "none": 2}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    output_directory: Path
    reconciliation_fingerprint: str
    cases: tuple[dict[str, Any], ...]
    proposals: tuple[dict[str, Any], ...]
    stats: dict[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        values.append(value)
    return values


def _jsonl(values: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for value in values
    )


def _normal(value: str) -> str:
    return _rendered_markdown_form_with_map(value)[0]


def _tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(_normal(value))


def _lexical_score(left: str, right: str) -> float:
    normalized_left, normalized_right = _normal(left), _normal(right)
    ratio = difflib.SequenceMatcher(
        None, normalized_left, normalized_right, autojunk=False,
    ).ratio()
    left_tokens, right_tokens = Counter(_tokens(left)), Counter(_tokens(right))
    overlap = sum((left_tokens & right_tokens).values())
    token_f1 = (
        2 * overlap / (sum(left_tokens.values()) + sum(right_tokens.values()))
        if left_tokens and right_tokens else 0.0
    )
    containment = float(
        bool(normalized_left and normalized_right)
        and (normalized_left in normalized_right or normalized_right in normalized_left)
    )
    return round(0.55 * ratio + 0.35 * token_f1 + 0.10 * containment, 6)


def _section_path_at(document: NormalizedDocument, block_index: int) -> list[str]:
    result: list[str] = []
    for start, end, path in _sections(document):
        if start <= block_index < end and len(path) >= len(result):
            result = list(path)
    return result


def _source_units(document: NormalizedDocument) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for block_index, block in enumerate(document.blocks):
        candidates = [sentence.text for sentence in block.sentences]
        candidates.extend(line.strip() for line in block.text.splitlines() if line.strip())
        if not candidates or len(block.text) <= 600:
            candidates.append(block.text)
        cursor_by_text: dict[str, int] = defaultdict(int)
        for text in candidates:
            start = block.text.find(text, cursor_by_text[text])
            if start < 0:
                start = block.text.find(text)
            if start < 0:
                continue
            end = start + len(text)
            cursor_by_text[text] = end
            key = (block_index, start, end)
            if key in seen:
                continue
            seen.add(key)
            units.append({
                "text": text, "block_index": block_index,
                "char_start": start, "char_end": end,
                "section_path": _section_path_at(document, block_index),
            })
    return units


def _context_excerpt(value: str | None, limit: int = 500) -> str | None:
    if value is None:
        return None
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def find_text_candidates(
    document: NormalizedDocument, authored_text: str, *, limit: int = 3,
) -> list[dict[str, Any]]:
    """Rank source-exact units lexically; candidates never become gate truth."""

    ranked = []
    for unit in _source_units(document):
        score = _lexical_score(authored_text, unit["text"])
        if score <= 0:
            continue
        block_index = unit["block_index"]
        ranked.append({
            **unit, "score": score,
            "previous_block": _context_excerpt(
                document.blocks[block_index - 1].text if block_index > 0 else None
            ),
            "next_block": _context_excerpt(
                document.blocks[block_index + 1].text
                if block_index + 1 < len(document.blocks) else None
            ),
        })
    ranked.sort(key=lambda item: (
        -item["score"], item["block_index"], item["char_start"], item["text"],
    ))
    return ranked[:limit]


def find_section_candidates(
    document: NormalizedDocument, authored_path: list[str],
    evidence_sentences: list[str], *, limit: int = 5,
) -> list[dict[str, Any]]:
    """Rank current canonical paths and report evidence-local disambiguation."""

    authored = " > ".join(authored_path)
    ranked: list[dict[str, Any]] = []
    for start, end, path in _sections(document):
        canonical = list(path)
        path_score = max(
            _lexical_score(authored, " > ".join(canonical)),
            max((_lexical_score(item, part) for item in authored_path for part in canonical), default=0.0),
        )
        section_text = "\n\n".join(
            document.blocks[index].text for index in range(start, end)
        )
        evidence_hits = sum(
            bool(_normal(sentence) and _normal(sentence) in _normal(section_text))
            for sentence in evidence_sentences
        )
        ranked.append({
            "section_path": canonical, "heading_block_index": start,
            "section_block_end": end, "path_score": round(path_score, 6),
            "evidence_sentence_hits": evidence_hits,
        })
    ranked.sort(key=lambda item: (
        -item["evidence_sentence_hits"], -item["path_score"],
        item["heading_block_index"], item["section_path"],
    ))
    return ranked[:limit]


def _candidate_confidence(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "none"
    first = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    if first["score"] >= 0.95 and first["score"] - second_score >= 0.03:
        return "high"
    if first["score"] >= 0.68 and first["score"] - second_score >= 0.03:
        return "medium"
    return "none"


def _normalization_candidate_confidence(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "none"
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    return (
        "high" if candidates[0]["score"] >= 0.88
        and candidates[0]["score"] - second_score >= 0.03 else "none"
    )


def _section_confidence(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "none"
    first = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    if first["evidence_sentence_hits"] > 0 and (
        second is None or first["evidence_sentence_hits"] > second["evidence_sentence_hits"]
    ):
        return "high"
    second_score = second["path_score"] if second else 0.0
    if first["path_score"] >= 0.72 and first["path_score"] - second_score >= 0.08:
        return "medium"
    return "none"


def _classify_case(
    failures: list[dict[str, Any]], text_candidates: dict[int, list[dict[str, Any]]],
    section_candidates: list[dict[str, Any]],
) -> tuple[str, str, str, str]:
    roots = {item.get("root_cause") for item in failures}
    if "ambiguous_section_mapping" in roots:
        if _section_confidence(section_candidates) == "high":
            return (
                "dataset_section_path_error", "high",
                "Replace the abbreviated section path with the uniquely evidence-supported full canonical path after human review.",
                "P1",
            )
        return (
            "ambiguous_source_content", "none",
            "Review every duplicate canonical section and select the intended full path.", "P4",
        )
    if "section_path_mismatch" in roots:
        confidence = _section_confidence(section_candidates)
        if confidence != "none":
            return (
                "dataset_section_path_error", confidence,
                "Review the ranked renamed/relocated canonical section path.",
                "P1" if confidence == "high" else "P3",
            )
        return (
            "dataset_from_different_corpus_version", "none",
            "Confirm the source snapshot or manually locate the retired section.", "P3",
        )
    confidences = [_candidate_confidence(value) for value in text_candidates.values()]
    if roots == {"evidence_text_normalization_issue"} and text_candidates and all(
        _normalization_candidate_confidence(value) == "high"
        for value in text_candidates.values()
    ):
        # Punctuation is source content. Ignoring it globally could merge distinct
        # occurrences, so this is a dataset correction rather than a resolver fix.
        return (
            "dataset_evidence_not_source_exact", "high",
            "Replace authored punctuation/formatting drift with the exact canonical source text.", "P1",
        )
    if confidences and all(value in {"high", "medium"} for value in confidences):
        return (
            "dataset_evidence_paraphrase", "medium",
            "Compare the authored statement with the lexical source candidate and verify semantics manually.", "P3",
        )
    return (
        "dataset_from_different_corpus_version", "none",
        "Identify the dataset's source revision or re-author evidence from the canonical corpus.", "P3",
    )


def _answer_support(text_candidates: dict[int, list[dict[str, Any]]]) -> str:
    if not text_candidates:
        return "support_unclear"
    scores = [values[0]["score"] if values else 0.0 for values in text_candidates.values()]
    if scores and min(scores) >= 0.80:
        return "support_likely_unchanged"
    if scores and max(scores) < 0.35:
        return "possible_answer_drift"
    return "support_unclear"


def _proposal(
    *, question_id: str, evidence_id: str, field: str,
    current_value: Any, proposed_value: Any, classification: str,
    confidence: str, reason: str, sentence_index: int | None = None,
    source_doc_id: str | None = None, candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "question_id": question_id, "evidence_id": evidence_id,
        "field": field, "current_value": current_value,
        "proposed_value": proposed_value, "classification": classification,
        "confidence": confidence, "reason": reason,
        "human_review_required": True, "auto_apply": False,
    }
    if sentence_index is not None:
        result["sentence_index"] = sentence_index
    if source_doc_id is not None:
        result["source_doc_id"] = source_doc_id
    if candidate is not None and "block_index" in candidate:
        result["source_span"] = {
            "block_index": candidate["block_index"],
            "char_start": candidate["char_start"], "char_end": candidate["char_end"],
        }
        result["source_section_path"] = candidate["section_path"]
    return result


def _validate_compatibility_lineage(
    *, dataset: QADataset, documents: list[NormalizedDocument],
    compatibility_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    manifest = _read_json_object(compatibility_directory / "manifest.json")
    report = _read_json_object(compatibility_directory / "compatibility_report.json")
    unresolved = _read_jsonl(compatibility_directory / "unresolved_cases.jsonl")
    if manifest.get("complete") is not True or manifest.get("schema_version") != COMPATIBILITY_SCHEMA_VERSION:
        raise ValueError("compatibility manifest is incomplete or incompatible")
    if manifest.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("compatibility audit dataset fingerprint is stale")
    expected_report = canonical_fingerprint({
        key: value for key, value in report.items() if key != "compatibility_fingerprint"
    })
    if report.get("compatibility_fingerprint") != expected_report:
        raise ValueError("compatibility report fingerprint mismatch")
    if manifest.get("compatibility_fingerprint") != expected_report:
        raise ValueError("compatibility manifest/report fingerprint mismatch")
    if manifest.get("unresolved_cases_fingerprint") != canonical_fingerprint(unresolved):
        raise ValueError("unresolved work queue fingerprint mismatch")
    corpus_fingerprint = canonical_fingerprint({
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "documents": [item.to_dict() for item in sorted(documents, key=lambda value: value.doc_id)],
    })
    if manifest.get("corpus_fingerprint") != corpus_fingerprint:
        raise ValueError("compatibility audit corpus fingerprint is stale")
    return manifest, report, unresolved, canonical_fingerprint(manifest)


def _breakdown(cases: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"questions": set(), "evidence_items": set()},
    )
    for case in cases:
        value = case["question_metadata"][field]
        result[str(value)]["questions"].add(case["question_id"])
        result[str(value)]["evidence_items"].add(case["evidence_id"])
    return {
        key: {
            "questions": len(value["questions"]),
            "evidence_items": len(value["evidence_items"]),
        }
        for key, value in sorted(result.items())
    }


def _report(stats: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    lines = [
        "# Dataset–Corpus Reconciliation Report", "",
        "## Executive summary", "",
        f"- Questions reviewed: {stats['questions_reviewed']}",
        f"- Evidence items affected: {stats['evidence_items_affected']}",
        f"- Evidence items unmapped to chunks: {stats['unmapped_evidence_items']}",
        f"- Unresolved evidence sentence occurrences: {stats['evidence_sentences_affected']}",
        f"- Compatibility failure rows: {stats['failure_rows']}",
        f"- High-confidence proposals: {stats['proposal_counts']['high']}",
        f"- Medium-confidence proposals: {stats['proposal_counts']['medium']}",
        f"- No-safe-correction evidence cases: {stats['no_safe_correction_cases']}",
        f"- Question semantic reviews required: {stats['question_semantics_review_required']}", "",
        "The exact source revision used to author the dataset is not recorded. The clustering of retired headings, punctuation drift, paraphrases, and absent source-exact statements is consistent with authorship against another documentation snapshot and/or non-source-exact curation.", "",
        "## Resolved pipeline finding", "",
        "One section resolver defect was found and fixed with a regression test: an exact full path was previously pooled with suffix matches. Exact full paths now win before suffix search. This legitimately changed compatibility from 92 to 94 questions and reduced the current failure queue from 68 to 65 rows.", "",
        "## Root-cause classification", "",
        "| Classification | Questions | Evidence items |",
        "| --- | ---: | ---: |",
    ]
    for name, value in stats["classification_counts"].items():
        lines.append(f"| {name} | {value['questions']} | {value['evidence_items']} |")
    lines.extend(["", "## Systematic patterns", ""])
    lines.append("Failure rows by compatibility root cause:")
    lines.append("")
    lines.extend(
        f"- {name}: {value['failure_rows']} rows, {value['evidence_items']} evidence items, {value['questions']} questions"
        for name, value in stats["compatibility_failure_counts"].items()
    )
    lines.extend(["", "Most affected documents:", ""])
    lines.extend(
        f"- `{name}`: {count} affected evidence items"
        for name, count in sorted(
            stats["document_failure_counts"].items(), key=lambda item: (-item[1], item[0]),
        )[:10]
    )
    lines.extend(["", "Repeated authored section paths:", ""])
    lines.extend(
        f"- `{name}`: {count} affected evidence items"
        for name, count in sorted(
            stats["authored_section_failure_counts"].items(), key=lambda item: (-item[1], item[0]),
        )
        if count > 1
    )
    lines.extend(["", "## Projected compatibility", ""])
    lines.extend([
        f"- Current: {stats['current_compatible_questions']} / {stats['total_questions']}",
        f"- PROJECTED — NOT ACTUAL GATE RESULT after all high-confidence proposals: {stats['projected_compatible_questions']} / {stats['total_questions']}",
        f"- Still requiring human review: {stats['projected_remaining_questions']}", "",
        "## Incompatible cross-document questions", "",
    ])
    for item in stats["cross_document_questions"]:
        evidence_summary = ", ".join(
            f"{evidence['evidence_id']}@{evidence['doc_id']}={evidence['compatibility_status']}"
            for evidence in item["evidence_items"]
        )
        lines.append(
            f"- `{item['question_id']}`: {item['failing_evidence_items']} failing evidence item(s); "
            f"documents resolved {item['resolved_documents']}/{item['required_documents']}; "
            f"evidence: {evidence_summary}; action: {item['recommended_action']}"
        )
    lines.extend(["", "## Per-question review queue", ""])
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_question[case["question_id"]].append(case)
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}
    for question_id in sorted(by_question, key=lambda value: (
        min(priority_order[item["review_priority"]] for item in by_question[value]), value,
    )):
        question_cases = by_question[question_id]
        lines.extend([f"### {question_id}", "", f"{question_cases[0]['question']}", ""])
        for case in question_cases:
            failures = ", ".join(
                f"{item['failure_stage']}"
                + (f"[{item['evidence_sentence_index']}]" if item['evidence_sentence_index'] is not None else "")
                for item in case["compatibility_failures"]
            )
            lines.append(
                f"- `{case['evidence_id']}` — {case['classification']} ({case['confidence']}); "
                f"failures: {failures}; answer support: {case['reference_answer_support']}; "
                f"action: {case['recommended_action']}"
            )
            lines.append(f"  - Canonical document: `{case['corpus']['canonical_doc_id']}`")
            if case["corpus"]["candidate_sections"]:
                candidate = case["corpus"]["candidate_sections"][0]
                lines.append(
                    f"  - Authored path: `{case['dataset']['section_path']}`; top current path: "
                    f"`{candidate['section_path']}` (path score {candidate['path_score']}, "
                    f"evidence hits {candidate['evidence_sentence_hits']})"
                )
            for source in case["corpus"]["candidate_source_spans"]:
                sentence_index = source["sentence_index"]
                authored_value = case["dataset"]["evidence_sentences"][sentence_index]
                authored = (
                    authored_value["text"] if isinstance(authored_value, dict)
                    else authored_value
                )
                if source["candidates"]:
                    candidate = source["candidates"][0]
                    lines.append(
                        f"  - Sentence {sentence_index}: authored `{authored}`; top source candidate "
                        f"`{candidate['text']}` (score {candidate['score']}, block {candidate['block_index']}, "
                        f"section `{candidate['section_path']}`)"
                    )
                else:
                    lines.append(f"  - Sentence {sentence_index}: no lexical source candidate for `{authored}`")
        lines.append("")
    lines.extend([
        "## Safety", "",
        "Every proposal requires human review and has `auto_apply: false`. No dataset, compatibility gate, retrieval behavior, or benchmark artifact was modified by this reconciliation run.", "",
        "BENCHMARK GATE: FAIL", "",
    ])
    return "\n".join(lines)


def reconcile_dataset(
    *, dataset_path: Path, documents: list[NormalizedDocument],
    compatibility_directory: Path, output_directory: Path,
) -> ReconciliationResult:
    """Build a deterministic proposal package from the authoritative failure queue."""

    dataset = load_team_qa_dataset(dataset_path)
    manifest, compatibility_report, failures, manifest_fingerprint = _validate_compatibility_lineage(
        dataset=dataset, documents=documents,
        compatibility_directory=compatibility_directory,
    )
    records = {item.id: item for item in dataset.records}
    documents_by_id = {item.doc_id: item for item in documents}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for failure in failures:
        key = (failure.get("question_id"), failure.get("evidence_id"))
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError("reconciliation requires question/evidence-scoped failure rows")
        grouped[key].append(failure)

    cases: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    for question_id, evidence_id in sorted(grouped):
        record = records.get(question_id)
        if record is None:
            raise ValueError(f"unresolved queue references unknown question {question_id}")
        evidence = next((item for item in record.evidence if item.evidence_id == evidence_id), None)
        if evidence is None:
            raise ValueError(f"unresolved queue references unknown evidence {evidence_id}")
        document = documents_by_id.get(evidence.doc_id)
        if document is None:
            raise ValueError(f"canonical document disappeared for {evidence_id}")
        case_failures = sorted(grouped[(question_id, evidence_id)], key=lambda item: (
            item["failure_stage"],
            -1 if item["evidence_sentence_index"] is None else item["evidence_sentence_index"],
        ))
        failed_sentence_indexes = sorted({
            item["evidence_sentence_index"] for item in case_failures
            if item["evidence_sentence_index"] is not None
        })
        text_candidates = {
            index: find_text_candidates(document, evidence.evidence_sentences[index].text)
            for index in failed_sentence_indexes
        }
        has_section_failure = any(
            item["failure_stage"] == "section_resolution" for item in case_failures
        )
        section_candidates = (
            find_section_candidates(
                document, evidence.section_path,
                [item.text for item in evidence.evidence_sentences],
            ) if has_section_failure else []
        )
        classification, confidence, action, priority = _classify_case(
            case_failures, text_candidates, section_candidates,
        )
        if record.metadata["evidence_scope"] == "cross_document" and priority not in {"P0", "P1"}:
            priority = "P2"
        support = _answer_support(text_candidates)
        case = {
            "question_id": question_id, "evidence_id": evidence_id,
            "question": record.question, "reference_answer": record.answer,
            "question_metadata": {
                "difficulty": record.difficulty, "question_type": record.question_type,
                "reasoning_type": record.reasoning_type,
                "evidence_scope": record.metadata["evidence_scope"],
            },
            "dataset": evidence.to_dict(),
            "compatibility_failures": case_failures,
            "corpus": {
                "canonical_doc_id": document.doc_id,
                "resolved_source_path": document.relative_path,
                "document_fingerprint": document.source_sha256,
                "normalized_schema_version": document.schema_version,
                "parser_identity": document.metadata.get("parser"),
                "candidate_sections": section_candidates,
                "candidate_source_spans": [
                    {"sentence_index": index, "candidates": text_candidates[index]}
                    for index in failed_sentence_indexes
                ],
            },
            "classification": classification, "confidence": confidence,
            "recommended_action": action, "review_priority": priority,
            "reference_answer_support": support,
            "question_semantics_review_required": support != "support_likely_unchanged",
            "human_review_required": True,
        }
        cases.append(case)

        if has_section_failure:
            section_confidence = _section_confidence(section_candidates)
            if section_confidence != "none" and section_candidates[0]["section_path"] != evidence.section_path:
                proposals.append(_proposal(
                    question_id=question_id, evidence_id=evidence_id,
                    field="section_path", current_value=evidence.section_path,
                    proposed_value=section_candidates[0]["section_path"],
                    classification="dataset_section_path_error",
                    confidence=section_confidence,
                    reason="Highest-ranked unique canonical path from deterministic heading/evidence inspection.",
                    source_doc_id=document.doc_id,
                ))
        for index in failed_sentence_indexes:
            candidates = text_candidates[index]
            failure_root = next(
                item.get("root_cause") for item in case_failures
                if item.get("evidence_sentence_index") == index
            )
            candidate_confidence = (
                _normalization_candidate_confidence(candidates)
                if failure_root == "evidence_text_normalization_issue"
                else _candidate_confidence(candidates)
            )
            if candidate_confidence == "none":
                continue
            best = candidates[0]
            proposals.append(_proposal(
                question_id=question_id, evidence_id=evidence_id,
                sentence_index=index, field="evidence_sentences",
                current_value=evidence.evidence_sentences[index].text,
                proposed_value=best["text"], source_doc_id=document.doc_id,
                candidate=best, confidence=candidate_confidence,
                classification=(
                    "dataset_evidence_not_source_exact"
                    if candidate_confidence == "high" else "dataset_evidence_paraphrase"
                ),
                reason="Exact canonical source unit selected by deterministic lexical ranking; semantic acceptance remains human-reviewed.",
            ))

    proposals.sort(key=lambda item: (
        item["question_id"], item["evidence_id"], item.get("sentence_index", -1), item["field"],
    ))
    proposal_keys = {
        (item["question_id"], item["evidence_id"], item["field"], item.get("sentence_index"))
        for item in proposals if item["confidence"] == "high"
    }
    projected_fixed_questions: set[str] = set()
    for question_id in sorted({item["question_id"] for item in cases}):
        question_failures = [item for item in failures if item["question_id"] == question_id]
        all_covered = all(
            (
                (item["question_id"], item["evidence_id"], "section_path", None)
                if item["failure_stage"] == "section_resolution" else
                (item["question_id"], item["evidence_id"], "evidence_sentences", item["evidence_sentence_index"])
            ) in proposal_keys
            for item in question_failures
        )
        if all_covered:
            projected_fixed_questions.add(question_id)

    classification_sets: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"questions": set(), "evidence_items": set()},
    )
    for case in cases:
        group = classification_sets[case["classification"]]
        group["questions"].add(case["question_id"])
        group["evidence_items"].add(case["evidence_id"])
    classifications = {
        key: {"questions": len(value["questions"]), "evidence_items": len(value["evidence_items"])}
        for key, value in sorted(classification_sets.items())
    }
    incompatible_cross_document = sorted({
        case["question_id"] for case in cases
        if case["question_metadata"]["evidence_scope"] == "cross_document"
    })
    cross_document = []
    for question_id in incompatible_cross_document:
        record = records[question_id]
        question_cases = [case for case in cases if case["question_id"] == question_id]
        failed_ids = {case["evidence_id"] for case in question_cases}
        cross_document.append({
            "question_id": question_id, "required_documents": len({item.doc_id for item in record.evidence}),
            "resolved_documents": len({item.doc_id for item in record.evidence if item.doc_id in documents_by_id}),
            "required_evidence_items": len(record.evidence),
            "failing_evidence_items": len(question_cases),
            "failed_evidence_ids": sorted(failed_ids),
            "evidence_items": [
                {
                    "evidence_id": item.evidence_id, "doc_id": item.doc_id,
                    "document_resolved": item.doc_id in documents_by_id,
                    "compatibility_status": "failed" if item.evidence_id in failed_ids else "resolved",
                }
                for item in record.evidence
            ],
            "one_failure_invalidates_question": True,
            "recommended_action": "; ".join(sorted({case["recommended_action"] for case in question_cases})),
        })
    stats: dict[str, Any] = {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "questions_reviewed": len({item["question_id"] for item in cases}),
        "evidence_items_affected": len(cases),
        "evidence_sentences_affected": len({
            (item["question_id"], item["evidence_id"], item["evidence_sentence_index"])
            for item in failures if item["evidence_sentence_index"] is not None
        }),
        "failure_rows": len(failures),
        "unmapped_evidence_items": next(iter(
            compatibility_report["chunk_mapping"].values()
        ))["unmapped_evidence"],
        "classification_counts": classifications,
        "compatibility_failure_counts": {
            root: {
                "failure_rows": sum(item.get("root_cause") == root for item in failures),
                "questions": len({
                    item["question_id"] for item in failures if item.get("root_cause") == root
                }),
                "evidence_items": len({
                    (item["question_id"], item["evidence_id"])
                    for item in failures if item.get("root_cause") == root
                }),
            }
            for root in sorted({item.get("root_cause") for item in failures})
        },
        "proposal_counts": {
            confidence: sum(item["confidence"] == confidence for item in proposals)
            for confidence in ("high", "medium")
        },
        "no_safe_correction_cases": sum(
            not any(
                proposal["question_id"] == case["question_id"]
                and proposal["evidence_id"] == case["evidence_id"]
                for proposal in proposals
            ) for case in cases
        ),
        "question_semantics_review_required": len({
            item["question_id"] for item in cases if item["question_semantics_review_required"]
        }),
        "current_compatible_questions": compatibility_report["compatible_question_count"],
        "total_questions": compatibility_report["question_count"],
        "projected_compatible_questions": compatibility_report["compatible_question_count"] + len(projected_fixed_questions),
        "projected_remaining_questions": compatibility_report["incompatible_question_count"] - len(projected_fixed_questions),
        "projected_newly_compatible_question_ids": sorted(projected_fixed_questions),
        "projection_label": "PROJECTED — NOT ACTUAL GATE RESULT",
        "cross_document_questions": cross_document,
        "breakdown": {
            field: _breakdown(cases, field)
            for field in ("difficulty", "question_type", "reasoning_type", "evidence_scope")
        },
        "document_failure_counts": dict(sorted(Counter(
            case["dataset"]["doc_id"] for case in cases
        ).items())),
        "authored_section_failure_counts": dict(sorted(Counter(
            " > ".join(case["dataset"]["section_path"])
            for case in cases
            if any(
                item["failure_stage"] == "section_resolution"
                for item in case["compatibility_failures"]
            )
        ).items())),
        "reference_answer_support_counts": dict(sorted(Counter(
            case["reference_answer_support"] for case in cases
        ).items())),
        "review_priority_counts": dict(sorted(Counter(
            case["review_priority"] for case in cases
        ).items())),
        "resolver_bug_found": True,
        "resolved_pipeline_findings": 1,
        "resolver_fix": {
            "issue": "exact full section path was combined with suffix candidates",
            "fix": "exact full paths now take precedence over suffix matching",
            "compatibility_effect": "two additional questions became compatible; one mixed-failure question lost its section ambiguity",
        },
        "corpus_revision": {
            "exact_upstream_revision": None,
            "available_identity": "per-document source_sha256 plus processed corpus fingerprint",
            "dataset_source_revision": None,
        },
    }
    identity = {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "algorithm_version": RECONCILIATION_ALGORITHM_VERSION,
        "dataset_fingerprint": dataset.fingerprint,
        "compatibility_manifest_fingerprint": manifest_fingerprint,
        "compatibility_fingerprint": manifest["compatibility_fingerprint"],
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "document_fingerprints": {
            case["corpus"]["canonical_doc_id"]: case["corpus"]["document_fingerprint"]
            for case in cases
        },
        "cases_fingerprint": canonical_fingerprint(cases),
        "proposals_fingerprint": canonical_fingerprint(proposals),
        "stats_fingerprint": canonical_fingerprint(stats),
    }
    reconciliation_fingerprint = canonical_fingerprint(identity)
    output_manifest = {
        **identity, "reconciliation_fingerprint": reconciliation_fingerprint,
        "complete": True, "case_count": len(cases), "proposal_count": len(proposals),
        "output_artifacts": [
            "reconciliation_cases.jsonl", "proposed_corrections.jsonl",
            "stats.json", "RECONCILIATION_REPORT.md", "manifest.json",
        ],
    }
    write_artifact_set(output_directory, {
        "reconciliation_cases.jsonl": _jsonl(cases),
        "proposed_corrections.jsonl": _jsonl(proposals),
        "stats.json": serialize_json(stats),
        "RECONCILIATION_REPORT.md": _report(stats, cases),
        "manifest.json": serialize_json(output_manifest),
    })
    return ReconciliationResult(
        output_directory, reconciliation_fingerprint,
        tuple(cases), tuple(proposals), stats,
    )
