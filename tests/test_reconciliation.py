"""Deterministic reconciliation candidates remain separate from benchmark truth."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.data.models import DocumentBlock, NormalizedDocument
from rag_chunking.data.writer import write_processed_corpus
from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.evaluation.compatibility import COMPATIBILITY_SCHEMA_VERSION
from rag_chunking.evaluation.qa_dataset import load_team_qa_dataset
from rag_chunking.evaluation.reconciliation import (
    _classify_case, find_section_candidates, find_text_candidates, reconcile_dataset,
)


def document() -> NormalizedDocument:
    return NormalizedDocument(
        doc_id="angular:guide.md", source="angular", relative_path="guide.md",
        filename="guide.md", source_sha256="source-fixture", blocks=[
            DocumentBlock(type="heading", text="Current routing", level=1),
            DocumentBlock(type="paragraph", text="Angular loads this route lazily:"),
            DocumentBlock(type="heading", text="Duplicate", level=2),
            DocumentBlock(type="paragraph", text="First duplicate fact."),
            DocumentBlock(type="heading", text="Duplicate", level=2),
            DocumentBlock(type="paragraph", text="Second duplicate fact."),
        ], metadata={"parser": "fixture-parser"},
    )


def record() -> dict:
    return {
        "question_id": "q1", "question": "How is the route loaded?",
        "reference_answer": "It is loaded lazily.", "difficulty": "hard",
        "question_type": "behavior", "reasoning_type": "multi_fact_synthesis",
        "evidence": [
            {
                "evidence_id": "q1_e1", "doc_id": "angular:guide.md",
                "section_path": ["Old routing"],
                "evidence_sentences": ["Angular loads this route lazily."],
            },
            {
                "evidence_id": "q1_e2", "doc_id": "angular:guide.md",
                "section_path": ["Duplicate"],
                "evidence_sentences": ["Second duplicate fact."],
            },
        ],
        "metadata": {
            "evidence_scope": "cross_section", "num_evidence": 2,
            "semantic_competition": False, "content_type": "prose",
            "answerable_from_evidence": True,
        },
    }


def write_fixture(root: Path):
    dataset_path = root / "qa.jsonl"
    dataset_path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    doc = document()
    processed = root / "processed"
    write_processed_corpus([doc], processed)
    dataset = load_team_qa_dataset(dataset_path)
    failures = [
        {
            "question_id": "q1", "evidence_id": "q1_e1",
            "evidence_sentence_index": None, "original_doc_id": doc.doc_id,
            "canonical_doc_candidate": doc.doc_id, "section_path": ["Old routing"],
            "failure_stage": "section_resolution", "failure_reason": "missing",
            "strategy": None, "candidate_chunk_ids": [], "candidates": [],
            "root_cause": "section_path_mismatch",
        },
        {
            "question_id": "q1", "evidence_id": "q1_e1",
            "evidence_sentence_index": 0, "original_doc_id": doc.doc_id,
            "canonical_doc_candidate": doc.doc_id, "section_path": ["Old routing"],
            "failure_stage": "evidence_sentence_resolution", "failure_reason": "punctuation",
            "strategy": None, "candidate_chunk_ids": [], "candidates": [],
            "root_cause": "evidence_text_normalization_issue",
        },
        {
            "question_id": "q1", "evidence_id": "q1_e2",
            "evidence_sentence_index": None, "original_doc_id": doc.doc_id,
            "canonical_doc_candidate": doc.doc_id, "section_path": ["Duplicate"],
            "failure_stage": "section_resolution", "failure_reason": "ambiguous",
            "strategy": None, "candidate_chunk_ids": [],
            "candidates": ["Current routing > Duplicate", "Current routing > Duplicate"],
            "root_cause": "ambiguous_section_mapping",
        },
    ]
    report = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "question_count": 1, "compatible_question_count": 0,
        "incompatible_question_count": 1,
        "chunk_mapping": {"fixed_size": {"unmapped_evidence": 2}},
    }
    report["compatibility_fingerprint"] = canonical_fingerprint(report)
    corpus_fingerprint = canonical_fingerprint({
        "schema_version": "normalized_document_v2", "documents": [doc.to_dict()],
    })
    manifest = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION, "complete": True,
        "dataset_fingerprint": dataset.fingerprint,
        "compatibility_fingerprint": report["compatibility_fingerprint"],
        "unresolved_cases_fingerprint": canonical_fingerprint(failures),
        "corpus_fingerprint": corpus_fingerprint,
    }
    compatibility = root / "compatibility"
    write_artifact_set(compatibility, {
        "compatibility_report.json": serialize_json(report),
        "unresolved_cases.jsonl": "".join(json.dumps(item, sort_keys=True) + "\n" for item in failures),
        "manifest.json": serialize_json(manifest),
    })
    return dataset_path, processed / "documents.jsonl", compatibility, doc


def test_candidate_section_search_ranked_renamed_duplicate_and_no_candidate():
    doc = document()
    renamed = find_section_candidates(doc, ["Old routing"], ["Angular loads this route lazily."])
    assert renamed[0]["section_path"] == ["Current routing"]
    duplicates = find_section_candidates(doc, ["Duplicate"], ["unknown"])
    assert sum(item["path_score"] == 1.0 for item in duplicates) == 2
    unrelated = find_section_candidates(doc, ["ZZZ absent"], ["unknown"])
    assert all(item["path_score"] < 0.5 for item in unrelated)


def test_candidate_text_search_exact_normalized_lexical_multiple_and_none():
    doc = document()
    exact = find_text_candidates(doc, "First duplicate fact.")
    assert exact[0]["text"] == "First duplicate fact." and exact[0]["score"] == 1.0
    normalized = find_text_candidates(doc, "Angular loads this route lazily.")
    assert normalized[0]["text"].endswith(":")
    lexical = find_text_candidates(doc, "The route is lazily loaded by Angular")
    assert "lazily" in lexical[0]["text"]
    multiple = find_text_candidates(doc, "duplicate fact")
    assert {item["block_index"] for item in multiple[:2]} == {3, 5}
    assert find_text_candidates(doc, "") == []


@pytest.mark.parametrize("failures,text_candidates,sections,expected", [
    (
        [{"root_cause": "section_path_mismatch"}], {},
        [{"section_path": ["Renamed"], "evidence_sentence_hits": 1, "path_score": 0.8}],
        "dataset_section_path_error",
    ),
    (
        [{"root_cause": "corpus_version_mismatch"}],
        {0: [{"score": 0.8}, {"score": 0.2}]}, [], "dataset_evidence_paraphrase",
    ),
    (
        [{"root_cause": "corpus_version_mismatch"}],
        {0: [{"score": 0.2}]}, [], "dataset_from_different_corpus_version",
    ),
    (
        [{"root_cause": "evidence_text_normalization_issue"}],
        {0: [{"score": 0.9}, {"score": 0.2}]}, [], "dataset_evidence_not_source_exact",
    ),
    (
        [{"root_cause": "ambiguous_section_mapping"}], {},
        [
            {"section_path": ["One"], "evidence_sentence_hits": 0, "path_score": 1.0},
            {"section_path": ["Two"], "evidence_sentence_hits": 0, "path_score": 1.0},
        ], "ambiguous_source_content",
    ),
])
def test_controlled_classification(failures, text_candidates, sections, expected):
    assert _classify_case(failures, text_candidates, sections)[0] == expected


def test_reconciliation_groups_rows_proposes_without_applying_and_is_deterministic(tmp_path: Path):
    dataset_path, documents_path, compatibility, doc = write_fixture(tmp_path)
    before = dataset_path.read_bytes()
    output = tmp_path / "reconciliation"
    first = reconcile_dataset(
        dataset_path=dataset_path, documents=[doc],
        compatibility_directory=compatibility, output_directory=output,
    )
    assert len(first.cases) == 2
    assert [item["evidence_id"] for item in first.cases] == ["q1_e1", "q1_e2"]
    assert len(first.cases[0]["compatibility_failures"]) == 2
    assert any(item["field"] == "section_path" for item in first.proposals)
    assert all(item["auto_apply"] is False and item["human_review_required"] for item in first.proposals)
    assert dataset_path.read_bytes() == before
    assert any(item["classification"] == "ambiguous_source_content" for item in first.cases)
    assert not any(
        item["evidence_id"] == "q1_e2" and item["field"] == "evidence_text"
        for item in first.proposals
    )
    first_bytes = {item.name: item.read_bytes() for item in output.iterdir()}
    second = reconcile_dataset(
        dataset_path=dataset_path, documents=[doc],
        compatibility_directory=compatibility, output_directory=output,
    )
    assert first.reconciliation_fingerprint == second.reconciliation_fingerprint
    assert first_bytes == {item.name: item.read_bytes() for item in output.iterdir()}


def test_reconciliation_rejects_stale_work_queue(tmp_path: Path):
    dataset_path, _, compatibility, doc = write_fixture(tmp_path)
    path = compatibility / "unresolved_cases.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="work queue fingerprint"):
        reconcile_dataset(
            dataset_path=dataset_path, documents=[doc],
            compatibility_directory=compatibility,
            output_directory=tmp_path / "output",
        )
