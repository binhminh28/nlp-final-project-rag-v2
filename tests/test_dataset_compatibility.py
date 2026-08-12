"""Offline tests for the real team dataset adapter and strict compatibility gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.tokenizer import TiktokenTokenizer
from rag_chunking.data.models import DocumentBlock, NormalizedDocument
from rag_chunking.evaluation.compatibility import (
    audit_dataset_compatibility, resolve_document, resolve_section, resolve_sentence,
)
from rag_chunking.evaluation.qa_dataset import adapt_team_qa_record, load_team_qa_dataset


STRATEGIES = ("fixed_size", "structure_aware", "prompt_based")


def team_record(**changes):
    value = {
        "question_id": "q1", "question": "What is alpha?",
        "reference_answer": "Alpha.", "difficulty": "easy",
        "question_type": "definition", "reasoning_type": "single_fact",
        "evidence": [{
            "evidence_id": "q1_e1", "doc_id": "angular:a.md",
            "section_path": ["Alpha `API`"],
            "evidence_sentences": ["Alpha  is evidence."],
        }],
        "metadata": {
            "evidence_scope": "single_section", "num_evidence": 1,
            "semantic_competition": False, "content_type": "prose",
            "answerable_from_evidence": True,
        },
    }
    value.update(changes)
    return value


def document(doc_id="angular:a.md", heading="Alpha `API`", text="Alpha is evidence."):
    relative = doc_id.split(":", 1)[-1]
    return NormalizedDocument(
        doc_id=doc_id, source="angular", relative_path=relative,
        filename=Path(relative).name, source_sha256="fixture", blocks=[
            DocumentBlock(type="heading", text=heading, level=1),
            DocumentBlock(type="paragraph", text=text),
        ],
    )


def chunks_for(documents, *, omit=None):
    tokenizer = TiktokenTokenizer()
    result = {}
    for strategy in STRATEGIES:
        result[strategy] = []
        for doc in documents:
            if omit == (strategy, doc.doc_id):
                continue
            for index, block in enumerate(doc.blocks):
                result[strategy].append(Chunk(
                    chunk_id=f"{doc.doc_id}::{strategy}::{index:06d}",
                    strategy=strategy, doc_id=doc.doc_id, source=doc.source,
                    relative_path=doc.relative_path, chunk_index=index,
                    text=block.text, token_start=None, token_end=None,
                    token_count=len(tokenizer.encode(block.text)), chunk_size=512,
                    chunk_overlap=0, tokenizer=tokenizer.name,
                    title_path=[doc.blocks[0].text], metadata={"block_fragments": [{
                        "source_block_index": index, "char_start": 0,
                        "char_end": len(block.text),
                    }]},
                ))
    return result


def manifests(chunks, documents):
    return {
        strategy: {
            "strategy": strategy, "chunks": len(values),
            "documents": len(documents),
            "source_schema_version": "normalized_document_v2",
        }
        for strategy, values in chunks.items()
    }


def write_dataset(path, records):
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return path


def test_adapter_preserves_multi_evidence_cross_document_and_fingerprint(tmp_path: Path):
    second = dict(team_record()["evidence"][0])
    second.update(evidence_id="q1_e2", doc_id="angular:b.md")
    raw = team_record(
        evidence=[team_record()["evidence"][0], second],
        metadata={**team_record()["metadata"], "evidence_scope": "cross_document", "num_evidence": 2},
    )
    record = adapt_team_qa_record(raw)
    assert [item.doc_id for item in record.evidence] == ["angular:a.md", "angular:b.md"]
    assert record.answer == "Alpha." and record.reasoning_type == "single_fact"
    one = load_team_qa_dataset(write_dataset(tmp_path / "one.jsonl", [raw]))
    two = load_team_qa_dataset(write_dataset(tmp_path / "two.jsonl", [dict(reversed(list(raw.items())))]))
    assert one.fingerprint == two.fingerprint


@pytest.mark.parametrize("change,match", [
    ({"question_id": ""}, "id must"), ({"question": ""}, "empty"),
    ({"reference_answer": ""}, "answer must"), ({"evidence": ["bad"]}, "objects"),
    ({"evidence": [{"evidence_id": "e", "doc_id": "d", "section_path": ["s"], "evidence_sentences": []}]}, "non-empty"),
])
def test_adapter_rejects_missing_or_invalid_required_values(change, match):
    with pytest.raises(ValueError, match=match):
        adapt_team_qa_record(team_record(**change))


def test_adapter_rejects_metadata_count_mismatch_and_unknown_fields():
    with pytest.raises(ValueError, match="num_evidence"):
        adapt_team_qa_record(team_record(metadata={**team_record()["metadata"], "num_evidence": 2}))
    with pytest.raises(ValueError, match="unknown team QA"):
        adapt_team_qa_record({**team_record(), "surprise": True})


def test_document_resolution_exact_namespace_unknown_ambiguous_and_no_filename_fallback():
    a = document("angular:path/a.md")
    assert resolve_document(a.doc_id, [a]).method == "exact_doc_id"
    assert resolve_document("path/a.md", [a]).method == "add_corpus_namespace"
    assert resolve_document("missing.md", [a]).status == "unresolved"
    assert resolve_document(a.doc_id, [a, a]).status == "ambiguous"
    collision = document("angular:other/a.md")
    assert resolve_document("a.md", [a, collision]).status == "unresolved"


def test_section_resolution_exact_normalized_missing_and_duplicate():
    doc = NormalizedDocument(
        doc_id="angular:a.md", source="angular", relative_path="a.md", filename="a.md",
        source_sha256="fixture", blocks=[
            DocumentBlock(type="heading", text="Root", level=1),
            DocumentBlock(type="heading", text="Nested `API` {#api}", level=2),
            DocumentBlock(type="paragraph", text="fact"),
        ],
    )
    assert resolve_section(doc, ["Root", "Nested `API` {#api}"]).method == "exact_path"
    assert resolve_section(doc, ["Nested API"]).method == "normalized_heading_suffix"
    assert resolve_section(doc, ["Missing"]).status == "unresolved"
    duplicate = document(heading="Same")
    duplicate.blocks.extend([DocumentBlock(type="heading", text="Same", level=1)])
    assert resolve_section(duplicate, ["Same"]).status == "ambiguous"


def test_exact_full_section_path_wins_over_same_named_descendant():
    doc = NormalizedDocument(
        doc_id="angular:a.md", source="angular", relative_path="a.md", filename="a.md",
        source_sha256="fixture", blocks=[
            DocumentBlock(type="heading", text="Interceptors", level=1),
            DocumentBlock(type="paragraph", text="Root evidence."),
            DocumentBlock(type="heading", text="Interceptors", level=2),
            DocumentBlock(type="paragraph", text="Nested evidence."),
        ],
    )
    resolved = resolve_section(doc, ["Interceptors"])
    assert resolved.status == "resolved"
    assert resolved.method == "exact_path"
    assert resolved.block_start == 0


def test_sentence_resolution_exact_normalized_missing_and_section_disambiguation():
    doc = NormalizedDocument(
        doc_id="angular:a.md", source="angular", relative_path="a.md", filename="a.md",
        source_sha256="fixture", blocks=[
            DocumentBlock(type="heading", text="One", level=1),
            DocumentBlock(type="paragraph", text="Repeated fact."),
            DocumentBlock(type="heading", text="Two", level=1),
            DocumentBlock(type="paragraph", text="Repeated   fact."),
        ],
    )
    one = resolve_section(doc, ["One"])
    two = resolve_section(doc, ["Two"])
    assert resolve_sentence(doc, "Repeated fact.", one).status == "exact_match"
    assert resolve_sentence(doc, "Repeated fact.", two).status == "normalized_exact_match"
    assert resolve_sentence(doc, "Missing", one).status == "unresolved"
    doc.blocks[3].text = "Repeated fact."
    unresolved_scope = resolve_section(doc, ["Missing"])
    assert resolve_sentence(doc, "Repeated fact.", unresolved_scope).status == "ambiguous"


def test_full_cross_document_gate_passes_and_writes_deterministic_artifacts(tmp_path: Path):
    docs = [document(), document("angular:b.md", "Beta", "Beta is evidence.")]
    evidence = [team_record()["evidence"][0], {
        "evidence_id": "q1_e2", "doc_id": "angular:b.md",
        "section_path": ["Beta"], "evidence_sentences": ["Beta is evidence."],
    }]
    raw = team_record(
        evidence=evidence,
        metadata={**team_record()["metadata"], "evidence_scope": "cross_document", "num_evidence": 2},
    )
    path = write_dataset(tmp_path / "qa.jsonl", [raw])
    chunks = chunks_for(docs)
    output = tmp_path / "compatibility"
    first = audit_dataset_compatibility(
        dataset_path=path, documents=docs, chunks_by_strategy=chunks,
        chunk_manifests=manifests(chunks, docs), output_directory=output,
    )
    assert first.passed and first.report["compatible_question_count"] == 1
    assert all(item["mapped_evidence"] == 2 for item in first.report["chunk_mapping"].values())
    before = {item.name: item.read_bytes() for item in output.iterdir() if item.is_file()}
    second = audit_dataset_compatibility(
        dataset_path=path, documents=docs, chunks_by_strategy=chunks,
        chunk_manifests=manifests(chunks, docs), output_directory=output,
    )
    assert second.passed
    assert before == {item.name: item.read_bytes() for item in output.iterdir() if item.is_file()}


def test_gate_fails_for_unknown_document_missing_sentence_and_one_strategy_gap(tmp_path: Path):
    doc = document()
    cases = [
        team_record(evidence=[{**team_record()["evidence"][0], "doc_id": "angular:missing.md"}]),
        team_record(evidence=[{**team_record()["evidence"][0], "evidence_sentences": ["Absent."]}]),
    ]
    for index, raw in enumerate(cases):
        raw["question_id"] = f"q{index}"
        raw["evidence"][0]["evidence_id"] = f"q{index}_e1"
        path = write_dataset(tmp_path / f"bad-{index}.jsonl", [raw])
        chunks = chunks_for([doc])
        result = audit_dataset_compatibility(
            dataset_path=path, documents=[doc], chunks_by_strategy=chunks,
            chunk_manifests=manifests(chunks, [doc]),
        )
        assert not result.passed
    path = write_dataset(tmp_path / "gap.jsonl", [team_record()])
    chunks = chunks_for([doc], omit=("prompt_based", doc.doc_id))
    result = audit_dataset_compatibility(
        dataset_path=path, documents=[doc], chunks_by_strategy=chunks,
        chunk_manifests=None,
    )
    assert not result.passed
    assert result.report["chunk_mapping"]["prompt_based"]["unmapped_evidence"] == 1


def test_gate_fails_corrupt_manifest_lineage(tmp_path: Path):
    doc = document()
    path = write_dataset(tmp_path / "qa.jsonl", [team_record()])
    chunks = chunks_for([doc])
    broken = manifests(chunks, [doc])
    broken["fixed_size"]["chunks"] += 1
    result = audit_dataset_compatibility(
        dataset_path=path, documents=[doc], chunks_by_strategy=chunks,
        chunk_manifests=broken,
    )
    assert not result.passed
    assert result.report["chunk_mapping"]["fixed_size"]["artifact_errors"]
