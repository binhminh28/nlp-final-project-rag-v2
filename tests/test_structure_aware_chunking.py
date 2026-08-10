from __future__ import annotations

import json
from pathlib import Path

from rag_chunking.chunking.structure_aware import (
    StructureAwareChunker,
    StructureAwareChunkingConfig,
    build_sections,
)
from rag_chunking.chunking.structure_statistics import structure_corpus_statistics
from rag_chunking.chunking.structure_validation import validate_structure_aware_chunks
from rag_chunking.chunking.structure_writer import write_structure_aware_artifacts
from rag_chunking.chunking.writer import read_chunks_jsonl
from rag_chunking.data.models import DocumentBlock, NormalizedDocument, Sentence


def document(blocks: list[DocumentBlock]) -> NormalizedDocument:
    return NormalizedDocument(
        doc_id="angular:test.md",
        source="angular",
        relative_path="test.md",
        filename="test.md",
        source_sha256="abc123",
        blocks=blocks,
    )


def paragraph(text: str) -> DocumentBlock:
    return DocumentBlock(type="paragraph", text=text, sentences=[Sentence("s", text)])


def exact_tokens(chunker: StructureAwareChunker, count: int, word: str = "a") -> str:
    text = f" {word}" * count
    assert len(chunker.tokenizer.encode(text)) == count
    return text


def test_heading_hierarchy_paths() -> None:
    doc = document(
        [
            DocumentBlock(type="heading", text="A", level=1), paragraph("intro"),
            DocumentBlock(type="heading", text="B", level=2), paragraph("text"),
            DocumentBlock(type="heading", text="C", level=3), paragraph("text"),
            DocumentBlock(type="heading", text="D", level=2), paragraph("text"),
        ]
    )
    assert [[heading.text for heading in section.path] for section in build_sections(doc)] == [
        ["A"], ["A", "B"], ["A", "B", "C"], ["A", "D"]
    ]
    assert [chunk.metadata["section_path"] for chunk in StructureAwareChunker().chunk(doc)] == [
        ["A"], ["A", "B"], ["A", "B", "C"], ["A", "D"]
    ]


def test_heading_level_jump_does_not_create_fake_heading() -> None:
    doc = document(
        [
            DocumentBlock(type="heading", text="A", level=1),
            DocumentBlock(type="heading", text="C", level=3), paragraph("text"),
            DocumentBlock(type="heading", text="B", level=2), paragraph("text"),
        ]
    )
    assert [[h.level for h in section.path] for section in build_sections(doc)] == [[1], [1, 3], [1, 2]]


def test_preamble_is_not_lost() -> None:
    doc = document(
        [paragraph("Intro before heading."), DocumentBlock(type="heading", text="Heading", level=1), paragraph("Body.")]
    )
    chunks = StructureAwareChunker().chunk(doc)
    assert chunks[0].text == "Intro before heading."
    assert chunks[0].metadata["section_path"] == []
    assert chunks[1].text == "Heading\n\nBody."


def test_small_section_keeps_heading_and_content_together() -> None:
    doc = document([DocumentBlock(type="heading", text="Inputs", level=2), paragraph("Short body")])
    chunks = StructureAwareChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Inputs\n\nShort body"


def test_sibling_sections_never_merge_even_when_both_fit() -> None:
    doc = document(
        [
            DocumentBlock(type="heading", text="Inputs", level=2), paragraph("short text"),
            DocumentBlock(type="heading", text="Outputs", level=2), paragraph("short text"),
        ]
    )
    chunks = StructureAwareChunker().chunk(doc)
    assert len(chunks) == 2
    assert [chunk.metadata["section_path"] for chunk in chunks] == [["Inputs"], ["Outputs"]]


def test_block_aware_greedy_packing_does_not_split_fitting_code() -> None:
    probe = StructureAwareChunker()
    a = exact_tokens(probe, 20, "a")
    code = exact_tokens(probe, 20, "c")
    b = exact_tokens(probe, 20, "b")
    doc = document(
        [
            DocumentBlock(type="heading", text="H", level=1), paragraph(a),
            DocumentBlock(type="code_block", text=code, language="ts"), paragraph(b),
        ]
    )
    chunker = StructureAwareChunker(StructureAwareChunkingConfig(max_chunk_tokens=50))
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].metadata["block_types"] == ["heading", "paragraph", "code_block"]
    assert chunks[1].metadata["block_types"] == ["paragraph"]
    assert chunks[0].metadata["source_block_metadata"]["2"]["language"] == "ts"


def test_oversized_paragraph_uses_sentences_without_loss_or_overlap() -> None:
    probe = StructureAwareChunker()
    sentences = [exact_tokens(probe, 24, word) for word in ("one", "two", "three")]
    text = " ".join(sentences)
    block = DocumentBlock(
        type="paragraph",
        text=text,
        sentences=[Sentence(f"s{i}", value) for i, value in enumerate(sentences)],
    )
    doc = document([block])
    chunker = StructureAwareChunker(StructureAwareChunkingConfig(max_chunk_tokens=40))
    chunks = chunker.chunk(doc)
    fragments = [item for chunk in chunks for item in chunk.metadata["block_fragments"]]
    assert len(chunks) == 3
    assert all(chunk.token_count <= 40 for chunk in chunks)
    assert all(item["split_reason"] == "oversized_paragraph_sentence_split" for item in fragments)
    assert "".join(text[item["char_start"]:item["char_end"]] for item in fragments) == text


def test_oversized_code_splits_by_lines_and_preserves_language() -> None:
    probe = StructureAwareChunker()
    lines = [exact_tokens(probe, 24, word) + "\n" for word in ("a", "b", "c")]
    text = "".join(lines)
    doc = document([DocumentBlock(type="code_block", text=text, language="angular-ts")])
    chunker = StructureAwareChunker(StructureAwareChunkingConfig(max_chunk_tokens=30))
    chunks = chunker.chunk(doc)
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(chunk.token_count <= 30 for chunk in chunks)
    assert all(chunk.metadata["source_block_metadata"]["0"]["language"] == "angular-ts" for chunk in chunks)
    assert all(chunk.metadata["split_reason"] == "oversized_code_line_split" for chunk in chunks)


def test_single_oversized_sentence_and_code_line_use_unicode_safe_fallback() -> None:
    for block in (
        DocumentBlock(type="paragraph", text="🌷" * 80, sentences=[Sentence("s", "🌷" * 80)]),
        DocumentBlock(type="code_block", text="const x='🌷';" * 80, language="ts"),
    ):
        doc = document([block])
        chunker = StructureAwareChunker(StructureAwareChunkingConfig(max_chunk_tokens=30))
        chunks = chunker.chunk(doc)
        assert "".join(chunk.text for chunk in chunks) == block.text
        assert all("\ufffd" not in chunk.text and chunk.token_count <= 30 for chunk in chunks)
        assert all(chunk.metadata["split_reason"] == "oversized_token_fallback" for chunk in chunks)


def test_content_coverage_source_order_and_determinism_for_all_types() -> None:
    blocks = [
        DocumentBlock(type="heading", text="All", level=1),
        paragraph("paragraph"),
        DocumentBlock(type="list", text="- one\n- two", ordered=False),
        DocumentBlock(type="code_block", text="x = 1", language="py"),
        DocumentBlock(type="table", text="| A |\n|---|\n| B |"),
        DocumentBlock(type="blockquote", text="> quote", sentences=[Sentence("q", "> quote")]),
        DocumentBlock(type="custom_block", text="custom", metadata={"tag": "docs-callout"}),
    ]
    doc = document(blocks)
    chunker = StructureAwareChunker(StructureAwareChunkingConfig(max_chunk_tokens=12))
    first = chunker.chunk(doc)
    second = chunker.chunk(doc)
    assert [chunk.to_dict() for chunk in first] == [chunk.to_dict() for chunk in second]
    report = validate_structure_aware_chunks([doc], first, chunker.config, chunker.tokenizer)
    assert report.valid, report.errors
    assert report.coverage_gaps == 0
    assert all(chunk.token_count <= 12 for chunk in first)


def test_heading_uses_metadata_instead_of_tiny_standalone_chunk_when_pair_overflows() -> None:
    probe = StructureAwareChunker()
    body = exact_tokens(probe, 10)
    doc = document([DocumentBlock(type="heading", text="Heading", level=1), paragraph(body)])
    chunker = StructureAwareChunker(StructureAwareChunkingConfig(max_chunk_tokens=10))
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == body
    assert chunks[0].metadata["local_heading_in_text"] is False
    assert chunks[0].metadata["context_heading_block_index"] == 0
    assert validate_structure_aware_chunks([doc], chunks, chunker.config, chunker.tokenizer).valid


def test_structure_artifacts_are_deterministic_and_policy_identified(tmp_path: Path) -> None:
    doc = document([DocumentBlock(type="heading", text="H", level=1), paragraph("Body")])
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(doc)
    stats = structure_corpus_statistics([doc], chunks)
    output = tmp_path / "structure"
    args = (chunks, output, chunker.config, chunker.tokenizer, stats, "input.jsonl")
    write_structure_aware_artifacts(*args)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    write_structure_aware_artifacts(*args)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first
    assert read_chunks_jsonl(output / "chunks.jsonl")[0].to_dict() == chunks[0].to_dict()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["overlap_policy"] == "none"
    assert manifest["sibling_section_merge_policy"] == "never"
    assert manifest["hierarchy_policy"] == "markdown_stack_pop_level_gte_current_v1"
    assert manifest["source_schema_version"] == "normalized_document_v2"


def test_structure_validation_rejects_text_that_disagrees_with_provenance() -> None:
    doc = document([DocumentBlock(type="heading", text="H", level=1), paragraph("Body")])
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(doc)
    chunks[0].text += " corrupted"
    chunks[0].token_count = len(chunker.tokenizer.encode(chunks[0].text))

    report = validate_structure_aware_chunks(
        [doc], chunks, chunker.config, chunker.tokenizer
    )

    assert not report.valid
    assert any("exact local source slicing" in error for error in report.errors)
