from __future__ import annotations

from pathlib import Path

from rag_chunking.chunking.structure_aware import StructureAwareChunker
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


def by_title_path(chunks, path: list[str]):
    return next(chunk for chunk in chunks if chunk.title_path == path)


def test_parent_child_and_sibling_hierarchy_from_sections() -> None:
    doc = document(
        [
            DocumentBlock(type="heading", text="A", level=1),
            paragraph("intro"),
            DocumentBlock(type="heading", text="B", level=2),
            paragraph("b text"),
            DocumentBlock(type="heading", text="C", level=3),
            paragraph("c text"),
            DocumentBlock(type="heading", text="D", level=2),
            paragraph("d text"),
        ]
    )
    chunks = StructureAwareChunker().chunk(doc)
    a = by_title_path(chunks, ["A"])
    b = by_title_path(chunks, ["A", "B"])
    c = by_title_path(chunks, ["A", "B", "C"])
    d = by_title_path(chunks, ["A", "D"])

    # section cha-con
    assert b.parent_id == a.chunk_id
    assert c.parent_id == b.chunk_id

    # sibling chunks (B, D) share the same parent (A) and appear in document order
    assert d.parent_id == a.chunk_id
    assert a.children_ids == [b.chunk_id, d.chunk_id]
    assert b.children_ids == [c.chunk_id]
    assert c.children_ids == []
    assert d.children_ids == []


def test_top_level_heading_is_root_even_with_a_preceding_preamble() -> None:
    doc = document(
        [
            paragraph("preamble text before any heading"),
            DocumentBlock(type="heading", text="A", level=1),
            paragraph("body"),
        ]
    )
    chunks = StructureAwareChunker().chunk(doc)
    preamble = by_title_path(chunks, [])
    a = by_title_path(chunks, ["A"])

    # root-level chunk: preamble has no heading ancestry, and a top-level
    # heading must not be attached to the preamble bucket either.
    assert preamble.parent_id is None
    assert preamble.children_ids == []
    assert a.parent_id is None
    assert a.chunk_id not in preamble.children_ids


def test_section_with_no_generated_chunk_leaves_relations_unlinked() -> None:
    # A heading with empty text produces zero fragments (split_block returns
    # [] for empty text), so its section generates no chunk at all.
    doc = document(
        [
            DocumentBlock(type="heading", text="A", level=1),
            paragraph("intro"),
            DocumentBlock(type="heading", text="", level=2),
            DocumentBlock(type="heading", text="C", level=3),
            paragraph("c text"),
        ]
    )
    chunks = StructureAwareChunker().chunk(doc)
    a = by_title_path(chunks, ["A"])
    c = by_title_path(chunks, ["A", "", "C"])

    # The empty-heading section produced zero chunks, so C's parent link is
    # left unset rather than substituting A's chunk as a guessed ancestor.
    assert not any(chunk.title_path == ["A", ""] for chunk in chunks)
    assert c.parent_id is None
    assert c.chunk_id not in a.children_ids


def test_hierarchy_survives_jsonl_round_trip(tmp_path: Path) -> None:
    doc = document(
        [
            DocumentBlock(type="heading", text="A", level=1),
            paragraph("intro"),
            DocumentBlock(type="heading", text="B", level=2),
            paragraph("b text"),
        ]
    )
    chunks = StructureAwareChunker().chunk(doc)
    path = tmp_path / "chunks.jsonl"
    import json

    with path.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk.to_dict()))
            stream.write("\n")

    reloaded = read_chunks_jsonl(path)
    a = by_title_path(chunks, ["A"])
    b = by_title_path(chunks, ["A", "B"])
    reloaded_a = by_title_path(reloaded, ["A"])
    reloaded_b = by_title_path(reloaded, ["A", "B"])

    assert reloaded_b.parent_id == b.parent_id == a.chunk_id
    assert reloaded_a.children_ids == a.children_ids == [b.chunk_id]
