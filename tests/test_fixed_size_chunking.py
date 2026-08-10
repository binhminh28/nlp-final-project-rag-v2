from __future__ import annotations

from rag_chunking.chunking.fixed_size import FixedSizeChunker, FixedSizeChunkingConfig
from rag_chunking.chunking.serialization import BLOCK_SEPARATOR, document_to_text
from rag_chunking.data.models import DocumentBlock, NormalizedDocument
from rag_chunking.chunking.tokenizer import is_utf8_safe_boundary


def make_document(text: str, *, blocks: list[DocumentBlock] | None = None) -> NormalizedDocument:
    return NormalizedDocument(
        doc_id="angular:test.md",
        source="angular",
        relative_path="test.md",
        filename="test.md",
        source_sha256="abc123",
        blocks=blocks if blocks is not None else [DocumentBlock(type="paragraph", text=text)],
    )


def exact_token_text(chunker: FixedSizeChunker, count: int) -> str:
    text = " a" * count
    assert len(chunker.tokenizer.encode(text)) == count
    return text


def test_short_document_creates_one_chunk() -> None:
    chunker = FixedSizeChunker()
    text = exact_token_text(chunker, 20)
    chunks = chunker.chunk(make_document(text))
    assert len(chunks) == 1
    assert (chunks[0].token_start, chunks[0].token_end) == (0, 20)


def test_exactly_512_tokens_does_not_create_overlap_only_chunk() -> None:
    chunker = FixedSizeChunker()
    chunks = chunker.chunk(make_document(exact_token_text(chunker, 512)))
    assert len(chunks) == 1
    assert chunks[0].token_count == 512


def test_513_tokens_creates_expected_last_window() -> None:
    chunker = FixedSizeChunker()
    chunks = chunker.chunk(make_document(exact_token_text(chunker, 513)))
    assert [(chunk.token_start, chunk.token_end) for chunk in chunks] == [
        (0, 512),
        (448, 513),
    ]
    assert chunks[1].token_count == 65


def test_multiple_windows_use_fixed_stride() -> None:
    chunker = FixedSizeChunker()
    chunks = chunker.chunk(make_document(exact_token_text(chunker, 1400)))
    assert [chunk.token_start for chunk in chunks] == [0, 448, 896]


def test_zero_overlap_uses_chunk_size_as_stride() -> None:
    config = FixedSizeChunkingConfig(chunk_overlap=0)
    chunker = FixedSizeChunker(config)
    chunks = chunker.chunk(make_document(exact_token_text(chunker, 1025)))
    assert config.stride == 512
    assert [chunk.token_start for chunk in chunks] == [0, 512, 1024]


def test_empty_document_explicitly_produces_zero_chunks() -> None:
    assert FixedSizeChunker().chunk(make_document("")) == []


def test_deterministic_output() -> None:
    chunker = FixedSizeChunker(FixedSizeChunkingConfig(chunk_size=8, chunk_overlap=2))
    document = make_document("deterministic technical text " * 20)
    assert [c.to_dict() for c in chunker.chunk(document)] == [
        c.to_dict() for c in chunker.chunk(document)
    ]


def test_source_metadata_is_preserved() -> None:
    chunk = FixedSizeChunker().chunk(make_document("metadata"))[0]
    assert (chunk.doc_id, chunk.source, chunk.relative_path) == (
        "angular:test.md",
        "angular",
        "test.md",
    )
    assert chunk.metadata["text_token_roundtrip"] is True


def test_mixed_structural_input_is_only_linearly_serialized() -> None:
    blocks = [
        DocumentBlock(type="heading", text="Heading", level=1),
        DocumentBlock(type="paragraph", text="Paragraph"),
        DocumentBlock(type="code_block", text="const x = 1;", language="ts"),
        DocumentBlock(type="list", text="- one", ordered=False),
        DocumentBlock(type="table", text="A | B"),
        DocumentBlock(type="custom_block", text="Custom"),
    ]
    document = make_document("", blocks=blocks)
    assert document_to_text(document) == BLOCK_SEPARATOR.join(block.text for block in blocks)
    config = FixedSizeChunkingConfig(chunk_size=4, chunk_overlap=1)
    chunker = FixedSizeChunker(config)
    source_tokens = chunker.tokenizer.encode(document_to_text(document))
    chunks = chunker.chunk(document)
    assert [chunk.token_start for chunk in chunks] == list(
        range(0, len(source_tokens) - 1, config.stride)
    )[: len(chunks)]
    assert all(chunk.token_count <= 4 for chunk in chunks)


def test_unicode_split_boundary_is_adjusted_without_corruption_or_gaps() -> None:
    probe = FixedSizeChunker()
    text = "A tulip 🌷 remains valid after an arbitrary token boundary."
    source_tokens = probe.tokenizer.encode(text)
    unsafe_boundaries = [
        position
        for position in range(1, len(source_tokens))
        if not is_utf8_safe_boundary(source_tokens, position, probe.tokenizer)
    ]
    assert unsafe_boundaries, "Test fixture must contain a split Unicode code point"
    unsafe_end = unsafe_boundaries[0]
    assert "\ufffd" in probe.tokenizer.decode(source_tokens[:unsafe_end])

    config = FixedSizeChunkingConfig(chunk_size=unsafe_end, chunk_overlap=0)
    chunker = FixedSizeChunker(config)
    document = make_document(text)
    first = [chunk.to_dict() for chunk in chunker.chunk(document)]
    second = [chunk.to_dict() for chunk in chunker.chunk(document)]

    assert first == second
    assert all("\ufffd" not in chunk["text"] for chunk in first)
    assert all(chunk["token_count"] == chunk["token_end"] - chunk["token_start"] for chunk in first)
    assert all(chunk["token_count"] <= config.chunk_size for chunk in first)
    assert any(chunk["metadata"]["boundary_adjusted"] for chunk in first)
    covered = {position for chunk in first for position in range(chunk["token_start"], chunk["token_end"])}
    assert covered == set(range(len(source_tokens)))


def test_bpe_context_boundary_keeps_persisted_token_count_canonical() -> None:
    # This suffix previously produced a valid UTF-8 source-token slice whose
    # standalone BPE encoding was one token shorter.
    text = """    // Close the popup
    await combobox.close();
    expect(await combobox.isOpen()).toBe(false);
  });
});

API reference

For detailed API documentation, inspect the following API references:

- [`Combobox`](/api/aria/combobox/Combobox)
- [`ComboboxPopup`](/api/aria/combobox/ComboboxPopup)
- [`ComboboxWidget`](/api/aria/combobox/ComboboxWidget)

Related patterns and directives

Combobox is the primitive directive for these documented patterns:

- [Autocomplete](guide/aria/autocomplete) - Filtering and suggestions pattern
- [Select](guide/aria/select) - Single selection dropdown pattern
- [Multiselect](guide/aria/multiselect) - Multiple selection pattern

Combobox typically combines with:

- [Listbox](guide/aria/listbox) - Most common popup content
- [Tree](guide/aria/tree) - Hierarchical popup content"""
    chunker = FixedSizeChunker(FixedSizeChunkingConfig(chunk_size=54, chunk_overlap=10))

    chunks = chunker.chunk(make_document(text))

    assert all(chunk.metadata["text_token_roundtrip"] is True for chunk in chunks)
    assert all(
        chunk.token_count == len(chunker.tokenizer.encode(chunk.text)) for chunk in chunks
    )
