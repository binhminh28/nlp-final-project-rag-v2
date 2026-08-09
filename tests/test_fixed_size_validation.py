import pytest

from rag_chunking.chunking.fixed_size import FixedSizeChunker, FixedSizeChunkingConfig
from rag_chunking.chunking.validation import validate_fixed_size_chunks
from rag_chunking.data.models import DocumentBlock, NormalizedDocument


def make_document(text: str) -> NormalizedDocument:
    return NormalizedDocument(
        doc_id="angular:validation.md",
        source="angular",
        relative_path="validation.md",
        filename="validation.md",
        source_sha256="hash",
        blocks=[DocumentBlock(type="paragraph", text=text)],
    )


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "greater than 0"),
        (10, -1, "greater than or equal to 0"),
        (10, 10, "smaller than chunk_size"),
        (10, 11, "smaller than chunk_size"),
    ],
)
def test_invalid_config_raises_clear_error(chunk_size: int, overlap: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FixedSizeChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap)


def test_validation_checks_source_token_slices_and_spans() -> None:
    document = make_document("token slice validation " * 200)
    chunker = FixedSizeChunker(FixedSizeChunkingConfig(chunk_size=32, chunk_overlap=8))
    chunks = chunker.chunk(document)
    report = validate_fixed_size_chunks(
        [document], chunks, chunker.config, chunker.tokenizer
    )
    assert report.valid, report.errors
    chunks[1].token_start += 1
    report = validate_fixed_size_chunks(
        [document], chunks, chunker.config, chunker.tokenizer
    )
    assert not report.valid
    assert any("span" in error for error in report.errors)


def test_empty_document_validates_with_zero_chunks() -> None:
    document = make_document("")
    chunker = FixedSizeChunker()
    assert validate_fixed_size_chunks(
        [document], [], chunker.config, chunker.tokenizer
    ).valid
