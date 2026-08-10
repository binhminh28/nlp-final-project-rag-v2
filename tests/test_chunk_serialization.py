import json
from pathlib import Path

from rag_chunking.chunking.fixed_size import FixedSizeChunker
from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.statistics import chunk_corpus_statistics
from rag_chunking.chunking.writer import read_chunks_jsonl, write_fixed_size_artifacts
from rag_chunking.data.models import DocumentBlock, NormalizedDocument


def test_chunk_model_round_trip() -> None:
    chunk = Chunk(
        chunk_id="angular:a.md::fixed::000000",
        strategy="fixed_size",
        doc_id="angular:a.md",
        source="angular",
        relative_path="a.md",
        chunk_index=0,
        text="Text",
        token_start=0,
        token_end=1,
        token_count=1,
        chunk_size=512,
        chunk_overlap=64,
        tokenizer="tiktoken:cl100k_base",
    )
    assert Chunk.from_dict(chunk.to_dict()).to_dict() == chunk.to_dict()


def test_artifacts_are_deterministic_and_readable(tmp_path: Path) -> None:
    document = NormalizedDocument(
        doc_id="angular:a.md",
        source="angular",
        relative_path="a.md",
        filename="a.md",
        source_sha256="hash",
        blocks=[DocumentBlock(type="paragraph", text="Some technical text.")],
    )
    chunker = FixedSizeChunker()
    chunks = chunker.chunk(document)
    stats = chunk_corpus_statistics([document], chunks, chunker.tokenizer)
    output = tmp_path / "fixed_size"
    args = (chunks, output, chunker.config, chunker.tokenizer, stats, "input.jsonl")
    write_fixed_size_artifacts(*args)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    write_fixed_size_artifacts(*args)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first
    assert read_chunks_jsonl(output / "chunks.jsonl")[0].to_dict() == chunks[0].to_dict()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stride"] == 448
    assert manifest["tokenizer"] == "tiktoken:cl100k_base"
    assert manifest["boundary_policy"] == "utf8_safe_minimal_backoff"
