"""CLI for deterministic structure-aware chunking."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_chunking.chunking.structure_aware import StructureAwareChunker, StructureAwareChunkingConfig
from rag_chunking.chunking.structure_statistics import structure_corpus_statistics
from rag_chunking.chunking.structure_validation import validate_structure_aware_chunks
from rag_chunking.chunking.structure_writer import write_structure_aware_artifacts
from rag_chunking.data.writer import read_documents_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk normalized documents by structural sections")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chunk-tokens", type=int, default=512)
    parser.add_argument("--tokenizer", default="cl100k_base")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = StructureAwareChunkingConfig(args.max_chunk_tokens, args.tokenizer)
        documents = read_documents_jsonl(args.input)
        chunker = StructureAwareChunker(config)
        chunks = chunker.chunk_corpus(documents)
        report = validate_structure_aware_chunks(documents, chunks, config, chunker.tokenizer)
        if not report.valid:
            for error in report.errors:
                print(f"ERROR: {error}")
            return 1
        stats = structure_corpus_statistics(documents, chunks)
        write_structure_aware_artifacts(chunks, args.output, config, chunker.tokenizer, stats, str(args.input))
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1

    chunk_doc = stats["chunks_per_document"]
    token_stats = stats["tokens_per_chunk"]
    oversized = stats["oversized_blocks"]
    structural = stats["structural"]
    print(f"Documents processed: {stats['documents']}")
    print("Documents failed: 0")
    print(f"Chunks generated: {stats['chunks']}")
    print(f"Tokenizer: {chunker.tokenizer.name}")
    print(f"Max chunk tokens: {config.max_chunk_tokens}")
    print(f"Chunks/document: min={chunk_doc['min']} mean={chunk_doc['mean']:.2f} median={chunk_doc['median']:.2f} max={chunk_doc['max']}")
    print("Tokens/chunk: " + " ".join(f"{key}={token_stats[key]:.2f}" for key in ("min", "mean", "median", "max", "p25", "p75", "p95")))
    print(f"Oversized blocks encountered: {oversized['total']}")
    print(f"Paragraph splits: {oversized['paragraph']}")
    print(f"Callout splits: {oversized['callout']}")
    print(f"Code splits: {oversized['code_block']}")
    print(f"Code-reference splits: {oversized['code_reference']}")
    print(f"List splits: {oversized['list']}")
    print(f"Table splits: {oversized['table']}")
    print(f"HTML-block splits: {oversized['html_block']}")
    print(f"Custom-block splits: {oversized['custom_block']}")
    print(f"Token-fallback splits: {oversized['token_fallbacks']}")
    print(f"Section-boundary chunks: {structural['section_boundary_chunks']}")
    print(f"Block-boundary chunks: {structural['block_boundary_chunks']}")
    print(f"Internal-block-split chunks: {structural['internal_block_split_chunks']}")
    print(f"Output path: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
