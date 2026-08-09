"""CLI for deterministic fixed-size token chunking."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_chunking.chunking.fixed_size import FixedSizeChunker, FixedSizeChunkingConfig
from rag_chunking.chunking.statistics import chunk_corpus_statistics
from rag_chunking.chunking.validation import validate_fixed_size_chunks
from rag_chunking.chunking.writer import write_fixed_size_artifacts
from rag_chunking.data.writer import read_documents_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk normalized documents into fixed token windows")
    parser.add_argument("--input", type=Path, required=True, help="Normalized documents JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Strategy-specific output directory")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--tokenizer", default="cl100k_base", help="tiktoken encoding name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = FixedSizeChunkingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            tokenizer_name=args.tokenizer,
        )
        documents = read_documents_jsonl(args.input)
        chunker = FixedSizeChunker(config)
        chunks = chunker.chunk_corpus(documents)
        report = validate_fixed_size_chunks(documents, chunks, config, chunker.tokenizer)
        if not report.valid:
            for error in report.errors:
                print(f"ERROR: {error}")
            return 1
        stats = chunk_corpus_statistics(documents, chunks, chunker.tokenizer)
        write_fixed_size_artifacts(
            chunks,
            args.output,
            config,
            chunker.tokenizer,
            stats,
            str(args.input),
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1

    chunk_stats = stats["tokens_per_chunk"]
    unicode_stats = stats["unicode_safe_boundaries"]
    print(f"Documents processed: {stats['documents']}")
    print("Documents failed: 0")
    print(f"Chunks generated: {stats['chunks']}")
    print(f"Tokenizer: {chunker.tokenizer.name}")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Overlap: {config.chunk_overlap}")
    print(f"Stride: {config.stride}")
    print(f"Average tokens/chunk: {chunk_stats['mean']:.2f}")
    print(f"Median tokens/chunk: {chunk_stats['median']:.2f}")
    print(f"Min tokens/chunk: {chunk_stats['min']}")
    print(f"Max tokens/chunk: {chunk_stats['max']}")
    print(f"Total token occurrences: {stats['total_token_occurrences']}")
    print(f"Unique source tokens: {stats['source_tokens']}")
    print(f"Boundary-adjusted chunks: {unicode_stats['boundary_adjusted_chunks']}")
    print(f"Documents with boundary adjustments: {unicode_stats['documents_affected']}")
    print(f"Maximum boundary adjustment: {unicode_stats['maximum_adjustment_tokens']}")
    print(
        "Generated replacement-character chunks: "
        f"{unicode_stats['generated_replacement_character_chunks']}"
    )
    print(f"Token coverage gap positions: {unicode_stats['token_coverage_gap_positions']}")
    print(f"Output path: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
