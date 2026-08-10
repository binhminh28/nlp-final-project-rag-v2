"""Build resumable embedding artifacts from one Unified Chunk artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_chunking.embedding.models import EmbeddingConfig
from rag_chunking.embedding.pipeline import run_embedding_pipeline
from rag_chunking.embedding.provider import DeterministicFakeEmbeddingProvider, OpenRouterEmbeddingProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed a Unified Chunk artifact with a persistent cache")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--provider", choices=("openrouter", "fake"), default="openrouter")
    parser.add_argument("--model", default="openai/text-embedding-3-small")
    parser.add_argument("--dimension", type=int, default=1536)
    parser.add_argument("--max-batch-items", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--input-type")
    parser.add_argument("--limit", type=int, help="Explicit complete sample build; never labeled as full")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = EmbeddingConfig(
            provider=args.provider, model=args.model, dimension=args.dimension,
            max_batch_items=args.max_batch_items, max_batch_tokens=args.max_batch_tokens,
            max_input_tokens=args.max_input_tokens, tokenizer=args.tokenizer,
            input_type=args.input_type,
        )
        provider = DeterministicFakeEmbeddingProvider(config) if args.provider == "fake" else OpenRouterEmbeddingProvider(config)
        result = run_embedding_pipeline(
            args.input, args.output, args.cache, provider, corpus=args.corpus, limit=args.limit
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    print(
        f"strategy={result.manifest['chunk_strategy']} chunks={len(result.records)} "
        f"dimension={config.dimension} cache_hits={result.stats['cache_hits']} "
        f"cache_misses={result.stats['cache_misses']} model_calls={result.stats['model_calls']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
