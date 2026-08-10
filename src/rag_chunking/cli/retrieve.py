"""Machine-readable canonical retrieval command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.embedding.config import load_embedding_config
from rag_chunking.embedding.provider import OpenRouterEmbeddingProvider
from rag_chunking.retrieval.models import RetrievalRequest
from rag_chunking.retrieval.service import RetrievalService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search one validated dense-vector index")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--strategy", required=True, choices=("fixed_size", "structure_aware", "prompt_based"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--filter", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--query-cache", type=Path, default=Path("data/query-embedding-cache"))
    return parser


def _filters(values: list[str]) -> dict[str, str] | None:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("filters must use FIELD=VALUE")
        key, item = value.split("=", 1)
        if key in result:
            raise ValueError(f"duplicate filter {key!r}")
        result[key] = item
    return result or None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_embedding_config(args.embedding_config)
        index = args.index or Path("data/indexes") / args.corpus / args.strategy / config.fingerprint
        service = RetrievalService(
            corpus=args.corpus, index_directories={args.strategy: index}, embedding_config=config,
            provider=OpenRouterEmbeddingProvider(config), query_cache_directory=args.query_cache,
        )
        result = service.retrieve(RetrievalRequest(args.query, args.strategy, args.top_k, _filters(args.filter)))
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
