"""Run and publish the canonical multi-strategy dense retrieval benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.embedding.config import load_embedding_config
from rag_chunking.embedding.provider import OpenRouterEmbeddingProvider
from rag_chunking.evaluation.dataset import load_evaluation_dataset
from rag_chunking.evaluation.runner import run_retrieval_benchmark
from rag_chunking.retrieval.service import RetrievalService


STRATEGIES = ("fixed_size", "structure_aware", "prompt_based")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pure dense retrieval across chunking strategies")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--indexes-root", type=Path, default=Path("data/indexes"))
    parser.add_argument("--query-cache", type=Path, default=Path("data/query-embedding-cache"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval"))
    parser.add_argument("--baseline-name", default="baseline_v1")
    parser.add_argument("--plan-only", action="store_true", help="Validate and report workload without provider calls")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_embedding_config(args.embedding_config)
        directories = {strategy: args.indexes_root / args.corpus / strategy / config.fingerprint for strategy in args.strategies}
        provider = OpenRouterEmbeddingProvider(config)
        service = RetrievalService(
            corpus=args.corpus, index_directories=directories, embedding_config=config,
            provider=provider, query_cache_directory=args.query_cache,
        )
        source_sets = [{record.relative_path for record in records.values()} for records in service.records.values()]
        common_sources = set.intersection(*source_sets)
        dataset = load_evaluation_dataset(args.dataset, common_sources)
        cached = sum(service.cache.get(record.query) is not None for record in dataset.records)
        plan = {
            "dataset_fingerprint": dataset.fingerprint, "queries": len(dataset.records),
            "unique_normalized_queries": len({record.query for record in dataset.records}),
            "query_cache_hits": cached, "expected_query_cache_misses": len(dataset.records) - cached,
            "expected_provider_calls_upper_bound": len(dataset.records) - cached,
        }
        if args.plan_only:
            print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
            return 0
        result = run_retrieval_benchmark(
            service, dataset, args.output / args.corpus, strategies=args.strategies,
            depth=args.depth, baseline_name=args.baseline_name,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({**plan, "benchmark_fingerprint": result.benchmark_fingerprint, "output": result.output_directory.as_posix(), **result.stats}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
