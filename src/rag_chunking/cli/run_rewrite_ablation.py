"""Run isolated Q1/Q2 retrieval rewrite ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.chunking.prompt_config import load_project_dotenv
from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.embedding.config import load_embedding_config
from rag_chunking.embedding.provider import OpenRouterEmbeddingProvider
from rag_chunking.evaluation.dataset import load_evaluation_dataset
from rag_chunking.evaluation.dataset import EvaluationDataset
from rag_chunking.retrieval.cache import QueryEmbeddingCache
from rag_chunking.retrieval.models import RetrievalConfig
from rag_chunking.tuning.rewrite_experiment import (
    load_rewrite_records, prepare_rewrite_embeddings, publish_rewrite_experiments,
    run_rewrite_retrieval,
)


STRATEGIES = ["fixed_size", "structure_aware", "prompt_based"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run query rewrite retrieval ablations")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rewrites", type=Path, required=True)
    parser.add_argument("--rewrite-config-fingerprint", required=True)
    parser.add_argument("--reference-per-query", type=Path, required=True)
    parser.add_argument("--reference-candidates", type=Path)
    parser.add_argument("--populate-embeddings", action="store_true")
    parser.add_argument("--populate-only", action="store_true")
    parser.add_argument("--embedding-limit", type=int)
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--documents", type=Path, default=Path("data/processed/angular/documents.jsonl"))
    parser.add_argument("--indexes-root", type=Path, default=Path("data/indexes"))
    parser.add_argument("--query-cache", type=Path, default=Path("data/query-embedding-cache"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval"))
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--query-limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_project_dotenv()
        embedding = load_embedding_config(args.embedding_config)
        sources = {document.relative_path for document in read_documents_jsonl(args.documents)}
        dataset = load_evaluation_dataset(args.dataset, sources)
        rewrites = load_rewrite_records(args.rewrites, dataset)
        provider = OpenRouterEmbeddingProvider(embedding)
        population = prepare_rewrite_embeddings(
            rewrites, QueryEmbeddingCache(args.query_cache, embedding), provider,
            populate=args.populate_embeddings, limit=args.embedding_limit,
        )
        if args.populate_only:
            print(json.dumps({"dataset_fingerprint": dataset.fingerprint, **population}, ensure_ascii=False, sort_keys=True))
            return 0
        if args.query_offset or args.query_limit is not None:
            if not args.raw_only:
                raise ValueError("query slicing is permitted only for raw memory-bounded runs")
            selected = dataset.records[args.query_offset:None if args.query_limit is None else args.query_offset + args.query_limit]
            if not selected:
                raise ValueError("query slice is empty")
            dataset = EvaluationDataset(selected, dataset.fingerprint)
        strategies = args.strategies
        index_dirs = {name: args.indexes_root / args.corpus / name / embedding.fingerprint for name in strategies}
        original_candidates = None
        if args.reference_candidates:
            original_candidates = {}
            with args.reference_candidates.open(encoding="utf-8") as stream:
                for line in stream:
                    value = json.loads(line)
                    original_candidates[(value["query_id"], value["strategy"])] = value["hits"]
        variants, fingerprints, retrieval_stats = run_rewrite_retrieval(
            corpus=args.corpus, dataset=dataset, rewrites=rewrites, strategies=strategies,
            index_directories=index_dirs, embedding_config=embedding,
            provider_factory=lambda: OpenRouterEmbeddingProvider(embedding),
            query_cache_directory=args.query_cache, repository_root=Path.cwd(),
            original_candidates=original_candidates,
        )
        if args.raw_output:
            args.raw_output.parent.mkdir(parents=True, exist_ok=True)
            raw = {
                "index_fingerprints": fingerprints, "stats": retrieval_stats,
                "variants": {variant: [
                    {"query_id": query_id, "strategy": strategy, "hits": hits}
                    for (query_id, strategy), hits in sorted(values.items())
                ] for variant, values in variants.items()},
            }
            args.raw_output.write_text(json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        if args.raw_only:
            print(json.dumps({"dataset_fingerprint": dataset.fingerprint, "raw_output": args.raw_output.as_posix() if args.raw_output else None, **population, **retrieval_stats}, ensure_ascii=False, sort_keys=True))
            return 0
        outputs = publish_rewrite_experiments(
            corpus=args.corpus, dataset=dataset, rewrites=rewrites, variants=variants,
            strategies=strategies, index_fingerprints=fingerprints,
            embedding_config=embedding, retrieval_config=RetrievalConfig(),
            rewrite_config_fingerprint=args.rewrite_config_fingerprint,
            reference_per_query=args.reference_per_query,
            output_root=args.output / args.corpus / "experiments",
            runtime_stats={**population, **retrieval_stats},
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({"dataset_fingerprint": dataset.fingerprint, "outputs": [path.as_posix() for path in outputs], **population, **retrieval_stats}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
