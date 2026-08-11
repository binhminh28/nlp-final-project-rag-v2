"""Run E1 dense candidate-depth experiments with bounded memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.embedding.config import load_embedding_config
from rag_chunking.embedding.provider import OpenRouterEmbeddingProvider
from rag_chunking.evaluation.dataset import load_evaluation_dataset
from rag_chunking.retrieval.models import RetrievalConfig
from rag_chunking.tuning.dense import publish_dense_depth_experiments, retrieve_dense_candidates


STRATEGIES = ["fixed_size", "structure_aware", "prompt_based"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run memory-bounded dense candidate-depth ablation")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--documents", type=Path, default=Path("data/processed/angular/documents.jsonl"))
    parser.add_argument("--indexes-root", type=Path, default=Path("data/indexes"))
    parser.add_argument("--query-cache", type=Path, default=Path("data/query-embedding-cache"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_embedding_config(args.embedding_config)
        index_dirs = {name: args.indexes_root / args.corpus / name / config.fingerprint for name in STRATEGIES}
        sources = {document.relative_path for document in read_documents_jsonl(args.documents)}
        dataset = load_evaluation_dataset(args.dataset, sources)
        max_depth = max(args.depths)
        hits, fingerprints, stats = retrieve_dense_candidates(
            corpus=args.corpus, dataset=dataset, strategies=STRATEGIES,
            index_directories=index_dirs, embedding_config=config,
            provider_factory=lambda: OpenRouterEmbeddingProvider(config),
            query_cache_directory=args.query_cache, depth=max_depth,
            repository_root=Path.cwd(),
        )
        outputs = publish_dense_depth_experiments(
            corpus=args.corpus, dataset=dataset, strategies=STRATEGIES,
            all_hits=hits, index_fingerprints=fingerprints, embedding_config=config,
            retrieval_config=RetrievalConfig(), depths=sorted(args.depths),
            output_root=args.output / args.corpus / "experiments", runtime_stats=stats,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({"dataset_fingerprint": dataset.fingerprint, "outputs": [path.as_posix() for path in outputs], **stats}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
