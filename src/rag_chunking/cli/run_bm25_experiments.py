"""Run E5 BM25 reranking and E6 full-corpus lexical retrieval."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.embedding.config import load_embedding_config
from rag_chunking.evaluation.dataset import load_evaluation_dataset
from rag_chunking.retrieval.models import RetrievalConfig
from rag_chunking.tuning.config import ExperimentConfig
from rag_chunking.tuning.lexical import BM25Config, BM25Index, load_dense_rows
from rag_chunking.tuning.publish import publish_ranked_experiment


STRATEGIES = ["fixed_size", "structure_aware", "prompt_based"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BM25 reranking and lexical experiments")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dense-depth-5", type=Path, required=True)
    parser.add_argument("--dense-depth-50", type=Path, required=True)
    parser.add_argument("--documents", type=Path, default=Path("data/processed/angular/documents.jsonl"))
    parser.add_argument("--chunks-root", type=Path, default=Path("data/chunks"))
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval"))
    args = parser.parse_args(argv)
    try:
        sources = {document.relative_path for document in read_documents_jsonl(args.documents)}
        dataset = load_evaluation_dataset(args.dataset, sources)
        embedding = load_embedding_config(args.embedding_config)
        dense5 = load_dense_rows(args.dense_depth_5 / "per_query.jsonl")
        dense50 = load_dense_rows(args.dense_depth_50 / "per_query.jsonl")
        dense_config = json.loads((args.dense_depth_50 / "config.json").read_text(encoding="utf-8"))
        bm25_config = BM25Config()
        indexes = {}
        build_seconds = {}
        for strategy in STRATEGIES:
            started = time.monotonic()
            indexes[strategy] = BM25Index.from_path(args.chunks_root / args.corpus / strategy / "chunks.jsonl", bm25_config)
            build_seconds[strategy] = time.monotonic() - started
        lexical_fingerprints = {strategy: index.fingerprint for strategy, index in indexes.items()}
        outputs = []
        for pool_depth in (5, 10, 20, 50):
            started = time.monotonic()
            rankings = {}
            for record in dataset.records:
                for strategy in STRATEGIES:
                    candidates = dense50[(record.query_id, strategy)]["hits"][:pool_depth]
                    rankings[(record.query_id, strategy)] = indexes[strategy].rerank(record.query, candidates, 5)
            config = ExperimentConfig(
                experiment_id=f"E5-bm25-rerank-{pool_depth}-to-5", experiment_name=f"bm25_rerank_dense_{pool_depth}_to_5",
                experiment_family="E5", dataset_fingerprint=dataset.fingerprint,
                retrieval_config_fingerprint=RetrievalConfig().fingerprint,
                embedding_config_fingerprint=embedding.fingerprint,
                index_fingerprints=dense_config["index_fingerprints"], candidate_depth=pool_depth,
                ranking_method="bm25_candidate_rerank",
                reranker={"type": "bm25", "config_fingerprint": bm25_config.fingerprint,
                          "lexical_index_fingerprints": lexical_fingerprints, "candidate_pool_depth": pool_depth, "output_depth": 5},
            )
            outputs.append(publish_ranked_experiment(
                config=config, corpus=args.corpus, dataset=dataset, strategies=STRATEGIES, rankings=rankings,
                output_depth=5, output_root=args.output / args.corpus / "experiments", reference_rows=dense5,
                stats={"provider_calls": 0, "build_seconds": build_seconds, "ranking_seconds": time.monotonic() - started},
            ))
        started = time.monotonic()
        lexical_rankings = {(record.query_id, strategy): indexes[strategy].search(record.query, 50)
                            for record in dataset.records for strategy in STRATEGIES}
        lexical_config = {"method": "bm25", "config": bm25_config.identity(),
                          "config_fingerprint": bm25_config.fingerprint,
                          "lexical_index_fingerprints": lexical_fingerprints}
        config = ExperimentConfig(
            experiment_id="E6-bm25-depth-50", experiment_name="bm25_full_corpus_depth_50",
            experiment_family="E6", dataset_fingerprint=dataset.fingerprint,
            retrieval_config_fingerprint=RetrievalConfig().fingerprint,
            embedding_config_fingerprint=embedding.fingerprint,
            index_fingerprints=lexical_fingerprints, candidate_depth=50, ranking_method="bm25",
            lexical_config=lexical_config,
        )
        outputs.append(publish_ranked_experiment(
            config=config, corpus=args.corpus, dataset=dataset, strategies=STRATEGIES, rankings=lexical_rankings,
            output_depth=50, output_root=args.output / args.corpus / "experiments", reference_rows=dense50,
            stats={"provider_calls": 0, "build_seconds": build_seconds, "ranking_seconds": time.monotonic() - started},
        ))
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({"dataset_fingerprint": dataset.fingerprint, "bm25_config_fingerprint": bm25_config.fingerprint,
                      "lexical_index_fingerprints": lexical_fingerprints, "outputs": [path.as_posix() for path in outputs]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
