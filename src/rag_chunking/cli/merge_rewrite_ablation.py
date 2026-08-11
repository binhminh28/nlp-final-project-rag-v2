"""Merge memory-bounded per-strategy rewrite runs into canonical E4 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.embedding.config import load_embedding_config
from rag_chunking.evaluation.dataset import load_evaluation_dataset
from rag_chunking.retrieval.models import RetrievalConfig
from rag_chunking.tuning.rewrite_experiment import load_rewrite_records, publish_rewrite_experiments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rewrites", type=Path, required=True)
    parser.add_argument("--rewrite-config-fingerprint", required=True)
    parser.add_argument("--reference-per-query", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--documents", type=Path, default=Path("data/processed/angular/documents.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval"))
    args = parser.parse_args(argv)
    try:
        embedding = load_embedding_config(args.embedding_config)
        sources = {document.relative_path for document in read_documents_jsonl(args.documents)}
        dataset = load_evaluation_dataset(args.dataset, sources)
        rewrites = load_rewrite_records(args.rewrites, dataset)
        variants = {"Q1": {}, "Q2": {}}
        fingerprints = {}
        stats = {"provider_calls": 0, "per_strategy_runs": []}
        for path in args.raw:
            raw = json.loads(path.read_text(encoding="utf-8"))
            fingerprints.update(raw["index_fingerprints"])
            stats["per_strategy_runs"].append(raw["stats"])
            for variant in variants:
                for row in raw["variants"][variant]:
                    key = (row["query_id"], row["strategy"])
                    if key in variants[variant]:
                        raise ValueError(f"duplicate raw result {key}")
                    variants[variant][key] = row["hits"]
        strategies = sorted(fingerprints)
        expected = len(dataset.records) * len(strategies)
        if any(len(values) != expected for values in variants.values()):
            raise ValueError("raw runs do not provide complete strategy-query coverage")
        outputs = publish_rewrite_experiments(
            corpus=args.corpus, dataset=dataset, rewrites=rewrites, variants=variants,
            strategies=strategies, index_fingerprints=fingerprints, embedding_config=embedding,
            retrieval_config=RetrievalConfig(), rewrite_config_fingerprint=args.rewrite_config_fingerprint,
            reference_per_query=args.reference_per_query, output_root=args.output / args.corpus / "experiments",
            runtime_stats=stats,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({"dataset_fingerprint": dataset.fingerprint, "outputs": [path.as_posix() for path in outputs]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
