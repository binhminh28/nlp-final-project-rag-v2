"""Run justified E7 dense plus BM25 reciprocal-rank fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.embedding.config import load_embedding_config
from rag_chunking.evaluation.dataset import load_evaluation_dataset
from rag_chunking.retrieval.models import RetrievalConfig
from rag_chunking.tuning.config import ExperimentConfig
from rag_chunking.tuning.fusion import RRF_SCHEMA_VERSION, reciprocal_rank_fusion
from rag_chunking.tuning.lexical import load_dense_rows
from rag_chunking.tuning.publish import publish_ranked_experiment


STRATEGIES = ["fixed_size", "structure_aware", "prompt_based"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run E7 dense plus BM25 RRF")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--documents", type=Path, default=Path("data/processed/angular/documents.jsonl"))
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval"))
    args = parser.parse_args(argv)
    try:
        sources = {document.relative_path for document in read_documents_jsonl(args.documents)}
        dataset = load_evaluation_dataset(args.dataset, sources)
        embedding = load_embedding_config(args.embedding_config)
        dense = load_dense_rows(args.dense / "per_query.jsonl")
        lexical = load_dense_rows(args.lexical / "per_query.jsonl")
        dense_config = json.loads((args.dense / "config.json").read_text(encoding="utf-8"))
        lexical_config = json.loads((args.lexical / "config.json").read_text(encoding="utf-8"))
        rankings = {}
        for key in sorted(dense):
            if key not in lexical:
                raise ValueError(f"lexical results missing {key}")
            rankings[key] = reciprocal_rank_fusion(
                [dense[key]["hits"], lexical[key]["hits"]], rank_constant=args.rank_constant, limit=50,
            )
        config = ExperimentConfig(
            experiment_id="E7-dense-bm25-rrf-50", experiment_name="dense_bm25_rrf_depth_50",
            experiment_family="E7", dataset_fingerprint=dataset.fingerprint,
            retrieval_config_fingerprint=RetrievalConfig().fingerprint,
            embedding_config_fingerprint=embedding.fingerprint,
            index_fingerprints=dense_config["index_fingerprints"], candidate_depth=50,
            ranking_method="reciprocal_rank_fusion", lexical_config=lexical_config["lexical_config"],
            fusion_config={"method": "rrf", "schema_version": RRF_SCHEMA_VERSION,
                           "rank_constant": args.rank_constant, "dense_depth": 50,
                           "lexical_depth": 50, "output_depth": 50},
        )
        output = publish_ranked_experiment(
            config=config, corpus=args.corpus, dataset=dataset, strategies=STRATEGIES,
            rankings=rankings, output_depth=50, output_root=args.output / args.corpus / "experiments",
            reference_rows=dense, stats={"provider_calls": 0},
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({"dataset_fingerprint": dataset.fingerprint, "output": output.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
