"""Prepare gold-free retrieval/protocol/context inputs for answer generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rag_chunking.benchmark import prepare_answer_benchmark_inputs, project_benchmark_queries
from rag_chunking.chunking.prompt_config import load_project_dotenv
from rag_chunking.chunking.writer import read_chunks_jsonl
from rag_chunking.context import ContextConfig
from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.embedding.config import load_embedding_config
from rag_chunking.embedding.provider import OpenRouterEmbeddingProvider
from rag_chunking.evaluation.compatibility import audit_dataset_compatibility
from rag_chunking.evaluation.qa_dataset import (
    is_team_qa_dataset, validate_canonical_qa_dataset,
)
from rag_chunking.retrieval import RetrievalProtocolConfig, RetrievalService
from rag_chunking.retrieval.protocols import SAME_TOKEN_BUDGET, SAME_TOP_K


STRATEGIES = ("fixed_size", "structure_aware", "prompt_based")


class CacheOnlyEmbeddingProvider:
    """Offline guard: cache hits are allowed; a miss fails before network access."""

    def __init__(self, config):
        self.config = config
        self.calls = self.retries = self.input_tokens = 0

    def embed_texts(self, texts):
        raise ValueError("query embedding cache miss in cache-only mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare canonical answer generation inputs")
    parser.add_argument("--corpus", default="angular")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--chunks-root", type=Path, default=Path("data/chunks"))
    parser.add_argument("--indexes-root", type=Path, default=Path("data/indexes"))
    parser.add_argument("--query-cache", type=Path, default=Path("data/query-embedding-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-mode", choices=("openrouter", "cache_only"), default="cache_only")
    parser.add_argument("--protocol", choices=(SAME_TOP_K, SAME_TOKEN_BUDGET), default=SAME_TOKEN_BUDGET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        embedding = load_embedding_config(args.embedding_config)
        documents_path = args.processed_root / args.corpus / "documents.jsonl"
        documents_list = read_documents_jsonl(documents_path)
        documents = {item.doc_id: item for item in documents_list}
        if len(documents) != len(documents_list):
            raise ValueError("canonical corpus contains duplicate document IDs")
        if is_team_qa_dataset(args.dataset):
            chunks = {
                strategy: read_chunks_jsonl(
                    args.chunks_root / args.corpus / strategy / "chunks.jsonl"
                ) for strategy in STRATEGIES
            }
            manifests = {
                strategy: json.loads((
                    args.chunks_root / args.corpus / strategy / "manifest.json"
                ).read_text(encoding="utf-8")) for strategy in STRATEGIES
            }
            compatibility = audit_dataset_compatibility(
                dataset_path=args.dataset, documents=documents_list,
                chunks_by_strategy=chunks, chunk_manifests=manifests,
                raw_root=args.raw_root / args.corpus,
            )
            if not compatibility.passed:
                raise ValueError(
                    "benchmark compatibility gate failed: "
                    + "; ".join(compatibility.report["gate_reasons"])
                )
            dataset = compatibility.dataset
        else:
            dataset, semantic = validate_canonical_qa_dataset(args.dataset, documents)
            if not semantic.valid:
                raise ValueError(f"canonical QA semantic validation failed: {semantic.errors}")
        if args.embedding_mode == "openrouter":
            load_project_dotenv()
            provider = OpenRouterEmbeddingProvider(embedding)
        else:
            provider = CacheOnlyEmbeddingProvider(embedding)
        index_dirs = {
            strategy: args.indexes_root / args.corpus / strategy / embedding.fingerprint
            for strategy in STRATEGIES
        }
        service = RetrievalService(
            corpus=args.corpus, index_directories=index_dirs,
            embedding_config=embedding, provider=provider,
            query_cache_directory=args.query_cache,
        )
        protocol = RetrievalProtocolConfig(
            args.protocol, top_k=args.top_k, candidate_k=args.candidate_k,
            token_budget=args.token_budget if args.protocol == SAME_TOKEN_BUDGET else None,
        )
        context_config = ContextConfig(context_token_budget=args.context_token_budget)
        result = prepare_answer_benchmark_inputs(
            service, project_benchmark_queries(dataset.records),
            dataset_fingerprint=dataset.fingerprint,
            corpus_fingerprint=hashlib.sha256(documents_path.read_bytes()).hexdigest(),
            protocol=protocol, context_config=context_config,
            output_directory=args.output,
        )
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({
        "status": "PASS", "dataset_fingerprint": dataset.fingerprint,
        "preparation_fingerprint": result.preparation_fingerprint,
        "query_count": len(dataset.records), "strategies": list(STRATEGIES),
        "reused": result.reused, "output": result.output_directory.as_posix(),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
