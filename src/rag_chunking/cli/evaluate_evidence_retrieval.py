"""Run the controlled evidence-aware retrieval protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rag_chunking.chunking.writer import read_chunks_jsonl
from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.embedding.config import load_embedding_config
from rag_chunking.embedding.provider import OpenRouterEmbeddingProvider
from rag_chunking.evaluation.evidence_runner import run_evidence_retrieval_benchmark
from rag_chunking.evaluation.compatibility import audit_dataset_compatibility
from rag_chunking.evaluation.qa_dataset import (
    is_team_qa_dataset, load_qa_dataset, validate_qa_semantics,
)
from rag_chunking.retrieval.protocols import SAME_TOKEN_BUDGET, SAME_TOP_K, RetrievalProtocolConfig
from rag_chunking.retrieval.service import RetrievalService


STRATEGIES = ("fixed_size", "structure_aware", "prompt_based")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval with top-k and token-budget controls")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--chunks-root", type=Path, default=Path("data/chunks"))
    parser.add_argument("--indexes-root", type=Path, default=Path("data/indexes"))
    parser.add_argument("--query-cache", type=Path, default=Path("data/query-embedding-cache"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval"))
    parser.add_argument("--run-name", default="evidence_dev_v1")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        embedding = load_embedding_config(args.embedding_config)
        index_dirs = {name: args.indexes_root / args.corpus / name / embedding.fingerprint for name in args.strategies}
        service = RetrievalService(
            corpus=args.corpus, index_directories=index_dirs, embedding_config=embedding,
            provider=OpenRouterEmbeddingProvider(embedding), query_cache_directory=args.query_cache,
        )
        documents_path = args.processed_root / args.corpus / "documents.jsonl"
        document_list = read_documents_jsonl(documents_path)
        documents = {document.doc_id: document for document in document_list}
        chunks = {}
        chunk_fingerprints = {}
        chunk_manifests = {}
        for strategy in args.strategies:
            chunk_path = args.chunks_root / args.corpus / strategy / "chunks.jsonl"
            chunks[strategy] = read_chunks_jsonl(chunk_path)
            chunk_fingerprints[strategy] = _sha256(chunk_path)
            chunk_manifests[strategy] = json.loads(
                (chunk_path.parent / "manifest.json").read_text(encoding="utf-8")
            )
        if is_team_qa_dataset(args.dataset):
            compatibility = audit_dataset_compatibility(
                dataset_path=args.dataset, documents=document_list,
                chunks_by_strategy=chunks, chunk_manifests=chunk_manifests,
                raw_root=args.raw_root / args.corpus,
            )
            dataset = compatibility.dataset
            semantic_errors = [] if compatibility.passed else compatibility.report["gate_reasons"]
            semantic_warnings = compatibility.report["warnings"]
        else:
            dataset = load_qa_dataset(args.dataset, set(documents))
            semantic = validate_qa_semantics(dataset, documents)
            semantic_errors = semantic.errors
            semantic_warnings = semantic.warnings
        protocols = [
            RetrievalProtocolConfig(SAME_TOP_K, top_k=args.top_k, candidate_k=args.candidate_k),
            RetrievalProtocolConfig(SAME_TOKEN_BUDGET, candidate_k=args.candidate_k, token_budget=args.token_budget),
        ]
        plan = {
            "dataset_fingerprint": dataset.fingerprint, "queries": len(dataset.records),
            "semantic_errors": semantic_errors, "semantic_warnings": semantic_warnings,
            "protocol_fingerprints": {protocol.mode: protocol.fingerprint for protocol in protocols},
        }
        if semantic_errors:
            raise ValueError(f"QA compatibility validation failed: {semantic_errors}")
        if args.plan_only:
            print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
            return 0
        result = run_evidence_retrieval_benchmark(
            service, dataset, documents, chunks, args.output / args.corpus,
            strategies=args.strategies, protocols=protocols,
            corpus_fingerprint=_sha256(documents_path),
            chunk_artifact_fingerprints=chunk_fingerprints, run_name=args.run_name,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({**plan, "benchmark_fingerprint": result.benchmark_fingerprint, "output": result.output_directory.as_posix()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
