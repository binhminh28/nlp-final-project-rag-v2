"""Audit frozen retrieval candidates and publish a revised dataset when needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.evaluation.audit import build_relevance_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic relevance-audit artifacts")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revised-dataset-name", default="baseline_v2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sources = {document.relative_path for document in read_documents_jsonl(args.documents)}
        result = build_relevance_audit(
            dataset_path=args.dataset, baseline_directory=args.baseline,
            decisions_path=args.decisions, corpus_sources=sources,
            output_directory=args.output, revised_dataset_name=args.revised_dataset_name,
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({
        "audit_fingerprint": result.audit_fingerprint,
        "candidate_count": result.candidate_count,
        "dataset_fingerprint": result.dataset_fingerprint,
        "labels_added": result.labels_added, "labels_removed": result.labels_removed,
        "output": result.output_directory.as_posix(), "query_count": result.query_count,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
