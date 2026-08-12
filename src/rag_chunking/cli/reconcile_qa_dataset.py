"""Create an offline, human-reviewed dataset/corpus reconciliation package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.evaluation.reconciliation import reconcile_dataset


DEFAULT_DATASET = Path("data/evaluation/angular/qa_dataset.jsonl")
DEFAULT_COMPATIBILITY = Path("data/evaluation/angular/compatibility/qa_dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic reconciliation candidates without modifying gold data",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--compatibility", type=Path, default=DEFAULT_COMPATIBILITY)
    parser.add_argument(
        "--documents", type=Path,
        default=Path("data/processed/angular/documents.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or args.compatibility / "reconciliation"
    try:
        result = reconcile_dataset(
            dataset_path=args.dataset,
            documents=read_documents_jsonl(args.documents),
            compatibility_directory=args.compatibility,
            output_directory=output,
        )
        payload = {
            "status": "COMPLETE", "questions": result.stats["questions_reviewed"],
            "evidence_items": result.stats["evidence_items_affected"],
            "failure_rows": result.stats["failure_rows"],
            "proposals": len(result.proposals),
            "reconciliation_fingerprint": result.reconciliation_fingerprint,
            "output": result.output_directory.as_posix(),
            "benchmark_gate": "FAIL",
        }
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        payload = {"status": "FAILED", "error": str(error), "benchmark_gate": "FAIL"}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
