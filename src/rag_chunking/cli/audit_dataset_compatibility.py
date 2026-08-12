"""Audit the real team QA dataset without retrieval or generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.chunking.writer import read_chunks_jsonl
from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.evaluation.compatibility import audit_dataset_compatibility


STRATEGIES = ("fixed_size", "structure_aware", "prompt_based")
DEFAULT_DATASET = Path("data/evaluation/angular/qa_dataset.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/angular/compatibility/qa_dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline evidence-to-chunk dataset compatibility gate",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", default="angular")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunks-root", type=Path, default=Path("data/chunks"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        documents = read_documents_jsonl(
            args.processed_root / args.corpus / "documents.jsonl"
        )
        chunks = {}
        manifests = {}
        for strategy in args.strategies:
            directory = args.chunks_root / args.corpus / strategy
            chunks[strategy] = read_chunks_jsonl(directory / "chunks.jsonl")
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError(f"{strategy} manifest must be an object")
            manifests[strategy] = manifest
        result = audit_dataset_compatibility(
            dataset_path=args.dataset, documents=documents,
            chunks_by_strategy=chunks, chunk_manifests=manifests,
            raw_root=args.raw_root / args.corpus,
            output_directory=None if args.no_write else args.output,
        )
        payload = {
            "status": result.report["gate_decision"],
            "dataset_fingerprint": result.dataset.fingerprint,
            "compatibility_fingerprint": result.report["compatibility_fingerprint"],
            "questions": result.report["question_count"],
            "compatible_questions": result.report["compatible_question_count"],
            "unresolved_cases": result.report["unresolved_case_count"],
            "output": result.output_directory.as_posix() if result.output_directory else None,
        }
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        payload = {"status": "FAIL", "error": str(error)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
