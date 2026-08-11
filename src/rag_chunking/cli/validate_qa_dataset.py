"""Strict, read-only validation for the teammate-owned canonical QA handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.evaluation.qa_dataset import (
    QA_DATASET_SCHEMA_VERSION, validate_canonical_qa_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a canonical evidence QA dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--documents", type=Path,
        default=Path("data/processed/angular/documents.jsonl"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        documents = read_documents_jsonl(args.documents)
        by_id = {document.doc_id: document for document in documents}
        if len(by_id) != len(documents):
            raise ValueError("canonical corpus contains duplicate document IDs")
        dataset, report = validate_canonical_qa_dataset(args.dataset, by_id)
        result = {
            "status": "PASS" if report.valid else "BLOCKED",
            "schema_version": QA_DATASET_SCHEMA_VERSION,
            "dataset_fingerprint": dataset.fingerprint,
            "query_count": len(dataset.records),
            "errors": report.errors, "warnings": report.warnings,
        }
    except (OSError, UnicodeError, ValueError) as error:
        result = {
            "status": "BLOCKED", "schema_version": QA_DATASET_SCHEMA_VERSION,
            "errors": [str(error)], "warnings": [],
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
