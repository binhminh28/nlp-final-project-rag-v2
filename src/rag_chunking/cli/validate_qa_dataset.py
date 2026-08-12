"""Strict, read-only validation for the teammate-owned canonical QA handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.evaluation.qa_dataset import (
    QA_DATASET_SCHEMA_VERSION, TEAM_QA_DATASET_SCHEMA_VERSION,
    is_team_qa_dataset, load_team_qa_dataset, validate_canonical_qa_dataset,
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
        if is_team_qa_dataset(args.dataset):
            dataset = load_team_qa_dataset(args.dataset)
            report_errors = []
            report_warnings = [
                "Schema validation passed; run audit-dataset-compatibility for the required provenance/chunk gate."
            ]
            schema_version = TEAM_QA_DATASET_SCHEMA_VERSION
        else:
            dataset, report = validate_canonical_qa_dataset(args.dataset, by_id)
            report_errors = report.errors
            report_warnings = report.warnings
            schema_version = QA_DATASET_SCHEMA_VERSION
        result = {
            "status": "PASS" if not report_errors else "BLOCKED",
            "schema_version": schema_version,
            "dataset_fingerprint": dataset.fingerprint,
            "query_count": len(dataset.records),
            "errors": report_errors, "warnings": report_warnings,
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
