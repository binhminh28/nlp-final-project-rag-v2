"""Deterministic JSON/JSONL output for preprocessed corpora."""

from __future__ import annotations

import json
from pathlib import Path

from .models import NORMALIZED_SCHEMA_VERSION, NormalizedDocument
from .validation import corpus_statistics


def write_processed_corpus(documents: list[NormalizedDocument], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.jsonl"
    manifest_path = output_dir / "manifest.json"

    with documents_path.open("w", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            stream.write(
                json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            stream.write("\n")

    manifest = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "parser": sorted(
            {str(document.metadata.get("parser", "unknown")) for document in documents}
        ),
        "source": "angular",
        "statistics": corpus_statistics(documents),
        "audit": {
            "unresolved_code_references": sum(
                int(document.metadata.get("audit", {}).get("unresolved_code_references", 0))
                for document in documents
            ),
            "documents_with_warnings": sum(
                bool(document.metadata.get("audit", {}).get("warnings"))
                for document in documents
            ),
            "unknown_angular_tags": sorted(
                {
                    tag
                    for document in documents
                    for tag in document.metadata.get("audit", {}).get("unknown_angular_tags", [])
                }
            ),
        },
        "documents": [
            {
                "doc_id": document.doc_id,
                "source": document.source,
                "relative_path": document.relative_path,
                "filename": document.filename,
                "source_sha256": document.source_sha256,
            }
            for document in documents
        ],
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def read_documents_jsonl(path: Path) -> list[NormalizedDocument]:
    documents: list[NormalizedDocument] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                document = NormalizedDocument.from_dict(json.loads(line))
                if document.schema_version != NORMALIZED_SCHEMA_VERSION:
                    raise ValueError(
                        f"unsupported normalized schema {document.schema_version!r}; "
                        f"expected {NORMALIZED_SCHEMA_VERSION!r}"
                    )
                documents.append(document)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return documents
