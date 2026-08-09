"""Deterministic JSON/JSONL output for preprocessed corpora."""

from __future__ import annotations

import json
from pathlib import Path

from .models import NormalizedDocument
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
        "source": "angular",
        "statistics": corpus_statistics(documents),
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
                documents.append(NormalizedDocument.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return documents

