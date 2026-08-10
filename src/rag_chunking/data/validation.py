"""Validation and corpus-level statistics for normalized documents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re

from .models import NORMALIZED_SCHEMA_VERSION, NormalizedDocument


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_corpus(
    documents: list[NormalizedDocument], expected_paths: list[str]
) -> ValidationReport:
    """Validate identity, coverage, structure, and serialization invariants."""

    report = ValidationReport()
    actual_paths = [document.relative_path for document in documents]
    doc_ids = [document.doc_id for document in documents]

    _report_duplicates(doc_ids, "doc_id", report)
    _report_duplicates(actual_paths, "relative_path", report)

    if len(documents) != len(expected_paths):
        report.errors.append(
            f"Expected {len(expected_paths)} documents, received {len(documents)}"
        )
    if actual_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))
        unexpected = sorted(set(actual_paths) - set(expected_paths))
        report.errors.append(
            f"Document coverage/order mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    for document in documents:
        for warning in document.metadata.get("audit", {}).get("warnings", []):
            report.warnings.append(f"{document.relative_path}: {warning}")
        if document.schema_version != NORMALIZED_SCHEMA_VERSION:
            report.errors.append(
                f"Unsupported schema for {document.relative_path}: {document.schema_version}"
            )
        if not document.blocks:
            report.errors.append(f"Empty normalized document: {document.relative_path}")
        previous_line = 0
        for position, block in enumerate(document.blocks):
            if block.type == "heading" and (block.level is None or not 1 <= block.level <= 6):
                report.errors.append(
                    f"Invalid heading level at {document.relative_path} block {position}: {block.level}"
                )
            if block.type == "code_block" and block.text is None:
                report.errors.append(
                    f"Missing code text at {document.relative_path} block {position}"
                )
            if (block.source_line_start is None) != (block.source_line_end is None):
                report.errors.append(
                    f"Incomplete source line span at {document.relative_path} block {position}"
                )
            if block.source_line_start is not None and block.source_line_end is not None:
                if block.source_line_start <= 0 or block.source_line_end < block.source_line_start:
                    report.errors.append(
                        f"Invalid source line span at {document.relative_path} block {position}"
                    )
                if block.source_line_start < previous_line:
                    report.errors.append(
                        f"Source block order regression at {document.relative_path} block {position}"
                    )
                previous_line = block.source_line_start
            if block.type == "code_reference":
                if not block.metadata.get("path") or block.metadata.get("resolved") is not False:
                    report.errors.append(
                        f"Invalid unresolved code reference at {document.relative_path} block {position}"
                    )
            if block.type == "table":
                schema = block.metadata
                if not isinstance(schema.get("header"), list) or not isinstance(
                    schema.get("rows"), list
                ):
                    report.errors.append(
                        f"Missing table schema at {document.relative_path} block {position}"
                    )
                header = schema.get("header", [])
                rows = schema.get("rows", [])
                if isinstance(header, list) and isinstance(rows, list) and any(
                    not isinstance(row, list) or len(row) != len(header) for row in rows
                ):
                    report.errors.append(
                        f"Unnormalized table width at {document.relative_path} block {position}"
                    )
            visible_for_tag_check = re.sub(r"`+[^`]*`+", "", block.text)
            if (
                block.type not in {"code_block", "html_block", "table"}
                and "<docs-" in visible_for_tag_check
            ):
                report.errors.append(
                    f"Raw Angular tag leaked at {document.relative_path} block {position}"
                )
        restored = NormalizedDocument.from_dict(document.to_dict())
        if restored.to_dict() != document.to_dict():
            report.errors.append(f"Serialization round trip changed {document.relative_path}")
    return report


def corpus_statistics(documents: list[NormalizedDocument]) -> dict[str, int]:
    counts = Counter(block.type for document in documents for block in document.blocks)
    return {
        "documents": len(documents),
        "total_blocks": sum(counts.values()),
        **{f"{block_type}_blocks": count for block_type, count in sorted(counts.items())},
        "sentences": sum(
            len(block.sentences) for document in documents for block in document.blocks
        ),
        "unresolved_code_references": sum(
            block.type == "code_reference" for document in documents for block in document.blocks
        ),
    }


def expected_relative_paths(files: list[Path], root: Path) -> list[str]:
    resolved_root = root.resolve()
    return [path.resolve().relative_to(resolved_root).as_posix() for path in files]


def _report_duplicates(values: list[str], label: str, report: ValidationReport) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        report.errors.append(f"Duplicate {label}: {duplicates[:5]}")
