"""Validation and corpus-level statistics for normalized documents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .models import NormalizedDocument


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
        if not document.blocks:
            report.errors.append(f"Empty normalized document: {document.relative_path}")
        for position, block in enumerate(document.blocks):
            if block.type == "heading" and (block.level is None or not 1 <= block.level <= 6):
                report.errors.append(
                    f"Invalid heading level at {document.relative_path} block {position}: {block.level}"
                )
            if block.type == "code_block" and block.text is None:
                report.errors.append(
                    f"Missing code text at {document.relative_path} block {position}"
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
    }


def expected_relative_paths(files: list[Path], root: Path) -> list[str]:
    resolved_root = root.resolve()
    return [path.resolve().relative_to(resolved_root).as_posix() for path in files]


def _report_duplicates(values: list[str], label: str, report: ValidationReport) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        report.errors.append(f"Duplicate {label}: {duplicates[:5]}")

