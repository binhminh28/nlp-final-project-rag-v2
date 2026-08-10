"""CLI for deterministic preprocessing of the Angular Markdown corpus."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from rag_chunking.data.loader import discover_markdown_files, load_document
from rag_chunking.data.validation import expected_relative_paths, validate_corpus
from rag_chunking.data.writer import write_processed_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Root containing Markdown files")
    parser.add_argument("--output", type=Path, required=True, help="Processed output directory")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first file error (default: continue and report every failure)",
    )
    return parser


def run(input_dir: Path, output_dir: Path, *, fail_fast: bool = False) -> int:
    files = discover_markdown_files(input_dir)
    documents = []
    failures: list[tuple[str, str]] = []

    for path in files:
        relative_path = path.resolve().relative_to(input_dir.resolve()).as_posix()
        try:
            documents.append(load_document(path, input_dir))
        except Exception as error:  # each source error must be visible in the report
            failures.append((relative_path, f"{type(error).__name__}: {error}"))
            print(f"ERROR {relative_path}: {type(error).__name__}: {error}", file=sys.stderr)
            if fail_fast:
                break

    expected = expected_relative_paths(files, input_dir)
    validation = validate_corpus(documents, expected)
    if failures:
        validation.errors.append(f"{len(failures)} source document(s) failed preprocessing")

    if validation.valid:
        write_processed_corpus(documents, output_dir)

    counts = Counter(block.type for document in documents for block in document.blocks)
    print(f"Documents discovered: {len(files)}")
    print(f"Documents processed:  {len(documents)}")
    print(f"Documents failed:     {len(failures)}")
    print(f"Total blocks:         {sum(counts.values())}")
    for block_type in (
        "heading",
        "paragraph",
        "code_block",
        "code_reference",
        "list",
        "blockquote",
        "table",
        "callout",
        "html_block",
        "custom_block",
    ):
        print(f"{block_type.replace('_', ' ').title()} blocks: {counts[block_type]}")
    print(f"Output location:      {output_dir.resolve()}")

    if validation.errors:
        print("Validation: FAILED", file=sys.stderr)
        for error in validation.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    for warning in validation.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print("Validation: PASSED")
    return 0


def main() -> None:
    arguments = build_parser().parse_args()
    raise SystemExit(run(arguments.input, arguments.output, fail_fast=arguments.fail_fast))


if __name__ == "__main__":
    main()
