"""Deterministic discovery and loading of Markdown source documents."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import NormalizedDocument
from .preprocess import preprocess_markdown_document


def discover_markdown_files(input_dir: Path) -> list[Path]:
    """Return recursive Markdown files in stable relative-path order."""

    root = input_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    files = [path for path in root.rglob("*.md") if path.is_file()]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().casefold())


def make_doc_id(relative_path: str, source: str = "angular") -> str:
    """Build a readable identifier independent of host paths and scan order."""

    canonical_path = Path(relative_path.replace("\\", "/")).as_posix()
    return f"{source}:{canonical_path}"


def load_document(path: Path, input_dir: Path, source: str = "angular") -> NormalizedDocument:
    """Load and normalize one UTF-8 Markdown file."""

    root = input_dir.resolve()
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Source file is outside input directory: {path}") from error

    raw_bytes = resolved_path.read_bytes()
    try:
        markdown = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise UnicodeError(f"Expected UTF-8 Markdown: {relative_path}") from error

    doc_id = make_doc_id(relative_path, source)
    parsed = preprocess_markdown_document(markdown, doc_id=doc_id)
    return NormalizedDocument(
        doc_id=doc_id,
        source=source,
        relative_path=relative_path,
        filename=resolved_path.name,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        blocks=parsed.blocks,
        front_matter=parsed.front_matter,
        metadata=parsed.metadata,
    )
