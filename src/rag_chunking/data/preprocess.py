"""Public preprocessing API backed by the normalized-document v2 parser."""

from __future__ import annotations

from typing import Any

from .models import DocumentBlock
from .parser_v2 import ParsedMarkdown, parse_markdown, segment_sentences


def preprocess_markdown(
    markdown: str, *, doc_id: str
) -> tuple[list[DocumentBlock], dict[str, Any]]:
    """Compatibility API returning blocks and front matter.

    Loaders should call :func:`preprocess_markdown_document` to retain document-level
    parser metadata such as resolved link definitions and corpus audit counters.
    """

    result = parse_markdown(markdown, doc_id=doc_id)
    return result.blocks, result.front_matter


def preprocess_markdown_document(markdown: str, *, doc_id: str) -> ParsedMarkdown:
    """Return blocks, front matter, and v2 document metadata."""

    return parse_markdown(markdown, doc_id=doc_id)


__all__ = [
    "preprocess_markdown",
    "preprocess_markdown_document",
    "segment_sentences",
]
