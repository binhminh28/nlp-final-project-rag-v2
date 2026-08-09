"""Canonical linearization of normalized source content."""

from __future__ import annotations

from rag_chunking.data.models import NormalizedDocument


BLOCK_SEPARATOR = "\n\n"


def document_to_text(document: NormalizedDocument) -> str:
    """Join every source-ordered block text with one deterministic separator."""

    return BLOCK_SEPARATOR.join(block.text for block in document.blocks)
