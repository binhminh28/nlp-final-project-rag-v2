"""Document loading and normalization primitives."""

from .loader import discover_markdown_files, load_document
from .models import DocumentBlock, NormalizedDocument, Sentence
from .preprocess import preprocess_markdown

__all__ = [
    "DocumentBlock",
    "NormalizedDocument",
    "Sentence",
    "discover_markdown_files",
    "load_document",
    "preprocess_markdown",
]

