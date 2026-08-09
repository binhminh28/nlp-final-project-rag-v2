"""Deterministic corpus statistics for chunking artifacts."""

from __future__ import annotations

import statistics
from collections import Counter
from rag_chunking.data.models import NormalizedDocument

from .models import Chunk
from .serialization import document_to_text
from .tokenizer import TiktokenTokenizer


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {key: 0 for key in ("min", "mean", "median", "max")}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def chunk_corpus_statistics(
    documents: list[NormalizedDocument],
    chunks: list[Chunk],
    tokenizer: TiktokenTokenizer,
) -> dict[str, object]:
    source_tokens = {
        document.doc_id: len(tokenizer.encode(document_to_text(document)))
        for document in documents
    }
    source_has_replacement = {
        document.doc_id: "\ufffd" in document_to_text(document) for document in documents
    }
    chunk_counts = Counter(chunk.doc_id for chunk in chunks)
    chunks_per_document = [chunk_counts[document.doc_id] for document in documents]
    tokens_per_chunk = [chunk.token_count for chunk in chunks]
    top_documents = sorted(
        (
            {
                "relative_path": document.relative_path,
                "source_tokens": source_tokens[document.doc_id],
                "chunks": chunk_counts[document.doc_id],
            }
            for document in documents
        ),
        key=lambda item: (-item["chunks"], -item["source_tokens"], item["relative_path"]),
    )[:10]
    token_distribution = _distribution(tokens_per_chunk)
    token_distribution.update(
        {
            "p25": _percentile(tokens_per_chunk, 0.25),
            "p75": _percentile(tokens_per_chunk, 0.75),
            "p95": _percentile(tokens_per_chunk, 0.95),
        }
    )
    adjusted_chunks = [chunk for chunk in chunks if chunk.metadata["boundary_adjusted"]]
    affected_documents = sorted({chunk.doc_id for chunk in adjusted_chunks})
    chunks_by_doc: dict[str, list[Chunk]] = {
        document.doc_id: [chunk for chunk in chunks if chunk.doc_id == document.doc_id]
        for document in documents
    }
    coverage_gap_tokens = 0
    for document in documents:
        covered_until = 0
        for chunk in chunks_by_doc[document.doc_id]:
            coverage_gap_tokens += max(0, chunk.token_start - covered_until)
            covered_until = max(covered_until, chunk.token_end)
        coverage_gap_tokens += max(0, source_tokens[document.doc_id] - covered_until)
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "source_tokens": sum(source_tokens.values()),
        "total_token_occurrences": sum(tokens_per_chunk),
        "chunks_per_document": _distribution(chunks_per_document),
        "tokens_per_chunk": token_distribution,
        "unicode_safe_boundaries": {
            "boundary_adjusted_chunks": len(adjusted_chunks),
            "documents_affected": len(affected_documents),
            "maximum_adjustment_tokens": max(
                (
                    max(chunk.metadata["start_adjustment"], chunk.metadata["end_adjustment"])
                    for chunk in adjusted_chunks
                ),
                default=0,
            ),
            "text_token_roundtrip_issues": sum(
                not chunk.metadata["text_token_roundtrip"] for chunk in chunks
            ),
            "generated_replacement_character_chunks": sum(
                "\ufffd" in chunk.text and not source_has_replacement[chunk.doc_id]
                for chunk in chunks
            ),
            "token_coverage_gap_positions": coverage_gap_tokens,
        },
        "top_documents_by_chunks": top_documents,
    }
