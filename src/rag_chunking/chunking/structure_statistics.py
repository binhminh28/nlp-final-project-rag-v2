"""Descriptive statistics for structure-aware chunk artifacts."""

from __future__ import annotations

import statistics
from collections import Counter

from rag_chunking.data.models import NormalizedDocument

from .models import Chunk
from .statistics import _distribution, _percentile


def structure_corpus_statistics(
    documents: list[NormalizedDocument], chunks: list[Chunk]
) -> dict[str, object]:
    chunk_counts = Counter(chunk.doc_id for chunk in chunks)
    chunks_per_document = [chunk_counts[document.doc_id] for document in documents]
    tokens = [chunk.token_count for chunk in chunks]
    token_distribution = _distribution(tokens)
    token_distribution.update(
        {"p25": _percentile(tokens, 0.25), "p75": _percentile(tokens, 0.75), "p95": _percentile(tokens, 0.95)}
    )
    section_counts = Counter(str(chunk.metadata["section_id"]) for chunk in chunks)
    boundary_counts = Counter(str(chunk.metadata["boundary_reason"]) for chunk in chunks)
    containing = Counter()
    oversized_blocks: dict[tuple[str, int], str] = {}
    split_chunks: set[str] = set()
    token_fallback_blocks: set[tuple[str, int]] = set()
    block_boundary_chunks = 0
    document_map = {document.doc_id: document for document in documents}
    for chunk in chunks:
        containing.update(set(chunk.metadata["block_types"]))
        fragments = chunk.metadata["block_fragments"]
        for fragment in fragments:
            block_index = int(fragment["source_block_index"])
            if int(fragment["fragment_count"]) > 1:
                key = (chunk.doc_id, block_index)
                oversized_blocks[key] = document_map[chunk.doc_id].blocks[block_index].type
                split_chunks.add(chunk.chunk_id)
            if fragment["token_fallback"]:
                token_fallback_blocks.add((chunk.doc_id, block_index))
        if fragments:
            last = fragments[-1]
            last_index = int(last["source_block_index"])
            block_boundary_chunks += int(
                int(last["char_end"]) == len(document_map[chunk.doc_id].blocks[last_index].text)
            )
    oversized_by_type = Counter(oversized_blocks.values())
    section_total = len(section_counts)
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "chunks_per_document": _distribution(chunks_per_document),
        "tokens_per_chunk": token_distribution,
        "structural": {
            "unique_sections": section_total,
            "average_chunks_per_section": statistics.fmean(section_counts.values()) if section_counts else 0.0,
            "sections_split_into_multiple_chunks": sum(count > 1 for count in section_counts.values()),
            "section_boundary_chunks": boundary_counts["section_end"],
            "block_boundary_chunks": block_boundary_chunks,
            "internal_block_split_chunks": len(split_chunks),
            "section_boundary_percentage": 100.0 * boundary_counts["section_end"] / len(chunks) if chunks else 0.0,
            "block_boundary_percentage": 100.0 * block_boundary_chunks / len(chunks) if chunks else 0.0,
            "internal_block_split_percentage": 100.0 * len(split_chunks) / len(chunks) if chunks else 0.0,
            "boundary_reasons": dict(sorted(boundary_counts.items())),
            "chunks_containing": dict(sorted(containing.items())),
        },
        "oversized_blocks": {
            "total": len(oversized_blocks),
            "paragraph": oversized_by_type["paragraph"],
            "blockquote": oversized_by_type["blockquote"],
            "code_block": oversized_by_type["code_block"],
            "list": oversized_by_type["list"],
            "table": oversized_by_type["table"],
            "custom_block": oversized_by_type["custom_block"],
            "heading": oversized_by_type["heading"],
            "token_fallbacks": len(token_fallback_blocks),
        },
    }
