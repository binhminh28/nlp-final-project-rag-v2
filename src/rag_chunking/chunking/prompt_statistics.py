"""Deterministic descriptive statistics for prompt-based chunks."""

from __future__ import annotations

from collections import Counter

from rag_chunking.data.models import NormalizedDocument

from .models import Chunk
from .prompt_based import PromptRunMetrics
from .statistics import _distribution, _percentile


def prompt_corpus_statistics(
    documents: list[NormalizedDocument], chunks: list[Chunk], metrics: PromptRunMetrics
) -> dict[str, object]:
    counts = Counter(chunk.doc_id for chunk in chunks)
    document_map = {document.doc_id: document for document in documents}
    successful = [document for document in documents if document.doc_id not in metrics.failed_documents]
    chunk_distribution = _distribution([counts[document.doc_id] for document in successful])
    token_values = [chunk.token_count for chunk in chunks]
    token_distribution = _distribution(token_values)
    token_distribution.update(
        {key: _percentile(token_values, percentile) for key, percentile in (("p25", .25), ("p75", .75), ("p95", .95))}
    )
    groups_adjusted = {
        (chunk.doc_id, chunk.metadata["planner_group_index"])
        for chunk in chunks if chunk.metadata["locally_adjusted"]
    }
    planner_batches = {
        (chunk.doc_id, chunk.metadata["planner_batch_index"]) for chunk in chunks
    }
    planner_groups = {
        (chunk.doc_id, chunk.metadata["planner_group_index"]) for chunk in chunks
    }
    budget_adjusted_batches = {
        (chunk.doc_id, chunk.metadata["planner_batch_index"])
        for chunk in chunks
        if chunk.metadata.get("local_budget_adjustment") is not None
    }
    budget_adjustments = Counter(
        int(chunk.metadata["local_budget_adjustment"]["used_max_response_tokens"])
        for chunk in chunks
        if chunk.metadata.get("local_budget_adjustment") is not None
    )
    oversized_blocks = {
        (chunk.doc_id, fragment["source_block_index"])
        for chunk in chunks for fragment in chunk.metadata["block_fragments"]
        if fragment["fragment_count"] > 1
    }
    fallback_fragments = [
        fragment for chunk in chunks for fragment in chunk.metadata["block_fragments"]
        if fragment["fragment_count"] > 1
    ]
    block_boundary = sum(
        chunk.metadata["block_fragments"][-1]["char_end"]
        == len(document_map[chunk.doc_id].blocks[
            chunk.metadata["block_fragments"][-1]["source_block_index"]
        ].text)
        for chunk in chunks
    )
    internal_split = sum(
        any(fragment["char_start"] > 0 or fragment["char_end"] < len(
            document_map[chunk.doc_id].blocks[fragment["source_block_index"]].text
        ) for fragment in chunk.metadata["block_fragments"])
        for chunk in chunks
    )
    section_crossing = sum(chunk.metadata["crosses_section_boundary"] for chunk in chunks)
    group_total = len(planner_groups)
    adjusted_total = len(groups_adjusted)
    chunk_total = len(chunks)
    adjustment_reasons = Counter(
        str(chunk.metadata["adjustment_reason"])
        for chunk in chunks
        if chunk.metadata["locally_adjusted"]
    )
    section_boundary_distribution = Counter(
        max(0, len(chunk.metadata["section_paths"]) - 1) for chunk in chunks
    )
    prompt = {
        "planner_batches": len(planner_batches),
        "planner_groups": group_total,
        "cache_hits": metrics.cache_hits,
        "cache_misses": metrics.cache_misses,
        "model_calls": metrics.model_calls,
        "retries": metrics.retries,
        "invalid_model_responses": metrics.invalid_model_responses,
        "capability_fallbacks": metrics.capability_fallbacks,
        "documents_requiring_retry": len(metrics.documents_requiring_retry),
        "documents_failed": len(metrics.failed_documents),
        "planner_groups_accepted_unchanged": group_total - adjusted_total,
        "planner_groups_locally_adjusted": len(groups_adjusted),
        "planner_group_adjustment_rate": adjusted_total / group_total if group_total else 0.0,
        "adjustment_reasons": dict(sorted(adjustment_reasons.items())),
        "locally_adjusted_chunks": sum(chunk.metadata["locally_adjusted"] for chunk in chunks),
        "oversized_source_blocks": len(oversized_blocks),
        "deterministic_fallback_fragment_count": len(fallback_fragments),
        "unicode_safe_token_fallback_count": sum(item["token_fallback"] for item in fallback_fragments),
        "section_crossing_chunks": section_crossing,
        "section_crossing_rate": section_crossing / chunk_total if chunk_total else 0.0,
        "section_boundaries_crossed_distribution": {
            str(key): value for key, value in sorted(section_boundary_distribution.items())
        },
        "block_boundary_chunks": block_boundary,
        "block_boundary_rate": block_boundary / chunk_total if chunk_total else 0.0,
        "internal_block_split_chunks": internal_split,
        "internal_block_split_rate": internal_split / chunk_total if chunk_total else 0.0,
        "cache_hit_rate": metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
        if metrics.cache_hits + metrics.cache_misses else 0.0,
        "structured_output_modes": dict(sorted(Counter(
            str(chunk.metadata["structured_output_mode"]) for chunk in chunks
        ).items())),
        "planner_batches_with_output_budget_adjustment": len(budget_adjusted_batches),
        "output_budget_adjusted_chunk_distribution": {
            str(key): value for key, value in sorted(budget_adjustments.items())
        },
    }
    return {
        "documents": len(documents),
        "documents_chunked": len(successful),
        "documents_failed": len(metrics.failed_documents),
        "chunks": len(chunks),
        "chunks_per_document": chunk_distribution,
        "tokens_per_chunk": token_distribution,
        "prompt": prompt,
    }
