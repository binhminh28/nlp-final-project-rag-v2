"""Corpus-level invariants for prompt-based chunks and exact provenance."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Iterable

from rag_chunking.data.models import NormalizedDocument

from .models import Chunk
from .prompt_based import PromptBasedChunkingConfig
from .prompt_cache import canonical_json
from .prompt_client import PlannerModelConfig
from .tokenizer import TiktokenTokenizer
from .validation import ChunkValidationReport


def _render(document: NormalizedDocument, fragments: Iterable[dict[str, object]]) -> str:
    pieces: list[str] = []
    previous: int | None = None
    for fragment in fragments:
        index = int(fragment["source_block_index"])
        if previous is not None and previous != index:
            pieces.append("\n\n")
        pieces.append(document.blocks[index].text[int(fragment["char_start"]) : int(fragment["char_end"])])
        previous = index
    return "".join(pieces)


def validate_prompt_based_chunks(
    documents: list[NormalizedDocument],
    chunks: list[Chunk],
    config: PromptBasedChunkingConfig,
    model_config: PlannerModelConfig,
    tokenizer: TiktokenTokenizer,
    failed_document_ids: set[str] | None = None,
) -> ChunkValidationReport:
    report = ChunkValidationReport()
    failed = failed_document_ids or set()
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)
    duplicates = sorted(key for key, count in Counter(c.chunk_id for c in chunks).items() if count > 1)
    if duplicates:
        report.errors.append(f"Duplicate chunk_id: {duplicates[:5]}")
    known = {document.doc_id for document in documents}
    if not failed <= known:
        report.errors.append(f"Unknown failed documents: {sorted(failed - known)[:5]}")
    unexpected = sorted(set(by_doc) - known)
    if unexpected:
        report.errors.append(f"Chunks reference unknown documents: {unexpected[:5]}")

    for document in documents:
        document_chunks = by_doc.get(document.doc_id, [])
        if document.doc_id in failed:
            if document_chunks:
                report.errors.append(f"Failed document has chunks: {document.doc_id}")
            continue
        if document.blocks and not document_chunks:
            report.errors.append(f"Non-empty document has no chunks: {document.doc_id}")
            continue
        coverage: dict[int, list[tuple[int, int]]] = defaultdict(list)
        normalized_hash = hashlib.sha256(canonical_json(document.to_dict()).encode("utf-8")).hexdigest()
        previous_position = (-1, -1)
        for expected_index, chunk in enumerate(document_chunks):
            prefix = f"{document.doc_id} chunk {expected_index}"
            if chunk.chunk_index != expected_index:
                report.errors.append(f"{prefix}: non-contiguous chunk_index {chunk.chunk_index}")
            if chunk.chunk_id != f"{document.doc_id}::prompt::{expected_index:06d}":
                report.errors.append(f"{prefix}: non-deterministic chunk_id")
            if chunk.strategy != "prompt_based" or chunk.chunk_overlap != 0:
                report.errors.append(f"{prefix}: incorrect strategy or overlap")
            if chunk.token_start is not None or chunk.token_end is not None:
                report.errors.append(f"{prefix}: fabricated token span")
            actual_tokens = len(tokenizer.encode(chunk.text))
            if not chunk.text or actual_tokens != chunk.token_count or actual_tokens > config.max_chunk_tokens:
                report.errors.append(f"{prefix}: empty, miscounted, or oversized text")
            if chunk.tokenizer != tokenizer.name:
                report.errors.append(f"{prefix}: incorrect tokenizer")
            if "\ufffd" not in "".join(block.text for block in document.blocks) and "\ufffd" in chunk.text:
                report.unicode_decoding_issues += 1
                report.errors.append(f"{prefix}: generated Unicode replacement character")
            expected_provenance = {
                "source_sha256": document.source_sha256,
                "normalized_sha256": normalized_hash,
                "planner_provider": model_config.provider,
                "planner_model": model_config.model,
                "planner_base_url": model_config.base_url.rstrip("/"),
                "structured_output_policy": model_config.structured_output_policy,
                "prompt_version": config.prompt_version,
                "planner_schema_version": config.planner_schema_version,
            }
            for key, value in expected_provenance.items():
                if chunk.metadata.get(key) != value:
                    report.errors.append(f"{prefix}: inconsistent {key}")
            request_hash = chunk.metadata.get("planner_request_sha256")
            if not isinstance(request_hash, str) or len(request_hash) != 64 or any(
                character not in "0123456789abcdef" for character in request_hash
            ):
                report.errors.append(f"{prefix}: invalid planner request hash")
            if type(chunk.metadata.get("cache_hit")) is not bool:
                report.errors.append(f"{prefix}: invalid cache_hit provenance")
            if type(chunk.metadata.get("model_attempt_count")) is not int:
                report.errors.append(f"{prefix}: invalid model attempt provenance")
            if chunk.metadata.get("structured_output_mode") not in {
                "json_schema", "prompt_json", "test_or_custom"
            }:
                report.errors.append(f"{prefix}: invalid structured-output mode")
            if type(chunk.metadata.get("capability_fallback_used")) is not bool:
                report.errors.append(f"{prefix}: invalid capability fallback provenance")
            fragments = chunk.metadata.get("block_fragments")
            if not isinstance(fragments, list) or not fragments:
                report.errors.append(f"{prefix}: missing block fragments")
                continue
            valid_fragments: list[dict[str, object]] = []
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    report.errors.append(f"{prefix}: malformed fragment")
                    continue
                block_index = fragment.get("source_block_index")
                start = fragment.get("char_start")
                end = fragment.get("char_end")
                if type(block_index) is not int or type(start) is not int or type(end) is not int:
                    report.errors.append(f"{prefix}: non-integer fragment span")
                    continue
                if not 0 <= block_index < len(document.blocks) or not 0 <= start < end <= len(document.blocks[block_index].text):
                    report.errors.append(f"{prefix}: impossible fragment span")
                    continue
                source_text = document.blocks[block_index].text[start:end]
                if fragment.get("fragment_sha256") != hashlib.sha256(source_text.encode("utf-8")).hexdigest():
                    report.errors.append(f"{prefix}: fragment hash mismatch")
                if fragment.get("block_type") != document.blocks[block_index].type:
                    report.errors.append(f"{prefix}: fragment block type mismatch")
                position = (block_index, start)
                if position < previous_position:
                    report.errors.append(f"{prefix}: source order regression")
                previous_position = position
                coverage[block_index].append((start, end))
                valid_fragments.append(fragment)
            if valid_fragments and chunk.text != _render(document, valid_fragments):
                report.errors.append(f"{prefix}: text is not exact local source slicing")

        for block_index, block in enumerate(document.blocks):
            cursor = 0
            for start, end in sorted(coverage.get(block_index, [])):
                if start != cursor:
                    report.coverage_gaps += abs(start - cursor)
                    report.errors.append(
                        f"{document.doc_id} block {block_index}: coverage discontinuity at {cursor}"
                    )
                    break
                cursor = end
            if cursor != len(block.text):
                report.coverage_gaps += max(0, len(block.text) - cursor)
                report.errors.append(f"{document.doc_id} block {block_index}: incomplete coverage")
    return report
