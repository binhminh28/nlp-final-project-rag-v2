"""Corpus invariants for structure-aware chunks and structural provenance."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from rag_chunking.data.models import NormalizedDocument

from .models import Chunk
from .structure_aware import StructureAwareChunkingConfig, build_sections
from .tokenizer import TiktokenTokenizer
from .validation import ChunkValidationReport


def validate_structure_aware_chunks(
    documents: list[NormalizedDocument],
    chunks: list[Chunk],
    config: StructureAwareChunkingConfig,
    tokenizer: TiktokenTokenizer,
) -> ChunkValidationReport:
    report = ChunkValidationReport()
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)

    duplicates = [key for key, count in Counter(c.chunk_id for c in chunks).items() if count > 1]
    if duplicates:
        report.errors.append(f"Duplicate chunk_id: {sorted(duplicates)[:5]}")
    known = {document.doc_id for document in documents}
    unexpected = sorted(set(by_doc) - known)
    if unexpected:
        report.errors.append(f"Chunks reference unknown documents: {unexpected[:5]}")

    for document in documents:
        document_chunks = by_doc.get(document.doc_id, [])
        if document.blocks and not document_chunks:
            report.errors.append(f"Non-empty document has no chunks: {document.doc_id}")
            continue
        expected_sections = {
            f"{document.doc_id}::section::{section.section_index:06d}": section
            for section in build_sections(document)
        }
        coverage: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        contextual_headings: Counter[int] = Counter()
        previous_position = (-1, -1)
        for index, chunk in enumerate(document_chunks):
            prefix = f"{document.doc_id} chunk {index}"
            if chunk.chunk_index != index:
                report.errors.append(f"{prefix}: non-contiguous chunk_index {chunk.chunk_index}")
            if chunk.chunk_id != f"{document.doc_id}::structure::{index:06d}":
                report.errors.append(f"{prefix}: non-deterministic chunk_id")
            if chunk.strategy != "structure_aware":
                report.errors.append(f"{prefix}: incorrect strategy")
            if chunk.token_start is not None or chunk.token_end is not None:
                report.errors.append(f"{prefix}: fabricated token span")
            actual_tokens = len(tokenizer.encode(chunk.text))
            if not chunk.text or chunk.token_count != actual_tokens:
                report.errors.append(f"{prefix}: empty text or incorrect token_count")
            if actual_tokens > config.max_chunk_tokens:
                report.errors.append(f"{prefix}: exceeds max token budget ({actual_tokens})")
            if chunk.chunk_overlap != 0:
                report.errors.append(f"{prefix}: non-zero overlap")
            if chunk.tokenizer != tokenizer.name:
                report.errors.append(f"{prefix}: incorrect tokenizer")
            if "\ufffd" not in "".join(block.text for block in document.blocks) and "\ufffd" in chunk.text:
                report.unicode_decoding_issues += 1
                report.errors.append(f"{prefix}: generated Unicode replacement character")
            section_id = chunk.metadata.get("section_id")
            section = expected_sections.get(str(section_id))
            if section is None:
                report.errors.append(f"{prefix}: unknown section_id {section_id}")
            else:
                expected_path = [heading.text for heading in section.path]
                expected_levels = [heading.level for heading in section.path]
                if chunk.metadata.get("section_path") != expected_path:
                    report.errors.append(f"{prefix}: incorrect section_path")
                if chunk.metadata.get("heading_levels") != expected_levels:
                    report.errors.append(f"{prefix}: incorrect heading_levels")
            omitted = chunk.metadata.get("context_heading_block_index")
            if isinstance(omitted, int):
                contextual_headings[omitted] += 1
            for item in chunk.metadata.get("block_fragments", []):
                block_index = item.get("source_block_index")
                start = item.get("char_start")
                end = item.get("char_end")
                if not all(isinstance(value, int) for value in (block_index, start, end)):
                    report.errors.append(f"{prefix}: malformed block fragment provenance")
                    continue
                if not 0 <= block_index < len(document.blocks):
                    report.errors.append(f"{prefix}: invalid source block index {block_index}")
                    continue
                block = document.blocks[block_index]
                if not 0 <= start < end <= len(block.text):
                    report.errors.append(f"{prefix}: invalid character fragment")
                    continue
                fragment_text = block.text[start:end]
                expected_hash = hashlib.sha256(fragment_text.encode("utf-8")).hexdigest()
                if item.get("fragment_sha256") != expected_hash:
                    report.errors.append(f"{prefix}: fragment hash mismatch")
                coverage[block_index].append((start, end, index))
                position = (block_index, start)
                if position < previous_position:
                    report.errors.append(f"{prefix}: source order regression")
                previous_position = position

        for block_index, block in enumerate(document.blocks):
            spans = sorted(coverage.get(block_index, []))
            if block.type == "heading" and not spans:
                if contextual_headings[block_index] != 1:
                    report.errors.append(
                        f"{document.doc_id} block {block_index}: heading not represented exactly once"
                    )
                continue
            cursor = 0
            for start, end, _ in spans:
                if start != cursor:
                    report.coverage_gaps += abs(start - cursor)
                    report.errors.append(
                        f"{document.doc_id} block {block_index}: coverage discontinuity at {cursor}"
                    )
                    break
                cursor = end
            if cursor != len(block.text):
                report.coverage_gaps += max(0, len(block.text) - cursor)
                report.errors.append(
                    f"{document.doc_id} block {block_index}: incomplete content coverage"
                )
    return report
