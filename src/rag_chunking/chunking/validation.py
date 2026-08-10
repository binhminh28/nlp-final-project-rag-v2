"""Validation invariants for fixed-size chunk artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rag_chunking.data.models import NormalizedDocument

from .fixed_size import FixedSizeChunkingConfig, fixed_size_windows
from .models import Chunk
from .serialization import document_to_text
from .tokenizer import TiktokenTokenizer


@dataclass(slots=True)
class ChunkValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage_gaps: int = 0
    unicode_decoding_issues: int = 0

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_unified_chunk_contract(
    documents: list[NormalizedDocument], chunks: list[Chunk]
) -> list[str]:
    """Validate schema rules shared by every chunking strategy."""

    errors: list[str] = []
    documents_by_id = {document.doc_id: document for document in documents}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for chunk in chunks:
        prefix = f"{chunk.doc_id} chunk {chunk.chunk_index}"
        try:
            chunk.validate()
        except ValueError as error:
            errors.append(f"{prefix}: invalid unified chunk: {error}")
            continue
        document = documents_by_id.get(chunk.doc_id)
        if document is None:
            continue
        if (
            chunk.source != document.source
            or chunk.relative_path != document.relative_path
            or chunk.metadata.get("source_sha256") != document.source_sha256
        ):
            errors.append(f"{prefix}: incorrect source provenance")
        if chunk.parent_id == chunk.chunk_id or chunk.chunk_id in chunk.children_ids:
            errors.append(f"{prefix}: self-referential hierarchy")
        if chunk.parent_id is not None:
            parent = chunks_by_id.get(chunk.parent_id)
            if parent is None:
                errors.append(f"{prefix}: parent_id does not resolve")
            elif parent.doc_id != chunk.doc_id or chunk.chunk_id not in parent.children_ids:
                errors.append(f"{prefix}: parent relationship is not reciprocal in the same document")
        for child_id in chunk.children_ids:
            child = chunks_by_id.get(child_id)
            if child is None:
                errors.append(f"{prefix}: child id does not resolve: {child_id}")
            elif child.doc_id != chunk.doc_id or child.parent_id != chunk.chunk_id:
                errors.append(f"{prefix}: child relationship is not reciprocal in the same document")
    return errors


def validate_fixed_size_chunks(
    documents: list[NormalizedDocument],
    chunks: list[Chunk],
    config: FixedSizeChunkingConfig,
    tokenizer: TiktokenTokenizer,
) -> ChunkValidationReport:
    report = ChunkValidationReport()
    report.errors.extend(validate_unified_chunk_contract(documents, chunks))
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)

    duplicate_ids = sorted(
        chunk_id for chunk_id, count in Counter(c.chunk_id for c in chunks).items() if count > 1
    )
    if duplicate_ids:
        report.errors.append(f"Duplicate chunk_id: {duplicate_ids[:5]}")

    known_doc_ids = {document.doc_id for document in documents}
    unexpected = sorted(set(by_doc) - known_doc_ids)
    if unexpected:
        report.errors.append(f"Chunks reference unknown documents: {unexpected[:5]}")

    for document in documents:
        source_text = document_to_text(document)
        source_tokens = tokenizer.encode(source_text)
        document_chunks = by_doc.get(document.doc_id, [])
        if source_tokens and not document_chunks:
            report.errors.append(f"Non-empty document has no chunks: {document.doc_id}")
            continue
        if not source_tokens and document_chunks:
            report.errors.append(f"Empty document has chunks: {document.doc_id}")
            continue

        expected_windows = fixed_size_windows(source_tokens, config, tokenizer)

        if len(document_chunks) != len(expected_windows):
            report.errors.append(
                f"Wrong chunk count for {document.doc_id}: "
                f"{len(document_chunks)} != {len(expected_windows)}"
            )
        covered_until = 0
        previous_end = 0
        for index, chunk in enumerate(document_chunks):
            try:
                chunk.validate()
            except ValueError:
                continue
            expected_window = expected_windows[index] if index < len(expected_windows) else None
            expected_start = expected_window.token_start if expected_window else None
            expected_end = expected_window.token_end if expected_window else None
            prefix = f"{document.doc_id} chunk {index}"
            if chunk.chunk_index != index:
                report.errors.append(f"{prefix}: non-contiguous chunk_index {chunk.chunk_index}")
            if chunk.chunk_id != f"{document.doc_id}::fixed::{index:06d}":
                report.errors.append(f"{prefix}: non-deterministic chunk_id")
            if chunk.token_start != expected_start or chunk.token_end != expected_end:
                report.errors.append(
                    f"{prefix}: span [{chunk.token_start},{chunk.token_end}) != "
                    f"[{expected_start},{expected_end})"
                )
            if chunk.token_count != chunk.token_end - chunk.token_start:
                report.errors.append(f"{prefix}: token_count does not match span")
            if chunk.token_count > config.chunk_size or chunk.token_count == 0:
                report.errors.append(f"{prefix}: invalid token_count {chunk.token_count}")
            if chunk.token_start > covered_until:
                report.coverage_gaps += chunk.token_start - covered_until
                report.errors.append(
                    f"{prefix}: source token gap [{covered_until},{chunk.token_start})"
                )
            covered_until = max(covered_until, chunk.token_end)
            token_slice = source_tokens[chunk.token_start : chunk.token_end]
            if chunk.token_count != len(token_slice):
                report.errors.append(f"{prefix}: token_count does not match source token slice")
            try:
                strict_text = tokenizer.decode_strict(token_slice)
            except UnicodeDecodeError:
                strict_text = None
                report.unicode_decoding_issues += 1
                report.errors.append(f"{prefix}: token slice is not valid UTF-8")
            if strict_text is not None and chunk.text != strict_text:
                report.errors.append(f"{prefix}: text does not strictly decode from source token slice")
            try:
                chunk.text.encode("utf-8", errors="strict")
            except UnicodeEncodeError:
                report.unicode_decoding_issues += 1
                report.errors.append(f"{prefix}: chunk text is not valid UTF-8")
            if "\ufffd" not in source_text and "\ufffd" in chunk.text:
                report.unicode_decoding_issues += 1
                report.errors.append(f"{prefix}: generated Unicode replacement character")
            expected_roundtrip = tokenizer.encode(chunk.text) == token_slice
            if chunk.metadata.get("text_token_roundtrip") is not expected_roundtrip:
                report.errors.append(f"{prefix}: incorrect text_token_roundtrip metadata")
            expected_overlap = previous_end - chunk.token_start if index else 0
            expected_metadata = {
                "nominal_chunk_size": config.chunk_size,
                "nominal_overlap": config.chunk_overlap,
                "actual_token_count": chunk.token_count,
                "actual_overlap": expected_overlap,
                "boundary_adjusted": expected_window.boundary_adjusted if expected_window else None,
                "start_adjustment": expected_window.start_adjustment if expected_window else None,
                "end_adjustment": expected_window.end_adjustment if expected_window else None,
            }
            for key, expected_value in expected_metadata.items():
                if chunk.metadata.get(key) != expected_value:
                    report.errors.append(
                        f"{prefix}: incorrect {key} metadata "
                        f"{chunk.metadata.get(key)!r} != {expected_value!r}"
                    )
            if chunk.strategy != "fixed_size":
                report.errors.append(f"{prefix}: incorrect strategy {chunk.strategy}")
            if chunk.chunk_size != config.chunk_size or chunk.chunk_overlap != config.chunk_overlap:
                report.errors.append(f"{prefix}: incorrect chunk configuration")
            if (
                chunk.level != 0
                or chunk.parent_id is not None
                or chunk.children_ids
                or chunk.title_path
            ):
                report.errors.append(f"{prefix}: fixed-size chunk has hierarchy data")
            if chunk.tokenizer != tokenizer.name:
                report.errors.append(f"{prefix}: incorrect tokenizer {chunk.tokenizer}")
            previous_end = chunk.token_end
        if covered_until < len(source_tokens):
            report.coverage_gaps += len(source_tokens) - covered_until
            report.errors.append(
                f"{document.doc_id}: source token gap [{covered_until},{len(source_tokens)})"
            )
    return report
