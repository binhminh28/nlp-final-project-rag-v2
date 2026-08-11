"""Deterministic mapping from source-grounded QA evidence to chunk artifacts."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass

from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.serialization import BLOCK_SEPARATOR, document_to_text
from rag_chunking.chunking.tokenizer import TiktokenTokenizer
from rag_chunking.data.models import NormalizedDocument

from .qa_dataset import EvidenceSpec, QARecord


EVIDENCE_MAPPING_SCHEMA_VERSION = "evidence_chunk_mapping_v1"


@dataclass(frozen=True, slots=True)
class ChunkEvidenceMatch:
    chunk_id: str
    evidence_coverage: float
    evidence_char_start: int | None = None
    evidence_char_end: int | None = None


@dataclass(frozen=True, slots=True)
class EvidenceMapping:
    evidence_id: str
    doc_id: str
    strategy: str
    evidence_text: str
    matched_chunk_ids: list[str]
    match_method: str
    coverage: float
    matches: list[ChunkEvidenceMatch]
    schema_version: str = EVIDENCE_MAPPING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _block_offsets(document: NormalizedDocument) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for index, block in enumerate(document.blocks):
        offsets.append(cursor)
        cursor += len(block.text)
        if index + 1 < len(document.blocks):
            cursor += len(BLOCK_SEPARATOR)
    return offsets


def _chunk_source_spans(
    chunk: Chunk, document: NormalizedDocument, tokenizer: TiktokenTokenizer,
) -> list[tuple[int, int]]:
    """Resolve the source characters represented by one chunk."""

    source_text = document_to_text(document)
    if chunk.token_start is not None and chunk.token_end is not None:
        tokens = tokenizer.encode(source_text)
        return [
            (len(tokenizer.decode(tokens[: chunk.token_start])),
             len(tokenizer.decode(tokens[: chunk.token_end])))
        ]
    offsets = _block_offsets(document)
    spans: list[tuple[int, int]] = []
    fragments = chunk.metadata.get("block_fragments", [])
    if isinstance(fragments, list):
        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue
            block_index = fragment.get("source_block_index")
            start = fragment.get("char_start")
            end = fragment.get("char_end")
            if type(block_index) is int and type(start) is int and type(end) is int:
                if 0 <= block_index < len(offsets):
                    spans.append((offsets[block_index] + start, offsets[block_index] + end))
    return spans


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _find_all(text: str, needle: str) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    while True:
        start = text.find(needle, cursor)
        if start < 0:
            return spans
        spans.append((start, start + len(needle)))
        cursor = start + max(1, len(needle))


def _source_evidence_spans(
    evidence: EvidenceSpec, document: NormalizedDocument,
) -> tuple[list[tuple[int, int]], str]:
    offsets = _block_offsets(document)
    if evidence.block_index is not None and evidence.char_start is not None and evidence.char_end is not None:
        if evidence.block_index >= len(document.blocks):
            return [], "source_span"
        block = document.blocks[evidence.block_index]
        if evidence.char_end > len(block.text):
            return [], "source_span"
        return [(offsets[evidence.block_index] + evidence.char_start,
                 offsets[evidence.block_index] + evidence.char_end)], "source_span"
    spans = _find_all(document_to_text(document), evidence.text)
    return spans, "canonical_text_overlap"


def _merge_coverage(intervals: list[tuple[int, int]], required: tuple[int, int]) -> float:
    clipped = sorted((max(start, required[0]), min(end, required[1])) for start, end in intervals if _overlap((start, end), required))
    if not clipped:
        return 0.0
    covered = 0
    start, end = clipped[0]
    for next_start, next_end in clipped[1:]:
        if next_start > end:
            covered += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    covered += end - start
    return min(1.0, covered / (required[1] - required[0]))


def map_evidence_to_chunks(
    record: QARecord, document: NormalizedDocument, chunks: list[Chunk], strategy: str,
    *, tokenizer: TiktokenTokenizer | None = None,
) -> list[EvidenceMapping]:
    """Derive strategy-specific chunk relevance without changing the QA record."""

    if record.doc_id != document.doc_id:
        raise ValueError("QA record and source document doc_id differ")
    tokenizer = tokenizer or TiktokenTokenizer()
    strategy_chunks = sorted(
        (chunk for chunk in chunks if chunk.doc_id == record.doc_id and chunk.strategy == strategy),
        key=lambda chunk: chunk.chunk_index,
    )
    span_map = {chunk.chunk_id: _chunk_source_spans(chunk, document, tokenizer) for chunk in strategy_chunks}
    mappings: list[EvidenceMapping] = []
    evidence_units = record.evidence_sentences
    if evidence_units:
        for index, evidence in enumerate(evidence_units):
            evidence_spans, method = _source_evidence_spans(evidence, document)
            matches: list[ChunkEvidenceMatch] = []
            if evidence_spans:
                # Multiple identical occurrences are treated as deterministic
                # alternatives; a chunk may match any occurrence.
                for chunk in strategy_chunks:
                    best = 0.0
                    best_relative: tuple[int, int] | None = None
                    for required in evidence_spans:
                        overlaps = []
                        for chunk_span in span_map[chunk.chunk_id]:
                            if _overlap(chunk_span, required):
                                overlaps.append(chunk_span)
                        coverage = _merge_coverage(overlaps, required)
                        if coverage > best:
                            best = coverage
                            covered_start = max(required[0], min((span[0] for span in overlaps), default=required[0]))
                            covered_end = min(required[1], max((span[1] for span in overlaps), default=required[0]))
                            best_relative = (covered_start - required[0], covered_end - required[0])
                    if best > 0:
                        matches.append(ChunkEvidenceMatch(
                            chunk.chunk_id, best,
                            best_relative[0] if best_relative else None,
                            best_relative[1] if best_relative else None,
                        ))
            else:
                method = "normalized_text"
                normalized_evidence = _normalized(evidence.text)
                for chunk in strategy_chunks:
                    if normalized_evidence and normalized_evidence in _normalized(chunk.text):
                        matches.append(ChunkEvidenceMatch(chunk.chunk_id, 1.0, 0, len(evidence.text)))
            coverage = max((match.evidence_coverage for match in matches), default=0.0)
            # Boundary-spanning evidence can be fully represented by the union
            # of partial chunk matches even when no single chunk covers it.
            if matches and evidence_spans:
                for required in evidence_spans:
                    intervals = []
                    for chunk in strategy_chunks:
                        for span in span_map[chunk.chunk_id]:
                            if _overlap(span, required):
                                intervals.append(span)
                    coverage = max(coverage, _merge_coverage(intervals, required))
            mappings.append(EvidenceMapping(
                evidence_id=f"{record.id}:sentence:{index}", doc_id=record.doc_id,
                strategy=strategy, evidence_text=evidence.text,
                matched_chunk_ids=[match.chunk_id for match in matches],
                match_method=method, coverage=coverage, matches=matches,
            ))
    else:
        for index, section in enumerate(record.evidence_sections):
            target = _normalized(section)
            matches = []
            for chunk in strategy_chunks:
                paths: list[str] = list(chunk.title_path)
                raw_paths = chunk.metadata.get("section_paths", [])
                if isinstance(raw_paths, list):
                    paths.extend(" > ".join(path) for path in raw_paths if isinstance(path, list))
                if any(_normalized(path) == target or _normalized(path).endswith(f" > {target}") for path in paths):
                    matches.append(ChunkEvidenceMatch(chunk.chunk_id, 1.0))
            mappings.append(EvidenceMapping(
                evidence_id=f"{record.id}:section:{index}", doc_id=record.doc_id,
                strategy=strategy, evidence_text=section,
                matched_chunk_ids=[match.chunk_id for match in matches],
                match_method="section_path", coverage=1.0 if matches else 0.0, matches=matches,
            ))
    return mappings


def retrieved_evidence_coverage(
    mapping: EvidenceMapping, retrieved_chunk_ids: set[str],
) -> float:
    """Return coverage of one evidence unit by the selected chunks."""

    selected = [match for match in mapping.matches if match.chunk_id in retrieved_chunk_ids]
    if not selected:
        return 0.0
    intervals = [
        (match.evidence_char_start, match.evidence_char_end)
        for match in selected
        if match.evidence_char_start is not None and match.evidence_char_end is not None
    ]
    if intervals:
        required_end = max((match.evidence_char_end or 0) for match in mapping.matches)
        return _merge_coverage(intervals, (0, max(1, required_end)))
    return max(match.evidence_coverage for match in selected)
