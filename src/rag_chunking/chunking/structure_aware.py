"""Deterministic section-first, block-aware chunking for normalized documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

from rag_chunking.data.models import DocumentBlock, NormalizedDocument

from .models import Chunk
from .tokenizer import TiktokenTokenizer, retreat_to_utf8_safe_boundary


@dataclass(frozen=True, slots=True)
class StructureAwareChunkingConfig:
    max_chunk_tokens: int = 512
    tokenizer_name: str = "cl100k_base"
    include_local_heading: bool = True

    def __post_init__(self) -> None:
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")


@dataclass(frozen=True, slots=True)
class HeadingRef:
    level: int
    text: str
    block_index: int


@dataclass(slots=True)
class Section:
    section_index: int
    path: tuple[HeadingRef, ...]
    block_indices: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BlockFragment:
    block_index: int
    block_type: str
    char_start: int
    char_end: int
    fragment_index: int = 0
    fragment_count: int = 1
    split_reason: str | None = None
    token_fallback: bool = False


@dataclass(slots=True)
class _PackedChunk:
    fragments: list[BlockFragment]
    text: str
    boundary_reason: str
    omitted_heading_index: int | None = None


def build_sections(document: NormalizedDocument) -> list[Section]:
    """Group preamble and each local heading body without merging sections."""

    sections: list[Section] = []
    stack: list[HeadingRef] = []
    current: Section | None = None
    for block_index, block in enumerate(document.blocks):
        if block.type == "heading":
            if block.level is None or not 1 <= block.level <= 6:
                raise ValueError(f"Invalid heading level at {document.doc_id} block {block_index}")
            while stack and stack[-1].level >= block.level:
                stack.pop()
            heading = HeadingRef(block.level, block.text, block_index)
            stack.append(heading)
            current = Section(len(sections), tuple(stack), [block_index])
            sections.append(current)
        else:
            if current is None:
                current = Section(len(sections), (), [])
                sections.append(current)
            current.block_indices.append(block_index)
    return sections


def _token_count(text: str, tokenizer: TiktokenTokenizer) -> int:
    return len(tokenizer.encode(text))


def _exact_sentence_spans(block: DocumentBlock) -> list[tuple[int, int]] | None:
    if not block.sentences:
        return None
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in block.sentences:
        position = block.text.find(sentence.text, cursor)
        if position < 0:
            return None
        end = position + len(sentence.text)
        spans.append((cursor, end))
        cursor = end
    if spans:
        spans[-1] = (spans[-1][0], len(block.text))
    return spans


def _line_spans(text: str) -> list[tuple[int, int]]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return [(0, len(text))]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for line in lines:
        end = cursor + len(line)
        spans.append((cursor, end))
        cursor = end
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return spans


def _list_item_spans(text: str) -> list[tuple[int, int]]:
    """Keep continuation lines attached to their Markdown list item."""

    lines = text.splitlines(keepends=True)
    if not lines:
        return [(0, len(text))]
    item_re = re.compile(r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+")
    spans: list[tuple[int, int]] = []
    cursor = 0
    current_start = 0
    seen_item = False
    for line in lines:
        if item_re.match(line) and seen_item:
            spans.append((current_start, cursor))
            current_start = cursor
        if item_re.match(line):
            seen_item = True
        cursor += len(line)
    if current_start < len(text):
        spans.append((current_start, len(text)))
    return spans


def _unicode_safe_token_spans(
    text: str, max_tokens: int, tokenizer: TiktokenTokenizer
) -> list[tuple[int, int]]:
    """Partition exact text with token-budgeted, UTF-8-safe boundaries."""

    tokens = tokenizer.encode(text)
    byte_offsets = [0]
    for token in tokens:
        byte_offsets.append(byte_offsets[-1] + len(tokenizer.token_bytes(token)))
    raw = text.encode("utf-8")
    spans: list[tuple[int, int]] = []
    start_token = 0
    start_char = 0
    while start_token < len(tokens):
        end_token = retreat_to_utf8_safe_boundary(
            tokens, min(start_token + max_tokens, len(tokens)), tokenizer
        )
        if end_token <= start_token:
            raise ValueError("max_chunk_tokens cannot contain one Unicode code point")
        while True:
            part_bytes = raw[byte_offsets[start_token] : byte_offsets[end_token]]
            part = part_bytes.decode("utf-8", errors="strict")
            if _token_count(part, tokenizer) <= max_tokens:
                break
            end_token = retreat_to_utf8_safe_boundary(tokens, end_token - 1, tokenizer)
            if end_token <= start_token:
                raise ValueError("Unable to create a token-budgeted Unicode-safe fragment")
        end_char = start_char + len(part)
        spans.append((start_char, end_char))
        start_char = end_char
        start_token = end_token
    return spans


def _pack_spans(
    block: DocumentBlock,
    spans: list[tuple[int, int]],
    max_tokens: int,
    tokenizer: TiktokenTokenizer,
) -> list[tuple[int, int, bool]]:
    packed: list[tuple[int, int, bool]] = []
    current_start: int | None = None
    current_end = 0
    for start, end in spans:
        unit = block.text[start:end]
        if _token_count(unit, tokenizer) > max_tokens:
            if current_start is not None:
                packed.append((current_start, current_end, False))
                current_start = None
            for sub_start, sub_end in _unicode_safe_token_spans(unit, max_tokens, tokenizer):
                packed.append((start + sub_start, start + sub_end, True))
            continue
        candidate_start = start if current_start is None else current_start
        if current_start is None or _token_count(
            block.text[candidate_start:end], tokenizer
        ) <= max_tokens:
            current_start = candidate_start
            current_end = end
        else:
            packed.append((current_start, current_end, False))
            current_start, current_end = start, end
    if current_start is not None:
        packed.append((current_start, current_end, False))
    return packed


def _split_reason(block_type: str) -> str:
    return {
        "paragraph": "oversized_paragraph_sentence_split",
        "blockquote": "oversized_blockquote_sentence_split",
        "callout": "oversized_callout_sentence_split",
        "code_block": "oversized_code_line_split",
        "code_reference": "oversized_code_reference_line_split",
        "list": "oversized_list_item_split",
        "table": "oversized_table_row_split",
        "html_block": "oversized_html_line_split",
        "custom_block": "oversized_custom_line_split",
        "heading": "oversized_heading_line_split",
    }[block_type]


def split_block(
    block: DocumentBlock,
    block_index: int,
    max_tokens: int,
    tokenizer: TiktokenTokenizer,
) -> list[BlockFragment]:
    """Keep a fitting block atomic; otherwise use type-specific exact spans."""

    if not block.text:
        return []
    if _token_count(block.text, tokenizer) <= max_tokens:
        return [BlockFragment(block_index, block.type, 0, len(block.text))]
    if block.type in ("paragraph", "blockquote", "callout"):
        spans = _exact_sentence_spans(block) or _line_spans(block.text)
    elif block.type == "list":
        spans = []
        for item_start, item_end in _list_item_spans(block.text):
            item_text = block.text[item_start:item_end]
            if _token_count(item_text, tokenizer) <= max_tokens:
                spans.append((item_start, item_end))
            else:
                spans.extend(
                    (item_start + line_start, item_start + line_end)
                    for line_start, line_end in _line_spans(item_text)
                )
    else:
        spans = _line_spans(block.text)
    reason = _split_reason(block.type)
    packed = _pack_spans(block, spans, max_tokens, tokenizer)
    count = len(packed)
    return [
        BlockFragment(
            block_index=block_index,
            block_type=block.type,
            char_start=start,
            char_end=end,
            fragment_index=index,
            fragment_count=count,
            split_reason="oversized_token_fallback" if fallback else reason,
            token_fallback=fallback,
        )
        for index, (start, end, fallback) in enumerate(packed)
    ]


def _render_fragments(
    document: NormalizedDocument, fragments: Iterable[BlockFragment]
) -> str:
    pieces: list[str] = []
    previous_block: int | None = None
    for fragment in fragments:
        if previous_block is not None and previous_block != fragment.block_index:
            pieces.append("\n\n")
        pieces.append(
            document.blocks[fragment.block_index].text[fragment.char_start : fragment.char_end]
        )
        previous_block = fragment.block_index
    return "".join(pieces)


def _pack_section(
    document: NormalizedDocument,
    section: Section,
    config: StructureAwareChunkingConfig,
    tokenizer: TiktokenTokenizer,
) -> list[_PackedChunk]:
    fragments = [
        fragment
        for block_index in section.block_indices
        for fragment in split_block(
            document.blocks[block_index], block_index, config.max_chunk_tokens, tokenizer
        )
    ]
    if not fragments:
        return []

    omitted_heading: int | None = None
    if (
        config.include_local_heading
        and len(fragments) > 1
        and fragments[0].block_type == "heading"
        and _token_count(_render_fragments(document, fragments[:2]), tokenizer)
        > config.max_chunk_tokens
    ):
        # Keep a fitting content block atomic. The local heading remains explicit
        # hierarchy metadata instead of becoming a tiny standalone chunk.
        omitted_heading = fragments[0].block_index
        fragments = fragments[1:]

    packed: list[_PackedChunk] = []
    current: list[BlockFragment] = []
    for fragment in fragments:
        candidate = [*current, fragment]
        candidate_text = _render_fragments(document, candidate)
        if current and _token_count(candidate_text, tokenizer) > config.max_chunk_tokens:
            next_same_block = current[-1].block_index == fragment.block_index
            reason = current[-1].split_reason if next_same_block else "block_budget"
            packed.append(
                _PackedChunk(
                    current,
                    _render_fragments(document, current),
                    reason or "block_budget",
                    omitted_heading if not packed else None,
                )
            )
            current = [fragment]
        else:
            current = candidate
    if current:
        packed.append(
            _PackedChunk(
                current,
                _render_fragments(document, current),
                "section_end",
                omitted_heading if not packed else None,
            )
        )
    return packed


def _fragment_metadata(document: NormalizedDocument, fragment: BlockFragment) -> dict[str, object]:
    text = document.blocks[fragment.block_index].text[fragment.char_start : fragment.char_end]
    return {
        "source_block_index": fragment.block_index,
        "fragment_index": fragment.fragment_index,
        "fragment_count": fragment.fragment_count,
        "char_start": fragment.char_start,
        "char_end": fragment.char_end,
        "fragment_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "split_reason": fragment.split_reason,
        "token_fallback": fragment.token_fallback,
    }


def _link_section_hierarchy(
    sections: list[Section],
    section_chunk_ids: dict[int, list[str]],
    chunk_by_id: dict[str, Chunk],
) -> None:
    """Populate parent_id/children_ids purely from Section.path prefixes.

    A section's immediate parent is the section whose path equals this
    section's path with the last heading dropped -- unique because HeadingRef
    carries the heading's block_index, so no two sections share a path. The
    link is anchored on the parent's last generated chunk and the child's
    first generated chunk (the two chunks adjacent to the section transition
    in document order). Sections that produced zero chunks are left
    unlinked rather than substituting a guessed anchor.

    Sections with a path shorter than two headings never get a parent, even
    when a same-document preamble section (path == ()) exists: that
    zero-length path is "no heading yet", not an ancestor heading, so a
    top-level (single-heading) section must stay root-level rather than
    being attached to the preamble.
    """

    path_to_section_index = {section.path: section.section_index for section in sections}
    for section in sections:
        if len(section.path) < 2:
            continue
        parent_index = path_to_section_index.get(section.path[:-1])
        if parent_index is None:
            continue
        child_ids = section_chunk_ids.get(section.section_index, [])
        parent_ids = section_chunk_ids.get(parent_index, [])
        if not child_ids or not parent_ids:
            continue
        child_chunk = chunk_by_id[child_ids[0]]
        parent_chunk = chunk_by_id[parent_ids[-1]]
        child_chunk.parent_id = parent_chunk.chunk_id
        parent_chunk.children_ids.append(child_chunk.chunk_id)


class StructureAwareChunker:
    strategy = "structure_aware"

    def __init__(
        self,
        config: StructureAwareChunkingConfig | None = None,
        tokenizer: TiktokenTokenizer | None = None,
    ) -> None:
        self.config = config or StructureAwareChunkingConfig()
        self.tokenizer = tokenizer or TiktokenTokenizer(self.config.tokenizer_name)

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        sections = build_sections(document)
        section_chunk_ids: dict[int, list[str]] = {}
        for section in sections:
            section_chunk_ids[section.section_index] = []
            section_id = f"{document.doc_id}::section::{section.section_index:06d}"
            for packed in _pack_section(document, section, self.config, self.tokenizer):
                token_count = _token_count(packed.text, self.tokenizer)
                if not packed.text or token_count > self.config.max_chunk_tokens:
                    raise ValueError(f"Invalid packed chunk in {document.doc_id}: {token_count} tokens")
                index = len(chunks)
                path = [heading.text for heading in section.path]
                levels = [heading.level for heading in section.path]
                block_indices = [fragment.block_index for fragment in packed.fragments]
                types = list(dict.fromkeys(fragment.block_type for fragment in packed.fragments))
                fragment_metadata = [
                    _fragment_metadata(document, fragment) for fragment in packed.fragments
                ]
                metadata: dict[str, object] = {
                    "source_sha256": document.source_sha256,
                    "section_id": section_id,
                    "section_path": path,
                    "heading_levels": levels,
                    "block_start_index": min(block_indices),
                    "block_end_index": max(block_indices),
                    "block_types": types,
                    "boundary_reason": packed.boundary_reason,
                    "block_fragments": fragment_metadata,
                    "source_block_metadata": {
                        str(block_index): {
                            key: value
                            for key, value in {
                                "language": document.blocks[block_index].language,
                                "ordered": document.blocks[block_index].ordered,
                                "metadata": document.blocks[block_index].metadata or None,
                            }.items()
                            if value is not None
                        }
                        for block_index in dict.fromkeys(block_indices)
                        if any(
                            value is not None
                            for value in (
                                document.blocks[block_index].language,
                                document.blocks[block_index].ordered,
                                document.blocks[block_index].metadata or None,
                            )
                        )
                    },
                    "local_heading_in_text": packed.omitted_heading_index is None
                    and bool(packed.fragments)
                    and packed.fragments[0].block_type == "heading",
                }
                if packed.omitted_heading_index is not None:
                    metadata["context_heading_block_index"] = packed.omitted_heading_index
                    metadata["local_heading_in_text"] = False
                oversized = [f for f in packed.fragments if f.fragment_count > 1]
                oversized_blocks = {f.block_index for f in oversized}
                if len(oversized_blocks) == 1:
                    first = oversized[0]
                    metadata.update(
                        {
                            "source_block_index": first.block_index,
                            "fragment_index": first.fragment_index,
                            "fragment_count": first.fragment_count,
                            "split_reason": first.split_reason,
                        }
                    )
                chunk = Chunk(
                    chunk_id=f"{document.doc_id}::structure::{index:06d}",
                    strategy=self.strategy,
                    doc_id=document.doc_id,
                    source=document.source,
                    relative_path=document.relative_path,
                    chunk_index=index,
                    text=packed.text,
                    token_start=None,
                    token_end=None,
                    token_count=token_count,
                    chunk_size=self.config.max_chunk_tokens,
                    chunk_overlap=0,
                    tokenizer=self.tokenizer.name,
                    level=levels[-1] if levels else 0,
                    title_path=path,
                    metadata=metadata,
                )
                chunks.append(chunk)
                section_chunk_ids[section.section_index].append(chunk.chunk_id)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        _link_section_hierarchy(sections, section_chunk_ids, chunk_by_id)
        return chunks

    def chunk_corpus(self, documents: list[NormalizedDocument]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk(document)]
