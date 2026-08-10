"""Deterministic artifact writer for structure-aware chunking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_chunking.data.models import NORMALIZED_SCHEMA_VERSION

from .models import Chunk
from .structure_aware import StructureAwareChunkingConfig
from .tokenizer import TiktokenTokenizer
from .writer import _write_text, serialize_chunks_jsonl, serialize_json


def write_structure_aware_artifacts(
    chunks: list[Chunk],
    output_dir: Path,
    config: StructureAwareChunkingConfig,
    tokenizer: TiktokenTokenizer,
    statistics: dict[str, Any],
    source_input: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "source_schema_version": NORMALIZED_SCHEMA_VERSION,
        "strategy": "structure_aware",
        "max_chunk_tokens": config.max_chunk_tokens,
        "tokenizer": tokenizer.name,
        "overlap_policy": "none",
        "heading_policy": "local_heading_first_chunk_if_atomic_budget_allows_v1",
        "hierarchy_policy": "markdown_stack_pop_level_gte_current_v1",
        "packing_policy": "section_local_source_order_greedy_atomic_blocks_v1",
        "sibling_section_merge_policy": "never",
        "oversized_block_policy": "sentence_or_line_then_utf8_safe_token_fallback_v1",
        "source_input": Path(source_input).as_posix(),
        "documents": statistics["documents"],
        "chunks": statistics["chunks"],
    }
    serialized = {
        "chunks.jsonl": serialize_chunks_jsonl(chunks),
        "manifest.json": serialize_json(manifest),
        "stats.json": serialize_json(statistics),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in serialized.items():
        _write_text(output_dir / name, value)
