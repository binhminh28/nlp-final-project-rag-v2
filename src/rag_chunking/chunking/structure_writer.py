"""Deterministic artifact writer for structure-aware chunking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Chunk
from .structure_aware import StructureAwareChunkingConfig
from .tokenizer import TiktokenTokenizer


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def write_structure_aware_artifacts(
    chunks: list[Chunk],
    output_dir: Path,
    config: StructureAwareChunkingConfig,
    tokenizer: TiktokenTokenizer,
    statistics: dict[str, Any],
    source_input: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    manifest = {
        "schema_version": 1,
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
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "stats.json", statistics)
