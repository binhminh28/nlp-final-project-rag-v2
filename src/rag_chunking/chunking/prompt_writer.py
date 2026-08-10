"""Byte-stable Prompt-based chunk artifacts and reproducibility manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag_chunking.data.models import NormalizedDocument

from .models import Chunk
from .prompt_based import PromptBasedChunkingConfig
from .prompt_cache import CACHE_VERSION, canonical_json
from .prompt_client import PlannerModelConfig
from .tokenizer import TiktokenTokenizer


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def write_prompt_based_artifacts(
    chunks: list[Chunk],
    documents: list[NormalizedDocument],
    output_dir: Path,
    config: PromptBasedChunkingConfig,
    model_config: PlannerModelConfig,
    tokenizer: TiktokenTokenizer,
    statistics: dict[str, Any],
    source_input: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    corpus_hash = hashlib.sha256(canonical_json([document.to_dict() for document in documents]).encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "strategy": "prompt_based",
        "tokenizer": tokenizer.name,
        "max_chunk_tokens": config.max_chunk_tokens,
        "planner": model_config.identity(),
        "structured_output_policy": model_config.structured_output_policy,
        "structured_output_modes": statistics["prompt"]["structured_output_modes"],
        "fallback_policy": "prompt_json_on_json_schema_capability_rejection_v1",
        "prompt_version": config.prompt_version,
        "planner_schema_version": config.planner_schema_version,
        "planner_input_tokens": config.planner_input_tokens,
        "block_preview_tokens": config.block_preview_tokens,
        "cache_policy": {"version": CACHE_VERSION, "validated_reads": True, "default": "reuse"},
        "retry_policy": {"max_retries": config.max_retries, "on_exhaustion": "fail_document"},
        "packing_policy": "planner_contiguous_groups_then_local_greedy_v1",
        "oversized_block_policy": "sentence_or_line_then_utf8_safe_token_fallback_v1",
        "source_content_policy": "exact_normalized_source_slices_only",
        "source_input": Path(source_input).as_posix(),
        "source_corpus_sha256": corpus_hash,
        "documents": statistics["documents"],
        "documents_failed": statistics["documents_failed"],
        "chunks": statistics["chunks"],
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "stats.json", statistics)
