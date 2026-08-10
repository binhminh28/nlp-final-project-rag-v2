"""Byte-stable Prompt-based chunk artifacts and reproducibility manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rag_chunking.data.models import NORMALIZED_SCHEMA_VERSION, NormalizedDocument

from .models import Chunk
from .prompt_based import PromptBasedChunkingConfig
from .prompt_cache import CACHE_VERSION, canonical_json
from .prompt_client import PlannerModelConfig
from .tokenizer import TiktokenTokenizer
from .writer import _write_text, serialize_chunks_jsonl, serialize_json


def _validate_artifact_inputs(
    chunks: list[Chunk], documents: list[NormalizedDocument], statistics: dict[str, Any]
) -> None:
    """Reject internally inconsistent prompt artifact sets before touching disk."""

    document_order = {document.doc_id: index for index, document in enumerate(documents)}
    if len(document_order) != len(documents):
        raise ValueError("prompt artifacts contain duplicate document IDs")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("prompt artifacts contain duplicate chunk IDs")

    previous_position = (-1, -1)
    next_index: dict[str, int] = {}
    for chunk in chunks:
        chunk.validate()
        if chunk.strategy != "prompt_based":
            raise ValueError("prompt artifacts contain a non-prompt chunk")
        if chunk.doc_id not in document_order:
            raise ValueError(f"prompt chunk references unknown document {chunk.doc_id!r}")
        expected_index = next_index.get(chunk.doc_id, 0)
        if chunk.chunk_index != expected_index:
            raise ValueError(
                f"prompt chunks for {chunk.doc_id!r} are not contiguous and source ordered"
            )
        position = (document_order[chunk.doc_id], chunk.chunk_index)
        if position <= previous_position:
            raise ValueError("prompt chunks are not in deterministic document/source order")
        previous_position = position
        next_index[chunk.doc_id] = expected_index + 1

    expected = {
        "documents": len(documents),
        "chunks": len(chunks),
    }
    for key, value in expected.items():
        if statistics.get(key) != value:
            raise ValueError(
                f"prompt artifact statistics disagree for {key}: "
                f"{statistics.get(key)!r} != {value}"
            )


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
    _validate_artifact_inputs(chunks, documents, statistics)
    corpus_hash = hashlib.sha256(canonical_json([document.to_dict() for document in documents]).encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "source_schema_version": NORMALIZED_SCHEMA_VERSION,
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
        "response_budget_policy": {
            "base_max_response_tokens": model_config.max_response_tokens,
            "on_empty_content_length": "double_with_cap",
            "maximum_multiplier": 4,
        },
        "packing_policy": "planner_contiguous_groups_then_local_greedy_v1",
        "oversized_block_policy": "sentence_or_line_then_utf8_safe_token_fallback_v1",
        "source_content_policy": "exact_normalized_source_slices_only",
        "source_input": Path(source_input).as_posix(),
        "source_corpus_sha256": corpus_hash,
        "documents": statistics["documents"],
        "documents_failed": statistics["documents_failed"],
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
