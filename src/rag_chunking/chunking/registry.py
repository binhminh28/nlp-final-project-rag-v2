"""Strategy registry: which chunking strategies are runnable vs. schema-supported.

`Chunk` (models.py) is already the unified representation produced by every
chunker. What was missing was a single place that dispatches by strategy name
instead of "run the CLI you want". `CHUNKER_REGISTRY` holds strategies with a
working implementation; `SUPPORTED_STRATEGIES` is the larger set the schema is
designed for (includes "prompt_based", which has no LLM provider wired up in
this environment). Requesting an unregistered-but-supported strategy raises
`UnavailableStrategyError` with an actionable message instead of a KeyError.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rag_chunking.data.models import NormalizedDocument

from .fixed_size import FixedSizeChunker, FixedSizeChunkingConfig
from .models import Chunk
from .statistics import chunk_corpus_statistics
from .structure_aware import StructureAwareChunker, StructureAwareChunkingConfig
from .structure_statistics import structure_corpus_statistics
from .structure_validation import validate_structure_aware_chunks
from .structure_writer import write_structure_aware_artifacts
from .validation import validate_fixed_size_chunks
from .writer import write_fixed_size_artifacts

SUPPORTED_STRATEGIES = frozenset({"fixed_size", "structure_aware", "prompt_based"})


@dataclass(frozen=True, slots=True)
class StrategyRunSummary:
    strategy: str
    documents: int
    chunks: int
    output_dir: Path


class UnavailableStrategyError(ValueError):
    """Raised when a strategy is schema-supported but has no registered chunker."""


def _run_fixed_size(
    documents: list[NormalizedDocument],
    config: FixedSizeChunkingConfig,
    output_dir: Path,
    source_input: str,
) -> StrategyRunSummary:
    chunker = FixedSizeChunker(config)
    chunks: list[Chunk] = chunker.chunk_corpus(documents)
    report = validate_fixed_size_chunks(documents, chunks, config, chunker.tokenizer)
    if not report.valid:
        raise ValueError("fixed_size validation failed: " + "; ".join(report.errors))
    stats = chunk_corpus_statistics(documents, chunks, chunker.tokenizer)
    write_fixed_size_artifacts(chunks, output_dir, config, chunker.tokenizer, stats, source_input)
    return StrategyRunSummary("fixed_size", stats["documents"], stats["chunks"], output_dir)


def _run_structure_aware(
    documents: list[NormalizedDocument],
    config: StructureAwareChunkingConfig,
    output_dir: Path,
    source_input: str,
) -> StrategyRunSummary:
    chunker = StructureAwareChunker(config)
    chunks: list[Chunk] = chunker.chunk_corpus(documents)
    report = validate_structure_aware_chunks(documents, chunks, config, chunker.tokenizer)
    if not report.valid:
        raise ValueError("structure_aware validation failed: " + "; ".join(report.errors))
    stats = structure_corpus_statistics(documents, chunks)
    write_structure_aware_artifacts(chunks, output_dir, config, chunker.tokenizer, stats, source_input)
    return StrategyRunSummary("structure_aware", stats["documents"], stats["chunks"], output_dir)


def _build_fixed_size_config(options: dict[str, Any]) -> FixedSizeChunkingConfig:
    return FixedSizeChunkingConfig(
        chunk_size=options.get("chunk_size", 512),
        chunk_overlap=options.get("overlap", options.get("chunk_overlap", 64)),
        tokenizer_name=options.get("tokenizer", "cl100k_base"),
    )


def _build_structure_aware_config(options: dict[str, Any]) -> StructureAwareChunkingConfig:
    return StructureAwareChunkingConfig(
        max_chunk_tokens=options.get("max_chunk_tokens", 512),
        tokenizer_name=options.get("tokenizer", "cl100k_base"),
        include_local_heading=options.get("preserve_heading_context", True),
    )


@dataclass(frozen=True, slots=True)
class ChunkerRegistration:
    build_config: Callable[[dict[str, Any]], Any]
    run: Callable[[list[NormalizedDocument], Any, Path, str], StrategyRunSummary]


CHUNKER_REGISTRY: dict[str, ChunkerRegistration] = {
    "fixed_size": ChunkerRegistration(_build_fixed_size_config, _run_fixed_size),
    "structure_aware": ChunkerRegistration(_build_structure_aware_config, _run_structure_aware),
}


def get_registration(strategy: str) -> ChunkerRegistration:
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
    registration = CHUNKER_REGISTRY.get(strategy)
    if registration is None:
        raise UnavailableStrategyError(
            f"Strategy {strategy!r} is supported by the unified schema but is not available "
            "in this environment. Enable it after configuring an LLM provider."
        )
    return registration
