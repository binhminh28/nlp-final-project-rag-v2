"""YAML-driven chunking run configuration.

Decouples "which strategies run" from "which CLI you invoke": a config file
lists `enabled_strategies` plus per-strategy options, and the orchestrator
CLI (`rag_chunking.cli.chunk_documents`) reads it instead of hard-coding a
dispatch table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .registry import SUPPORTED_STRATEGIES


@dataclass(frozen=True, slots=True)
class ChunkingRunConfig:
    enabled_strategies: tuple[str, ...]
    strategy_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def options_for(self, strategy: str) -> dict[str, Any]:
        return dict(self.strategy_options.get(strategy, {}))


def load_chunking_config(path: Path) -> ChunkingRunConfig:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    section = raw.get("chunking", raw)
    if not isinstance(section, dict):
        raise ValueError(f"Config {path} must contain a 'chunking' mapping")

    enabled = section.get("enabled_strategies")
    if not enabled:
        raise ValueError(f"Config {path} must define a non-empty chunking.enabled_strategies list")

    unknown = [strategy for strategy in enabled if strategy not in SUPPORTED_STRATEGIES]
    if unknown:
        raise ValueError(f"Config {path} lists unknown chunking strategies: {unknown}")

    strategy_options = {
        key: value
        for key, value in section.items()
        if key != "enabled_strategies" and isinstance(value, dict)
    }
    return ChunkingRunConfig(enabled_strategies=tuple(enabled), strategy_options=strategy_options)
