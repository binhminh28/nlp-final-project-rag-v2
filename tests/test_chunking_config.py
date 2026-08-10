from __future__ import annotations

from pathlib import Path

import pytest

from rag_chunking.chunking.config import load_chunking_config


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "chunking.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_enabled_strategies_and_options(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
chunking:
  enabled_strategies:
    - fixed_size
    - structure_aware
  fixed_size:
    chunk_size: 500
    overlap: 50
  structure_aware:
    max_chunk_tokens: 400
""",
    )
    config = load_chunking_config(path)
    assert config.enabled_strategies == ("fixed_size", "structure_aware")
    assert config.options_for("fixed_size") == {"chunk_size": 500, "overlap": 50}
    assert config.options_for("structure_aware") == {"max_chunk_tokens": 400}
    assert config.options_for("prompt_based") == {}


def test_missing_enabled_strategies_raises(tmp_path: Path) -> None:
    path = write_config(tmp_path, "chunking:\n  fixed_size:\n    chunk_size: 500\n")
    with pytest.raises(ValueError, match="enabled_strategies"):
        load_chunking_config(path)


def test_unknown_strategy_raises(tmp_path: Path) -> None:
    path = write_config(tmp_path, "chunking:\n  enabled_strategies:\n    - not_a_real_strategy\n")
    with pytest.raises(ValueError, match="unknown chunking strategies"):
        load_chunking_config(path)


def test_schema_supported_but_unregistered_strategy_loads_from_config(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        "chunking:\n  enabled_strategies:\n    - prompt_based\n",
    )
    config = load_chunking_config(path)
    assert config.enabled_strategies == ("prompt_based",)
