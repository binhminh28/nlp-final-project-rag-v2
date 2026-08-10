from __future__ import annotations

import pytest

from rag_chunking.chunking.registry import (
    CHUNKER_REGISTRY,
    SUPPORTED_STRATEGIES,
    UnavailableStrategyError,
    get_registration,
)


def test_available_strategies_are_registered() -> None:
    assert set(CHUNKER_REGISTRY) == {"fixed_size", "structure_aware"}


def test_get_registration_returns_handler_for_available_strategy() -> None:
    for strategy in ("fixed_size", "structure_aware"):
        registration = get_registration(strategy)
        assert registration.build_config({})
        assert callable(registration.run)


def test_get_registration_raises_for_supported_but_unavailable_strategy() -> None:
    assert "prompt_based" in SUPPORTED_STRATEGIES
    assert "prompt_based" not in CHUNKER_REGISTRY
    with pytest.raises(UnavailableStrategyError, match="not available"):
        get_registration("prompt_based")


def test_get_registration_raises_for_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_registration("does_not_exist")


def test_build_fixed_size_config_reads_overlap_alias() -> None:
    registration = get_registration("fixed_size")
    config = registration.build_config({"chunk_size": 500, "overlap": 50, "tokenizer": "cl100k_base"})
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50


def test_build_structure_aware_config_reads_preserve_heading_context() -> None:
    registration = get_registration("structure_aware")
    config = registration.build_config({"max_chunk_tokens": 400, "preserve_heading_context": False})
    assert config.max_chunk_tokens == 400
    assert config.include_local_heading is False
