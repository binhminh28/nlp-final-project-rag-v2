"""Embedding configuration loader shared by indexing and retrieval CLIs."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import EmbeddingConfig


def load_embedding_config(path: Path) -> EmbeddingConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("embedding"), dict):
        raise ValueError("embedding config must contain an embedding object")
    raw = value["embedding"]
    allowed = {
        "provider", "model", "dimension", "max_batch_items", "max_batch_tokens",
        "max_input_tokens", "tokenizer", "input_type", "encoding_format",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown embedding configuration fields: {unknown}")
    return EmbeddingConfig(**raw)
