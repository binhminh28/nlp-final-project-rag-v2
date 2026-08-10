"""Typed, deterministic chunk representation shared by later experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    strategy: str
    doc_id: str
    source: str
    relative_path: str
    chunk_index: int
    text: str
    token_start: int | None
    token_end: int | None
    token_count: int
    chunk_size: int
    chunk_overlap: int
    tokenizer: str
    level: int = 0
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    title_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Chunk:
        return cls(**value)
