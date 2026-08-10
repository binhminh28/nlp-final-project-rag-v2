"""Typed, deterministic chunk representation shared by later experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def validate_json_value(value: Any, path: str = "value") -> None:
    """Reject values that JSON would coerce, lose, or emit non-standardly."""

    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains non-JSON value {type(value).__name__}")


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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("chunk_id", "strategy", "doc_id", "source", "relative_path", "text", "tokenizer"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("chunk_index", "token_count", "chunk_size", "chunk_overlap", "level"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.chunk_size <= 0 or self.token_count > self.chunk_size:
            raise ValueError("chunk_size must be positive and at least token_count")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.level < 0:
            raise ValueError("level must be non-negative")
        if (self.token_start is None) != (self.token_end is None):
            raise ValueError("token_start and token_end must both be null or both be integers")
        if self.token_start is not None and self.token_end is not None:
            if type(self.token_start) is not int or type(self.token_end) is not int:
                raise ValueError("token_start and token_end must be integers when present")
            if not 0 <= self.token_start <= self.token_end:
                raise ValueError("token span must satisfy 0 <= token_start <= token_end")
            if self.token_count != self.token_end - self.token_start:
                raise ValueError("token span length must equal token_count")
        if self.parent_id is not None and (not isinstance(self.parent_id, str) or not self.parent_id):
            raise ValueError("parent_id must be null or a non-empty string")
        if not isinstance(self.children_ids, list) or not all(
            isinstance(item, str) and item for item in self.children_ids
        ):
            raise ValueError("children_ids must be a list of non-empty strings")
        if len(self.children_ids) != len(set(self.children_ids)):
            raise ValueError("children_ids must not contain duplicates")
        if not isinstance(self.title_path, list) or not all(
            isinstance(item, str) for item in self.title_path
        ):
            raise ValueError("title_path must be a list of strings")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")
        validate_json_value(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Chunk:
        if not isinstance(value, dict):
            raise ValueError("chunk representation must be an object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown chunk fields: {unknown}")
        return cls(**value)
