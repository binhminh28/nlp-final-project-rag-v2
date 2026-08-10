"""Typed, JSON-serializable normalized document representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


NORMALIZED_SCHEMA_VERSION = "normalized_document_v2"


BlockType = Literal[
    "heading",
    "paragraph",
    "code_block",
    "code_reference",
    "list",
    "blockquote",
    "table",
    "callout",
    "html_block",
    "custom_block",
]


@dataclass(slots=True)
class Sentence:
    """A stable sentence within one normalized document."""

    sentence_id: str
    text: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Sentence:
        return cls(sentence_id=value["sentence_id"], text=value["text"])


@dataclass(slots=True)
class DocumentBlock:
    """One source-ordered structural block.

    Optional fields only apply to relevant block types. ``metadata`` retains
    dialect-specific details without coupling later experiments to the parser.
    """

    type: BlockType
    text: str
    level: int | None = None
    language: str | None = None
    ordered: bool | None = None
    source_line_start: int | None = None
    source_line_end: int | None = None
    sentences: list[Sentence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {key: item for key, item in value.items() if item not in (None, [], {})}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DocumentBlock:
        data = dict(value)
        data["sentences"] = [Sentence.from_dict(item) for item in data.get("sentences", [])]
        return cls(**data)


@dataclass(slots=True)
class NormalizedDocument:
    """A normalized view of one Markdown source document."""

    doc_id: str
    source: str
    relative_path: str
    filename: str
    source_sha256: str
    blocks: list[DocumentBlock]
    front_matter: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = NORMALIZED_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "doc_id": self.doc_id,
            "source": self.source,
            "relative_path": self.relative_path,
            "filename": self.filename,
            "source_sha256": self.source_sha256,
            "schema_version": self.schema_version,
            "blocks": [block.to_dict() for block in self.blocks],
        }
        if self.front_matter:
            value["front_matter"] = self.front_matter
        if self.metadata:
            value["metadata"] = self.metadata
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NormalizedDocument:
        data = dict(value)
        data.setdefault("schema_version", "normalized_document_v1")
        data["blocks"] = [DocumentBlock.from_dict(item) for item in data["blocks"]]
        return cls(**data)
