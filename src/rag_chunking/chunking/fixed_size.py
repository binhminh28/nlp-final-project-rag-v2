"""Structure-independent fixed-token-window baseline."""

from __future__ import annotations

from dataclasses import dataclass

from rag_chunking.data.models import NormalizedDocument

from .models import Chunk
from .serialization import document_to_text
from .tokenizer import TiktokenTokenizer, retreat_to_utf8_safe_boundary


@dataclass(frozen=True, slots=True)
class FixedSizeChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    tokenizer_name: str = "cl100k_base"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    @property
    def stride(self) -> int:
        return self.chunk_size - self.chunk_overlap


@dataclass(frozen=True, slots=True)
class FixedSizeWindow:
    """One actual window plus the nominal boundaries used to derive it."""

    desired_start: int
    nominal_end: int
    token_start: int
    token_end: int

    @property
    def start_adjustment(self) -> int:
        return self.desired_start - self.token_start

    @property
    def end_adjustment(self) -> int:
        return self.nominal_end - self.token_end

    @property
    def boundary_adjusted(self) -> bool:
        return bool(self.start_adjustment or self.end_adjustment)


def fixed_size_windows(
    source_tokens: list[int],
    config: FixedSizeChunkingConfig,
    tokenizer: TiktokenTokenizer,
) -> list[FixedSizeWindow]:
    """Create fixed windows whose starts and ends are valid UTF-8 boundaries."""

    windows: list[FixedSizeWindow] = []
    desired_start = 0
    start = 0
    while start < len(source_tokens):
        nominal_end = min(start + config.chunk_size, len(source_tokens))
        end = retreat_to_utf8_safe_boundary(source_tokens, nominal_end, tokenizer)
        if end <= start:
            raise ValueError(
                "chunk_size is too small to contain a complete UTF-8 code point "
                f"at source token {start}"
            )
        windows.append(
            FixedSizeWindow(
                desired_start=desired_start,
                nominal_end=nominal_end,
                token_start=start,
                token_end=end,
            )
        )
        if end == len(source_tokens):
            break

        desired_start = max(0, end - config.chunk_overlap)
        next_start = retreat_to_utf8_safe_boundary(source_tokens, desired_start, tokenizer)
        if next_start <= start:
            raise ValueError(
                "Unicode-safe boundary adjustment cannot make forward progress; "
                "use a larger chunk_size or smaller chunk_overlap"
            )
        start = next_start
    return windows


class FixedSizeChunker:
    strategy = "fixed_size"

    def __init__(
        self,
        config: FixedSizeChunkingConfig | None = None,
        tokenizer: TiktokenTokenizer | None = None,
    ) -> None:
        self.config = config or FixedSizeChunkingConfig()
        self.tokenizer = tokenizer or TiktokenTokenizer(self.config.tokenizer_name)

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        source_tokens = self.tokenizer.encode(document_to_text(document))
        chunks: list[Chunk] = []
        previous_end = 0
        for index, window in enumerate(
            fixed_size_windows(source_tokens, self.config, self.tokenizer)
        ):
            token_slice = source_tokens[window.token_start : window.token_end]
            chunk_text = self.tokenizer.decode_strict(token_slice)
            actual_overlap = previous_end - window.token_start if index else 0
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::fixed::{index:06d}",
                    strategy=self.strategy,
                    doc_id=document.doc_id,
                    source=document.source,
                    relative_path=document.relative_path,
                    chunk_index=index,
                    text=chunk_text,
                    token_start=window.token_start,
                    token_end=window.token_end,
                    token_count=len(token_slice),
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    tokenizer=self.tokenizer.name,
                    metadata={
                        "source_sha256": document.source_sha256,
                        "text_token_roundtrip": self.tokenizer.encode(chunk_text) == token_slice,
                        "nominal_chunk_size": self.config.chunk_size,
                        "nominal_overlap": self.config.chunk_overlap,
                        "actual_token_count": len(token_slice),
                        "actual_overlap": actual_overlap,
                        "boundary_adjusted": window.boundary_adjusted,
                        "start_adjustment": window.start_adjustment,
                        "end_adjustment": window.end_adjustment,
                    },
                )
            )
            previous_end = window.token_end
        return chunks

    def chunk_corpus(self, documents: list[NormalizedDocument]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk(document)]
