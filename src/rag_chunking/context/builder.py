"""Single controlled formatter from authoritative selected hits to context."""

from __future__ import annotations

from dataclasses import dataclass

from rag_chunking.chunking.tokenizer import TiktokenTokenizer
from rag_chunking.embedding.models import canonical_fingerprint

from .models import ContextBuildInput, ContextConfig, ContextPiece, ContextResult, context_identity


@dataclass(frozen=True, slots=True)
class ContextOverflowError(ValueError):
    context_token_budget: int
    rendered_context_tokens: int
    raw_selected_chunk_tokens: int
    formatting_overhead_tokens: int
    selected_chunk_count: int

    def __str__(self) -> str:
        return (
            "rendered context exceeds context token budget: "
            f"budget={self.context_token_budget}, actual={self.rendered_context_tokens}, "
            f"raw_selected={self.raw_selected_chunk_tokens}, "
            f"formatting_overhead={self.formatting_overhead_tokens}, "
            f"chunks={self.selected_chunk_count}"
        )


class ContextBuilder:
    """Render hits exactly once, without retrieval, transformation, or deduplication."""

    def __init__(self, config: ContextConfig, *, tokenizer: TiktokenTokenizer | None = None) -> None:
        if not isinstance(config, ContextConfig):
            raise ValueError("config must be a ContextConfig")
        self.config = config
        self.tokenizer = tokenizer or TiktokenTokenizer()
        if self.tokenizer.name != config.tokenizer:
            raise ValueError("tokenizer does not match ContextConfig")

    def build(self, build_input: ContextBuildInput) -> ContextResult:
        if not isinstance(build_input, ContextBuildInput):
            raise ValueError("build_input must be a ContextBuildInput")

        blocks: list[str] = []
        pieces: list[ContextPiece] = []
        for ordinal, hit in enumerate(build_input.selected_hits, 1):
            label = self.config.label_format.format(ordinal=ordinal)
            block = f"{label}\n{hit.text}"
            blocks.append(block)
            pieces.append(ContextPiece(
                ordinal=ordinal, retrieval_rank=hit.rank, chunk_id=hit.chunk_id,
                doc_id=hit.doc_id, source=hit.source, relative_path=hit.relative_path,
                score=float(hit.score), raw_chunk_token_count=hit.token_count,
                rendered_block_token_count=len(self.tokenizer.encode(block)),
                strategy=build_input.strategy,
            ))

        rendered = self.config.separator.join(blocks)
        raw_tokens = sum(hit.token_count for hit in build_input.selected_hits)
        rendered_tokens = len(self.tokenizer.encode(rendered))
        if rendered_tokens > self.config.context_token_budget:
            raise ContextOverflowError(
                context_token_budget=self.config.context_token_budget,
                rendered_context_tokens=rendered_tokens,
                raw_selected_chunk_tokens=raw_tokens,
                formatting_overhead_tokens=rendered_tokens - raw_tokens,
                selected_chunk_count=len(build_input.selected_hits),
            )

        frozen_pieces = tuple(pieces)
        fingerprint = canonical_fingerprint(
            context_identity(self.config.fingerprint, frozen_pieces, rendered)
        )
        return ContextResult(
            query_id=build_input.query_id, strategy=build_input.strategy,
            context_config_fingerprint=self.config.fingerprint,
            retrieval_config_fingerprint=build_input.retrieval_config_fingerprint,
            protocol_config_fingerprint=build_input.protocol_config_fingerprint,
            embedding_config_fingerprint=build_input.embedding_config_fingerprint,
            index_fingerprint=build_input.index_fingerprint,
            dataset_fingerprint=build_input.dataset_fingerprint,
            pieces=frozen_pieces, rendered_context=rendered,
            raw_selected_chunk_tokens=raw_tokens,
            rendered_context_tokens=rendered_tokens,
            context_token_budget=self.config.context_token_budget,
            budget_utilization=rendered_tokens / self.config.context_token_budget,
            context_fingerprint=fingerprint,
        )
