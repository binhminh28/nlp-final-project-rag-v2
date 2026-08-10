"""LLM-planned contiguous block grouping with deterministic source enforcement."""

from __future__ import annotations

import hashlib
import json
import statistics
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rag_chunking.data.models import NormalizedDocument

from .models import Chunk
from .prompt_cache import CACHE_VERSION, PromptResponseCache, canonical_json, request_digest
from .prompt_client import (
    BoundaryPlanner,
    PlannerModelConfig,
    PlannerResponse,
    PlannerTransportError,
)
from .prompt_prompts import PROMPT_VERSION, SYSTEM_PROMPT
from .prompt_schema import PLANNER_SCHEMA_VERSION, BoundaryPlan, PlanValidationError, parse_boundary_plan
from .serialization import BLOCK_SEPARATOR
from .structure_aware import BlockFragment, build_sections, split_block
from .tokenizer import TiktokenTokenizer


@dataclass(frozen=True, slots=True)
class PromptBasedChunkingConfig:
    max_chunk_tokens: int = 512
    tokenizer_name: str = "cl100k_base"
    prompt_version: str = PROMPT_VERSION
    planner_schema_version: str = PLANNER_SCHEMA_VERSION
    planner_input_tokens: int = 12_000
    block_preview_tokens: int = 1_024
    max_retries: int = 2

    def __post_init__(self) -> None:
        if min(self.max_chunk_tokens, self.planner_input_tokens, self.block_preview_tokens) <= 0:
            raise ValueError("token limits must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass(slots=True)
class PromptRunMetrics:
    cache_hits: int = 0
    cache_misses: int = 0
    model_calls: int = 0
    retries: int = 0
    invalid_model_responses: int = 0
    capability_fallbacks: int = 0
    documents_requiring_retry: set[str] = field(default_factory=set)
    failed_documents: list[str] = field(default_factory=list)
    planner_jobs_total: int = 0
    jobs_deduplicated: int = 0
    successful_planner_jobs: int = 0
    failed_planner_jobs: int = 0
    max_concurrency: int = 1
    peak_in_flight: int = 0
    transport_retries: int = 0
    http_429_responses: int = 0
    http_5xx_responses: int = 0
    network_errors: int = 0
    empty_content_responses: int = 0
    empty_content_retries: int = 0
    local_budget_adjustments: int = 0
    max_response_tokens_used: int = 0
    request_latencies: list[float] = field(default_factory=list)

    def operational_summary(self) -> dict[str, object]:
        latencies = self.request_latencies
        return {
            "planner_jobs_total": self.planner_jobs_total,
            "jobs_deduplicated": self.jobs_deduplicated,
            "successful_planner_jobs": self.successful_planner_jobs,
            "failed_planner_jobs": self.failed_planner_jobs,
            "max_concurrency": self.max_concurrency,
            "peak_in_flight": self.peak_in_flight,
            "transport_retries": self.transport_retries,
            "http_429_responses": self.http_429_responses,
            "http_5xx_responses": self.http_5xx_responses,
            "network_errors": self.network_errors,
            "empty_content_responses": self.empty_content_responses,
            "empty_content_retries": self.empty_content_retries,
            "local_budget_adjustments": self.local_budget_adjustments,
            "max_response_tokens_used": self.max_response_tokens_used,
            "latency_seconds": {
                "min": min(latencies) if latencies else 0.0,
                "mean": statistics.fmean(latencies) if latencies else 0.0,
                "median": statistics.median(latencies) if latencies else 0.0,
                "p95": _percentile_float(latencies, 0.95),
                "max": max(latencies) if latencies else 0.0,
            },
        }


@dataclass(frozen=True, slots=True)
class PlannerJobFailure:
    document_order: int
    document_path: str
    batch_index: int
    block_start: int
    block_end: int
    cache_key: str
    error_category: str
    error: str


@dataclass(slots=True)
class ConcurrentCorpusResult:
    chunks: list[Chunk]
    failures: list[PlannerJobFailure]


@dataclass(frozen=True, slots=True)
class _PlannedBatch:
    start: int
    end: int
    candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _ResolvedPlan:
    plan: BoundaryPlan
    cache_hit: bool
    attempt_count: int
    request_sha256: str
    response_metadata: dict[str, Any]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _percentile_float(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _render_fragments(document: NormalizedDocument, fragments: Iterable[BlockFragment]) -> str:
    pieces: list[str] = []
    previous: int | None = None
    for fragment in fragments:
        if previous is None:
            # The separator before every non-first block belongs to the source
            # slice that starts at that block. This keeps independently planned
            # groups and locally packed chunks a lossless partition of the
            # canonical NormalizedDocument linearization.
            if fragment.block_index > 0 and fragment.char_start == 0:
                pieces.append(BLOCK_SEPARATOR)
        elif previous != fragment.block_index:
            pieces.append(BLOCK_SEPARATOR)
        block = document.blocks[fragment.block_index]
        pieces.append(block.text[fragment.char_start : fragment.char_end])
        previous = fragment.block_index
    return "".join(pieces)


def _heading_context(document: NormalizedDocument) -> dict[int, tuple[tuple[str, ...], tuple[int, ...]]]:
    result: dict[int, tuple[tuple[str, ...], tuple[int, ...]]] = {}
    for section in build_sections(document):
        path = tuple(heading.text for heading in section.path)
        levels = tuple(heading.level for heading in section.path)
        for block_index in section.block_indices:
            result[block_index] = (path, levels)
    return result


def _planner_structure(block: object) -> dict[str, Any]:
    metadata = getattr(block, "metadata", {})
    block_type = getattr(block, "type", "")
    value: dict[str, Any] = {}
    if metadata.get("container_path"):
        value["container_path"] = metadata["container_path"]
    if block_type == "table":
        value["table"] = {
            "header": metadata.get("header", []),
            "column_count": metadata.get("column_count", 0),
            "row_count": metadata.get("row_count", 0),
        }
    elif block_type == "list":
        items = metadata.get("items", [])
        value["list"] = {
            "item_count": len(items),
            "max_level": max((int(item.get("level", 0)) for item in items), default=0),
            "nested_table_count": len(metadata.get("nested_tables", [])),
        }
    elif block_type == "callout":
        value["callout_kind"] = metadata.get("callout_kind", "note")
    elif block_type == "code_reference":
        value["code_reference"] = {
            "language": getattr(block, "language", None),
            "resolved": metadata.get("resolved", False),
        }
    return value


class PromptBasedChunker:
    strategy = "prompt_based"

    def __init__(
        self,
        planner: BoundaryPlanner,
        cache_dir: Path,
        config: PromptBasedChunkingConfig | None = None,
        model_config: PlannerModelConfig | None = None,
        tokenizer: TiktokenTokenizer | None = None,
        *,
        force_refresh: bool = False,
    ) -> None:
        self.config = config or PromptBasedChunkingConfig()
        self.model_config = model_config or PlannerModelConfig()
        self.tokenizer = tokenizer or TiktokenTokenizer(self.config.tokenizer_name)
        self.planner = planner
        self.cache = PromptResponseCache(cache_dir)
        self.force_refresh = force_refresh
        self.metrics = PromptRunMetrics()
        self._metrics_lock = threading.Lock()
        self._active_requests = 0

    def _add_metric(self, name: str, amount: int = 1) -> None:
        with self._metrics_lock:
            setattr(self.metrics, name, getattr(self.metrics, name) + amount)

    def _record_transport(self, values: dict[str, int | float]) -> None:
        with self._metrics_lock:
            self.metrics.model_calls += int(values.get("transport_calls", 1))
            self.metrics.transport_retries += int(values.get("transport_retries", 0))
            self.metrics.http_429_responses += int(values.get("http_429_responses", 0))
            self.metrics.http_5xx_responses += int(values.get("http_5xx_responses", 0))
            self.metrics.network_errors += int(values.get("network_errors", 0))
            self.metrics.empty_content_responses += int(
                values.get("empty_content_responses", 0)
            )
            self.metrics.empty_content_retries += int(
                values.get("empty_content_retries", 0)
            )
            self.metrics.local_budget_adjustments += int(
                values.get("local_budget_adjustments", 0)
            )
            self.metrics.max_response_tokens_used = max(
                self.metrics.max_response_tokens_used,
                int(values.get("max_response_tokens_used", 0)),
            )
            latency = values.get("latency_seconds")
            if isinstance(latency, (int, float)):
                self.metrics.request_latencies.append(float(latency))

    def _begin_request(self) -> None:
        with self._metrics_lock:
            self._active_requests += 1
            self.metrics.peak_in_flight = max(
                self.metrics.peak_in_flight, self._active_requests
            )

    def _end_request(self) -> None:
        with self._metrics_lock:
            self._active_requests -= 1

    def _preview(self, text: str, token_count: int) -> tuple[str, bool]:
        if token_count <= self.config.block_preview_tokens:
            return text, False
        # Character slicing is intrinsically Unicode-safe. Head and tail receive
        # conservative sub-budgets; a final binary search proves the preview cap.
        marker = "\n[… deterministic preview omitted …]\n"
        unit_budget = max(1, self.config.block_preview_tokens // 3)

        def bounded(value: str, *, suffix: bool = False) -> str:
            low, high = 0, len(value)
            while low < high:
                middle = (low + high + 1) // 2
                candidate = value[-middle:] if suffix else value[:middle]
                if len(self.tokenizer.encode(candidate)) <= unit_budget:
                    low = middle
                else:
                    high = middle - 1
            return value[-low:] if suffix and low else value[:low]

        preview = bounded(text) + marker + bounded(text, suffix=True)
        if len(self.tokenizer.encode(preview)) > self.config.block_preview_tokens:
            low, high = 0, len(preview)
            while low < high:
                middle = (low + high + 1) // 2
                if len(self.tokenizer.encode(preview[:middle])) <= self.config.block_preview_tokens:
                    low = middle
                else:
                    high = middle - 1
            preview = preview[:low]
        return preview, True

    def _candidates(self, document: NormalizedDocument) -> list[dict[str, Any]]:
        contexts = _heading_context(document)
        candidates: list[dict[str, Any]] = []
        for index, block in enumerate(document.blocks):
            token_count = len(self.tokenizer.encode(block.text))
            preview, truncated = self._preview(block.text, token_count)
            path, levels = contexts[index]
            candidates.append(
                {
                    "block_index": index,
                    "block_type": block.type,
                    "heading_path": list(path),
                    "heading_levels": list(levels),
                    "structure": _planner_structure(block),
                    "token_count": token_count,
                    "text": preview,
                    "text_preview_truncated": truncated,
                    "text_sha256": _sha256_text(block.text),
                }
            )
        return candidates

    def _batches(self, document: NormalizedDocument) -> list[_PlannedBatch]:
        candidates = self._candidates(document)
        batches: list[_PlannedBatch] = []
        current: list[dict[str, Any]] = []
        # Per-candidate accounting is conservative by including JSON punctuation
        # independently. It avoids quadratic re-tokenization of every prefix.
        envelope_tokens = len(self.tokenizer.encode(canonical_json(
            {"target_max_tokens": self.config.max_chunk_tokens, "blocks": []}
        )))
        current_tokens = envelope_tokens
        for candidate in candidates:
            candidate_tokens = len(self.tokenizer.encode(canonical_json(candidate))) + 2
            if current and current_tokens + candidate_tokens > self.config.planner_input_tokens:
                batches.append(_PlannedBatch(current[0]["block_index"], current[-1]["block_index"], tuple(current)))
                current = [candidate]
                current_tokens = envelope_tokens + candidate_tokens
            else:
                current.append(candidate)
                current_tokens += candidate_tokens
        if current:
            batches.append(_PlannedBatch(current[0]["block_index"], current[-1]["block_index"], tuple(current)))
        return batches

    def _request(self, document: NormalizedDocument, batch: _PlannedBatch) -> dict[str, Any]:
        normalized_hash = _sha256_text(canonical_json(document.to_dict()))
        candidates = list(batch.candidates)
        return {
            "cache_version": CACHE_VERSION,
            "document": {
                "doc_id": document.doc_id,
                "relative_path": document.relative_path,
                "source_sha256": document.source_sha256,
                "normalized_sha256": normalized_hash,
            },
            "model": self.model_config.identity(),
            "prompt_version": self.config.prompt_version,
            "planner_schema_version": self.config.planner_schema_version,
            "target_max_tokens": self.config.max_chunk_tokens,
            "batch": {"start_block_index": batch.start, "end_block_index": batch.end},
            "candidate_sha256": _sha256_text(canonical_json(candidates)),
            "candidates": candidates,
        }

    def _resolve(self, document: NormalizedDocument, batch: _PlannedBatch) -> _ResolvedPlan:
        request = self._request(document, batch)
        request_hash = _sha256_text(canonical_json(request))
        if not self.force_refresh:
            cached = self.cache.get(request)
            if cached is not None:
                plan = parse_boundary_plan(cached.response, batch.start, batch.end)
                self._add_metric("cache_hits")
                return _ResolvedPlan(plan, True, 0, request_hash, cached.metadata)
        self._add_metric("cache_misses")
        base_prompt = canonical_json(
            {
                "target_max_tokens": self.config.max_chunk_tokens,
                "batch_start_block_index": batch.start,
                "batch_end_block_index": batch.end,
                "blocks": list(batch.candidates),
            }
        )
        error_message: str | None = None
        for attempt in range(1, self.config.max_retries + 2):
            prompt = base_prompt
            if error_message is not None:
                prompt += "\nPrevious response was invalid. Correct this schema error: " + error_message
                self._add_metric("retries")
                with self._metrics_lock:
                    self.metrics.documents_requiring_retry.add(document.doc_id)
            self._begin_request()
            try:
                planner_output = self.planner.plan(SYSTEM_PROMPT, prompt, self.model_config)
            except PlannerTransportError as error:
                self._record_transport(error.telemetry)
                raise
            finally:
                self._end_request()
            if isinstance(planner_output, PlannerResponse):
                self._record_transport(planner_output.operational_metadata)
                raw = planner_output.text
                response_metadata = planner_output.cache_metadata()
            else:
                self._add_metric("model_calls")
                raw = planner_output
                response_metadata = {
                    "response_mode": "test_or_custom",
                    "requested_model": self.model_config.model,
                    "resolved_model": None,
                    "capability_fallback_used": False,
                }
            try:
                plan = parse_boundary_plan(raw, batch.start, batch.end)
            except PlanValidationError as error:
                self._add_metric("invalid_model_responses")
                error_message = str(error)
                continue
            if response_metadata.get("capability_fallback_used") is True:
                self._add_metric("capability_fallbacks")
            self.cache.put(request, raw, response_metadata)
            return _ResolvedPlan(plan, False, attempt, request_hash, response_metadata)
        raise PlanValidationError(
            f"Planner failed {document.doc_id} blocks {batch.start}-{batch.end} after "
            f"{self.config.max_retries + 1} attempts: {error_message}"
        )

    def _pack_group(
        self, document: NormalizedDocument, start: int, end: int
    ) -> list[list[BlockFragment]]:
        separator_tokens = len(self.tokenizer.encode(BLOCK_SEPARATOR))
        fragments: list[BlockFragment] = []
        for block_index in range(start, end + 1):
            block = document.blocks[block_index]
            if not block.text:
                # Empty normalized blocks still occupy a position in the
                # canonical block-separated document. Retain a zero-length
                # provenance fragment so their separator is not lost.
                fragments.append(BlockFragment(block_index, block.type, 0, 0))
                continue
            # A chunk beginning at a non-first block owns its leading canonical
            # separator, so reserve that budget before splitting the block.
            block_budget = self.config.max_chunk_tokens
            if block_index > 0:
                block_budget -= separator_tokens
                if block_budget <= 0:
                    raise ValueError(
                        "max_chunk_tokens cannot fit a canonical block separator and content"
                    )
            fragments.extend(
                split_block(block, block_index, block_budget, self.tokenizer)
            )
        packed: list[list[BlockFragment]] = []
        current: list[BlockFragment] = []
        for fragment in fragments:
            candidate = [*current, fragment]
            if current and len(self.tokenizer.encode(_render_fragments(document, candidate))) > self.config.max_chunk_tokens:
                packed.append(current)
                current = [fragment]
            else:
                current = candidate
        if current:
            packed.append(current)
        return packed

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        if not document.blocks:
            return []
        batches = self._batches(document)
        resolved = [self._resolve(document, batch) for batch in batches]
        return self._assemble_document(document, batches, resolved)

    def _assemble_document(
        self,
        document: NormalizedDocument,
        batches: list[_PlannedBatch],
        resolved_batches: list[_ResolvedPlan],
    ) -> list[Chunk]:
        contexts = _heading_context(document)
        normalized_hash = _sha256_text(canonical_json(document.to_dict()))
        chunks: list[Chunk] = []
        planner_group_index = 0
        for batch_index, (batch, resolved) in enumerate(zip(batches, resolved_batches, strict=True)):
            for group in resolved.plan.groups:
                packed_group = self._pack_group(document, group.start_block_index, group.end_block_index)
                adjusted = len(packed_group) > 1
                for local_part_index, fragments in enumerate(packed_group):
                    text = _render_fragments(document, fragments)
                    token_count = len(self.tokenizer.encode(text))
                    if not text or token_count > self.config.max_chunk_tokens:
                        raise ValueError(f"Local prompt chunk enforcement failed in {document.doc_id}")
                    index = len(chunks)
                    block_indices = list(dict.fromkeys(fragment.block_index for fragment in fragments))
                    section_paths = list(dict.fromkeys(contexts[item][0] for item in block_indices))
                    block_metadata = []
                    for fragment in fragments:
                        source_text = document.blocks[fragment.block_index].text[
                            fragment.char_start : fragment.char_end
                        ]
                        block_metadata.append(
                            {
                                "source_block_index": fragment.block_index,
                                "block_type": fragment.block_type,
                                "char_start": fragment.char_start,
                                "char_end": fragment.char_end,
                                "fragment_index": fragment.fragment_index,
                                "fragment_count": fragment.fragment_count,
                                "fragment_sha256": _sha256_text(source_text),
                                "split_reason": fragment.split_reason,
                                "token_fallback": fragment.token_fallback,
                            }
                        )
                    path, levels = contexts[block_indices[0]]
                    metadata: dict[str, Any] = {
                        "source_sha256": document.source_sha256,
                        "normalized_sha256": normalized_hash,
                        "planner_provider": self.model_config.provider,
                        "planner_model": self.model_config.model,
                        "planner_base_url": self.model_config.base_url.rstrip("/"),
                        "structured_output_policy": self.model_config.structured_output_policy,
                        "structured_output_mode": resolved.response_metadata.get("response_mode"),
                        "capability_fallback_used": resolved.response_metadata.get(
                            "capability_fallback_used", False
                        ),
                        "resolved_model": resolved.response_metadata.get("resolved_model"),
                        "local_budget_adjustment": resolved.response_metadata.get(
                            "local_budget_adjustment"
                        ),
                        "prompt_version": self.config.prompt_version,
                        "planner_schema_version": self.config.planner_schema_version,
                        "planner_request_sha256": resolved.request_sha256,
                        "cache_hit": resolved.cache_hit,
                        "model_attempt_count": resolved.attempt_count,
                        "planner_batch_index": batch_index,
                        "planner_group_index": planner_group_index,
                        "planner_group_part_index": local_part_index,
                        "planner_reason": group.reason,
                        "planner_block_start": group.start_block_index,
                        "planner_block_end": group.end_block_index,
                        "source_block_start": min(block_indices),
                        "source_block_end": max(block_indices),
                        "source_block_indices": block_indices,
                        "source_block_types": [document.blocks[item].type for item in block_indices],
                        "block_count": len(block_indices),
                        "block_fragments": block_metadata,
                        "leading_block_separator": (
                            fragments[0].block_index > 0 and fragments[0].char_start == 0
                        ),
                        "section_paths": [list(item) for item in section_paths],
                        "crosses_section_boundary": len(section_paths) > 1,
                        "locally_adjusted": adjusted,
                        "adjustment_reason": "hard_token_budget" if adjusted else None,
                        "oversized_fallback": any(item.fragment_count > 1 for item in fragments),
                        "fallback_policy": "type_aware_then_utf8_safe_token_v1",
                    }
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document.doc_id}::prompt::{index:06d}",
                            strategy=self.strategy,
                            doc_id=document.doc_id,
                            source=document.source,
                            relative_path=document.relative_path,
                            chunk_index=index,
                            text=text,
                            token_start=None,
                            token_end=None,
                            token_count=token_count,
                            chunk_size=self.config.max_chunk_tokens,
                            chunk_overlap=0,
                            tokenizer=self.tokenizer.name,
                            level=levels[-1] if levels else 0,
                            title_path=list(path),
                            metadata=metadata,
                        )
                    )
                planner_group_index += 1
        return chunks

    def chunk_corpus_concurrent(
        self, documents: list[NormalizedDocument], *, max_concurrency: int = 8
    ) -> ConcurrentCorpusResult:
        """Resolve unique planner jobs concurrently, then assemble in source order."""

        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        batches_by_document = [self._batches(document) for document in documents]
        jobs_by_digest: dict[
            str, list[tuple[int, int, NormalizedDocument, _PlannedBatch]]
        ] = {}
        for document_order, (document, batches) in enumerate(
            zip(documents, batches_by_document, strict=True)
        ):
            for batch_index, batch in enumerate(batches):
                digest = request_digest(self._request(document, batch))
                jobs_by_digest.setdefault(digest, []).append(
                    (document_order, batch_index, document, batch)
                )

        job_total = sum(len(references) for references in jobs_by_digest.values())
        with self._metrics_lock:
            self.metrics.planner_jobs_total = job_total
            self.metrics.jobs_deduplicated = job_total - len(jobs_by_digest)
            self.metrics.max_concurrency = max_concurrency

        resolved: dict[tuple[int, int], _ResolvedPlan] = {}
        failures: list[PlannerJobFailure] = []
        network_jobs: list[
            tuple[str, list[tuple[int, int, NormalizedDocument, _PlannedBatch]]]
        ] = []
        for digest, references in jobs_by_digest.items():
            document_order, batch_index, document, batch = references[0]
            request = self._request(document, batch)
            if not self.force_refresh:
                try:
                    cached = self.cache.get(request)
                    if cached is not None:
                        plan = parse_boundary_plan(cached.response, batch.start, batch.end)
                        item = _ResolvedPlan(plan, True, 0, digest, cached.metadata)
                        for ref_document_order, ref_batch_index, _, _ in references:
                            resolved[(ref_document_order, ref_batch_index)] = item
                        self._add_metric("cache_hits", len(references))
                        self._add_metric("successful_planner_jobs", len(references))
                        continue
                except (OSError, ValueError) as error:
                    for ref_document_order, ref_batch_index, ref_document, ref_batch in references:
                        failures.append(
                            PlannerJobFailure(
                                ref_document_order,
                                ref_document.relative_path,
                                ref_batch_index,
                                ref_batch.start,
                                ref_batch.end,
                                digest,
                                type(error).__name__,
                                str(error)[:500],
                            )
                        )
                    self._add_metric("failed_planner_jobs", len(references))
                    continue
            network_jobs.append((digest, references))

        future_jobs: dict[
            Future[_ResolvedPlan],
            tuple[str, list[tuple[int, int, NormalizedDocument, _PlannedBatch]]],
        ] = {}
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            for digest, references in network_jobs:
                _, _, document, batch = references[0]
                future_jobs[executor.submit(self._resolve, document, batch)] = (
                    digest,
                    references,
                )
            for future in as_completed(future_jobs):
                digest, references = future_jobs[future]
                try:
                    item = future.result()
                except (OSError, ValueError, RuntimeError) as error:
                    for document_order, batch_index, document, batch in references:
                        failures.append(
                            PlannerJobFailure(
                                document_order,
                                document.relative_path,
                                batch_index,
                                batch.start,
                                batch.end,
                                digest,
                                type(error).__name__,
                                str(error)[:500],
                            )
                        )
                    self._add_metric("failed_planner_jobs", len(references))
                    continue
                for document_order, batch_index, _, _ in references:
                    resolved[(document_order, batch_index)] = item
                self._add_metric("successful_planner_jobs", len(references))

        failed_document_orders = {failure.document_order for failure in failures}
        chunks: list[Chunk] = []
        for document_order, (document, batches) in enumerate(
            zip(documents, batches_by_document, strict=True)
        ):
            if document_order in failed_document_orders:
                self.metrics.failed_documents.append(document.doc_id)
                continue
            ordered = [resolved[(document_order, batch_index)] for batch_index in range(len(batches))]
            chunks.extend(self._assemble_document(document, batches, ordered))
        failures.sort(key=lambda item: (item.document_order, item.batch_index, item.cache_key))
        return ConcurrentCorpusResult(chunks, failures)

    def chunk_corpus(self, documents: list[NormalizedDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            try:
                chunks.extend(self.chunk(document))
            except (ValueError, RuntimeError):
                self.metrics.failed_documents.append(document.doc_id)
                raise
        return chunks
