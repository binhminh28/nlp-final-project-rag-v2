from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path

from rag_chunking.chunking.prompt_based import PromptBasedChunker
from rag_chunking.chunking.prompt_client import PlannerModelConfig
from rag_chunking.chunking.prompt_statistics import prompt_corpus_statistics
from rag_chunking.chunking.prompt_writer import write_prompt_based_artifacts
from rag_chunking.data.models import DocumentBlock, NormalizedDocument


def _document(index: int, text: str | None = None) -> NormalizedDocument:
    value = text or f"document-{index} content"
    return NormalizedDocument(
        doc_id=f"angular:concurrent-{index}.md",
        source="angular",
        relative_path=f"concurrent-{index}.md",
        filename=f"concurrent-{index}.md",
        source_sha256=f"hash-{index}",
        blocks=[DocumentBlock(type="paragraph", text=value)],
    )


def _prompt_value(user_prompt: str) -> dict[str, object]:
    value, _ = json.JSONDecoder().raw_decode(user_prompt)
    return value


class DelayedPlanner:
    def __init__(
        self,
        delays: dict[str, float] | None = None,
        *,
        fail_text: str | None = None,
        retry_text: str | None = None,
    ) -> None:
        self.delays = delays or {}
        self.fail_text = fail_text
        self.retry_text = retry_text
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls: Counter[str] = Counter()
        self.completion_order: list[str] = []

    def plan(self, system_prompt: str, user_prompt: str, config: PlannerModelConfig) -> str:
        value = _prompt_value(user_prompt)
        text = str(value["blocks"][0]["text"])
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls[text] += 1
            call_number = self.calls[text]
        try:
            time.sleep(self.delays.get(text, 0.01))
            if self.fail_text == text:
                raise RuntimeError("isolated fixture failure")
            if self.retry_text == text and call_number == 1:
                return "invalid"
            return json.dumps({"groups": [{
                "start_block_index": value["batch_start_block_index"],
                "end_block_index": value["batch_end_block_index"],
                "reason": f"group {text}",
            }]})
        finally:
            with self.lock:
                self.active -= 1
                self.completion_order.append(text)


def _chunker(tmp_path: Path, planner: DelayedPlanner) -> PromptBasedChunker:
    return PromptBasedChunker(
        planner,
        tmp_path / "cache",
        model_config=PlannerModelConfig(provider="fake", model="concurrency-fixture"),
    )


def test_bounded_concurrency_and_out_of_order_completion_preserve_document_order(
    tmp_path: Path,
) -> None:
    documents = [_document(index) for index in range(6)]
    delays = {f"document-{index} content": (6 - index) * 0.015 for index in range(6)}
    planner = DelayedPlanner(delays)
    chunker = _chunker(tmp_path, planner)
    result = chunker.chunk_corpus_concurrent(documents, max_concurrency=3)
    assert not result.failures
    assert 1 < planner.peak <= 3
    assert chunker.metrics.peak_in_flight == planner.peak
    assert planner.completion_order != [document.blocks[0].text for document in documents]
    assert [chunk.relative_path for chunk in result.chunks] == [
        document.relative_path for document in documents
    ]


def test_concurrency_one_and_four_write_byte_identical_artifacts(tmp_path: Path) -> None:
    documents = [_document(index) for index in range(5)]
    outputs: list[dict[str, bytes]] = []
    for concurrency in (1, 4):
        chunker = _chunker(tmp_path / str(concurrency), DelayedPlanner())
        result = chunker.chunk_corpus_concurrent(documents, max_concurrency=concurrency)
        assert not result.failures
        stats = prompt_corpus_statistics(documents, result.chunks, chunker.metrics)
        output = tmp_path / f"output-{concurrency}"
        write_prompt_based_artifacts(
            result.chunks,
            documents,
            output,
            chunker.config,
            chunker.model_config,
            chunker.tokenizer,
            stats,
            "input.jsonl",
        )
        outputs.append({path.name: path.read_bytes() for path in output.iterdir()})
    assert outputs[0] == outputs[1]


def test_cache_reuse_and_same_key_deduplication(tmp_path: Path) -> None:
    cached = _document(0)
    uncached = _document(1)
    _chunker(tmp_path, DelayedPlanner()).chunk(cached)
    planner = DelayedPlanner()
    chunker = _chunker(tmp_path, planner)
    result = chunker.chunk_corpus_concurrent(
        [cached, cached, uncached, uncached], max_concurrency=4
    )
    assert not result.failures
    assert sum(planner.calls.values()) == 1
    assert chunker.metrics.cache_hits == 2
    assert chunker.metrics.cache_misses == 1
    assert chunker.metrics.jobs_deduplicated == 2


def test_concurrent_distinct_cache_writes_are_complete_and_readable(tmp_path: Path) -> None:
    documents = [_document(index) for index in range(12)]
    chunker = _chunker(tmp_path, DelayedPlanner())
    result = chunker.chunk_corpus_concurrent(documents, max_concurrency=8)
    assert not result.failures
    cache_files = list((tmp_path / "cache").glob("*.json"))
    assert len(cache_files) == len(documents)
    assert not list((tmp_path / "cache").glob("*.tmp"))
    assert all(json.loads(path.read_text(encoding="utf-8"))["response"] for path in cache_files)


def test_failure_and_schema_retry_are_isolated(tmp_path: Path) -> None:
    good_a = _document(0)
    failed = _document(1, "FAIL")
    retried = _document(2, "RETRY")
    good_b = _document(3)
    planner = DelayedPlanner(fail_text="FAIL", retry_text="RETRY")
    chunker = _chunker(tmp_path, planner)
    result = chunker.chunk_corpus_concurrent(
        [good_a, failed, retried, good_b], max_concurrency=4
    )
    assert len(result.failures) == 1
    assert result.failures[0].document_path == failed.relative_path
    assert [chunk.relative_path for chunk in result.chunks] == [
        good_a.relative_path,
        retried.relative_path,
        good_b.relative_path,
    ]
    assert planner.calls["RETRY"] == 2
    assert planner.calls[good_a.blocks[0].text] == 1
    assert planner.calls[good_b.blocks[0].text] == 1
    assert chunker.metrics.retries == 1
