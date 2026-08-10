from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.prompt_based import PromptBasedChunker, PromptBasedChunkingConfig
from rag_chunking.chunking.prompt_client import PlannerModelConfig
from rag_chunking.chunking.prompt_schema import PlanValidationError
from rag_chunking.chunking.prompt_statistics import prompt_corpus_statistics
from rag_chunking.chunking.prompt_validation import validate_prompt_based_chunks
from rag_chunking.chunking.prompt_writer import write_prompt_based_artifacts
from rag_chunking.chunking.serialization import document_to_text
from rag_chunking.chunking.writer import read_chunks_jsonl
from rag_chunking.data.loader import load_document


class DeterministicPlanner:
    """Offline planner fixture covering valid and adversarial boundary responses."""

    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode
        self.calls = 0

    def plan(self, system_prompt: str, user_prompt: str, config: PlannerModelConfig) -> str:
        self.calls += 1
        value, _ = json.JSONDecoder().raw_decode(user_prompt)
        start = value["batch_start_block_index"]
        end = value["batch_end_block_index"]
        group = lambda left, right: {
            "start_block_index": left,
            "end_block_index": right,
            "reason": "deterministic fixture",
        }
        responses = {
            "valid": {"groups": [group(start, end)]},
            "empty": {"groups": []},
            "negative": {"groups": [group(-1, end)]},
            "out_of_range": {"groups": [group(start, end + 1)]},
            "overlap": {"groups": [group(start, start), group(start, end)]},
            "gap": {"groups": [group(start, start), group(start + 2, end)]},
            "out_of_order": {"groups": [group(end, end), group(start, end - 1)]},
            "duplicate_range": {"groups": [group(start, end), group(start, end)]},
            "rewritten_text": {
                "groups": [{**group(start, end), "text": "LLM-authored replacement"}]
            },
        }
        if self.mode == "malformed":
            return "not JSON"
        return json.dumps(responses[self.mode])


def _chunker(tmp_path: Path, planner: DeterministicPlanner, max_tokens: int = 48) -> PromptBasedChunker:
    return PromptBasedChunker(
        planner,
        tmp_path / "planner-cache",
        PromptBasedChunkingConfig(max_chunk_tokens=max_tokens, max_retries=0),
        PlannerModelConfig(provider="fake", model="offline-boundary-planner-v1"),
    )


def _markdown_document(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    markdown = (
        "# Unicode\n\n"
        "🌷 U+1F337; Tiếng Việt nguyên vẹn; e\u0301; 你好.\n\n"
        "| cột | giá trị |\n| --- | --- |\n| hoa | 🌷 |\n\n"
        "```python\nflower = '🌷'\nprint(flower)\n```\n"
    )
    path = source / "unicode.md"
    path.write_text(markdown, encoding="utf-8", newline="\n")
    return load_document(path, source)


def test_offline_markdown_to_canonical_artifacts_round_trip(tmp_path: Path) -> None:
    document = _markdown_document(tmp_path)
    planner = DeterministicPlanner()
    chunker = _chunker(tmp_path, planner)

    chunks = chunker.chunk(document)
    report = validate_prompt_based_chunks(
        [document], chunks, chunker.config, chunker.model_config, chunker.tokenizer
    )
    assert report.valid, report.errors
    assert all(type(chunk) is Chunk for chunk in chunks)
    assert all(set(chunk.to_dict()) == set(Chunk.__dataclass_fields__) for chunk in chunks)
    assert all(chunk.token_start is None and chunk.token_end is None for chunk in chunks)
    assert all(chunk.parent_id is None and chunk.children_ids == [] for chunk in chunks)
    assert all(chunk.strategy == "prompt_based" and chunk.chunk_overlap == 0 for chunk in chunks)
    assert all(chunk.token_count == len(chunker.tokenizer.encode(chunk.text)) for chunk in chunks)
    assert document_to_text(document) == "".join(chunk.text for chunk in chunks)
    assert "🌷" in "".join(chunk.text for chunk in chunks)
    assert "Tiếng Việt" in "".join(chunk.text for chunk in chunks)
    assert "e\u0301" in "".join(chunk.text for chunk in chunks)
    assert "你好" in "".join(chunk.text for chunk in chunks)
    assert "flower = '🌷'" in "".join(chunk.text for chunk in chunks)
    assert any(
        "code_block" in chunk.metadata["source_block_types"] for chunk in chunks
    )
    assert "\ufffd" not in "".join(chunk.text for chunk in chunks)
    assert "🌷".encode("utf-8") == bytes.fromhex("f0 9f 8c b7")

    stats = prompt_corpus_statistics([document], chunks, chunker.metrics)
    stats["validation"] = {"valid": True, "errors": []}
    output = tmp_path / "artifacts"
    write_prompt_based_artifacts(
        chunks,
        [document],
        output,
        chunker.config,
        chunker.model_config,
        chunker.tokenizer,
        stats,
        "source/unicode.md",
    )
    restored = read_chunks_jsonl(output / "chunks.jsonl")
    assert [chunk.to_dict() for chunk in restored] == [chunk.to_dict() for chunk in chunks]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    persisted_stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert manifest["chunks"] == persisted_stats["chunks"] == len(restored)
    assert manifest["documents"] == persisted_stats["documents"] == 1


@pytest.mark.parametrize(
    "mode",
    [
        "malformed",
        "empty",
        "negative",
        "out_of_range",
        "overlap",
        "gap",
        "out_of_order",
        "duplicate_range",
        "rewritten_text",
    ],
)
def test_untrusted_planner_output_cannot_create_chunks(tmp_path: Path, mode: str) -> None:
    document = _markdown_document(tmp_path)
    with pytest.raises(PlanValidationError):
        _chunker(tmp_path, DeterministicPlanner(mode)).chunk(document)


def test_prompt_writer_rejects_inconsistent_or_python_only_artifacts_before_writing(
    tmp_path: Path,
) -> None:
    document = _markdown_document(tmp_path)
    chunker = _chunker(tmp_path, DeterministicPlanner())
    chunks = chunker.chunk(document)
    stats = prompt_corpus_statistics([document], chunks, chunker.metrics)
    output = tmp_path / "artifacts"
    write_prompt_based_artifacts(
        chunks, [document], output, chunker.config, chunker.model_config,
        chunker.tokenizer, stats, "input.jsonl",
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    invalid_stats = dict(stats)
    invalid_stats["python_only"] = (1, 2)
    with pytest.raises(ValueError, match="non-JSON value tuple"):
        write_prompt_based_artifacts(
            chunks, [document], output, chunker.config, chunker.model_config,
            chunker.tokenizer, invalid_stats, "input.jsonl",
        )
    inconsistent_stats = dict(stats)
    inconsistent_stats["chunks"] = len(chunks) + 1
    with pytest.raises(ValueError, match="statistics disagree for chunks"):
        write_prompt_based_artifacts(
            chunks, [document], output, chunker.config, chunker.model_config,
            chunker.tokenizer, inconsistent_stats, "input.jsonl",
        )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_successful_planner_cache_survives_downstream_serialization_failure(
    tmp_path: Path,
) -> None:
    document = _markdown_document(tmp_path)
    planner = DeterministicPlanner()
    chunker = _chunker(tmp_path, planner)
    chunks = chunker.chunk(document)
    assert planner.calls > 0
    cache_before = {
        path.name: path.read_bytes() for path in (tmp_path / "planner-cache").glob("*.json")
    }
    stats = prompt_corpus_statistics([document], chunks, chunker.metrics)
    stats["invalid"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        write_prompt_based_artifacts(
            chunks, [document], tmp_path / "artifacts", chunker.config,
            chunker.model_config, chunker.tokenizer, stats, "input.jsonl",
        )
    assert not (tmp_path / "artifacts" / "chunks.jsonl").exists()
    assert {
        path.name: path.read_bytes() for path in (tmp_path / "planner-cache").glob("*.json")
    } == cache_before

    cache_only = DeterministicPlanner("malformed")
    rerun = _chunker(tmp_path, cache_only).chunk(document)
    assert cache_only.calls == 0
    assert [(chunk.chunk_id, chunk.text) for chunk in rerun] == [
        (chunk.chunk_id, chunk.text) for chunk in chunks
    ]


def test_empty_normalized_blocks_preserve_canonical_separators(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "empty-fence.md"
    path.write_text("Before\n\n```text\n```\n\nAfter\n\n```text\n```\n", encoding="utf-8")
    document = load_document(path, source)
    assert any(not block.text for block in document.blocks)
    chunker = _chunker(tmp_path, DeterministicPlanner())

    chunks = chunker.chunk(document)
    report = validate_prompt_based_chunks(
        [document], chunks, chunker.config, chunker.model_config, chunker.tokenizer
    )

    assert report.valid, report.errors
    assert "".join(chunk.text for chunk in chunks) == document_to_text(document)
    assert any(
        fragment["char_start"] == fragment["char_end"] == 0
        for chunk in chunks
        for fragment in chunk.metadata["block_fragments"]
    )
