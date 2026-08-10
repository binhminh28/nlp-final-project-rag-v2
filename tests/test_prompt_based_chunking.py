from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_chunking.chunking.prompt_based import PromptBasedChunker, PromptBasedChunkingConfig
from rag_chunking.chunking.prompt_client import PlannerModelConfig
from rag_chunking.chunking.prompt_schema import PlanValidationError, parse_boundary_plan
from rag_chunking.chunking.prompt_statistics import prompt_corpus_statistics
from rag_chunking.chunking.prompt_validation import validate_prompt_based_chunks
from rag_chunking.chunking.prompt_writer import write_prompt_based_artifacts
from rag_chunking.chunking.writer import read_chunks_jsonl
from rag_chunking.data.models import DocumentBlock, NormalizedDocument, Sentence


class FakePlanner:
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.calls = 0

    def plan(self, system_prompt: str, user_prompt: str, config: PlannerModelConfig) -> str:
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        value = json.loads(user_prompt)
        return json.dumps({"groups": [{
            "start_block_index": value["batch_start_block_index"],
            "end_block_index": value["batch_end_block_index"],
            "reason": "one coherent topic",
        }]})


def document(blocks: list[DocumentBlock], *, text_hash: str = "source-hash") -> NormalizedDocument:
    return NormalizedDocument(
        doc_id="angular:test.md", source="angular", relative_path="test.md",
        filename="test.md", source_sha256=text_hash, blocks=blocks,
    )


def paragraph(text: str) -> DocumentBlock:
    return DocumentBlock(type="paragraph", text=text, sentences=[Sentence("s", text)])


def plan(*ranges: tuple[int, int]) -> str:
    return json.dumps({"groups": [
        {"start_block_index": start, "end_block_index": end, "reason": f"group {i}"}
        for i, (start, end) in enumerate(ranges)
    ]})


def make_chunker(tmp_path: Path, planner: FakePlanner, **config: object) -> PromptBasedChunker:
    return PromptBasedChunker(
        planner, tmp_path / "cache", PromptBasedChunkingConfig(**config),
        PlannerModelConfig(provider="fake", model="deterministic-v1"),
    )


def test_normal_grouping_exact_coverage_order_and_section_crossing(tmp_path: Path) -> None:
    doc = document([
        DocumentBlock(type="heading", text="Inputs", level=1), paragraph("Input details."),
        DocumentBlock(type="heading", text="Outputs", level=1), paragraph("Output details."),
    ])
    chunker = make_chunker(tmp_path, FakePlanner([plan((0, 1), (2, 3))]))
    chunks = chunker.chunk(doc)
    assert [chunk.text for chunk in chunks] == ["Inputs\n\nInput details.", "Outputs\n\nOutput details."]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    report = validate_prompt_based_chunks([doc], chunks, chunker.config, chunker.model_config, chunker.tokenizer)
    assert report.valid, report.errors
    assert report.coverage_gaps == 0

    crossing = make_chunker(tmp_path / "cross", FakePlanner([plan((0, 3))])).chunk(doc)
    assert len(crossing) == 1
    assert crossing[0].metadata["crosses_section_boundary"] is True
    assert crossing[0].metadata["section_paths"] == [["Inputs"], ["Outputs"]]


@pytest.mark.parametrize("raw", [
    "not json",
    '{"groups":[]}',
    plan((0, 0), (2, 2)),
    plan((0, 1), (1, 2)),
    plan((0, 3)),
    '{"groups":[{"start_block_index":0,"end_block_index":2,"reason":"x","extra":1}]}',
])
def test_strict_plan_validation_rejects_malformed_gap_duplicate_unknown_and_extra(raw: str) -> None:
    with pytest.raises(PlanValidationError):
        parse_boundary_plan(raw, 0, 2)


def test_invalid_response_retries_and_valid_response_is_cached(tmp_path: Path) -> None:
    planner = FakePlanner(["bad", plan((0, 0))])
    chunker = make_chunker(tmp_path, planner)
    doc = document([paragraph("Retry me.")])
    first = chunker.chunk(doc)
    assert planner.calls == 2
    assert chunker.metrics.retries == 1
    assert chunker.metrics.invalid_model_responses == 1
    second_planner = FakePlanner(["must not be called"])
    second = make_chunker(tmp_path, second_planner).chunk(doc)
    assert second_planner.calls == 0
    assert [item.to_dict() for item in second] != [item.to_dict() for item in first]
    assert second[0].metadata["cache_hit"] is True
    assert first[0].metadata["cache_hit"] is False


def test_retry_exhaustion_and_corrupt_cache_fail_visibly(tmp_path: Path) -> None:
    doc = document([paragraph("Fail clearly.")])
    chunker = make_chunker(tmp_path, FakePlanner(["bad", "still bad"]), max_retries=1)
    with pytest.raises(PlanValidationError, match="after 2 attempts"):
        chunker.chunk(doc)
    assert chunker.metrics.invalid_model_responses == 2

    valid = make_chunker(tmp_path / "corrupt", FakePlanner([plan((0, 0))]))
    valid.chunk(doc)
    cache_file = next((tmp_path / "corrupt" / "cache").glob("*.json"))
    cache_file.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt prompt cache entry"):
        make_chunker(tmp_path / "corrupt", FakePlanner()).chunk(doc)


def test_cache_invalidates_on_source_prompt_and_model_configuration(tmp_path: Path) -> None:
    base_doc = document([paragraph("A")])
    make_chunker(tmp_path, FakePlanner()).chunk(base_doc)
    source_planner = FakePlanner()
    make_chunker(tmp_path, source_planner).chunk(document([paragraph("B")]))
    assert source_planner.calls == 1
    prompt_planner = FakePlanner()
    make_chunker(tmp_path, prompt_planner, prompt_version="prompt_based_test_variant").chunk(base_doc)
    assert prompt_planner.calls == 1
    model_planner = FakePlanner()
    PromptBasedChunker(
        model_planner, tmp_path / "cache", model_config=PlannerModelConfig(provider="fake", model="v2")
    ).chunk(base_doc)
    assert model_planner.calls == 1


def test_planner_candidates_include_compact_v2_structure_metadata(tmp_path: Path) -> None:
    block = DocumentBlock(
        type="table",
        text="| Name |\n| --- |\n| A |",
        metadata={
            "header": ["Name"],
            "column_count": 1,
            "row_count": 1,
            "container_path": [{"type": "step", "title": "Compare"}],
        },
    )
    chunker = make_chunker(tmp_path, FakePlanner())

    candidate = chunker._candidates(document([block]))[0]

    assert candidate["structure"] == {
        "container_path": [{"type": "step", "title": "Compare"}],
        "table": {"header": ["Name"], "column_count": 1, "row_count": 1},
    }


@pytest.mark.parametrize("block_type", ["code_block", "list", "table", "custom_block"])
def test_oversized_line_oriented_blocks_preserve_exact_content(tmp_path: Path, block_type: str) -> None:
    probe = make_chunker(tmp_path, FakePlanner())
    lines = [(" value" * 24) + "\n" for _ in range(4)]
    text = "".join(lines)
    doc = document([DocumentBlock(type=block_type, text=text)])
    chunker = make_chunker(tmp_path, FakePlanner([plan((0, 0))]), max_chunk_tokens=30)
    chunks = chunker.chunk(doc)
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(chunk.token_count <= 30 for chunk in chunks)
    assert all(chunk.metadata["oversized_fallback"] for chunk in chunks)


def test_oversized_paragraph_sentences_and_unicode_token_fallback(tmp_path: Path) -> None:
    probe = make_chunker(tmp_path, FakePlanner())
    sentences = [" word" * 24 for _ in range(3)]
    prose = " ".join(sentences)
    prose_doc = document([DocumentBlock(
        type="paragraph", text=prose,
        sentences=[Sentence(str(i), value) for i, value in enumerate(sentences)],
    )])
    prose_chunker = make_chunker(tmp_path / "prose", FakePlanner([plan((0, 0))]), max_chunk_tokens=30)
    prose_chunks = prose_chunker.chunk(prose_doc)
    assert "".join(chunk.text for chunk in prose_chunks) == prose

    unicode_text = "🌷" * 80
    unicode_doc = document([DocumentBlock(
        type="paragraph", text=unicode_text, sentences=[Sentence("u", unicode_text)]
    )])
    unicode_chunker = make_chunker(tmp_path / "unicode", FakePlanner([plan((0, 0))]), max_chunk_tokens=30)
    chunks = unicode_chunker.chunk(unicode_doc)
    assert "".join(chunk.text for chunk in chunks) == unicode_text
    assert all("\ufffd" not in chunk.text and chunk.token_count <= 30 for chunk in chunks)
    assert all(any(item["token_fallback"] for item in chunk.metadata["block_fragments"]) for chunk in chunks)


def test_planner_group_is_locally_adjusted_and_ids_are_deterministic(tmp_path: Path) -> None:
    doc = document([paragraph(" a" * 40), paragraph(" b" * 40), paragraph(" c" * 40)])
    first_chunker = make_chunker(tmp_path, FakePlanner([plan((0, 2))]), max_chunk_tokens=60)
    first = first_chunker.chunk(doc)
    second = make_chunker(tmp_path, FakePlanner(), max_chunk_tokens=60).chunk(doc)
    assert len(first) == 3
    assert all(chunk.metadata["locally_adjusted"] for chunk in first)
    assert [chunk.chunk_id for chunk in first] == [f"angular:test.md::prompt::{i:06d}" for i in range(3)]
    # Cache-related diagnostic fields intentionally differ, while IDs/text stay stable.
    assert [(c.chunk_id, c.text) for c in first] == [(c.chunk_id, c.text) for c in second]


def test_artifacts_are_byte_identical_with_fixed_cached_response(tmp_path: Path) -> None:
    doc = document([DocumentBlock(type="heading", text="H", level=1), paragraph("Body")])
    # Resolve once, then compare two complete cache-backed runs.
    make_chunker(tmp_path, FakePlanner([plan((0, 1))])).chunk(doc)
    first_chunker = make_chunker(tmp_path, FakePlanner())
    first_chunks = first_chunker.chunk(doc)
    first_stats = prompt_corpus_statistics([doc], first_chunks, first_chunker.metrics)
    output = tmp_path / "output"
    write_prompt_based_artifacts(
        first_chunks, [doc], output, first_chunker.config, first_chunker.model_config,
        first_chunker.tokenizer, first_stats, "input.jsonl",
    )
    first_bytes = {name: (output / name).read_bytes() for name in ("chunks.jsonl", "manifest.json", "stats.json")}
    second_chunker = make_chunker(tmp_path, FakePlanner())
    second_chunks = second_chunker.chunk(doc)
    second_stats = prompt_corpus_statistics([doc], second_chunks, second_chunker.metrics)
    write_prompt_based_artifacts(
        second_chunks, [doc], output, second_chunker.config, second_chunker.model_config,
        second_chunker.tokenizer, second_stats, "input.jsonl",
    )
    assert {name: (output / name).read_bytes() for name in first_bytes} == first_bytes
    assert read_chunks_jsonl(output / "chunks.jsonl")[0].to_dict() == first_chunks[0].to_dict()
