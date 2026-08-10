from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_chunking.chunking.fixed_size import FixedSizeChunker, FixedSizeChunkingConfig
from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.prompt_based import PromptBasedChunker, PromptBasedChunkingConfig
from rag_chunking.chunking.prompt_client import PlannerModelConfig
from rag_chunking.chunking.prompt_validation import validate_prompt_based_chunks
from rag_chunking.chunking.statistics import chunk_corpus_statistics
from rag_chunking.chunking.structure_aware import (
    StructureAwareChunker,
    StructureAwareChunkingConfig,
)
from rag_chunking.chunking.structure_validation import validate_structure_aware_chunks
from rag_chunking.chunking.validation import validate_fixed_size_chunks
from rag_chunking.chunking.writer import (
    read_chunks_jsonl,
    write_fixed_size_artifacts,
)
from rag_chunking.data.models import DocumentBlock, NormalizedDocument, Sentence


TULIP = "🌷"


class WholeDocumentPlanner:
    def plan(self, system_prompt: str, user_prompt: str, config: PlannerModelConfig) -> str:
        value = json.loads(user_prompt)
        return json.dumps(
            {
                "groups": [
                    {
                        "start_block_index": value["batch_start_block_index"],
                        "end_block_index": value["batch_end_block_index"],
                        "reason": "fixture",
                    }
                ]
            }
        )


def document(text: str) -> NormalizedDocument:
    return NormalizedDocument(
        doc_id="angular:unicode.md",
        source="angular",
        relative_path="unicode.md",
        filename="unicode.md",
        source_sha256="source-hash",
        blocks=[
            DocumentBlock(
                type="paragraph",
                text=text,
                sentences=[Sentence("unicode:0", text)],
                metadata={"nested": {"language": "vi", "flags": [True, None, 3]}},
            )
        ],
    )


def prompt_chunker(tmp_path: Path, max_tokens: int = 24) -> PromptBasedChunker:
    return PromptBasedChunker(
        WholeDocumentPlanner(),
        tmp_path / "cache",
        PromptBasedChunkingConfig(max_chunk_tokens=max_tokens),
        PlannerModelConfig(provider="fake", model="deterministic"),
    )


def test_all_strategies_emit_one_runtime_schema_and_round_trip(tmp_path: Path) -> None:
    text = (
        f"Tiếng Việt nguyên vẹn {TULIP}; e\u0301; 你好; مرحبا.\n\n"
        "| cột | giá trị |\n| --- | --- |\n| hoa | 🌷 |\n\n"
        "```ts\nconst flower = '🌷';\n```\n"
    ) * 8
    doc = document(text)
    fixed = FixedSizeChunker(FixedSizeChunkingConfig(chunk_size=24, chunk_overlap=3))
    structure = StructureAwareChunker(StructureAwareChunkingConfig(max_chunk_tokens=24))
    prompt = prompt_chunker(tmp_path)
    outputs = {
        "fixed_size": fixed.chunk(doc),
        "structure_aware": structure.chunk(doc),
        "prompt_based": prompt.chunk(doc),
    }

    expected_keys = set(Chunk.__dataclass_fields__)
    for strategy, chunks in outputs.items():
        assert chunks
        assert all(type(chunk) is Chunk and set(chunk.to_dict()) == expected_keys for chunk in chunks)
        assert all(chunk.strategy == strategy and chunk.token_count <= chunk.chunk_size for chunk in chunks)
        assert all("\ufffd" not in chunk.text for chunk in chunks)
        for chunk in chunks:
            encoded = json.dumps(chunk.to_dict(), ensure_ascii=False, allow_nan=False)
            restored = Chunk.from_dict(json.loads(encoded))
            assert restored.to_dict() == chunk.to_dict()

    source_tokens = fixed.tokenizer.encode(text)
    covered = {
        position
        for chunk in outputs["fixed_size"]
        for position in range(chunk.token_start, chunk.token_end)  # type: ignore[arg-type]
    }
    assert covered == set(range(len(source_tokens)))
    assert "".join(chunk.text for chunk in outputs["structure_aware"]) == text
    assert "".join(chunk.text for chunk in outputs["prompt_based"]) == text
    assert TULIP.encode("utf-8") == bytes.fromhex("f0 9f 8c b7")

    assert validate_fixed_size_chunks([doc], outputs["fixed_size"], fixed.config, fixed.tokenizer).valid
    assert validate_structure_aware_chunks(
        [doc], outputs["structure_aware"], structure.config, structure.tokenizer
    ).valid
    assert validate_prompt_based_chunks(
        [doc], outputs["prompt_based"], prompt.config, prompt.model_config, prompt.tokenizer
    ).valid


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.update(token_start=2, token_end=1), "token span"),
        (lambda value: value.update(token_count=-1), "token_count"),
        (lambda value: value["metadata"].update(value=float("nan")), "non-finite"),
        (lambda value: value["metadata"].update(value=(1, 2)), "non-JSON"),
        (lambda value: value.update(extra="legacy"), "unknown chunk fields"),
    ],
)
def test_chunk_model_rejects_invalid_or_lossy_representations(mutation, message: str) -> None:
    value = FixedSizeChunker().chunk(document("valid text"))[0].to_dict()
    mutation(value)
    with pytest.raises((TypeError, ValueError), match=message):
        Chunk.from_dict(value)


def test_reader_rejects_non_standard_json_constants(tmp_path: Path) -> None:
    value = FixedSizeChunker().chunk(document("valid text"))[0].to_dict()
    value["metadata"]["invalid"] = float("nan")
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant NaN"):
        read_chunks_jsonl(path)


def test_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    value = FixedSizeChunker().chunk(document("valid text"))[0].to_dict()
    encoded = json.dumps(value)
    path = tmp_path / "chunks.jsonl"
    path.write_text('{"chunk_id":"shadow",' + encoded[1:] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key 'chunk_id'"):
        read_chunks_jsonl(path)


def test_serialization_failure_preserves_existing_artifact_set(tmp_path: Path) -> None:
    doc = document("valid text")
    chunker = FixedSizeChunker()
    chunks = chunker.chunk(doc)
    stats = chunk_corpus_statistics([doc], chunks, chunker.tokenizer)
    output = tmp_path / "artifacts"
    write_fixed_size_artifacts(
        chunks, output, chunker.config, chunker.tokenizer, stats, "input.jsonl"
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    chunks[0].metadata["invalid"] = float("inf")

    with pytest.raises(ValueError, match="non-finite"):
        write_fixed_size_artifacts(
            chunks, output, chunker.config, chunker.tokenizer, stats, "input.jsonl"
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_shared_validation_rejects_wrong_id_source_hash_and_hierarchy() -> None:
    doc = document("enough text")
    chunker = FixedSizeChunker()
    chunks = chunker.chunk(doc)
    chunks[0].chunk_id = "arbitrary"
    chunks[0].metadata["source_sha256"] = "wrong"
    chunks[0].children_ids = ["missing"]

    report = validate_fixed_size_chunks([doc], chunks, chunker.config, chunker.tokenizer)

    assert not report.valid
    assert any("non-deterministic chunk_id" in error for error in report.errors)
    assert any("incorrect source provenance" in error for error in report.errors)
    assert any("child id does not resolve" in error for error in report.errors)


def test_structure_validation_rejects_semantically_wrong_reciprocal_hierarchy() -> None:
    doc = NormalizedDocument(
        doc_id="angular:hierarchy.md",
        source="angular",
        relative_path="hierarchy.md",
        filename="hierarchy.md",
        source_sha256="source-hash",
        blocks=[
            DocumentBlock(type="heading", text="Parent", level=1),
            DocumentBlock(type="paragraph", text="parent body"),
            DocumentBlock(type="heading", text="Child", level=2),
            DocumentBlock(type="paragraph", text="child body"),
        ],
    )
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(doc)
    parent, child = chunks
    parent.parent_id = child.chunk_id
    child.children_ids = [parent.chunk_id]

    report = validate_structure_aware_chunks([doc], chunks, chunker.config, chunker.tokenizer)

    assert not report.valid
    assert any("incorrect hierarchy links" in error for error in report.errors)
