from __future__ import annotations

from pathlib import Path

import pytest

from rag_chunking.chunking.writer import read_chunks_jsonl
from rag_chunking.cli import chunk_documents
from rag_chunking.data.models import DocumentBlock, NormalizedDocument
from rag_chunking.data.writer import write_processed_corpus


def make_document(doc_id: str) -> NormalizedDocument:
    return NormalizedDocument(
        doc_id=doc_id,
        source="angular",
        relative_path=f"{doc_id}.md",
        filename=f"{doc_id}.md",
        source_sha256="abc123",
        blocks=[
            DocumentBlock(type="heading", text="Title", level=1),
            DocumentBlock(type="paragraph", text="Some body text about routing and modules." * 5),
        ],
    )


@pytest.fixture
def documents_path(tmp_path: Path) -> Path:
    documents = [make_document("angular:a.md"), make_document("angular:b.md")]
    output_dir = tmp_path / "processed"
    write_processed_corpus(documents, output_dir)
    return output_dir / "documents.jsonl"


CONFIG_BOTH = """
chunking:
  enabled_strategies:
    - fixed_size
    - structure_aware
  fixed_size:
    chunk_size: 64
    overlap: 8
  structure_aware:
    max_chunk_tokens: 64
"""

CONFIG_LLM_ENABLED = """
chunking:
  enabled_strategies:
    - prompt_based
"""


def test_runs_all_enabled_strategies(tmp_path: Path, documents_path: Path) -> None:
    config_path = tmp_path / "chunking.yaml"
    config_path.write_text(CONFIG_BOTH, encoding="utf-8")
    output_root = tmp_path / "chunks"

    exit_code = chunk_documents.main(
        [
            "--input",
            str(documents_path),
            "--output-root",
            str(output_root),
            "--config",
            str(config_path),
            "--strategy",
            "all",
        ]
    )

    assert exit_code == 0
    for strategy in ("fixed_size", "structure_aware"):
        chunks = read_chunks_jsonl(output_root / strategy / "chunks.jsonl")
        assert len(chunks) > 0
        assert all(chunk.strategy == strategy for chunk in chunks)


def test_single_strategy_subset(tmp_path: Path, documents_path: Path) -> None:
    config_path = tmp_path / "chunking.yaml"
    config_path.write_text(CONFIG_BOTH, encoding="utf-8")
    output_root = tmp_path / "chunks"

    exit_code = chunk_documents.main(
        [
            "--input",
            str(documents_path),
            "--output-root",
            str(output_root),
            "--config",
            str(config_path),
            "--strategy",
            "fixed_size",
        ]
    )

    assert exit_code == 0
    assert (output_root / "fixed_size" / "chunks.jsonl").exists()
    assert not (output_root / "structure_aware").exists()


def test_requesting_disabled_strategy_fails(tmp_path: Path, documents_path: Path) -> None:
    config_path = tmp_path / "chunking.yaml"
    config_path.write_text(CONFIG_BOTH, encoding="utf-8")
    output_root = tmp_path / "chunks"

    exit_code = chunk_documents.main(
        [
            "--input",
            str(documents_path),
            "--output-root",
            str(output_root),
            "--config",
            str(config_path),
            "--strategy",
            "prompt_based",
        ]
    )

    assert exit_code == 1


def test_enabling_unimplemented_llm_strategy_fails_with_clear_message_and_no_network(
    tmp_path: Path, documents_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "chunking.yaml"
    config_path.write_text(CONFIG_LLM_ENABLED, encoding="utf-8")
    output_root = tmp_path / "chunks"

    exit_code = chunk_documents.main(
        [
            "--input",
            str(documents_path),
            "--output-root",
            str(output_root),
            "--config",
            str(config_path),
            "--strategy",
            "all",
        ]
    )

    assert exit_code == 1
    assert not output_root.exists()
    captured = capsys.readouterr()
    assert "not available in this environment" in captured.out
