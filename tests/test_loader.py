from pathlib import Path

import pytest

from rag_chunking.data.loader import discover_markdown_files, load_document, make_doc_id


def test_discovery_is_recursive_markdown_only_and_stable(tmp_path: Path) -> None:
    (tmp_path / "z.md").write_text("# Z", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.md").write_text("# B", encoding="utf-8")
    (tmp_path / "a" / "ignored.txt").write_text("ignored", encoding="utf-8")

    files = discover_markdown_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == ["a/b.md", "z.md"]


def test_deterministic_id_uses_relative_posix_path() -> None:
    assert make_doc_id("guide\\components\\inputs.md") == "angular:guide/components/inputs.md"
    assert make_doc_id("guide\\components\\inputs.md") == make_doc_id(
        "guide/components/inputs.md"
    )


def test_load_document_metadata_and_hash_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "guide" / "intro.md"
    path.parent.mkdir()
    path.write_text("# Intro\n\nText.\n", encoding="utf-8")

    first = load_document(path, tmp_path)
    second = load_document(path, tmp_path)

    assert first.doc_id == "angular:guide/intro.md"
    assert first.source == "angular"
    assert first.relative_path == "guide/intro.md"
    assert first.filename == "intro.md"
    assert first.to_dict() == second.to_dict()


def test_loader_rejects_path_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.md"
    outside.write_text("# Outside", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside input directory"):
            load_document(outside, tmp_path)
    finally:
        outside.unlink()

