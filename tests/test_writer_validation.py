import json
from pathlib import Path

from rag_chunking.data.loader import load_document
from rag_chunking.data.validation import validate_corpus
from rag_chunking.data.writer import read_documents_jsonl, write_processed_corpus


def test_write_read_and_repeat_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "processed"
    source.mkdir()
    path = source / "doc.md"
    path.write_text("# Title\n\nText.\n", encoding="utf-8")
    document = load_document(path, source)

    report = validate_corpus([document], ["doc.md"])
    assert report.valid

    write_processed_corpus([document], output)
    first_documents = (output / "documents.jsonl").read_bytes()
    first_manifest = (output / "manifest.json").read_bytes()
    write_processed_corpus([document], output)

    assert (output / "documents.jsonl").read_bytes() == first_documents
    assert (output / "manifest.json").read_bytes() == first_manifest
    assert read_documents_jsonl(output / "documents.jsonl")[0].to_dict() == document.to_dict()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["statistics"]["documents"] == 1


def test_validation_detects_duplicate_identity(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("# Title", encoding="utf-8")
    document = load_document(path, tmp_path)

    report = validate_corpus([document, document], ["doc.md", "doc.md"])

    assert not report.valid
    assert any("Duplicate doc_id" in error for error in report.errors)
    assert any("Duplicate relative_path" in error for error in report.errors)
