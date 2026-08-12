import json
from pathlib import Path

import pytest

from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.tokenizer import TiktokenTokenizer
from rag_chunking.data.models import DocumentBlock, NormalizedDocument
from rag_chunking.evaluation.evidence import map_evidence_to_chunks, retrieved_evidence_coverage
from rag_chunking.evaluation.evidence_runner import _map_record_evidence, run_evidence_retrieval_benchmark
from rag_chunking.evaluation.metrics import aggregate_evidence, evaluate_evidence_coverage
from rag_chunking.evaluation.qa_dataset import (
    EvidenceSpec, GoldEvidenceItem, QADataset, QARecord, load_qa_dataset, validate_qa_semantics,
)
from rag_chunking.embedding.provider import DeterministicFakeEmbeddingProvider
from rag_chunking.retrieval.protocols import SAME_TOKEN_BUDGET, SAME_TOP_K, RetrievalProtocolConfig
from rag_chunking.retrieval.service import RetrievalService
from test_retrieval import config, make_index, record as embedding_record


FIXTURES = Path(__file__).parent / "fixtures"


def document() -> NormalizedDocument:
    return NormalizedDocument(
        doc_id="test:guide.md", source="test", relative_path="guide.md", filename="guide.md",
        source_sha256="hash", blocks=[
            DocumentBlock(type="heading", text="Café Signals", level=1),
            DocumentBlock(type="paragraph", text="Alpha βeta crosses the boundary safely."),
            DocumentBlock(type="heading", text="Second", level=2),
            DocumentBlock(type="paragraph", text="Another required fact."),
        ],
    )


def structural_chunk(index: int, block_index: int, start: int, end: int, text: str) -> Chunk:
    tokenizer = TiktokenTokenizer()
    return Chunk(
        chunk_id=f"test:guide.md::structure::{index:06d}", strategy="structure_aware",
        doc_id="test:guide.md", source="test", relative_path="guide.md", chunk_index=index,
        text=text, token_start=None, token_end=None, token_count=len(tokenizer.encode(text)),
        chunk_size=512, chunk_overlap=0, tokenizer=tokenizer.name,
        title_path=["Café Signals"], metadata={"block_fragments": [{
            "source_block_index": block_index, "char_start": start, "char_end": end,
        }]},
    )


def qa(evidence, doc_id="test:guide.md") -> QARecord:
    return QARecord.from_dict({
        "id": "q1", "doc_id": doc_id, "question": "What crosses?", "answer": "βeta",
        "evidence_sentences": evidence, "evidence_sections": ["Café Signals"],
        "question_type": "multi_evidence", "difficulty": "medium",
    })


def test_exact_unicode_evidence_and_wrong_doc_id():
    doc = document()
    text = doc.blocks[1].text
    chunk = structural_chunk(0, 1, 0, len(text), text)
    mapping = map_evidence_to_chunks(qa([text]), doc, [chunk], "structure_aware")
    assert mapping[0].match_method == "canonical_text_overlap"
    assert mapping[0].matched_chunk_ids == [chunk.chunk_id]
    assert retrieved_evidence_coverage(mapping[0], {chunk.chunk_id}) == 1.0
    with pytest.raises(ValueError, match="doc_id differ"):
        map_evidence_to_chunks(qa([text], "wrong"), doc, [chunk], "structure_aware")


def test_source_span_boundary_maps_to_multiple_chunks_and_requires_union():
    doc = document()
    text = doc.blocks[1].text
    left = structural_chunk(0, 1, 0, 20, text[:20])
    right = structural_chunk(1, 1, 20, len(text), text[20:])
    record = qa([{"text": text, "block_index": 1, "char_start": 0, "char_end": len(text)}])
    mapping = map_evidence_to_chunks(record, doc, [left, right], "structure_aware")[0]
    assert mapping.match_method == "source_span"
    assert mapping.matched_chunk_ids == [left.chunk_id, right.chunk_id]
    assert 0 < retrieved_evidence_coverage(mapping, {left.chunk_id}) < 1
    assert retrieved_evidence_coverage(mapping, {left.chunk_id, right.chunk_id}) == 1.0


def test_multiple_evidence_units_and_hand_calculated_metrics():
    metric = evaluate_evidence_coverage([1.0, 0.5])
    assert metric == {"evidence_unit_count": 2, "covered_evidence_units": 1, "evidence_coverage": 0.75, "all_evidence_retrieved": 0}
    full = evaluate_evidence_coverage([1.0, 1.0])
    aggregate = aggregate_evidence([metric, full])
    assert aggregate["evidence_coverage"] == 0.875
    assert aggregate["all_evidence_retrieved_rate"] == 0.5


def test_qa_loader_schema_fingerprint_and_semantic_reporting(tmp_path: Path):
    value = {
        "id": "q1", "doc_id": "test:guide.md", "question": " Question ", "answer": "Answer",
        "evidence_sentences": ["not in source"], "evidence_sections": ["Missing"],
        "question_type": "new_team_defined_type", "difficulty": "team_defined", "notes": "development only",
    }
    one = tmp_path / "one.jsonl"; two = tmp_path / "two.jsonl"
    one.write_text(json.dumps(value) + "\n")
    two.write_text(json.dumps(dict(reversed(list(value.items())))) + "\n")
    first = load_qa_dataset(one, {"test:guide.md"})
    second = load_qa_dataset(two, {"test:guide.md"})
    assert first.fingerprint == second.fingerprint
    report = validate_qa_semantics(first, {"test:guide.md": document()})
    assert report.valid and len(report.warnings) == 2


def test_development_only_fixture_follows_external_contract():
    available = {"dev:signals.md", "dev:guide.md", "dev:boundaries.md", "dev:unicode.md"}
    dataset = load_qa_dataset(FIXTURES / "qa_development_only.jsonl", available)
    assert len(dataset.records) == 5
    assert all(record.notes and "DEVELOPMENT ONLY" in record.notes for record in dataset.records)


@pytest.mark.parametrize("change,match", [
    ({"id": ""}, "id must"), ({"question": " "}, "empty"), ({"answer": ""}, "answer must"),
    ({"evidence_sentences": "bad"}, "must be a list"), ({"evidence_sections": "bad"}, "must be a list"),
    ({"question_type": ""}, "question_type"), ({"difficulty": ""}, "difficulty"),
])
def test_qa_loader_rejects_schema_errors(tmp_path: Path, change, match):
    value = {
        "id": "q1", "doc_id": "test:guide.md", "question": "q", "answer": "a",
        "evidence_sentences": [], "evidence_sections": [], "question_type": "type", "difficulty": "easy",
    }
    value.update(change)
    path = tmp_path / "qa.jsonl"; path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match=match):
        load_qa_dataset(path, {"test:guide.md"})



def test_structured_team_evidence_maps_with_authored_identity():
    doc = document()
    text = doc.blocks[1].text
    chunk = structural_chunk(0, 1, 0, len(text), text)
    record = QARecord(
        id="q-team", doc_id=doc.doc_id, question="What crosses?", answer="beta",
        evidence_sentences=[], evidence_sections=[], question_type="multi_evidence",
        difficulty="medium", evidence=[GoldEvidenceItem(
            evidence_id="q-team_e01", doc_id=doc.doc_id, section_path=["Café Signals"],
            evidence_sentences=[EvidenceSpec(text)],
        )],
    )
    mappings = _map_record_evidence(
        record, {doc.doc_id: doc}, [chunk], "structure_aware",
    )
    assert len(mappings) == 1
    assert mappings[0].evidence_id == "q-team_e01:sentence:0"
    assert mappings[0].matched_chunk_ids == [chunk.chunk_id]


def test_three_strategy_two_protocol_dev_integration_is_deterministic(tmp_path: Path):
    cfg = config()
    doc = NormalizedDocument(
        doc_id="angular:a.md", source="angular", relative_path="a.md", filename="a.md",
        source_sha256="source", blocks=[DocumentBlock(type="paragraph", text="alpha evidence")],
    )
    raw = {
        "id": "dev-q", "doc_id": doc.doc_id, "question": "alpha", "answer": "alpha",
        "evidence_sentences": ["alpha evidence"], "evidence_sections": [],
        "question_type": "development", "difficulty": "easy",
    }
    qa_record = QARecord.from_dict(raw)
    dataset = QADataset([qa_record], "dataset-fingerprint")
    index_dirs = {}
    chunks_by_strategy = {}
    for strategy in ("fixed_size", "structure_aware", "prompt_based"):
        records = [embedding_record(cfg, 0, "alpha evidence", [1.0, 0.0, 0.0, 0.0], strategy, "a.md")]
        index_dirs[strategy], _ = make_index(tmp_path, cfg, strategy, records)
        chunks_by_strategy[strategy] = [Chunk(
            chunk_id=records[0].chunk_id, strategy=strategy, doc_id=doc.doc_id,
            source="angular", relative_path="a.md", chunk_index=0, text="alpha evidence",
            token_start=None, token_end=None, token_count=2, chunk_size=512, chunk_overlap=0,
            tokenizer="tiktoken:cl100k_base", metadata={"block_fragments": [{
                "source_block_index": 0, "char_start": 0, "char_end": len("alpha evidence"),
            }]},
        )]
    service = RetrievalService(
        corpus="angular", index_directories=index_dirs, embedding_config=cfg,
        provider=DeterministicFakeEmbeddingProvider(cfg), query_cache_directory=tmp_path / "cache",
        repository_root=tmp_path,
    )
    protocols = [
        RetrievalProtocolConfig(SAME_TOP_K, top_k=1, candidate_k=1),
        RetrievalProtocolConfig(SAME_TOKEN_BUDGET, candidate_k=1, token_budget=2),
    ]
    args = dict(
        strategies=list(index_dirs), protocols=protocols, corpus_fingerprint="corpus",
        chunk_artifact_fingerprints={strategy: f"chunks-{strategy}" for strategy in index_dirs},
    )
    first = run_evidence_retrieval_benchmark(
        service, dataset, {doc.doc_id: doc}, chunks_by_strategy, tmp_path / "results", **args,
    )
    assert len(first.per_query) == 6
    assert all(item["evidence_coverage"] == 1.0 for item in first.per_query)
    assert all(item["actual_selected_tokens"] <= 2 for item in first.per_query)
    deterministic = {name: (first.output_directory / name).read_bytes() for name in ("per_query.jsonl", "aggregate.json", "manifest.json")}
    second = run_evidence_retrieval_benchmark(
        service, dataset, {doc.doc_id: doc}, chunks_by_strategy, tmp_path / "results", **args,
    )
    assert deterministic == {name: (second.output_directory / name).read_bytes() for name in deterministic}
