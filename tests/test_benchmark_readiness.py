"""Synthetic/non-production plumbing tests for canonical benchmark readiness."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import urllib.request

import pytest

from rag_chunking.benchmark import (
    prepare_answer_benchmark_inputs, project_benchmark_queries,
    validate_generation_requests_against_preparation,
    validate_prepared_benchmark_inputs,
)
from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.tokenizer import TiktokenTokenizer
from rag_chunking.chunking.writer import serialize_chunks_jsonl, serialize_json, write_artifact_set
from rag_chunking.context import ContextConfig
from rag_chunking.data.models import DocumentBlock, NormalizedDocument
from rag_chunking.data.writer import write_processed_corpus
from rag_chunking.embedding.index import build_local_index
from rag_chunking.embedding.models import EmbeddingConfig
from rag_chunking.embedding.pipeline import run_embedding_pipeline
from rag_chunking.embedding.provider import DeterministicFakeEmbeddingProvider
from rag_chunking.evaluation import EvaluationConfig, run_answer_benchmark
from rag_chunking.evaluation.qa_dataset import (
    QA_DATASET_SCHEMA_VERSION, QADataset, load_qa_dataset,
    qa_dataset_fingerprint, validate_canonical_qa_dataset,
)
from rag_chunking.generation import (
    DeterministicFakeGenerationProvider, GenerationCache, GenerationConfig,
    GenerationInput, GenerationService, run_generation,
)
from rag_chunking.readiness import run_benchmark_preflight
from rag_chunking.retrieval import (
    RetrievalProtocolConfig, RetrievalService, SAME_TOP_K,
)


STRATEGIES = ("fixed_size", "structure_aware", "prompt_based")


def _qa_value(**changes):
    value = {
        "id": "synthetic-q1", "doc_id": "angular:fixture.md",
        "question": "What is alpha?", "answer": "Alpha is synthetic.",
        "evidence_sentences": ["Alpha is synthetic evidence."],
        "evidence_sections": ["Synthetic"], "question_type": "synthetic_category",
        "difficulty": "synthetic", "notes": "SYNTHETIC NON-PRODUCTION FIXTURE",
    }
    value.update(changes)
    return value


def _write_qa(path: Path, values=None) -> Path:
    values = values or [_qa_value()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")
    return path


def _setup_artifacts(root: Path):
    tokenizer = TiktokenTokenizer()
    document = NormalizedDocument(
        doc_id="angular:fixture.md", source="angular", relative_path="fixture.md",
        filename="fixture.md", source_sha256="synthetic-source",
        blocks=[
            DocumentBlock(type="heading", text="Synthetic", level=1),
            DocumentBlock(type="paragraph", text="Alpha is synthetic evidence."),
        ],
    )
    processed = root / "processed" / "angular"
    write_processed_corpus([document], processed)
    documents_path = processed / "documents.jsonl"
    chunks_root = root / "chunks"
    for strategy in STRATEGIES:
        text = "Synthetic\n\nAlpha is synthetic evidence."
        chunk = Chunk(
            chunk_id=f"angular:fixture.md::{strategy}::000000", strategy=strategy,
            doc_id=document.doc_id, source="angular", relative_path="fixture.md",
            chunk_index=0, text=text, token_start=None, token_end=None,
            token_count=len(tokenizer.encode(text)), chunk_size=512, chunk_overlap=0,
            tokenizer=tokenizer.name,
        )
        config_fingerprint = f"synthetic-chunks-{strategy}"
        manifest = {
            "schema_version": 1, "source_schema_version": "normalized_document_v2",
            "strategy": strategy, "config_fingerprint": config_fingerprint,
            "configuration": {"strategy": strategy, "fixture": "non-production"},
            "source_input": documents_path.as_posix(), "documents": 1, "chunks": 1,
        }
        write_artifact_set(chunks_root / "angular" / strategy, {
            "chunks.jsonl": serialize_chunks_jsonl([chunk]),
            "stats.json": serialize_json({"documents": 1, "chunks": 1}),
            "manifest.json": serialize_json(manifest),
        })
    embedding = EmbeddingConfig(
        provider="fake", model="synthetic-fake-v1", dimension=4,
        max_batch_items=4, max_batch_tokens=100, max_input_tokens=100,
    )
    embedding_config = root / "embedding.yaml"
    embedding_config.write_text(
        "embedding:\n  provider: fake\n  model: synthetic-fake-v1\n  dimension: 4\n"
        "  max_batch_items: 4\n  max_batch_tokens: 100\n  max_input_tokens: 100\n"
        "  tokenizer: cl100k_base\n  encoding_format: float\n",
        encoding="utf-8",
    )
    embeddings_root = root / "embeddings"
    indexes_root = root / "indexes"
    index_dirs = {}
    for strategy in STRATEGIES:
        embedding_dir = embeddings_root / "angular" / strategy / embedding.fingerprint
        run_embedding_pipeline(
            chunks_root / "angular" / strategy, embedding_dir,
            root / "embedding-cache", DeterministicFakeEmbeddingProvider(embedding),
            corpus="angular",
        )
        index_dir = indexes_root / "angular" / strategy / embedding.fingerprint
        build_local_index(embedding_dir, index_dir)
        index_dirs[strategy] = index_dir
    return {
        "document": document, "processed_root": root / "processed",
        "documents_path": documents_path, "chunks_root": chunks_root,
        "embeddings_root": embeddings_root, "indexes_root": indexes_root,
        "embedding": embedding, "embedding_config": embedding_config,
        "index_dirs": index_dirs,
    }


def _preflight(root: Path, artifacts, dataset_path: Path):
    return run_benchmark_preflight(
        dataset_path=dataset_path, processed_root=artifacts["processed_root"],
        chunks_root=artifacts["chunks_root"], embeddings_root=artifacts["embeddings_root"],
        indexes_root=artifacts["indexes_root"], embedding_config_path=artifacts["embedding_config"],
        generation_config=GenerationConfig(provider="fake", model="synthetic-fake-v1"),
        evaluation_config=EvaluationConfig(),
        protocol_config=RetrievalProtocolConfig(SAME_TOP_K, top_k=1, candidate_k=1),
        context_config=ContextConfig(context_token_budget=512),
    )


def test_strict_dataset_validation_and_fingerprint_contract(tmp_path: Path):
    artifacts = _setup_artifacts(tmp_path / "artifacts")
    first_path = _write_qa(tmp_path / "one.jsonl")
    reversed_value = dict(reversed(list(_qa_value().items())))
    second_path = _write_qa(tmp_path / "different-location.jsonl", [reversed_value])
    documents = {artifacts["document"].doc_id: artifacts["document"]}
    first, report = validate_canonical_qa_dataset(first_path, documents)
    second, second_report = validate_canonical_qa_dataset(second_path, documents)
    assert report.valid and second_report.valid
    assert first.schema_version == QA_DATASET_SCHEMA_VERSION
    assert first.fingerprint == second.fingerprint == qa_dataset_fingerprint(first.records)
    for field, changed in (
        ("question", "Changed?"), ("answer", "Changed answer"),
        ("evidence_sentences", ["Changed evidence"]),
        ("question_type", "changed_category"), ("difficulty", "changed"),
    ):
        path = _write_qa(tmp_path / f"{field}.jsonl", [_qa_value(**{field: changed})])
        changed_dataset = load_qa_dataset(path, set(documents))
        assert changed_dataset.fingerprint != first.fingerprint


@pytest.mark.parametrize("value,match", [
    (_qa_value(answer=""), "answer must be non-empty"),
    ({key: item for key, item in _qa_value().items() if key != "question_type"}, "missing QA fields"),
    (_qa_value(evidence_sentences=[{"text": "x", "block_index": 99, "char_start": 0, "char_end": 1}]), "outside"),
    (_qa_value(schema_version="evidence_qa_dataset_v2"), "unknown QA fields"),
])
def test_dataset_handoff_rejects_missing_malformed_or_schema_mismatch(tmp_path: Path, value, match):
    artifacts = _setup_artifacts(tmp_path / "artifacts")
    path = _write_qa(tmp_path / "bad.jsonl", [value])
    documents = {artifacts["document"].doc_id: artifacts["document"]}
    if "outside" in match:
        dataset, report = validate_canonical_qa_dataset(path, documents)
        assert not report.valid and any(match in item for item in report.errors)
    else:
        with pytest.raises(ValueError, match=match):
            validate_canonical_qa_dataset(path, documents)
    with pytest.raises(ValueError, match="unsupported QA dataset schema"):
        qa_dataset_fingerprint([], schema_version="evidence_qa_dataset_v2")


def test_duplicate_ids_and_duplicate_json_keys_fail_loudly(tmp_path: Path):
    artifacts = _setup_artifacts(tmp_path / "artifacts")
    path = _write_qa(tmp_path / "duplicate.jsonl", [_qa_value(), _qa_value()])
    with pytest.raises(ValueError, match="duplicate id"):
        load_qa_dataset(path, {artifacts["document"].doc_id})
    duplicate_key = tmp_path / "duplicate-key.jsonl"
    duplicate_key.write_text('{"id":"one","id":"two"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_qa_dataset(duplicate_key, {artifacts["document"].doc_id})


def test_preflight_missing_invalid_missing_strategy_incompatible_and_ready(tmp_path: Path):
    artifacts = _setup_artifacts(tmp_path / "artifacts")
    missing = _preflight(tmp_path, artifacts, tmp_path / "canonical-missing.jsonl")
    assert not missing.ready
    assert missing.blockers == ("Canonical production QA dataset not yet available.",)
    assert all(
        any(check.name == f"strategy:{strategy}" and check.status == "PASS" for check in missing.checks)
        for strategy in STRATEGIES
    )
    invalid_path = _write_qa(tmp_path / "invalid.jsonl", [_qa_value(answer="")])
    invalid = _preflight(tmp_path, artifacts, invalid_path)
    assert not invalid.ready and any("dataset invalid" in item.lower() for item in invalid.blockers)
    valid_path = _write_qa(tmp_path / "canonical.jsonl")
    ready = _preflight(tmp_path, artifacts, valid_path)
    assert ready.ready and not ready.blockers

    missing_manifest = artifacts["indexes_root"] / "angular" / "prompt_based" / artifacts["embedding"].fingerprint / "manifest.json"
    missing_manifest.unlink()
    broken = _preflight(tmp_path, artifacts, valid_path)
    assert not broken.ready and any("prompt_based artifacts invalid" in item for item in broken.blockers)

    incompatible = replace(artifacts["embedding"], model="other")
    incompatible_path = tmp_path / "incompatible.yaml"
    incompatible_path.write_text(
        f"embedding:\n  provider: fake\n  model: other\n  dimension: {incompatible.dimension}\n",
        encoding="utf-8",
    )
    report = run_benchmark_preflight(
        dataset_path=valid_path, processed_root=artifacts["processed_root"],
        chunks_root=artifacts["chunks_root"], embeddings_root=artifacts["embeddings_root"],
        indexes_root=artifacts["indexes_root"], embedding_config_path=incompatible_path,
        generation_config=GenerationConfig(provider="fake", model="synthetic-fake-v1"),
        evaluation_config=EvaluationConfig(),
        protocol_config=RetrievalProtocolConfig(SAME_TOP_K, top_k=1, candidate_k=1),
        context_config=ContextConfig(context_token_budget=512),
    )
    assert not report.ready


def test_synthetic_nonproduction_full_offline_three_strategy_dry_run(tmp_path: Path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("synthetic dry run must not use network")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    artifacts = _setup_artifacts(tmp_path / "artifacts")
    qa_path = _write_qa(tmp_path / "SYNTHETIC_NON_PRODUCTION_QA.jsonl", [
        _qa_value(),
        _qa_value(
            id="arbitrary-query-90210", question="Where is synthetic evidence?",
            answer="It is in the synthetic fixture.", question_type="another_synthetic_category",
        ),
    ])
    documents = {artifacts["document"].doc_id: artifacts["document"]}
    dataset, report = validate_canonical_qa_dataset(qa_path, documents)
    assert report.valid and len(dataset.records) == 2

    class CapturingFakeEmbeddingProvider(DeterministicFakeEmbeddingProvider):
        def __init__(self, config):
            super().__init__(config); self.seen = []
        def embed_texts(self, texts):
            self.seen.extend(texts)
            return super().embed_texts(texts)

    embedding_provider = CapturingFakeEmbeddingProvider(artifacts["embedding"])
    retrieval = RetrievalService(
        corpus="angular", index_directories=artifacts["index_dirs"],
        embedding_config=artifacts["embedding"], provider=embedding_provider,
        query_cache_directory=tmp_path / "query-cache", repository_root=tmp_path,
    )
    protocol = RetrievalProtocolConfig(SAME_TOP_K, top_k=1, candidate_k=1)
    context_config = ContextConfig(context_token_budget=512)
    prepared = prepare_answer_benchmark_inputs(
        retrieval, project_benchmark_queries(dataset.records),
        dataset_fingerprint=dataset.fingerprint,
        corpus_fingerprint="synthetic-corpus-fingerprint",
        protocol=protocol, context_config=context_config,
        output_directory=tmp_path / "prepared",
    )
    assert embedding_provider.seen == [item.question for item in dataset.records]
    serialized_prepared = {
        path.name: path.read_bytes() for path in (tmp_path / "prepared").iterdir()
        if path.name != "stats.json"
    }
    reused = prepare_answer_benchmark_inputs(
        retrieval, project_benchmark_queries(dataset.records),
        dataset_fingerprint=dataset.fingerprint,
        corpus_fingerprint="synthetic-corpus-fingerprint",
        protocol=protocol, context_config=context_config,
        output_directory=tmp_path / "prepared",
    )
    assert reused.reused and serialized_prepared == {
        path.name: path.read_bytes() for path in (tmp_path / "prepared").iterdir()
        if path.name != "stats.json"
    }
    calls_before_conflict = embedding_provider.calls
    with pytest.raises(ValueError, match="different identity"):
        prepare_answer_benchmark_inputs(
            retrieval, project_benchmark_queries(dataset.records),
            dataset_fingerprint=dataset.fingerprint,
            corpus_fingerprint="different-synthetic-corpus",
            protocol=protocol, context_config=context_config,
            output_directory=tmp_path / "prepared",
        )
    assert embedding_provider.calls == calls_before_conflict

    generation_config = GenerationConfig(provider="fake", model="synthetic-fake-v1")
    run_paths = {}
    for strategy in STRATEGIES:
        inputs = [
            GenerationInput.create(
                context.query_id,
                next(item.question for item in dataset.records if item.id == context.query_id),
                context, generation_config,
            )
            for context in prepared.contexts_by_strategy[strategy]
        ]
        generation_path = tmp_path / "generation" / strategy
        result = run_generation(
            inputs,
            GenerationService(
                generation_config, DeterministicFakeGenerationProvider(),
                cache=GenerationCache(tmp_path / "generation-cache"),
            ),
            generation_path,
        )
        assert result.complete
        run_paths[strategy] = generation_path

    validated_preparation = validate_prepared_benchmark_inputs(
        tmp_path / "prepared", dataset_fingerprint=dataset.fingerprint,
        expected_queries=project_benchmark_queries(dataset.records),
    )
    validate_generation_requests_against_preparation(validated_preparation, run_paths)
    evaluation = run_answer_benchmark(
        dataset, run_paths, tmp_path / "evaluation",
        source_corpus_fingerprint=validated_preparation.manifest["corpus_fingerprint"],
        preparation_fingerprint=validated_preparation.preparation_fingerprint,
    )
    assert len(evaluation.paired) == 2
    assert all(set(item["strategies"]) == set(STRATEGIES) for item in evaluation.paired)
    assert all(item["question_type"] == item["category"] and item["difficulty"] for item in evaluation.paired)
    before = {
        path.name: path.read_bytes() for path in (tmp_path / "evaluation").iterdir()
        if path.name != "stats.json"
    }
    repeated = run_answer_benchmark(
        dataset, run_paths, tmp_path / "evaluation",
        source_corpus_fingerprint=validated_preparation.manifest["corpus_fingerprint"],
        preparation_fingerprint=validated_preparation.preparation_fingerprint,
    )
    assert repeated.benchmark_fingerprint == evaluation.benchmark_fingerprint
    assert before == {
        path.name: path.read_bytes() for path in (tmp_path / "evaluation").iterdir()
        if path.name != "stats.json"
    }
    # Gold is absent from retrieval projection and generated prompts/contexts.
    assert all(not hasattr(item, "answer") for item in project_benchmark_queries(dataset.records))
    assert all(record.answer not in context.rendered_context for record in dataset.records for values in prepared.contexts_by_strategy.values() for context in values)
