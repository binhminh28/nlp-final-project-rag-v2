from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.evaluation import (
    AggregateEvaluationResult, EvaluationConfig, PerQueryEvaluationResult,
    evaluate_references, load_committed_generation_run, normalize_answer,
    normalized_containment, normalized_exact_match, run_answer_benchmark,
    token_prf, tokenize_answer, validate_answer_benchmark_artifacts,
)
from rag_chunking.evaluation.qa_dataset import QADataset, QARecord
from rag_chunking.generation.artifacts import GENERATION_RUN_SCHEMA_VERSION
from rag_chunking.generation.models import (
    AnswerResult, InputTokenAccounting, answer_result_fingerprint,
)


STRATEGIES = ("fixed_size", "structure_aware", "prompt_based")


def dataset() -> QADataset:
    records = [
        QARecord.from_dict({
            "id": "q1", "doc_id": "test:a.md", "question": "First question?",
            "answer": "10 mg is not optional.", "evidence_sentences": ["evidence"],
            "evidence_sections": [], "question_type": "conceptual", "difficulty": "easy",
        }),
        QARecord.from_dict({
            "id": "q2", "doc_id": "test:a.md", "question": "Second question?",
            "answer": "C++ v2.0", "evidence_sentences": ["evidence"],
            "evidence_sections": [], "question_type": "code_related", "difficulty": "medium",
        }),
        QARecord.from_dict({
            "id": "q3", "doc_id": "test:a.md", "question": "Third question?",
            "answer": "alpha beta", "evidence_sentences": ["evidence"],
            "evidence_sections": [], "question_type": "conceptual", "difficulty": "hard",
        }),
    ]
    identity = {"schema_version": "evidence_qa_dataset_v1", "records": [item.to_dict() for item in records]}
    return QADataset(records, canonical_fingerprint(identity))


def answer(query_id: str, text: str, data: QADataset, strategy: str) -> AnswerResult:
    prompt = f"prompt-{strategy}-{query_id}"
    result_fingerprint = answer_result_fingerprint(
        prompt_fingerprint=prompt, generation_config_fingerprint="generation-config",
        answer_text=text, status="success", finish_reason="stop", provider="fake", model="fake-v1",
    )
    return AnswerResult(
        query_id=query_id, answer_text=text, status="success",
        context_fingerprint=f"context-{strategy}-{query_id}",
        generation_config_fingerprint="generation-config", prompt_fingerprint=prompt,
        result_fingerprint=result_fingerprint, provider="fake", model="fake-v1",
        input_tokens=InputTokenAccounting(1, 1, 1, 1, 1, 5), output_tokens=2,
        finish_reason="stop", strategy=strategy, context_config_fingerprint="context-config",
        retrieval_config_fingerprint="retrieval-config", protocol_config_fingerprint="protocol-config",
        embedding_config_fingerprint="embedding-config", index_fingerprint=f"index-{strategy}",
        dataset_fingerprint=data.fingerprint,
    )


def write_run(
    path: Path, data: QADataset, strategy: str, *,
    answers: dict[str, str] | None = None, failures: tuple[str, ...] = (),
    request_ids: tuple[str, ...] | None = None,
) -> Path:
    answers = answers if answers is not None else {item.id: item.answer for item in data.records}
    request_ids = request_ids or tuple(item.id for item in data.records)
    records = {item.id: item for item in data.records}
    requests = [
        {"query_id": query_id, "question": records[query_id].question,
         "context_fingerprint": f"context-{strategy}-{query_id}"}
        for query_id in sorted(request_ids)
    ]
    identity = {
        "schema_version": GENERATION_RUN_SCHEMA_VERSION,
        "generation_config_fingerprint": "generation-config", "requests": requests,
    }
    answer_results = [answer(query_id, text, data, strategy) for query_id, text in sorted(answers.items())]
    failure_values = [
        {"query_id": query_id, "context_fingerprint": f"context-{strategy}-{query_id}",
         "generation_config_fingerprint": "generation-config", "error_type": "ProviderError",
         "error": "offline fixture failure"}
        for query_id in sorted(failures)
    ]
    manifest = {
        **identity, "run_fingerprint": canonical_fingerprint(identity),
        "answer_count": len(answer_results), "failure_count": len(failure_values), "complete": True,
    }
    stats = {
        "expected_queries": len(request_ids), "successful_queries": len(answer_results),
        "failed_queries": len(failure_values), "complete": True,
    }
    path.mkdir(parents=True)
    path.joinpath("answers.jsonl").write_text("".join(json.dumps(item.to_dict()) + "\n" for item in answer_results))
    path.joinpath("failures.jsonl").write_text("".join(json.dumps(item) + "\n" for item in failure_values))
    path.joinpath("stats.json").write_text(json.dumps(stats))
    path.joinpath("manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("value,expected", [
    ("  Café\tSIGNAL  ", "café signal"),
    ("10 mg is NOT optional.", "10 mg is not optional."),
    ("C++ v2.0", "c++ v2.0"),
    ("", ""),
])
def test_normalization_v1_is_conservative_and_deterministic(value, expected):
    assert normalize_answer(value) == expected
    assert normalize_answer(value) == normalize_answer(value)


def test_tokenizer_preserves_numbers_units_negation_and_symbols():
    assert tokenize_answer(normalize_answer("10 mg is NOT C++ v2.0")) == [
        "10", "mg", "is", "not", "c", "+", "+", "v2", ".", "0",
    ]


@pytest.mark.parametrize("prediction,reference,expected", [
    ("Exact", "Exact", 1.0), (" exact  ", "EXACT", 1.0),
    ("yes", "no", 0.0), ("", "", 1.0), ("", "value", 0.0),
])
def test_normalized_exact_match_cases(prediction, reference, expected):
    assert normalized_exact_match(prediction, reference) == expected


@pytest.mark.parametrize("prediction,reference,expected", [
    ("alpha beta", "alpha beta", (1.0, 1.0, 1.0)),
    ("alpha gamma", "alpha beta", (0.5, 0.5, 0.5)),
    ("gamma", "alpha beta", (0.0, 0.0, 0.0)),
    ("alpha alpha beta", "alpha beta beta", (2 / 3, 2 / 3, 2 / 3)),
    ("", "alpha", (0.0, 0.0, 0.0)), ("", "", (1.0, 1.0, 1.0)),
])
def test_token_prf_edge_cases(prediction, reference, expected):
    assert token_prf(prediction, reference) == expected


def test_multiple_references_best_per_metric_and_deterministic_tie():
    scores, indexes = evaluate_references(
        "the answer is beta", ("alpha", "beta", "beta"),
        ("normalized_exact_match", "token_f1", "normalized_containment"),
    )
    assert scores["token_f1"] > 0 and scores["normalized_containment"] == 1.0
    assert indexes["token_f1"] == 1 and indexes["normalized_containment"] == 1
    assert indexes["normalized_exact_match"] == 0  # all-zero tie uses lowest index


def test_containment_is_token_based_not_raw_substring():
    assert normalized_containment("The answer is alpha beta.", "alpha beta") == 1.0
    assert normalized_containment("partial", "art") == 0.0


def test_config_fingerprint_binds_all_semantic_metric_settings():
    base = EvaluationConfig()
    assert replace(base, enabled_metrics=("token_f1",)).fingerprint != base.fingerprint
    assert EvaluationConfig(enabled_metrics=tuple(reversed(base.enabled_metrics))).fingerprint == base.fingerprint
    changed_identity = {**base.identity(), "normalization_version": "answer_normalization_v2"}
    assert canonical_fingerprint(changed_identity) != base.fingerprint
    with pytest.raises(ValueError, match="unsupported normalization"):
        replace(base, normalization_version="v2")


def test_valid_complete_alignment_round_trip_and_lineage(tmp_path: Path):
    data = dataset(); path = write_run(tmp_path / "run", data, "fixed_size")
    run = load_committed_generation_run(path, data, strategy="fixed_size")
    result = run_answer_benchmark(data, {"fixed_size": path}, tmp_path / "evaluation")
    assert len(result.per_query) == 3
    assert result.aggregates["fixed_size"].metric_counts["token_f1"] == 3
    restored = PerQueryEvaluationResult.from_dict(result.per_query[0].to_dict())
    aggregate = AggregateEvaluationResult.from_dict(result.aggregates["fixed_size"].to_dict())
    assert restored == result.per_query[0] and aggregate == result.aggregates["fixed_size"]
    assert all(item.dataset_fingerprint == data.fingerprint for item in run.answers)
    assert validate_answer_benchmark_artifacts(tmp_path / "evaluation")["complete"] is True


def test_partial_run_and_invalid_manifest_are_rejected(tmp_path: Path):
    data = dataset(); path = write_run(tmp_path / "run", data, "fixed_size")
    path.joinpath("manifest.json").unlink()
    with pytest.raises(ValueError, match="committed generation manifest"):
        load_committed_generation_run(path, data, strategy="fixed_size")
    path = write_run(tmp_path / "other", data, "fixed_size")
    manifest = json.loads(path.joinpath("manifest.json").read_text())
    manifest["complete"] = False
    path.joinpath("manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="commit marker"):
        load_committed_generation_run(path, data, strategy="fixed_size")


def test_missing_query_is_visible_and_penalized_end_to_end(tmp_path: Path):
    data = dataset()
    path = write_run(
        tmp_path / "run", data, "fixed_size",
        answers={"q1": data.records[0].answer, "q2": data.records[1].answer},
        request_ids=("q1", "q2"),
    )
    result = run_answer_benchmark(data, {"fixed_size": path}, tmp_path / "evaluation")
    missing = [item for item in result.per_query if item.query_id == "q3"][0]
    aggregate = result.aggregates["fixed_size"]
    assert missing.evaluation_status == "missing_generation_result" and missing.metrics["token_f1"] is None
    assert aggregate.missing_queries == 1 and aggregate.metric_counts["token_f1"] == 2
    assert aggregate.success_aware_metric_counts["token_f1"] == 3


def test_generation_failure_is_visible_and_not_fabricated(tmp_path: Path):
    data = dataset()
    path = write_run(
        tmp_path / "run", data, "fixed_size",
        answers={"q1": data.records[0].answer, "q2": data.records[1].answer}, failures=("q3",),
    )
    result = run_answer_benchmark(data, {"fixed_size": path}, tmp_path / "evaluation")
    failed = [item for item in result.per_query if item.query_id == "q3"][0]
    assert failed.generation_status == "failed" and failed.generated_answer is None
    assert result.aggregates["fixed_size"].generation_failures == 1
    assert result.aggregates["fixed_size"].metric_counts["token_f1"] == 2
    assert result.aggregates["fixed_size"].success_aware_metric_means["token_f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("defect,match", [
    ("extra", "unexpected generation query"),
    ("duplicate", "duplicate query IDs"),
    ("dataset", "dataset fingerprint mismatch"),
    ("fingerprint", "artifact failed validation"),
])
def test_strict_alignment_and_result_validation(tmp_path: Path, defect, match):
    data = dataset(); path = write_run(tmp_path / "run", data, "fixed_size")
    if defect == "extra":
        manifest = json.loads(path.joinpath("manifest.json").read_text())
        manifest["requests"].append({"query_id": "extra", "question": "x", "context_fingerprint": "x"})
        path.joinpath("manifest.json").write_text(json.dumps(manifest))
    elif defect == "duplicate":
        manifest = json.loads(path.joinpath("manifest.json").read_text())
        manifest["requests"].append(manifest["requests"][0])
        identity = {key: manifest[key] for key in ("schema_version", "generation_config_fingerprint", "requests")}
        manifest["run_fingerprint"] = canonical_fingerprint(identity)
        path.joinpath("manifest.json").write_text(json.dumps(manifest))
    else:
        rows = path.joinpath("answers.jsonl").read_text().splitlines()
        row = json.loads(rows[0])
        if defect == "dataset":
            row["dataset_fingerprint"] = "wrong"
        else:
            row["answer_text"] = "tampered"
        rows[0] = json.dumps(row)
        path.joinpath("answers.jsonl").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match=match):
        load_committed_generation_run(path, data, strategy="fixed_size")


def test_identical_scoring_for_all_strategies_and_paired_shape(tmp_path: Path):
    data = dataset()
    runs = {
        strategy: write_run(
            tmp_path / strategy, data, strategy,
            answers={item.id: "alpha beta" for item in data.records},
        )
        for strategy in STRATEGIES
    }
    first = run_answer_benchmark(data, runs, tmp_path / "evaluation")
    values = {
        strategy: [item.metrics for item in first.per_query if item.strategy == strategy]
        for strategy in STRATEGIES
    }
    assert values["fixed_size"] == values["structure_aware"] == values["prompt_based"]
    assert len(first.paired) == 3 and set(first.paired[0]["strategies"]) == set(STRATEGIES)
    assert all(item["evidence_diagnostics"] is None for item in first.paired[0]["strategies"].values())
    second = run_answer_benchmark(data, runs, tmp_path / "evaluation")
    assert first.benchmark_fingerprint == second.benchmark_fingerprint
    assert first.manifest == second.manifest
    semantic_files = ("evaluations.jsonl", "summary.json", "paired.jsonl", "manifest.json")
    before = {name: (tmp_path / "evaluation" / name).read_bytes() for name in semantic_files}
    run_answer_benchmark(data, runs, tmp_path / "evaluation")
    assert before == {name: (tmp_path / "evaluation" / name).read_bytes() for name in semantic_files}


def test_fingerprints_change_with_answer_gold_and_config(tmp_path: Path):
    data = dataset()
    first_path = write_run(tmp_path / "first", data, "fixed_size")
    first = run_answer_benchmark(data, {"fixed_size": first_path}, tmp_path / "out1")
    changed_path = write_run(
        tmp_path / "changed", data, "fixed_size",
        answers={**{item.id: item.answer for item in data.records}, "q1": "changed"},
    )
    changed = run_answer_benchmark(data, {"fixed_size": changed_path}, tmp_path / "out2")
    assert first.per_query[0].evaluation_fingerprint != changed.per_query[0].evaluation_fingerprint
    assert first.aggregates["fixed_size"].aggregate_fingerprint != changed.aggregates["fixed_size"].aggregate_fingerprint
    changed_record = replace(data.records[0], answer="different gold")
    changed_data = QADataset([changed_record, *data.records[1:]], "changed-dataset")
    changed_gold_path = write_run(tmp_path / "gold", changed_data, "fixed_size")
    changed_gold = run_answer_benchmark(changed_data, {"fixed_size": changed_gold_path}, tmp_path / "out3")
    assert first.per_query[0].evaluation_fingerprint != changed_gold.per_query[0].evaluation_fingerprint
    metric_config = EvaluationConfig(enabled_metrics=("token_f1",))
    metric = run_answer_benchmark(data, {"fixed_size": first_path}, tmp_path / "out4", config=metric_config)
    assert first.per_query[0].evaluation_fingerprint != metric.per_query[0].evaluation_fingerprint


def test_manifest_is_published_last_conflicts_rejected_and_inputs_read_only(tmp_path: Path, monkeypatch):
    data = dataset(); path = write_run(tmp_path / "run", data, "fixed_size")
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    calls = []
    import rag_chunking.chunking.writer as writer
    original = writer.os.replace
    def recording(source, destination):
        calls.append(Path(destination).name)
        return original(source, destination)
    monkeypatch.setattr(writer.os, "replace", recording)
    output = tmp_path / "evaluation"
    run_answer_benchmark(data, {"fixed_size": path}, output)
    assert calls[-1] == "manifest.json"
    assert before == {item.name: item.read_bytes() for item in path.iterdir()}
    manifest = json.loads(output.joinpath("manifest.json").read_text())
    manifest["benchmark_fingerprint"] = "conflict"
    output.joinpath("manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different identity"):
        run_answer_benchmark(data, {"fixed_size": path}, output)


def test_evaluator_never_invokes_network_provider_or_retrieval(tmp_path: Path, monkeypatch):
    data = dataset(); path = write_run(tmp_path / "run", data, "fixed_size")
    def forbidden(*args, **kwargs):
        raise AssertionError("upstream/network code must not be called")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("rag_chunking.generation.service.GenerationService.generate", forbidden)
    monkeypatch.setattr("rag_chunking.retrieval.service.RetrievalService.retrieve", forbidden)
    result = run_answer_benchmark(data, {"fixed_size": path}, tmp_path / "evaluation")
    assert result.aggregates["fixed_size"].evaluated_queries == 3


def test_evaluation_validator_rejects_incomplete_and_tampered_artifacts(tmp_path: Path):
    data = dataset(); path = write_run(tmp_path / "run", data, "fixed_size")
    output = tmp_path / "evaluation"
    run_answer_benchmark(data, {"fixed_size": path}, output)
    output.joinpath("manifest.json").unlink()
    with pytest.raises(ValueError, match="not committed"):
        validate_answer_benchmark_artifacts(output)
    run_answer_benchmark(data, {"fixed_size": path}, output)
    rows = output.joinpath("evaluations.jsonl").read_text().splitlines()
    row = json.loads(rows[0]); row["generated_answer"] = "tampered"; rows[0] = json.dumps(row)
    output.joinpath("evaluations.jsonl").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="failed validation"):
        validate_answer_benchmark_artifacts(output)


def test_interrupted_evaluation_publication_never_commits(tmp_path: Path, monkeypatch):
    data = dataset(); path = write_run(tmp_path / "run", data, "fixed_size")
    import rag_chunking.chunking.writer as writer
    original = writer.os.replace
    def interrupt_manifest(source, destination):
        if Path(destination).name == "manifest.json":
            raise KeyboardInterrupt()
        return original(source, destination)
    monkeypatch.setattr(writer.os, "replace", interrupt_manifest)
    output = tmp_path / "evaluation"
    with pytest.raises(KeyboardInterrupt):
        run_answer_benchmark(data, {"fixed_size": path}, output)
    assert not output.joinpath("manifest.json").exists()
