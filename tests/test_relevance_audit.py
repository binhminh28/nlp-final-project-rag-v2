import json
from pathlib import Path

import pytest

from rag_chunking.evaluation.audit import build_relevance_audit
from rag_chunking.evaluation.dataset import load_evaluation_dataset


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, values) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "baseline_v1.jsonl"
    write_jsonl(dataset, [{
        "query_id": "q1", "query": "target query", "category": "conceptual",
        "relevant_sources": ["old.md"],
    }])
    loaded = load_evaluation_dataset(dataset, {"old.md", "new.md", "partial.md"})
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    write_json(baseline / "manifest.json", {
        "complete": True, "dataset_fingerprint": loaded.fingerprint,
        "benchmark_fingerprint": "frozen", "strategy_count": 2,
    })
    common = {"query_id": "q1", "query": "target query", "category": "conceptual"}
    write_jsonl(baseline / "per_query.jsonl", [
        {**common, "strategy": "fixed_size", "hits": [
            {"rank": 1, "relative_path": "old.md", "text": "old evidence"},
            {"rank": 2, "relative_path": "new.md", "text": "new evidence"},
        ]},
        {**common, "strategy": "structure_aware", "hits": [
            {"rank": 1, "relative_path": "partial.md", "text": "partial evidence"},
            {"rank": 2, "relative_path": "new.md", "text": "new evidence"},
        ]},
    ])
    decisions = tmp_path / "decisions.json"
    write_json(decisions, {
        "policy_version": "test_v1",
        "default_not_relevant_reason": "reviewed and not relevant",
        "queries_requiring_wording_correction": [],
        "decisions": [
            {"query_id": "q1", "source": "new.md", "classification": "RELEVANT", "reason": "direct"},
            {"query_id": "q1", "source": "partial.md", "classification": "PARTIALLY_RELEVANT", "reason": "incomplete"},
        ],
    })
    return dataset, baseline, decisions


def test_audit_adds_only_relevant_and_is_deterministic(tmp_path: Path) -> None:
    dataset, baseline, decisions = fixture(tmp_path)
    output = tmp_path / "audit"
    args = dict(
        dataset_path=dataset, baseline_directory=baseline, decisions_path=decisions,
        corpus_sources={"old.md", "new.md", "partial.md"}, output_directory=output,
    )
    first = build_relevance_audit(**args)
    manifest = json.loads((output / "manifest.json").read_text())
    revised = load_evaluation_dataset(output / "baseline_v2.jsonl", args["corpus_sources"])
    assert first.labels_added == 1 and first.labels_removed == 0
    assert revised.records[0].relevant_sources == ["new.md", "old.md"]
    assert manifest["candidate_labels_reviewed"] == 3
    deterministic = {path.name: path.read_bytes() for path in output.iterdir()}
    second = build_relevance_audit(**args)
    assert first.audit_fingerprint == second.audit_fingerprint
    assert deterministic == {path.name: path.read_bytes() for path in output.iterdir()}


def test_audit_rejects_decision_outside_frozen_candidates(tmp_path: Path) -> None:
    dataset, baseline, decisions = fixture(tmp_path)
    config = json.loads(decisions.read_text())
    config["decisions"].append({
        "query_id": "q1", "source": "missing.md", "classification": "RELEVANT", "reason": "bad",
    })
    write_json(decisions, config)
    with pytest.raises(ValueError, match="not baseline candidates"):
        build_relevance_audit(
            dataset_path=dataset, baseline_directory=baseline, decisions_path=decisions,
            corpus_sources={"old.md", "new.md", "partial.md", "missing.md"},
            output_directory=tmp_path / "audit",
        )
