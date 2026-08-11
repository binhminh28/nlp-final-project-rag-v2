"""Deterministic relevance audit artifacts derived from frozen benchmark hits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.embedding.models import canonical_fingerprint

from .dataset import EVALUATION_DATASET_SCHEMA_VERSION, EvaluationQuery, load_evaluation_dataset


AUDIT_SCHEMA_VERSION = "retrieval_relevance_audit_v1"
CLASSES = frozenset({"RELEVANT", "PARTIALLY_RELEVANT", "NOT_RELEVANT", "AMBIGUOUS"})


@dataclass(frozen=True, slots=True)
class AuditResult:
    output_directory: Path
    dataset_fingerprint: str
    audit_fingerprint: str
    query_count: int
    candidate_count: int
    labels_added: int
    labels_removed: int


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _jsonl(values: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for value in values
    )


def _decision_map(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    decisions: dict[tuple[str, str], dict[str, str]] = {}
    for item in config.get("decisions", []):
        classification = item.get("classification")
        if classification not in CLASSES:
            raise ValueError(f"unknown audit classification {classification!r}")
        key = (item.get("query_id"), item.get("source"))
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError("audit decisions require query_id and source")
        if key in decisions:
            raise ValueError(f"duplicate audit decision {key}")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"audit decision {key} requires a reason")
        decisions[key] = {"classification": classification, "reason": reason}
    return decisions


def build_relevance_audit(
    *, dataset_path: Path, baseline_directory: Path, decisions_path: Path,
    corpus_sources: set[str], output_directory: Path, revised_dataset_name: str = "baseline_v2",
) -> AuditResult:
    dataset = load_evaluation_dataset(dataset_path, corpus_sources)
    baseline_manifest = _read_json(baseline_directory / "manifest.json")
    if not baseline_manifest.get("complete"):
        raise ValueError("baseline manifest is not complete")
    if baseline_manifest.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("baseline and audit dataset fingerprints differ")
    rows = _read_jsonl(baseline_directory / "per_query.jsonl")
    if len(rows) != len(dataset.records) * baseline_manifest.get("strategy_count", 0):
        raise ValueError("baseline per-query record count is incomplete")

    config = _read_json(decisions_path)
    decisions = _decision_map(config)
    records_by_id = {record.query_id: record for record in dataset.records}
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        query_id = row["query_id"]
        if query_id not in records_by_id:
            raise ValueError(f"unknown baseline query {query_id!r}")
        for hit in row["hits"]:
            key = (query_id, hit["relative_path"])
            candidate = candidates.setdefault(key, {"ranks": {}, "evidence": []})
            strategy = row["strategy"]
            candidate["ranks"][strategy] = min(hit["rank"], candidate["ranks"].get(strategy, 10**9))
            excerpt = " ".join(hit["text"].split())[:600]
            if excerpt and excerpt not in candidate["evidence"]:
                candidate["evidence"].append(excerpt)

    unknown_decisions = sorted(set(decisions) - set(candidates))
    if unknown_decisions:
        raise ValueError(f"audit decisions are not baseline candidates: {unknown_decisions}")

    audit_rows: list[dict[str, Any]] = []
    additions: dict[str, list[str]] = {}
    removals: dict[str, list[str]] = {}
    for key in sorted(candidates):
        query_id, source = key
        record = records_by_id[query_id]
        was_relevant = source in record.relevant_sources
        explicit = decisions.get(key)
        if explicit is not None:
            classification = explicit["classification"]
            reason = explicit["reason"]
        elif was_relevant:
            classification = "RELEVANT"
            reason = "Existing baseline_v1 label confirmed by content review."
        else:
            classification = "NOT_RELEVANT"
            reason = config["default_not_relevant_reason"]
        binary_relevant = classification == "RELEVANT"
        if binary_relevant and not was_relevant:
            additions.setdefault(query_id, []).append(source)
        if was_relevant and not binary_relevant:
            removals.setdefault(query_id, []).append(source)
        candidate = candidates[key]
        audit_rows.append({
            "query_id": query_id, "query": record.query, "category": record.category,
            "source": source, "classification": classification,
            "binary_relevant": binary_relevant, "was_relevant_v1": was_relevant,
            "ranks": dict(sorted(candidate["ranks"].items())), "reason": reason,
            "evidence_excerpts": candidate["evidence"],
        })

    revised_records: list[EvaluationQuery] = []
    changes: list[dict[str, Any]] = []
    for record in dataset.records:
        old = set(record.relevant_sources)
        new = (old | set(additions.get(record.query_id, []))) - set(removals.get(record.query_id, []))
        revised = EvaluationQuery(
            query_id=record.query_id, query=record.query, category=record.category,
            relevant_sources=sorted(new), notes=record.notes, difficulty=record.difficulty,
        )
        revised_records.append(revised)
        if old != new:
            changes.append({
                "query_id": record.query_id, "old_relevant_sources": sorted(old),
                "new_relevant_sources": sorted(new), "added": sorted(new - old),
                "removed": sorted(old - new),
            })

    dataset_text = _jsonl([record.to_dict() for record in revised_records])
    revised_dataset_fingerprint = canonical_fingerprint({
        "schema_version": EVALUATION_DATASET_SCHEMA_VERSION,
        "ground_truth_level": "relative_path",
        "records": [record.to_dict() for record in sorted(revised_records, key=lambda item: item.query_id)],
    })
    audit_identity = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source_dataset_fingerprint": dataset.fingerprint,
        "source_benchmark_fingerprint": baseline_manifest["benchmark_fingerprint"],
        "decision_policy": config["policy_version"],
        "decisions": audit_rows,
    }
    audit_fingerprint = canonical_fingerprint(audit_identity)
    class_counts = {name: sum(row["classification"] == name for row in audit_rows) for name in sorted(CLASSES)}
    ambiguous_queries = sorted({row["query_id"] for row in audit_rows if row["classification"] == "AMBIGUOUS"})
    wording_corrections = sorted(config.get("queries_requiring_wording_correction", []))
    if set(wording_corrections) - set(records_by_id):
        raise ValueError("wording-correction list contains unknown query IDs")
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION, "complete": True,
        "audit_fingerprint": audit_fingerprint,
        "source_dataset": dataset_path.as_posix(), "source_dataset_fingerprint": dataset.fingerprint,
        "source_benchmark_fingerprint": baseline_manifest["benchmark_fingerprint"],
        "revised_dataset": revised_dataset_name, "revised_dataset_fingerprint": revised_dataset_fingerprint,
        "queries_reviewed": len(dataset.records), "candidate_labels_reviewed": len(audit_rows),
        "labels_added": sum(map(len, additions.values())), "labels_removed": sum(map(len, removals.values())),
        "queries_changed": len(changes), "ambiguous_queries": ambiguous_queries,
        "queries_requiring_wording_correction": wording_corrections,
        "classification_counts": class_counts, "policy_version": config["policy_version"],
    }
    # Publish all files transactionally; manifest remains the final completion marker.
    write_artifact_set(output_directory, {
        f"{revised_dataset_name}.jsonl": dataset_text,
        "audit.jsonl": _jsonl(audit_rows),
        "label_changes.json": serialize_json({"changes": changes}),
        "audit_summary.json": serialize_json({key: value for key, value in manifest.items() if key != "complete"}),
        "manifest.json": serialize_json(manifest),
    })
    return AuditResult(
        output_directory, revised_dataset_fingerprint, audit_fingerprint, len(dataset.records),
        len(audit_rows), manifest["labels_added"], manifest["labels_removed"],
    )
