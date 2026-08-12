"""Canonical-v2-specific gates built on the reusable freeze primitives."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from rag_chunking.benchmark import (
    CANONICAL_STRATEGIES, project_benchmark_queries,
    validate_generation_requests_against_preparation,
    validate_prepared_benchmark_inputs,
)
from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.evaluation.answer_artifacts import load_committed_generation_run
from rag_chunking.evaluation.answer_metrics import evaluate_references
from rag_chunking.evaluation.answer_models import EvaluationConfig, PerQueryEvaluationResult
from rag_chunking.evaluation.answer_runner import validate_answer_benchmark_artifacts
from rag_chunking.evaluation.metrics import aggregate, aggregate_evidence
from rag_chunking.evaluation.qa_dataset import load_team_qa_dataset

from rag_chunking.benchmark_freeze import (
    EXPECTED, EXPECTED_ANSWER, EXPECTED_DIFFICULTY, EXPECTED_PAIRS,
    EXPECTED_RETRIEVAL, EXPECTED_TYPES, FREEZE_SCHEMA_VERSION, PAIR_ORDER,
    FreezePaths, FreezeResult, FreezeValidationError, _close, _read_json,
    _read_jsonl, _require, _select_spot_checks, artifact_inventory,
    macro_means, summarize_pair, validate_id_set,
)


def _validate_retrieval_manifest(manifest: dict[str, Any]) -> None:
    identity_keys = (
        "schema_version", "qa_schema_version", "dataset_fingerprint", "corpus",
        "corpus_fingerprint", "chunk_config_fingerprints", "chunk_artifact_fingerprints",
        "embedding_config_fingerprint", "embedding_artifact_fingerprints", "index_fingerprints",
        "base_retrieval_config_fingerprint", "protocols", "metrics_versions",
        "historical_ground_truth_level", "evidence_ground_truth_level", "candidate_depth",
        "tie_breaking_rule",
    )
    identity = {key: manifest[key] for key in identity_keys}
    _require(manifest.get("complete") is True, "retrieval manifest is not complete")
    _require(canonical_fingerprint(identity) == manifest.get("benchmark_fingerprint"), "retrieval benchmark fingerprint does not match semantic manifest identity")
    _require(manifest["benchmark_fingerprint"] == EXPECTED["retrieval_benchmark_fingerprint"], "unexpected retrieval benchmark fingerprint")
    _require(manifest["dataset_fingerprint"] == EXPECTED["dataset_fingerprint"], "retrieval dataset fingerprint mismatch")
    _require(manifest.get("query_count") == 140 and manifest.get("strategy_count") == 3, "retrieval manifest coverage mismatch")


def validate_freeze(paths: FreezePaths) -> FreezeResult:
    dataset = load_team_qa_dataset(paths.dataset)
    _require(dataset.fingerprint == EXPECTED["dataset_fingerprint"], "canonical dataset fingerprint mismatch")
    _require(len(dataset.records) == 140, "canonical dataset must contain exactly 140 records")
    dataset_by_id = {row.id: row for row in dataset.records}
    _require(len(dataset_by_id) == 140, "canonical dataset contains duplicate IDs")
    ids = set(dataset_by_id)

    generation_dirs = {name: paths.benchmark_root / "generation" / name for name in CANONICAL_STRATEGIES}
    prepared = validate_prepared_benchmark_inputs(
        paths.benchmark_root / "inputs", dataset_fingerprint=dataset.fingerprint,
        expected_queries=project_benchmark_queries(dataset.records),
    )
    validate_generation_requests_against_preparation(prepared, generation_dirs)
    runs = {name: load_committed_generation_run(path, dataset, strategy=name) for name, path in generation_dirs.items()}

    completeness: dict[str, Any] = {}
    generation_by_strategy = {}
    for strategy, run in runs.items():
        requested = validate_id_set(run.expected_query_ids, ids, f"{strategy} generation requests")
        answered = validate_id_set((item.query_id for item in run.answers), ids, f"{strategy} answers")
        _require(len(run.failures) == 0, f"{strategy} has permanent generation failures")
        _require(run.generation_config_fingerprint == EXPECTED["generation_fingerprint"], f"{strategy} generation fingerprint mismatch")
        _require(all(item.finish_reason == "stop" for item in run.answers), f"{strategy} contains non-stop generation results")
        _require(all(item.answer_text.strip() for item in run.answers), f"{strategy} contains empty answers")
        completeness[strategy] = {
            "dataset_questions": 140, "requested": requested["count"], "answers": answered["count"],
            "failures": 0, "duplicates": 0, "missing": 0, "invalid": 0,
        }
        generation_by_strategy[strategy] = {item.query_id: item for item in run.answers}

    evaluation_manifest = validate_answer_benchmark_artifacts(paths.benchmark_root / "evaluation")
    _require(evaluation_manifest["dataset_fingerprint"] == EXPECTED["dataset_fingerprint"], "answer evaluation dataset fingerprint mismatch")
    _require(evaluation_manifest["evaluation_config_fingerprint"] == EXPECTED["evaluation_fingerprint"], "answer evaluation config fingerprint mismatch")
    _require(evaluation_manifest["benchmark_fingerprint"] == EXPECTED["answer_benchmark_fingerprint"], "answer benchmark fingerprint mismatch")
    _require(evaluation_manifest["preparation_fingerprint"] == prepared.preparation_fingerprint, "answer evaluation preparation lineage mismatch")
    _require(evaluation_manifest["source_corpus_fingerprint"] == prepared.manifest["corpus_fingerprint"], "answer evaluation corpus lineage mismatch")
    for strategy, run in runs.items():
        _require(evaluation_manifest["generation_run_identities"][strategy] == run.committed_identity, f"{strategy} evaluation generation lineage mismatch")

    evaluations = [PerQueryEvaluationResult.from_dict(row) for row in _read_jsonl(paths.benchmark_root / "evaluation/evaluations.jsonl")]
    _require(len(evaluations) == 420, "answer evaluation must contain exactly 420 records")
    eval_by_strategy: dict[str, dict[str, PerQueryEvaluationResult]] = {}
    for strategy in CANONICAL_STRATEGIES:
        rows = [item for item in evaluations if item.strategy == strategy]
        validate_id_set((item.query_id for item in rows), ids, f"{strategy} evaluations")
        eval_by_strategy[strategy] = {item.query_id: item for item in rows}
        completeness[strategy]["evaluation_rows"] = len(rows)

    paired = _read_jsonl(paths.benchmark_root / "evaluation/paired.jsonl")
    validate_id_set((row["query_id"] for row in paired), ids, "paired evaluations")
    for row in paired:
        canonical = dataset_by_id[row["query_id"]]
        _require(row["question"] == canonical.question, f"paired question mismatch for {canonical.id}")
        _require(row["gold_answers"] == [canonical.answer], f"paired gold mismatch for {canonical.id}")
        _require(row["difficulty"] == canonical.difficulty, f"paired difficulty mismatch for {canonical.id}")
        _require(row["question_type"] == row["category"] == canonical.question_type, f"paired question type mismatch for {canonical.id}")
        _require(set(row["strategies"]) == set(CANONICAL_STRATEGIES), f"paired strategy mismatch for {canonical.id}")
        for strategy in CANONICAL_STRATEGIES:
            item = eval_by_strategy[strategy][canonical.id]
            answer = generation_by_strategy[strategy][canonical.id]
            _require(item.question == canonical.question and item.gold_answers == (canonical.answer,), f"evaluation dataset binding mismatch for {strategy}/{canonical.id}")
            _require(item.generated_answer == answer.answer_text, f"evaluation answer binding mismatch for {strategy}/{canonical.id}")
            _require(item.generation_result_fingerprint == answer.result_fingerprint, f"evaluation result lineage mismatch for {strategy}/{canonical.id}")

    summary = _read_json(paths.benchmark_root / "evaluation/summary.json")["strategies"]
    metric_order = ("token_precision", "token_recall", "token_f1", "normalized_exact_match", "normalized_containment")
    answer_metrics = {}
    for strategy in CANONICAL_STRATEGIES:
        recomputed_rows = []
        for query_id in sorted(ids):
            item = eval_by_strategy[strategy][query_id]
            scores, _ = evaluate_references(item.generated_answer or "", item.gold_answers, EvaluationConfig().enabled_metrics)
            recomputed = {name: float(scores[name]) for name in metric_order}
            for name in metric_order:
                _require(_close(recomputed[name], float(item.metrics[name])), f"stored {name} mismatch for {strategy}/{query_id}")
            recomputed_rows.append(recomputed)
        means = macro_means(recomputed_rows, metric_order)
        for name in metric_order:
            _require(_close(means[name], float(summary[strategy]["metric_means"][name])), f"stored answer aggregate mismatch for {strategy}/{name}")
        published = tuple(round(means[name], 4) for name in metric_order)
        _require(published == EXPECTED_ANSWER[strategy], f"published answer metrics mismatch for {strategy}: {published}")
        answer_metrics[strategy] = means

    retrieval_manifest = _read_json(paths.retrieval_root / "manifest.json")
    _validate_retrieval_manifest(retrieval_manifest)
    _require(prepared.manifest["dataset_fingerprint"] == retrieval_manifest["dataset_fingerprint"], "prepared/retrieval dataset lineage mismatch")
    _require(prepared.manifest["corpus_fingerprint"] == retrieval_manifest["corpus_fingerprint"], "prepared/retrieval corpus lineage mismatch")
    _require(prepared.manifest["retrieval_config_fingerprint"] == retrieval_manifest["base_retrieval_config_fingerprint"], "prepared/retrieval config lineage mismatch")
    _require(prepared.manifest["embedding_config_fingerprint"] == retrieval_manifest["embedding_config_fingerprint"], "prepared/retrieval embedding lineage mismatch")
    _require(prepared.manifest["index_fingerprints"] == retrieval_manifest["index_fingerprints"], "prepared/retrieval index lineage mismatch")
    protocol = prepared.manifest["protocol_configuration"]
    _require(protocol == next(item for item in retrieval_manifest["protocols"] if item["mode"] == "same_token_budget"), "prepared/retrieval protocol configuration mismatch")
    _require(protocol["candidate_k"] == 50 and protocol["token_budget"] == 2048, "canonical retrieval budget mismatch")

    all_retrieval = _read_jsonl(paths.retrieval_root / "per_query.jsonl")
    retrieval_rows = [row for row in all_retrieval if row["protocol"] == "same_token_budget"]
    retrieval_by_strategy: dict[str, dict[str, dict[str, Any]]] = {}
    retrieval_aggregate = _read_json(paths.retrieval_root / "aggregate.json")["same_token_budget"]
    retrieval_metrics = {}
    for strategy in CANONICAL_STRATEGIES:
        rows = [row for row in retrieval_rows if row["strategy"] == strategy]
        validate_id_set((row["query_id"] for row in rows), ids, f"{strategy} canonical retrieval")
        retrieval_by_strategy[strategy] = {row["query_id"]: row for row in rows}
        for row in rows:
            canonical = dataset_by_id[row["query_id"]]
            _require(row["question"] == canonical.question and row["doc_id"] == canonical.doc_id, f"retrieval question/doc binding mismatch for {strategy}/{canonical.id}")
            _require(row["difficulty"] == canonical.difficulty and row["question_type"] == canonical.question_type, f"retrieval stratum binding mismatch for {strategy}/{canonical.id}")
            _require(row["candidate_k"] == 50 and row["requested_token_budget"] == 2048, f"retrieval budget mismatch for {strategy}/{canonical.id}")
            _require(row["index_fingerprint"] == retrieval_manifest["index_fingerprints"][strategy], f"retrieval index mismatch for {strategy}/{canonical.id}")
            _require(all(hit["strategy"] == strategy for hit in row["hits"]), f"cross-strategy retrieval hit for {strategy}/{canonical.id}")
        retrieval = aggregate(rows)
        evidence = aggregate_evidence(rows)
        for section_name, calculated in (("retrieval", retrieval), ("evidence", evidence)):
            for name, value in calculated.items():
                _require(_close(float(value), float(retrieval_aggregate[strategy][section_name][name])), f"retrieval aggregate mismatch for {strategy}/{section_name}/{name}")
        published = tuple(round(value, 4) for value in (
            retrieval["hit_at_1"], retrieval["hit_at_5"], retrieval["hit_at_10"], retrieval["mrr"],
            retrieval["recall_at_10"], evidence["evidence_coverage"], evidence["all_evidence_retrieved_rate"],
        ))
        _require(published == EXPECTED_RETRIEVAL[strategy], f"published retrieval metrics mismatch for {strategy}: {published}")
        retrieval_metrics[strategy] = {"retrieval": retrieval, "evidence": evidence}
        completeness[strategy]["retrieval_rows"] = len(rows)

    difficulty_counts = Counter(record.difficulty for record in dataset.records)
    type_counts = Counter(record.question_type for record in dataset.records)
    stratified: dict[str, Any] = {"difficulty": {}, "question_type": {}}
    for group, expected in EXPECTED_DIFFICULTY.items():
        _require(difficulty_counts[group] == expected[0], f"difficulty count mismatch for {group}")
        group_ids = sorted(item.id for item in dataset.records if item.difficulty == group)
        values = [sum(float(eval_by_strategy[strategy][query_id].metrics["token_f1"]) for query_id in group_ids) / len(group_ids) for strategy in CANONICAL_STRATEGIES]
        _require(tuple(round(value, 4) for value in values) == expected[1:], f"difficulty metric mismatch for {group}")
        stratified["difficulty"][group] = {"n": len(group_ids), **dict(zip(CANONICAL_STRATEGIES, values, strict=True))}
    _require(dict(sorted(type_counts.items())) == {key: value[0] for key, value in sorted(EXPECTED_TYPES.items())}, "question-type membership/count mismatch")
    for group, expected in EXPECTED_TYPES.items():
        group_ids = sorted(item.id for item in dataset.records if item.question_type == group)
        values = [sum(float(eval_by_strategy[strategy][query_id].metrics["token_f1"]) for query_id in group_ids) / len(group_ids) for strategy in CANONICAL_STRATEGIES]
        _require(tuple(round(value, 4) for value in values) == expected[1:], f"question-type metric mismatch for {group}")
        stratified["question_type"][group] = {"n": len(group_ids), **dict(zip(CANONICAL_STRATEGIES, values, strict=True))}

    pair_results = {}
    ordered_ids = sorted(ids)
    for pair in PAIR_ORDER:
        left, right = pair
        result = summarize_pair(
            [float(eval_by_strategy[left][query_id].metrics["token_f1"]) for query_id in ordered_ids],
            [float(eval_by_strategy[right][query_id].metrics["token_f1"]) for query_id in ordered_ids],
        )
        expected = EXPECTED_PAIRS[pair]
        _require((result["left_wins"], result["ties"], result["left_losses"]) == expected[:3], f"paired counts mismatch for {left} vs {right}")
        _require(round(float(result["mean_delta"]), 4) == expected[3], f"paired mean mismatch for {left} vs {right}")
        pair_results[f"{left}_vs_{right}"] = result

    health = _read_json(paths.benchmark_root / "generation_health.json")
    _require(health["generation_config_fingerprint"] == EXPECTED["generation_fingerprint"], "generation health fingerprint mismatch")
    combined = health["combined"]
    for name, value in {"requested": 420, "completed": 420, "cache_hits": 4, "provider_attempts": 416, "stop_answers": 420, "length_answers": 0, "empty_or_null": 0, "integrity_failures": 0, "permanent_failures": 0, "retries": 0}.items():
        _require(combined.get(name) == value, f"generation health {name} mismatch")
    for strategy, expected_calls in (("fixed_size", 136), ("structure_aware", 140), ("prompt_based", 140)):
        diagnostics = _read_jsonl(paths.benchmark_root / f"generation/{strategy}.provider_diagnostics.jsonl")
        _require(len(diagnostics) == expected_calls, f"provider diagnostic count mismatch for {strategy}")
        answer_prompts = {item.prompt_fingerprint for item in runs[strategy].answers}
        diagnostic_prompts = {item["prompt_fingerprint"] for item in diagnostics}
        _require(diagnostic_prompts <= answer_prompts, f"stale provider diagnostic for {strategy}")
        _require(len(answer_prompts - diagnostic_prompts) == (4 if strategy == "fixed_size" else 0), f"cache/provider reconciliation mismatch for {strategy}")

    spots = []
    paired_by_id = {row["query_id"]: row for row in paired}
    for query_id, reason in _select_spot_checks(paired):
        canonical = dataset_by_id[query_id]
        retrieval = {}
        answers = {}
        scores = {}
        for strategy in CANONICAL_STRATEGIES:
            answer = generation_by_strategy[strategy][query_id]
            row = retrieval_by_strategy[strategy][query_id]
            answers[strategy] = answer.answer_text[:280].replace("\n", " ")
            scores[strategy] = float(eval_by_strategy[strategy][query_id].metrics["token_f1"])
            retrieval[strategy] = {
                "first_chunk_id": row["hits"][0]["chunk_id"] if row["hits"] else None,
                "first_chunk_snippet": row["hits"][0]["text"][:180].replace("\n", " ") if row["hits"] else "",
                "evidence_coverage": row["evidence_coverage"],
            }
        spots.append({
            "selection": reason, "question_id": query_id, "question": canonical.question,
            "gold_answer": canonical.answer[:280].replace("\n", " "), "difficulty": canonical.difficulty,
            "question_type": canonical.question_type, "answers": answers, "token_f1": scores,
            "retrieval": retrieval, "paired_fingerprint": paired_by_id[query_id]["paired_fingerprint"],
        })

    inventory = artifact_inventory(paths)
    sections = {
        "canonical_identity": dict(EXPECTED), "completeness": completeness,
        "alignment": {"question_count": 140, "mismatches": 0},
        "generation_integrity": {**combined, "cache_identity": "validated through prompt/config/result fingerprints"},
        "answer_metrics": answer_metrics, "retrieval_metrics": retrieval_metrics,
        "stratified": stratified, "paired_comparisons": pair_results,
        "metric_semantics": {
            "normalization": "Unicode NFKC, casefold, whitespace split/rejoin; punctuation retained; no article removal, stemming, or lemmatization",
            "tokenization": "Unicode regex word runs or individual non-word/non-whitespace symbols",
            "overlap": "Counter/multiset intersection; precision=overlap/prediction tokens; recall=overlap/gold tokens; harmonic F1",
            "aggregation": "per-question scores followed by unweighted macro mean; raw floats retained",
            "empty_cases": "both token lists => PRF 1; exactly one empty => PRF 0",
            "exact_match": "equality after NFKC/casefold/whitespace normalization",
            "containment": "gold token sequence occurs contiguously in generated-answer token sequence",
            "ties": "exact raw-float equality (delta == 0); no rounded tie threshold",
            "rounding": "round to four decimals for published display checks only",
        },
        "spot_checks": spots,
        "excluded_nearby_artifacts": [
            "generation/fixed_size/manifest.inconsistent-overlap-audit.json (diagnostic, not referenced by commit manifest)",
            "retrieval sibling d50392b7.../failure.json (failed noncanonical run)",
            "same_top_k rows in canonical retrieval file (committed secondary protocol, excluded from same_token_budget answer comparison)",
        ],
        "freeze_policy": "Downstream work may read these artifacts and write derived outputs only under canonical_v2/statistical_analysis; canonical inputs, retrieval, answers, evaluation, and manifests are immutable.",
    }
    return FreezeResult("PASS", sections, inventory)


def build_freeze_manifest(result: FreezeResult, *, created_at: str | None = None) -> dict[str, Any]:
    _require(result.status == "PASS", "cannot build a freeze manifest from a failed validation")
    created_at = created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = {
        "freeze_schema_version": FREEZE_SCHEMA_VERSION, "created_at": created_at, **EXPECTED,
        "strategies": list(CANONICAL_STRATEGIES), "question_count": 140, "answer_count": 420,
        "canonical_artifacts": result.artifact_inventory, "validation_result": "PASS",
        "freeze_declaration": "CANONICAL PRODUCTION BENCHMARK V2: FROZEN FOR STATISTICAL ANALYSIS",
    }
    manifest["freeze_fingerprint"] = canonical_fingerprint({key: value for key, value in manifest.items() if key not in {"created_at", "freeze_fingerprint"}})
    return manifest


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
