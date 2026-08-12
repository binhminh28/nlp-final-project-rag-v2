"""Offline validation and immutable freeze declaration for canonical benchmark v2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from rag_chunking.benchmark import (
    CANONICAL_STRATEGIES,
    project_benchmark_queries,
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


FREEZE_SCHEMA_VERSION = "canonical_benchmark_freeze_v1"
EXPECTED = {
    "dataset_fingerprint": "9866799ea8f87c6a7c118cbaf0d8c298524757bd753db5d640e1a32234206d74",
    "generation_fingerprint": "c4f4768ec9b80361dfd0a1e252f74ff348aa4e4c953bcca02761ba345f38b301",
    "evaluation_fingerprint": "c6867bffbd9775d3ef9b4ce666ae09f1995a6ccb7a7ef14858bbcdb736c1fa55",
    "retrieval_benchmark_fingerprint": "9dab4015ca1ae4c4abda04ccc5809a5e030c793fcc7feff932d04ecfb116a6b7",
    "answer_benchmark_fingerprint": "375983ff4b3c4e84b303d7298c4dd93b44782430bbc1fd6dff41db6f3b60af23",
}
EXPECTED_ANSWER = {
    "fixed_size": (0.2858, 0.7420, 0.3830, 0.0071, 0.0143),
    "structure_aware": (0.2798, 0.7398, 0.3766, 0.0071, 0.0071),
    "prompt_based": (0.2707, 0.7555, 0.3721, 0.0000, 0.0143),
}
EXPECTED_RETRIEVAL = {
    "fixed_size": (0.8071, 0.9643, 0.9643, 0.8732, 0.9429, 0.6905, 0.5714),
    "structure_aware": (0.7786, 0.9786, 1.0000, 0.8692, 0.9857, 0.7476, 0.6357),
    "prompt_based": (0.7571, 0.9786, 0.9857, 0.8508, 0.9786, 0.7320, 0.6143),
}
EXPECTED_DIFFICULTY = {
    "easy": (60, 0.3892, 0.3831, 0.3723),
    "medium": (40, 0.3863, 0.3862, 0.3939),
    "hard": (40, 0.3702, 0.3571, 0.3501),
}
EXPECTED_TYPES = {
    "behavior": (22, 0.3823, 0.3770, 0.3682),
    "cause_effect": (2, 0.2993, 0.2976, 0.3085),
    "comparison": (26, 0.3333, 0.3493, 0.3400),
    "definition": (15, 0.3152, 0.3085, 0.2558),
    "fact": (21, 0.4836, 0.4564, 0.5103),
    "list": (4, 0.6402, 0.6101, 0.4812),
    "mechanism": (7, 0.3882, 0.3765, 0.3547),
    "procedure": (11, 0.3672, 0.3756, 0.3842),
    "security_mechanism": (1, 0.4649, 0.4521, 0.4961),
    "sequence": (3, 0.4080, 0.3608, 0.2929),
    "syntax": (1, 0.3000, 0.3077, 0.3200),
    "synthesis": (2, 0.3460, 0.3038, 0.3202),
    "tradeoff": (10, 0.3758, 0.3839, 0.3888),
    "why": (15, 0.3532, 0.3361, 0.3419),
}
PAIR_ORDER = (
    ("fixed_size", "structure_aware"),
    ("fixed_size", "prompt_based"),
    ("structure_aware", "prompt_based"),
)
EXPECTED_PAIRS = {
    ("fixed_size", "structure_aware"): (75, 3, 62, 0.0064),
    ("fixed_size", "prompt_based"): (81, 3, 56, 0.0108),
    ("structure_aware", "prompt_based"): (60, 3, 77, 0.0045),
}


class FreezeValidationError(ValueError):
    """A blocking canonical freeze validation failure."""


@dataclass(frozen=True)
class FreezePaths:
    repository_root: Path
    benchmark_root: Path
    retrieval_root: Path
    dataset: Path
    report: Path


@dataclass
class FreezeResult:
    status: str
    sections: dict[str, Any]
    artifact_inventory: dict[str, dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreezeValidationError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FreezeValidationError(f"{path}:{number} must contain an object")
        rows.append(value)
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeValidationError(message)


def _close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def validate_id_set(values: Iterable[str], expected: set[str], label: str) -> dict[str, Any]:
    ids = list(values)
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    missing = sorted(expected - set(ids))
    extra = sorted(set(ids) - expected)
    _require(not duplicates, f"{label} contains duplicate IDs: {duplicates}")
    _require(not missing, f"{label} is missing IDs: {missing}")
    _require(not extra, f"{label} contains extra IDs: {extra}")
    return {"count": len(ids), "duplicates": duplicates, "missing": missing, "extra": extra}


def macro_means(rows: Iterable[dict[str, Any]], names: Iterable[str]) -> dict[str, float]:
    values = list(rows)
    _require(bool(values), "cannot aggregate an empty record collection")
    return {name: sum(float(row[name]) for row in values) / len(values) for name in names}


def summarize_pair(left: list[float], right: list[float]) -> dict[str, float | int]:
    _require(len(left) == len(right) and bool(left), "paired inputs must be non-empty and aligned")
    deltas = [a - b for a, b in zip(left, right, strict=True)]
    positives = [value for value in deltas if value > 0]
    negatives = [value for value in deltas if value < 0]
    return {
        "left_wins": len(positives),
        "ties": sum(value == 0 for value in deltas),
        "left_losses": len(negatives),
        "mean_delta": sum(deltas) / len(deltas),
        "sum_positive_deltas": sum(positives),
        "sum_negative_deltas": sum(negatives),
        "mean_positive_win": sum(positives) / len(positives) if positives else 0.0,
        "mean_negative_loss": sum(negatives) / len(negatives) if negatives else 0.0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_paths(paths: FreezePaths) -> list[Path]:
    root = paths.benchmark_root
    files = [
        paths.dataset,
        paths.repository_root / "configs/generation_gpt5mini_v2.json",
        paths.repository_root / "FINAL_BENCHMARK_REPORT.md",
        paths.repository_root / "FINAL_BENCHMARK_HANDOFF.md",
        root / "inputs/manifest.json", root / "inputs/stats.json",
        root / "generation_health.json",
        root / "evaluation/evaluations.jsonl", root / "evaluation/paired.jsonl",
        root / "evaluation/summary.json", root / "evaluation/stats.json",
        root / "evaluation/manifest.json",
        paths.retrieval_root / "per_query.jsonl", paths.retrieval_root / "aggregate.json",
        paths.retrieval_root / "stats.json", paths.retrieval_root / "manifest.json",
    ]
    for strategy in CANONICAL_STRATEGIES:
        files.extend([
            root / f"inputs/{strategy}.generation_inputs.jsonl",
            root / f"generation/{strategy}/answers.jsonl",
            root / f"generation/{strategy}/failures.jsonl",
            root / f"generation/{strategy}/stats.json",
            root / f"generation/{strategy}/manifest.json",
            root / f"generation/{strategy}.provider_diagnostics.jsonl",
        ])
    return files


def artifact_inventory(paths: FreezePaths) -> dict[str, dict[str, Any]]:
    inventory = {}
    for path in sorted(_artifact_paths(paths)):
        _require(path.is_file(), f"canonical artifact is missing: {path}")
        relative = path.relative_to(paths.repository_root).as_posix()
        inventory[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    return inventory


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


def _select_spot_checks(paired: list[dict[str, Any]]) -> list[tuple[str, str]]:
    scored = []
    for row in paired:
        scores = {name: float(value["metrics"]["token_f1"]) for name, value in row["strategies"].items()}
        scored.append((row["query_id"], scores))
    selected: list[tuple[str, str]] = []
    for strategy in CANONICAL_STRATEGIES:
        advantages = sorted(
            ((min(score[strategy] - score[other] for other in CANONICAL_STRATEGIES if other != strategy), query_id)
             for query_id, score in scored),
            key=lambda item: (-item[0], item[1]),
        )
        selected.extend((query_id, f"strongest_{strategy}_advantage") for _, query_id in advantages[:5])
    ties = sorted(
        ((max(score.values()) - min(score.values()), query_id) for query_id, score in scored),
        key=lambda item: (item[0], item[1]),
    )
    selected.extend((query_id, "exact_or_near_tie") for _, query_id in ties[:5])
    return selected
