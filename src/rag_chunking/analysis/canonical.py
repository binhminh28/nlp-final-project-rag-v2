"""Canonical v2 statistical analysis orchestration over frozen artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import scipy
from scipy import stats

from rag_chunking.benchmark import CANONICAL_STRATEGIES
from rag_chunking.benchmark_freeze import EXPECTED
from rag_chunking.embedding.models import canonical_fingerprint

from .statistics import (
    bootstrap_joint_means, cohens_dz, describe, holm_adjust,
    paired_delta_summary, permutation_mean_test, rank_biserial,
    shape_diagnostics, wilcoxon_signed_rank,
)


ANALYSIS_SCHEMA_VERSION = "canonical_statistical_analysis_v1"
PAIR_SPECS = (
    ("fixed_size", "structure_aware", "fixed_minus_structure"),
    ("fixed_size", "prompt_based", "fixed_minus_prompt"),
    ("structure_aware", "prompt_based", "structure_minus_prompt"),
)
METRIC_FIELDS = {
    "token_f1": "token_f1", "precision": "token_precision", "recall": "token_recall",
    "exact": "normalized_exact_match", "containment": "normalized_containment",
}


@dataclass(frozen=True)
class AnalysisConfig:
    confidence_level: float = 0.95
    alpha: float = 0.05
    bootstrap_resamples: int = 50_000
    bootstrap_seed: int = 2026
    permutation_resamples: int = 100_000
    permutation_seed: int = 2026
    sensitivity_seeds: tuple[int, ...] = (42, 2026, 314159)
    sensitivity_resamples: tuple[int, ...] = (10_000, 50_000)
    primary_metric: str = "token_f1"
    bootstrap_method: str = "question_level_paired_percentile_v1"
    primary_test: str = "paired_monte_carlo_sign_flip_mean_v1"
    robustness_test: str = "scipy_wilcoxon_asymptotic_two_sided_wilcox_zeros_v1"
    multiple_testing_correction: str = "holm_bonferroni_within_each_three_pair_family_v1"
    effect_sizes: tuple[str, ...] = ("absolute_mean_delta", "paired_cohens_dz", "matched_rank_biserial")
    schema_version: str = ANALYSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1 or not 0 < self.alpha < 1:
            raise ValueError("confidence level and alpha must be between zero and one")
        for name in ("bootstrap_resamples", "permutation_resamples"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.bootstrap_seed) is not int or type(self.permutation_seed) is not int:
            raise ValueError("analysis seeds must be integers")
        if self.primary_metric != "token_f1" or self.schema_version != ANALYSIS_SCHEMA_VERSION:
            raise ValueError("unsupported canonical analysis configuration")

    def identity(self, freeze_fingerprint: str) -> dict[str, Any]:
        value = asdict(self)
        value["sensitivity_seeds"] = list(self.sensitivity_seeds)
        value["sensitivity_resamples"] = list(self.sensitivity_resamples)
        value["effect_sizes"] = list(self.effect_sizes)
        return {"source_freeze_fingerprint": freeze_fingerprint, **value}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} must contain an object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_inputs(repository_root: Path, benchmark_root: Path) -> dict[str, Any]:
    manifest_path = benchmark_root / "freeze_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("validation_result") != "PASS" or manifest.get("freeze_declaration") != "CANONICAL PRODUCTION BENCHMARK V2: FROZEN FOR STATISTICAL ANALYSIS":
        raise ValueError("canonical benchmark is not declared frozen for statistical analysis")
    for name, expected in EXPECTED.items():
        if manifest.get(name) != expected:
            raise ValueError(f"freeze manifest {name} mismatch")
    if manifest.get("question_count") != 140 or manifest.get("answer_count") != 420:
        raise ValueError("freeze manifest sample-size mismatch")
    artifacts = manifest.get("canonical_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("freeze manifest has no canonical artifact inventory")
    for relative, identity in sorted(artifacts.items()):
        path = repository_root / relative
        if not path.is_file():
            raise ValueError(f"frozen artifact is missing: {relative}")
        if path.stat().st_size != identity.get("size_bytes") or _sha256(path) != identity.get("sha256"):
            raise ValueError(f"frozen artifact hash mismatch: {relative}")
    semantic = {key: value for key, value in manifest.items() if key not in {"created_at", "freeze_fingerprint"}}
    if canonical_fingerprint(semantic) != manifest.get("freeze_fingerprint"):
        raise ValueError("freeze fingerprint does not match declaration contents")
    return {
        "manifest": manifest, "manifest_sha256": _sha256(manifest_path),
        "verified_artifact_count": len(artifacts),
    }


def build_paired_table(benchmark_root: Path, analysis_fingerprint: str, freeze_fingerprint: str) -> list[dict[str, Any]]:
    paired = _read_jsonl(benchmark_root / "evaluation/paired.jsonl")
    if len(paired) != 140 or len({row.get("query_id") for row in paired}) != 140:
        raise ValueError("analysis requires exactly 140 unique paired questions")
    rows = []
    for item in sorted(paired, key=lambda value: value["query_id"]):
        strategies = item.get("strategies", {})
        if set(strategies) != set(CANONICAL_STRATEGIES):
            raise ValueError(f"paired strategy coverage mismatch for {item.get('query_id')}")
        row: dict[str, Any] = {
            "statistical_analysis_fingerprint": analysis_fingerprint,
            "source_freeze_fingerprint": freeze_fingerprint,
            "question_id": item["query_id"], "difficulty": item["difficulty"],
            "question_type": item["question_type"],
        }
        for strategy in CANONICAL_STRATEGIES:
            metrics = strategies[strategy].get("metrics", {})
            for short, field in METRIC_FIELDS.items():
                value = metrics.get(field)
                if value is None or not isinstance(value, (int, float)):
                    raise ValueError(f"missing {field} for {strategy}/{item['query_id']}")
                row[f"{strategy}_{short}"] = float(value)
        row.update({
            "fixed_minus_structure": row["fixed_size_token_f1"] - row["structure_aware_token_f1"],
            "fixed_minus_prompt": row["fixed_size_token_f1"] - row["prompt_based_token_f1"],
            "structure_minus_prompt": row["structure_aware_token_f1"] - row["prompt_based_token_f1"],
        })
        rows.append(row)
    return rows


def _matrix(rows: list[dict[str, Any]], metric: str = "token_f1") -> np.ndarray:
    return np.asarray([[row[f"{strategy}_{metric}"] for row in rows] for strategy in CANONICAL_STRATEGIES], dtype=np.float64)


def _bootstrap_public(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "resamples": result["resamples"], "seed": result["seed"],
        "confidence_level": result["confidence_level"],
        "observed_means": [float(value) for value in result["observed_means"]],
        "bootstrap_means": [float(value) for value in result["bootstrap_means"]],
        "bootstrap_standard_errors": [float(value) for value in result["bootstrap_standard_errors"]],
        "ci_lower": [float(value) for value in result["ci_lower"]],
        "ci_upper": [float(value) for value in result["ci_upper"]],
    }


def _winner_frequencies(samples: np.ndarray) -> dict[str, float | int]:
    maxima = samples.max(axis=1)
    winners = samples == maxima[:, None]
    ties = winners.sum(axis=1) > 1
    denominator = samples.shape[0]
    result: dict[str, float | int] = {"resamples": denominator, "tie_frequency": float(ties.mean())}
    for index, strategy in enumerate(CANONICAL_STRATEGIES):
        result[f"{strategy}_winner_frequency"] = float(np.mean(winners[:, index] & ~ties))
    return result


def _pair_bootstrap(samples: np.ndarray, left_index: int, right_index: int, confidence: float) -> dict[str, float]:
    deltas = samples[:, left_index] - samples[:, right_index]
    tail = (1.0 - confidence) * 50.0
    lower, upper = np.percentile(deltas, [tail, 100.0 - tail], method="linear")
    return {
        "bootstrap_mean_delta": float(np.mean(deltas)),
        "bootstrap_standard_error": float(np.std(deltas, ddof=1)),
        "ci_lower": float(lower), "ci_upper": float(upper),
        "bootstrap_probability_gt_zero": float(np.mean(deltas > 0)),
        "bootstrap_probability_eq_zero": float(np.mean(deltas == 0)),
    }


def _difficulty_analysis(rows: list[dict[str, Any]], config: AnalysisConfig) -> dict[str, Any]:
    result = {}
    for offset, difficulty in enumerate(("easy", "medium", "hard")):
        subset = [row for row in rows if row["difficulty"] == difficulty]
        matrix = _matrix(subset)
        bootstrap = bootstrap_joint_means(
            matrix, resamples=config.bootstrap_resamples, seed=config.bootstrap_seed + offset + 1,
            confidence_level=config.confidence_level,
        )
        pairs = []
        for left_index, (left, right, _) in enumerate(PAIR_SPECS):
            right_index = CANONICAL_STRATEGIES.index(right)
            deltas = matrix[CANONICAL_STRATEGIES.index(left)] - matrix[right_index]
            pairs.append({"left": left, "right": right, **paired_delta_summary(deltas, np.zeros_like(deltas)), **_pair_bootstrap(bootstrap["samples"], CANONICAL_STRATEGIES.index(left), right_index, config.confidence_level)})
        result[difficulty] = {
            "n": len(subset), "strategy_means": {strategy: float(matrix[index].mean()) for index, strategy in enumerate(CANONICAL_STRATEGIES)},
            "pairs": pairs, "inference_scope": "secondary_exploratory",
            "bootstrap_configuration": {"method": config.bootstrap_method, "resamples": config.bootstrap_resamples, "seed": config.bootstrap_seed + offset + 1, "confidence_level": config.confidence_level},
        }
    return result


def _question_type_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for question_type in sorted({row["question_type"] for row in rows}):
        subset = [row for row in rows if row["question_type"] == question_type]
        matrix = _matrix(subset)
        means = {strategy: float(matrix[index].mean()) for index, strategy in enumerate(CANONICAL_STRATEGIES)}
        maximum = max(means.values())
        winners = [strategy for strategy, value in means.items() if value == maximum]
        result[question_type] = {
            "n": len(subset), "strategy_means": means, "observed_winner": winners[0] if len(winners) == 1 else "tie",
            "paired_mean_deltas": {name: float(np.mean([row[name] for row in subset])) for _, _, name in PAIR_SPECS},
            "interpretation": "LOW SAMPLE SIZE — DESCRIPTIVE ONLY" if len(subset) < 10 else "EXPLORATORY DESCRIPTIVE",
        }
    return result
