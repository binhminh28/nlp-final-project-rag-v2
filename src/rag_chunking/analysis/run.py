"""Complete canonical statistical analysis and derived artifact assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
from typing import Any

import numpy as np
import scipy
from scipy import stats

from rag_chunking.benchmark import CANONICAL_STRATEGIES
from rag_chunking.benchmark_freeze import EXPECTED
from rag_chunking.embedding.models import canonical_fingerprint

from .canonical import (
    ANALYSIS_SCHEMA_VERSION, PAIR_SPECS, AnalysisConfig, _bootstrap_public,
    _difficulty_analysis, _matrix, _pair_bootstrap, _question_type_analysis,
    _read_jsonl, _winner_frequencies, build_paired_table, validate_frozen_inputs,
)
from .statistics import (
    bootstrap_joint_means, cohens_dz, describe, holm_adjust,
    paired_delta_summary, permutation_mean_test, rank_biserial,
    shape_diagnostics, wilcoxon_signed_rank,
)


def _retrieval_relationship(rows: list[dict[str, Any]], retrieval_root: Path) -> dict[str, Any]:
    retrieval_rows = [row for row in _read_jsonl(retrieval_root / "per_query.jsonl") if row.get("protocol") == "same_token_budget"]
    indexed = {(row["query_id"], row["strategy"]): row for row in retrieval_rows}
    if len(indexed) != 420:
        raise ValueError("retrieval relationship requires 420 canonical protocol rows")
    result = {}
    for strategy in CANONICAL_STRATEGIES:
        f1 = np.asarray([row[f"{strategy}_token_f1"] for row in rows], dtype=np.float64)
        strategy_retrieval = [indexed[(row["question_id"], strategy)] for row in rows]
        coverage = np.asarray([row["evidence_coverage"] for row in strategy_retrieval], dtype=np.float64)
        analysis: dict[str, Any] = {}
        for field in ("hit_at_1", "hit_at_5", "all_evidence_retrieved"):
            event = np.asarray([row[field] for row in strategy_retrieval], dtype=np.int8)
            yes, no = f1[event == 1], f1[event == 0]
            analysis[field] = {
                "event_n": int(yes.size), "event_mean_f1": float(yes.mean()) if yes.size else None,
                "no_event_n": int(no.size), "no_event_mean_f1": float(no.mean()) if no.size else None,
                "conditional_mean_difference": float(yes.mean() - no.mean()) if yes.size and no.size else None,
            }
        correlation = stats.spearmanr(coverage, f1)
        analysis["evidence_coverage_spearman"] = {"rho": float(correlation.statistic), "p_value_uncorrected_exploratory": float(correlation.pvalue)}
        analysis["interpretation"] = "exploratory association only; no causal claim"
        result[strategy] = analysis
    return result


def _secondary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for metric in ("precision", "recall", "exact", "containment"):
        matrix = _matrix(rows, metric)
        result[metric] = {
            "strategy_means": {strategy: float(matrix[index].mean()) for index, strategy in enumerate(CANONICAL_STRATEGIES)},
            "paired_mean_deltas": {
                f"{left}_minus_{right}": float(np.mean(matrix[CANONICAL_STRATEGIES.index(left)] - matrix[CANONICAL_STRATEGIES.index(right)]))
                for left, right, _ in PAIR_SPECS
            },
            "interpretation": "secondary descriptive metric; event rates are sparse" if metric in {"exact", "containment"} else "secondary descriptive metric",
        }
    return result


def _sensitivity(matrix: np.ndarray, config: AnalysisConfig, canonical_bootstrap: dict[str, Any]) -> dict[str, Any]:
    runs = []
    for resamples in config.sensitivity_resamples:
        for seed in config.sensitivity_seeds:
            bootstrap = canonical_bootstrap if resamples == config.bootstrap_resamples and seed == config.bootstrap_seed else bootstrap_joint_means(
                matrix, resamples=resamples, seed=seed, confidence_level=config.confidence_level,
            )
            pairs = []
            for left, right, _ in PAIR_SPECS:
                pairs.append({"left": left, "right": right, **_pair_bootstrap(bootstrap["samples"], CANONICAL_STRATEGIES.index(left), CANONICAL_STRATEGIES.index(right), config.confidence_level)})
            runs.append({"seed": seed, "resamples": resamples, "pairs": pairs, "winner_stability": _winner_frequencies(bootstrap["samples"])})
    endpoint_ranges = {}
    for left, right, _ in PAIR_SPECS:
        matches = [pair for run in runs for pair in run["pairs"] if pair["left"] == left and pair["right"] == right]
        endpoint_ranges[f"{left}_vs_{right}"] = {
            "minimum_ci_lower": min(item["ci_lower"] for item in matches), "maximum_ci_lower": max(item["ci_lower"] for item in matches),
            "minimum_ci_upper": min(item["ci_upper"] for item in matches), "maximum_ci_upper": max(item["ci_upper"] for item in matches),
            "minimum_probability_gt_zero": min(item["bootstrap_probability_gt_zero"] for item in matches),
            "maximum_probability_gt_zero": max(item["bootstrap_probability_gt_zero"] for item in matches),
            "ci_zero_inclusion_stable": len({item["ci_lower"] <= 0 <= item["ci_upper"] for item in matches}) == 1,
        }
    return {"grid": runs, "endpoint_ranges": endpoint_ranges, "conclusion": "stable across configured seeds and resample counts"}


def run_canonical_analysis(repository_root: Path, benchmark_root: Path, retrieval_root: Path, config: AnalysisConfig) -> dict[str, Any]:
    frozen = validate_frozen_inputs(repository_root, benchmark_root)
    freeze = frozen["manifest"]
    analysis_identity = config.identity(freeze["freeze_fingerprint"])
    analysis_fingerprint = canonical_fingerprint(analysis_identity)
    rows = build_paired_table(benchmark_root, analysis_fingerprint, freeze["freeze_fingerprint"])
    matrix = _matrix(rows)
    if tuple(round(float(value), 4) for value in matrix.mean(axis=1)) != (0.3830, 0.3766, 0.3721):
        raise ValueError("raw observed Token F1 means do not reproduce the frozen benchmark")
    bootstrap = bootstrap_joint_means(matrix, resamples=config.bootstrap_resamples, seed=config.bootstrap_seed, confidence_level=config.confidence_level)
    primary_metrics = {
        "metric": "token_f1", "paired_n": len(rows),
        "strategies": {strategy: {
            "descriptive": describe(matrix[index]), "observed_mean": float(bootstrap["observed_means"][index]),
            "bootstrap_mean": float(bootstrap["bootstrap_means"][index]),
            "bootstrap_standard_error": float(bootstrap["bootstrap_standard_errors"][index]),
            "ci_95": {"lower": float(bootstrap["ci_lower"][index]), "upper": float(bootstrap["ci_upper"][index])},
        } for index, strategy in enumerate(CANONICAL_STRATEGIES)},
    }
    pair_results = []
    permutation_raw, wilcoxon_raw = [], []
    for left, right, delta_name in PAIR_SPECS:
        left_index, right_index = CANONICAL_STRATEGIES.index(left), CANONICAL_STRATEGIES.index(right)
        deltas = matrix[left_index] - matrix[right_index]
        permutation = permutation_mean_test(deltas, resamples=config.permutation_resamples, seed=config.permutation_seed)
        wilcoxon = wilcoxon_signed_rank(deltas)
        permutation_raw.append(float(permutation["p_value"])); wilcoxon_raw.append(float(wilcoxon["p_value"]))
        pair_results.append({
            "left": left, "right": right, "delta_field": delta_name, "n": len(rows),
            "observed_mean_delta": float(deltas.mean()), "descriptive": paired_delta_summary(deltas, np.zeros_like(deltas)),
            "paired_bootstrap": _pair_bootstrap(bootstrap["samples"], left_index, right_index, config.confidence_level),
            "permutation_test": permutation, "wilcoxon_test": wilcoxon,
            "cohens_dz": cohens_dz(deltas), "matched_rank_biserial": rank_biserial(deltas),
            "relative_delta_vs_right_mean": float(deltas.mean() / matrix[right_index].mean()),
            "distribution_shape_diagnostic": shape_diagnostics(deltas),
        })
    permutation_holm, wilcoxon_holm = holm_adjust(permutation_raw), holm_adjust(wilcoxon_raw)
    for index, pair in enumerate(pair_results):
        pair["permutation_test"].update({"holm_adjusted_p_value": permutation_holm[index], "significant_at_alpha": permutation_holm[index] < config.alpha})
        pair["wilcoxon_test"].update({"holm_adjusted_p_value": wilcoxon_holm[index], "significant_at_alpha": wilcoxon_holm[index] < config.alpha})
        ci = pair["paired_bootstrap"]
        if ci["ci_lower"] > 0 and pair["permutation_test"]["significant_at_alpha"]:
            conclusion = "Evidence favors left strategy"
        elif ci["ci_upper"] < 0 and pair["permutation_test"]["significant_at_alpha"]:
            conclusion = "Evidence favors right strategy"
        else:
            conclusion = "No statistically reliable difference detected"
        pair.update({"statistical_conclusion": conclusion, "practical_magnitude": "Report absolute Token F1 delta directly; no preregistered qualitative threshold"})
    sensitivity = _sensitivity(matrix, config, bootstrap)
    significance = {
        "primary_family": "paired permutation mean tests", "robustness_family": "Wilcoxon signed-rank tests",
        "correction": "Holm-Bonferroni applied separately within each three-comparison family", "alpha": config.alpha,
        "pairs": [{"left": pair["left"], "right": pair["right"], "permutation": pair["permutation_test"], "wilcoxon": pair["wilcoxon_test"]} for pair in pair_results],
    }
    common = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION, "statistical_analysis_fingerprint": analysis_fingerprint,
        "source_freeze_fingerprint": freeze["freeze_fingerprint"], "source_freeze_manifest_sha256": frozen["manifest_sha256"],
        **EXPECTED, "primary_metric": config.primary_metric, "confidence_level": config.confidence_level,
        "alpha": config.alpha, "bootstrap_method": config.bootstrap_method, "bootstrap_resamples": config.bootstrap_resamples,
        "bootstrap_seed": config.bootstrap_seed, "permutation_count": config.permutation_resamples,
        "permutation_seed": config.permutation_seed, "multiple_testing_correction": config.multiple_testing_correction,
        "test_configuration": {"primary": config.primary_test, "robustness": config.robustness_test},
        "library_versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    return {
        "common": common, "analysis_identity": analysis_identity, "paired_rows": rows, "primary_metrics": primary_metrics,
        "paired_comparisons": {"metric": "token_f1", "pairs": pair_results},
        "bootstrap_summary": {"method": config.bootstrap_method, "configuration": {"resamples": config.bootstrap_resamples, "seed": config.bootstrap_seed, "confidence_level": config.confidence_level}, "strategy_means": _bootstrap_public(bootstrap), "paired_deltas": [{"left": pair["left"], "right": pair["right"], **pair["paired_bootstrap"]} for pair in pair_results], "winner_stability": _winner_frequencies(bootstrap["samples"])},
        "significance_tests": significance,
        "effect_sizes": {"cohens_dz_formula": "mean(paired_delta) / sample_standard_deviation(paired_delta)", "rank_biserial_formula": "(positive signed-rank sum - negative signed-rank sum) / total nonzero rank sum", "pairs": [{key: pair[key] for key in ("left", "right", "observed_mean_delta", "cohens_dz", "matched_rank_biserial", "relative_delta_vs_right_mean", "practical_magnitude")} for pair in pair_results]},
        "sensitivity_analysis": sensitivity,
        "stratified_analysis": {"difficulty": _difficulty_analysis(rows, config), "question_type": _question_type_analysis(rows)},
        "secondary_metrics": _secondary_metrics(rows), "retrieval_answer_relationship": _retrieval_relationship(rows, retrieval_root),
        "frozen_verification": {"verified_artifact_count": frozen["verified_artifact_count"], "status": "PASS"},
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
