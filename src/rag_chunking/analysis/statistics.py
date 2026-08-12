"""Deterministic paired statistical primitives for canonical benchmark analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def _vector(values: list[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional vector")
    return result


def describe(values: list[float] | np.ndarray) -> dict[str, float | int]:
    vector = _vector(values, "values")
    q1, median, q3 = np.percentile(vector, [25.0, 50.0, 75.0], method="linear")
    return {
        "n": int(vector.size), "mean": float(np.mean(vector)), "median": float(median),
        "standard_deviation": float(np.std(vector, ddof=1)) if vector.size > 1 else 0.0,
        "variance": float(np.var(vector, ddof=1)) if vector.size > 1 else 0.0,
        "minimum": float(np.min(vector)), "maximum": float(np.max(vector)),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
    }


def paired_delta_summary(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> dict[str, Any]:
    left_values = _vector(left, "left")
    right_values = _vector(right, "right")
    if left_values.shape != right_values.shape:
        raise ValueError("paired vectors must be aligned")
    deltas = left_values - right_values
    positive = deltas[deltas > 0]
    negative = deltas[deltas < 0]
    return {
        **describe(deltas),
        "positive_count": int(positive.size), "zero_count": int(np.count_nonzero(deltas == 0)),
        "negative_count": int(negative.size), "sum_positive_delta": float(np.sum(positive)),
        "sum_negative_delta": float(np.sum(negative)),
        "mean_positive_magnitude": float(np.mean(positive)) if positive.size else 0.0,
        "mean_negative_magnitude": float(np.mean(np.abs(negative))) if negative.size else 0.0,
    }


def bootstrap_joint_means(
    values: list[list[float]] | np.ndarray, *, resamples: int, seed: int,
    confidence_level: float = 0.95, batch_size: int = 5000,
) -> dict[str, Any]:
    """Bootstrap all strategy means with one shared question-index sample per replicate."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("bootstrap values must be a non-empty finite strategy-by-question matrix")
    if type(resamples) is not int or resamples <= 0 or type(seed) is not int:
        raise ValueError("bootstrap resamples must be positive and seed must be an integer")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    rng = np.random.default_rng(seed)
    strategies, questions = matrix.shape
    samples = np.empty((resamples, strategies), dtype=np.float64)
    offset = 0
    while offset < resamples:
        count = min(batch_size, resamples - offset)
        indexes = rng.integers(0, questions, size=(count, questions))
        samples[offset:offset + count] = matrix[:, indexes].mean(axis=2).T
        offset += count
    tail = (1.0 - confidence_level) * 50.0
    intervals = np.percentile(samples, [tail, 100.0 - tail], axis=0, method="linear")
    return {
        "resamples": resamples, "seed": seed, "confidence_level": confidence_level,
        "observed_means": matrix.mean(axis=1), "bootstrap_means": samples.mean(axis=0),
        "bootstrap_standard_errors": samples.std(axis=0, ddof=1),
        "ci_lower": intervals[0], "ci_upper": intervals[1], "samples": samples,
    }


def permutation_mean_test(
    deltas: list[float] | np.ndarray, *, resamples: int, seed: int,
    batch_size: int = 5000,
) -> dict[str, float | int | str]:
    """Two-sided paired randomization test by Monte Carlo sign flipping."""

    values = _vector(deltas, "deltas")
    if type(resamples) is not int or resamples <= 0 or type(seed) is not int:
        raise ValueError("permutation resamples must be positive and seed must be an integer")
    nonzero = values[values != 0]
    observed = float(np.mean(values))
    if nonzero.size == 0:
        return {
            "observed_mean_delta": observed, "p_value": 1.0, "resamples": resamples,
            "seed": seed, "extreme_count": resamples, "method": "monte_carlo_sign_flip_two_sided_add_one_v1",
        }
    rng = np.random.default_rng(seed)
    threshold = abs(observed)
    extreme = 0
    completed = 0
    while completed < resamples:
        count = min(batch_size, resamples - completed)
        signs = rng.integers(0, 2, size=(count, nonzero.size), dtype=np.int8) * 2 - 1
        permuted_means = (signs * nonzero).sum(axis=1) / values.size
        extreme += int(np.count_nonzero(np.abs(permuted_means) >= threshold - 1e-15))
        completed += count
    return {
        "observed_mean_delta": observed, "p_value": (extreme + 1) / (resamples + 1),
        "resamples": resamples, "seed": seed, "extreme_count": extreme,
        "method": "monte_carlo_sign_flip_two_sided_add_one_v1",
    }


def wilcoxon_signed_rank(deltas: list[float] | np.ndarray) -> dict[str, float | int | str]:
    """SciPy Wilcoxon signed-rank wrapper with explicit exact-zero handling."""

    values = _vector(deltas, "deltas")
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return {
            "statistic": 0.0, "p_value": 1.0, "nonzero_pairs": 0,
            "zero_pairs": int(values.size), "method": "scipy_wilcoxon_asymptotic_two_sided",
            "zero_method": "wilcox", "continuity_correction": False,
        }
    result = stats.wilcoxon(
        nonzero, zero_method="wilcox", correction=False,
        alternative="two-sided", method="asymptotic",
    )
    return {
        "statistic": float(result.statistic), "p_value": float(result.pvalue),
        "nonzero_pairs": int(nonzero.size), "zero_pairs": int(values.size - nonzero.size),
        "method": "scipy_wilcoxon_asymptotic_two_sided", "zero_method": "wilcox",
        "continuity_correction": False,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values or any(not 0 <= value <= 1 or not math.isfinite(value) for value in p_values):
        raise ValueError("Holm adjustment requires finite probabilities")
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def cohens_dz(deltas: list[float] | np.ndarray) -> float | None:
    values = _vector(deltas, "deltas")
    if values.size < 2:
        return None
    deviation = float(np.std(values, ddof=1))
    mean = float(np.mean(values))
    if deviation == 0:
        return 0.0 if mean == 0 else None
    return mean / deviation


def rank_biserial(deltas: list[float] | np.ndarray) -> float:
    values = _vector(deltas, "deltas")
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def shape_diagnostics(values: list[float] | np.ndarray) -> dict[str, float]:
    vector = _vector(values, "values")
    shapiro = stats.shapiro(vector)
    return {
        "skewness": float(stats.skew(vector, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(vector, fisher=True, bias=False)),
        "shapiro_statistic": float(shapiro.statistic), "shapiro_p_value": float(shapiro.pvalue),
    }
