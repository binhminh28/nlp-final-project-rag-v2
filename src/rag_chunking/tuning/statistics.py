"""Deterministic query-level bootstrap uncertainty."""

from __future__ import annotations

import random
from typing import Callable


def percentile_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    if not values or not 0 < confidence < 1:
        raise ValueError("invalid percentile interval input")
    ordered = sorted(values)
    tail = (1.0 - confidence) / 2.0
    low = max(0, min(len(ordered) - 1, int(tail * len(ordered))))
    high = max(0, min(len(ordered) - 1, int((1.0 - tail) * len(ordered)) - 1))
    return ordered[low], ordered[high]


def paired_bootstrap(
    reference: list[float], experiment: list[float], *, seed: int = 20260811,
    samples: int = 5000,
) -> dict[str, float | int]:
    if len(reference) != len(experiment) or not reference or samples <= 0:
        raise ValueError("paired bootstrap inputs must be non-empty and aligned")
    randomizer = random.Random(seed)
    size = len(reference)
    reference_means, experiment_means, deltas = [], [], []
    for _ in range(samples):
        indexes = [randomizer.randrange(size) for _ in range(size)]
        left = sum(reference[index] for index in indexes) / size
        right = sum(experiment[index] for index in indexes) / size
        reference_means.append(left)
        experiment_means.append(right)
        deltas.append(right - left)
    ref_low, ref_high = percentile_interval(reference_means)
    exp_low, exp_high = percentile_interval(experiment_means)
    delta_low, delta_high = percentile_interval(deltas)
    return {
        "query_count": size, "samples": samples, "seed": seed,
        "reference_mean": sum(reference) / size, "reference_ci_low": ref_low, "reference_ci_high": ref_high,
        "experiment_mean": sum(experiment) / size, "experiment_ci_low": exp_low, "experiment_ci_high": exp_high,
        "delta": sum(experiment[index] - reference[index] for index in range(size)) / size,
        "delta_ci_low": delta_low, "delta_ci_high": delta_high,
    }
