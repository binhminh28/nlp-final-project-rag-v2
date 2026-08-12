"""Offline statistical analysis over frozen benchmark evidence."""

from .statistics import (
    bootstrap_joint_means, cohens_dz, describe, holm_adjust,
    paired_delta_summary, permutation_mean_test, wilcoxon_signed_rank,
)

__all__ = [
    "bootstrap_joint_means", "cohens_dz", "describe", "holm_adjust",
    "paired_delta_summary", "permutation_mean_test", "wilcoxon_signed_rank",
]
