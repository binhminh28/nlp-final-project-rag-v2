from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rag_chunking.analysis.canonical import validate_frozen_inputs
from rag_chunking.analysis.statistics import (
    bootstrap_joint_means, cohens_dz, describe, holm_adjust,
    paired_delta_summary, permutation_mean_test, rank_biserial,
    wilcoxon_signed_rank,
)
from rag_chunking.benchmark_freeze import EXPECTED
from rag_chunking.chunking.writer import write_artifact_set
from rag_chunking.embedding.models import canonical_fingerprint


def test_descriptive_and_paired_delta_statistics_are_raw_and_aligned():
    summary = describe([0.0, 1.0, 2.0, 3.0])
    assert summary["mean"] == 1.5 and summary["median"] == 1.5
    assert summary["q1"] == 0.75 and summary["q3"] == 2.25
    paired = paired_delta_summary([1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.5])
    assert (paired["positive_count"], paired["zero_count"], paired["negative_count"]) == (1, 1, 2)
    assert paired["mean"] == pytest.approx(-0.125)


def test_joint_bootstrap_is_deterministic_and_uses_shared_paired_indices():
    values = [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]
    first = bootstrap_joint_means(values, resamples=500, seed=7)
    second = bootstrap_joint_means(values, resamples=500, seed=7)
    assert np.array_equal(first["samples"], second["samples"])
    assert np.allclose(first["samples"][:, 1] - first["samples"][:, 0], 10.0)
    assert first["samples"].shape == (500, 2)


def test_percentile_interval_for_constant_distribution_is_exact():
    result = bootstrap_joint_means([[0.25, 0.25], [0.75, 0.75]], resamples=100, seed=42)
    assert result["ci_lower"].tolist() == [0.25, 0.75]
    assert result["ci_upper"].tolist() == [0.25, 0.75]


def test_permutation_test_is_deterministic_sign_flipping_and_handles_null():
    first = permutation_mean_test([1.0, 2.0, -0.5], resamples=1000, seed=9)
    second = permutation_mean_test([1.0, 2.0, -0.5], resamples=1000, seed=9)
    assert first == second and 0 < first["p_value"] <= 1
    null = permutation_mean_test([0.0, 0.0], resamples=100, seed=9)
    assert null["p_value"] == 1.0 and null["extreme_count"] == 100


def test_wilcoxon_wrapper_orientation_zeros_and_all_equal():
    positive = wilcoxon_signed_rank([0.1, 0.2, 0.0, 0.4])
    negative = wilcoxon_signed_rank([-0.1, -0.2, 0.0, -0.4])
    assert positive["p_value"] == negative["p_value"]
    assert positive["zero_pairs"] == 1 and positive["nonzero_pairs"] == 3
    assert rank_biserial([0.1, 0.2, 0.0, 0.4]) == 1.0
    assert rank_biserial([-0.1, -0.2, 0.0, -0.4]) == -1.0
    assert wilcoxon_signed_rank([0.0, 0.0])["p_value"] == 1.0


def test_holm_adjustment_known_order_and_monotonicity():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.8, 0.001, 0.04]) == pytest.approx([0.8, 0.003, 0.08])


def test_paired_cohens_dz_and_zero_variance_policy():
    assert cohens_dz([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert cohens_dz([0.0, 0.0]) == 0.0
    assert cohens_dz([1.0, 1.0]) is None


def _freeze_fixture(root: Path) -> tuple[Path, Path]:
    benchmark = root / "data/benchmark/angular/canonical_v2"
    artifact = root / "evidence.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}\n", encoding="utf-8")
    payload = artifact.read_bytes()
    manifest = {
        "freeze_schema_version": "canonical_benchmark_freeze_v1", "created_at": "2026-01-01T00:00:00Z",
        **EXPECTED, "strategies": ["fixed_size", "structure_aware", "prompt_based"],
        "question_count": 140, "answer_count": 420,
        "canonical_artifacts": {"evidence.json": {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}},
        "validation_result": "PASS", "freeze_declaration": "CANONICAL PRODUCTION BENCHMARK V2: FROZEN FOR STATISTICAL ANALYSIS",
    }
    manifest["freeze_fingerprint"] = canonical_fingerprint({key: value for key, value in manifest.items() if key not in {"created_at", "freeze_fingerprint"}})
    benchmark.mkdir(parents=True)
    benchmark.joinpath("freeze_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return benchmark, artifact


def test_frozen_hash_enforcement_passes_then_rejects_tampering(tmp_path: Path):
    benchmark, artifact = _freeze_fixture(tmp_path)
    assert validate_frozen_inputs(tmp_path, benchmark)["verified_artifact_count"] == 1
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_inputs(tmp_path, benchmark)


def test_derived_publication_never_modifies_frozen_input(tmp_path: Path):
    benchmark, artifact = _freeze_fixture(tmp_path)
    before = artifact.read_bytes()
    write_artifact_set(benchmark / "statistical_analysis", {"result.json": "{}\n", "manifest.json": "{\"complete\":true}\n"})
    assert artifact.read_bytes() == before
    assert validate_frozen_inputs(tmp_path, benchmark)["verified_artifact_count"] == 1
