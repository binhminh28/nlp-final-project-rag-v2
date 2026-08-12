from __future__ import annotations

import pytest

from rag_chunking.benchmark_freeze import FreezeResult, FreezeValidationError, macro_means, summarize_pair, validate_id_set
from rag_chunking.benchmark_freeze_validation import build_freeze_manifest


def test_artifact_completeness_exact_set_passes_and_defects_fail():
    assert validate_id_set(["q1", "q2"], {"q1", "q2"}, "fixture")["count"] == 2
    with pytest.raises(FreezeValidationError, match="duplicate"):
        validate_id_set(["q1", "q1", "q2"], {"q1", "q2"}, "fixture")
    with pytest.raises(FreezeValidationError, match="missing"):
        validate_id_set(["q1"], {"q1", "q2"}, "fixture")
    with pytest.raises(FreezeValidationError, match="extra"):
        validate_id_set(["q1", "q2", "q3"], {"q1", "q2"}, "fixture")


def test_raw_macro_reaggregation_is_unrounded():
    result = macro_means([
        {"token_f1": 0.1, "token_precision": 0.2},
        {"token_f1": 0.6, "token_precision": 0.4},
    ], ("token_f1", "token_precision"))
    assert result == {"token_f1": 0.35, "token_precision": pytest.approx(0.3)}


def test_pair_orientation_can_have_fewer_wins_and_positive_mean():
    result = summarize_pair([1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.1, 0.1])
    assert result["left_wins"] == 2 and result["left_losses"] == 3
    assert result["mean_delta"] == pytest.approx(0.34)
    assert result["sum_positive_deltas"] == 2.0
    assert result["sum_negative_deltas"] == pytest.approx(-0.3)


def test_freeze_manifest_is_canonical_and_requires_pass():
    result = FreezeResult("PASS", {}, {"z": {"sha256": "2", "size_bytes": 2}, "a": {"sha256": "1", "size_bytes": 1}})
    first = build_freeze_manifest(result, created_at="2026-01-01T00:00:00Z")
    second = build_freeze_manifest(result, created_at="2027-01-01T00:00:00Z")
    assert first["canonical_artifacts"] == result.artifact_inventory
    assert first["freeze_fingerprint"] == second["freeze_fingerprint"]
    assert first["question_count"] == 140 and first["answer_count"] == 420
    with pytest.raises(FreezeValidationError, match="failed validation"):
        build_freeze_manifest(FreezeResult("FAIL", {}, {}))
