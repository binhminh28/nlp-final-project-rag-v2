"""Offline final statistical validation for frozen canonical benchmark v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rag_chunking.analysis.canonical import AnalysisConfig, validate_frozen_inputs
from rag_chunking.analysis.report import render_report
from rag_chunking.analysis.run import run_canonical_analysis, utc_now
from rag_chunking.benchmark_freeze import EXPECTED
from rag_chunking.chunking.writer import write_artifact_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired statistical analysis over frozen canonical v2 evidence")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--benchmark-root", type=Path, default=Path("data/benchmark/angular/canonical_v2"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("data/retrieval/angular/canonical_production_v2") / EXPECTED["retrieval_benchmark_fingerprint"])
    parser.add_argument("--output", type=Path, default=Path("data/benchmark/angular/canonical_v2/statistical_analysis"))
    parser.add_argument("--report", type=Path, default=Path("FINAL_STATISTICAL_ANALYSIS.md"))
    parser.add_argument("--bootstrap-resamples", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--permutation-resamples", type=int, default=100_000)
    parser.add_argument("--permutation-seed", type=int, default=2026)
    parser.add_argument("--tests-summary", default="Focused and full-suite results pending final publication.")
    return parser


def _absolute(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _jsonl(values: list[dict]) -> str:
    return "".join(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for value in values)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = args.repository_root.resolve()
    benchmark = _absolute(repository, args.benchmark_root)
    retrieval = _absolute(repository, args.retrieval_root)
    output = _absolute(repository, args.output)
    report = _absolute(repository, args.report)
    config = AnalysisConfig(
        bootstrap_resamples=args.bootstrap_resamples, bootstrap_seed=args.bootstrap_seed,
        permutation_resamples=args.permutation_resamples, permutation_seed=args.permutation_seed,
    )
    try:
        frozen_before = validate_frozen_inputs(repository, benchmark)
        result = run_canonical_analysis(repository, benchmark, retrieval, config)
        created_at = utc_now()
        metadata = {**result["common"], "created_at": created_at}
        payloads = {
            "primary_metrics.json": _json({"metadata": metadata, **result["primary_metrics"]}),
            "paired_comparisons.json": _json({"metadata": metadata, **result["paired_comparisons"]}),
            "bootstrap_summary.json": _json({"metadata": metadata, **result["bootstrap_summary"]}),
            "significance_tests.json": _json({"metadata": metadata, **result["significance_tests"]}),
            "effect_sizes.json": _json({"metadata": metadata, **result["effect_sizes"]}),
            "sensitivity_analysis.json": _json({"metadata": metadata, **result["sensitivity_analysis"]}),
            "stratified_analysis.json": _json({"metadata": metadata, **result["stratified_analysis"]}),
            "secondary_metrics.json": _json({"metadata": metadata, "metrics": result["secondary_metrics"]}),
            "retrieval_answer_relationship.json": _json({"metadata": metadata, "strategies": result["retrieval_answer_relationship"]}),
            "per_question_deltas.jsonl": _jsonl([{**row, "created_at": created_at} for row in result["paired_rows"]]),
        }
        artifacts = {name: {"sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(), "size_bytes": len(value.encode("utf-8"))} for name, value in sorted(payloads.items())}
        manifest = {
            "metadata": metadata, "complete": True, "paired_question_count": 140,
            "analysis_identity": result["analysis_identity"], "artifacts": artifacts,
            "source_freeze_validation": result["frozen_verification"],
            "declaration": "FINAL STATISTICAL VALIDATION: COMPLETE",
        }
        if validate_frozen_inputs(repository, benchmark) != frozen_before:
            raise ValueError("frozen canonical identity changed during statistical computation")
        report.write_text(render_report(result, created_at=created_at, tests_summary=args.tests_summary), encoding="utf-8")
        if validate_frozen_inputs(repository, benchmark) != frozen_before:
            raise ValueError("frozen canonical identity changed before statistical publication")
        write_artifact_set(output, {**payloads, "manifest.json": _json(manifest)})
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"status": "FAILED", "error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({
        "status": "COMPLETE", "statistical_analysis_fingerprint": result["common"]["statistical_analysis_fingerprint"],
        "output": output.as_posix(), "report": report.as_posix(), "paired_questions": 140,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
