"""CLI for offline Canonical Production Benchmark v2 freeze validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.benchmark_freeze import EXPECTED, FreezePaths, FreezeValidationError, artifact_inventory
from rag_chunking.benchmark_freeze_report import render_report
from rag_chunking.benchmark_freeze_validation import atomic_write_json, build_freeze_manifest, validate_freeze


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate canonical benchmark v2 entirely offline")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--benchmark-root", type=Path, default=Path("data/benchmark/angular/canonical_v2"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("data/retrieval/angular/canonical_production_v2") / EXPECTED["retrieval_benchmark_fingerprint"])
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/angular/qa_dataset.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("BENCHMARK_V2_FREEZE_VALIDATION.md"))
    parser.add_argument("--write-freeze-manifest", action="store_true")
    parser.add_argument("--tests-summary", default="Relevant and full-suite results are recorded by the freeze operator after execution.")
    return parser


def _absolute(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = args.repository_root.resolve()
    paths = FreezePaths(
        repository, _absolute(repository, args.benchmark_root), _absolute(repository, args.retrieval_root),
        _absolute(repository, args.dataset), _absolute(repository, args.report),
    )
    manifest_path = paths.benchmark_root / "freeze_manifest.json"
    try:
        before = artifact_inventory(paths)
        result = validate_freeze(paths)
        if artifact_inventory(paths) != before:
            raise FreezeValidationError("canonical artifacts changed during read-only validation")
        paths.report.write_text(render_report(result, tests=args.tests_summary), encoding="utf-8")
        if args.write_freeze_manifest:
            if artifact_inventory(paths) != before:
                raise FreezeValidationError("canonical artifacts changed before freeze publication")
            atomic_write_json(manifest_path, build_freeze_manifest(result))
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"status": "NOT FROZEN", "error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({
        "status": "FROZEN" if args.write_freeze_manifest else "PASS (verify only)",
        "report": paths.report.as_posix(),
        "freeze_manifest": manifest_path.as_posix() if args.write_freeze_manifest else None,
        "canonical_artifact_count": len(result.artifact_inventory),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
