"""Calculate deterministic paired bootstrap intervals for major experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.chunking.writer import serialize_json
from rag_chunking.tuning.lexical import load_dense_rows
from rag_chunking.tuning.statistics import paired_bootstrap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args(argv)
    try:
        reference = load_dense_rows(args.reference / "per_query.jsonl")
        result = {"schema_version": "paired_bootstrap_v1", "reference_fingerprint": args.reference.name,
                  "samples": args.samples, "seed": 20260811, "experiments": {}}
        for directory in args.experiment:
            rows = load_dense_rows(directory / "per_query.jsonl")
            if set(rows) != set(reference):
                raise ValueError(f"query-strategy coverage mismatch: {directory}")
            by_strategy = {}
            for strategy in sorted({key[1] for key in reference}):
                keys = sorted(key for key in reference if key[1] == strategy)
                by_strategy[strategy] = {}
                for metric in ("hit_at_5", "reciprocal_rank"):
                    by_strategy[strategy]["mrr" if metric == "reciprocal_rank" else metric] = paired_bootstrap(
                        [float(reference[key][metric]) for key in keys],
                        [float(rows[key][metric]) for key in keys], samples=args.samples,
                    )
            result["experiments"][directory.name] = by_strategy
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialize_json(result), encoding="utf-8")
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"output": args.output.as_posix(), "experiments": len(args.experiment)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
