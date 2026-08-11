"""Run E2 max-chunks-per-source ablations from frozen E1 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.tuning.diversity import run_diversity_ablation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic source-cap diversification")
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[3, 2, 1])
    parser.add_argument("--result-depth", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/retrieval/angular/experiments"))
    args = parser.parse_args(argv)
    try:
        outputs = run_diversity_ablation(
            e1_depth50_directory=args.source_experiment, output_root=args.output,
            caps=args.caps, result_depth=args.result_depth,
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"outputs": [path.as_posix() for path in outputs]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
