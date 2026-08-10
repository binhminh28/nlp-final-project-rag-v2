"""CLI orchestrator: run configured chunking strategies over normalized documents.

Replaces "run chunk-fixed, then run chunk-structure" with one command driven
by a config file's `enabled_strategies`, so adding/removing a strategy is a
config change rather than a new invocation to remember.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_chunking.chunking.config import load_chunking_config
from rag_chunking.chunking.registry import UnavailableStrategyError, get_registration
from rag_chunking.data.writer import read_documents_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one or more configured chunking strategies over normalized documents"
    )
    parser.add_argument("--input", type=Path, required=True, help="Normalized documents JSONL")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory that receives one subdirectory per strategy",
    )
    parser.add_argument("--config", type=Path, required=True, help="Chunking config YAML")
    parser.add_argument(
        "--strategy",
        default="all",
        help="'all' (every strategy in config.enabled_strategies) or a comma-separated subset",
    )
    return parser


def resolve_strategies(requested: str, enabled_strategies: tuple[str, ...]) -> list[str]:
    if requested == "all":
        return list(enabled_strategies)
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = [name for name in names if name not in enabled_strategies]
    if unknown:
        raise ValueError(
            f"Strategies {unknown} are not in enabled_strategies {list(enabled_strategies)}; "
            "add them to the config first"
        )
    return names


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_config = load_chunking_config(args.config)
        strategies = resolve_strategies(args.strategy, run_config.enabled_strategies)
        documents = read_documents_jsonl(args.input)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1

    for strategy in strategies:
        try:
            registration = get_registration(strategy)
            config = registration.build_config(run_config.options_for(strategy))
            summary = registration.run(
                documents, config, args.output_root / strategy, str(args.input)
            )
        except (UnavailableStrategyError, ValueError) as error:
            print(f"ERROR [{strategy}]: {error}")
            return 1
        print(
            f"[{summary.strategy}] documents={summary.documents} "
            f"chunks={summary.chunks} output={summary.output_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
