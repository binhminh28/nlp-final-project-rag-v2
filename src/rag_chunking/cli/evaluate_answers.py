"""Offline deterministic evaluation of committed answer-generation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.evaluation.answer_models import EvaluationConfig, SUPPORTED_METRICS
from rag_chunking.evaluation.answer_runner import run_answer_benchmark
from rag_chunking.evaluation.qa_dataset import load_qa_dataset
from rag_chunking.retrieval.models import KNOWN_STRATEGIES


def _generation(value: str) -> tuple[str, Path]:
    strategy, separator, path = value.partition("=")
    if not separator or strategy not in KNOWN_STRATEGIES or not path:
        raise argparse.ArgumentTypeError("generation must be STRATEGY=PATH")
    return strategy, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen answers against frozen canonical references (offline)",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True, help="Canonical documents.jsonl")
    parser.add_argument(
        "--generation", type=_generation, action="append", required=True,
        metavar="STRATEGY=PATH", help="Committed generation directory; repeat per strategy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics", nargs="+", choices=SUPPORTED_METRICS,
        default=list(SUPPORTED_METRICS),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        directories = dict(args.generation)
        if len(directories) != len(args.generation):
            raise ValueError("duplicate generation strategy")
        documents = read_documents_jsonl(args.documents)
        dataset = load_qa_dataset(args.dataset, {item.doc_id for item in documents})
        config = EvaluationConfig(enabled_metrics=tuple(args.metrics))
        result = run_answer_benchmark(dataset, directories, args.output, config=config)
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({
        "benchmark_fingerprint": result.benchmark_fingerprint,
        "dataset_fingerprint": dataset.fingerprint,
        "evaluation_config_fingerprint": config.fingerprint,
        "output": result.output_directory.as_posix(),
        "queries": len(dataset.records), "strategies": sorted(directories),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
