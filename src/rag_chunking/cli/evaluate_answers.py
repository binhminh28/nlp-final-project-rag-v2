"""Offline deterministic evaluation of committed answer-generation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.benchmark import (
    project_benchmark_queries, validate_generation_requests_against_preparation,
    validate_prepared_benchmark_inputs,
)
from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.evaluation.answer_models import EvaluationConfig, SUPPORTED_METRICS
from rag_chunking.evaluation.answer_runner import run_answer_benchmark
from rag_chunking.evaluation.qa_dataset import (
    is_team_qa_dataset, load_team_qa_dataset, validate_canonical_qa_dataset,
)
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
        "--prepared-inputs", type=Path,
        help="Committed prepare-answer-inputs directory for corpus/context compatibility validation",
    )
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
        if set(directories) == set(KNOWN_STRATEGIES) and args.prepared_inputs is None:
            raise ValueError("three-strategy evaluation requires --prepared-inputs compatibility lineage")
        documents = read_documents_jsonl(args.documents)
        by_id = {item.doc_id: item for item in documents}
        if len(by_id) != len(documents):
            raise ValueError("canonical corpus contains duplicate document IDs")
        if is_team_qa_dataset(args.dataset):
            dataset = load_team_qa_dataset(args.dataset)
        else:
            dataset, semantic = validate_canonical_qa_dataset(args.dataset, by_id)
            if not semantic.valid:
                raise ValueError(f"canonical QA semantic validation failed: {semantic.errors}")
        config = EvaluationConfig(enabled_metrics=tuple(args.metrics))
        source_corpus_fingerprint = preparation_fingerprint = None
        if args.prepared_inputs is not None:
            prepared = validate_prepared_benchmark_inputs(
                args.prepared_inputs, dataset_fingerprint=dataset.fingerprint,
                expected_queries=project_benchmark_queries(dataset.records),
            )
            validate_generation_requests_against_preparation(prepared, directories)
            source_corpus_fingerprint = prepared.manifest["corpus_fingerprint"]
            preparation_fingerprint = prepared.preparation_fingerprint
        result = run_answer_benchmark(
            dataset, directories, args.output, config=config,
            source_corpus_fingerprint=source_corpus_fingerprint,
            preparation_fingerprint=preparation_fingerprint,
        )
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
