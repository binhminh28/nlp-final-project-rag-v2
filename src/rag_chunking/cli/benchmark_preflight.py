"""Read-only readiness report for the canonical three-strategy benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.chunking.prompt_config import load_project_dotenv
from rag_chunking.context import ContextConfig
from rag_chunking.evaluation import EvaluationConfig
from rag_chunking.generation import GenerationConfig
from rag_chunking.readiness import run_benchmark_preflight
from rag_chunking.retrieval import RetrievalProtocolConfig, SAME_TOKEN_BUDGET, SAME_TOP_K


DEFAULT_DATASET = Path("data/evaluation/angular/qa_dataset.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check canonical benchmark readiness without mutation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", default="angular")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--chunks-root", type=Path, default=Path("data/chunks"))
    parser.add_argument("--embeddings-root", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--indexes-root", type=Path, default=Path("data/indexes"))
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--protocol", choices=(SAME_TOP_K, SAME_TOKEN_BUDGET), default=SAME_TOKEN_BUDGET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument("--generation-provider", choices=("fake", "openrouter"), default="fake")
    parser.add_argument(
        "--generation-config", type=Path,
        help="Exact GenerationConfig JSON; overrides individual generation flags",
    )
    parser.add_argument("--generation-model", default="deterministic-fake-v1")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--context-window-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.5)
    parser.add_argument("--require-live-credentials", action="store_true")
    parser.add_argument("--output-path", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.require_live_credentials:
            load_project_dotenv()
        if args.generation_config is not None:
            value = json.loads(args.generation_config.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("--generation-config must contain a JSON object")
            generation = GenerationConfig(**value)
        else:
            generation = GenerationConfig(
                provider=args.generation_provider, model=args.generation_model,
                temperature=args.temperature, max_output_tokens=args.max_output_tokens,
                context_window_tokens=args.context_window_tokens, seed=args.seed,
                timeout_seconds=args.timeout_seconds, max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
        evaluation = EvaluationConfig()
        protocol = RetrievalProtocolConfig(
            args.protocol, top_k=args.top_k, candidate_k=args.candidate_k,
            token_budget=args.token_budget if args.protocol == SAME_TOKEN_BUDGET else None,
        )
        context = ContextConfig(context_token_budget=args.context_token_budget)
        report = run_benchmark_preflight(
            dataset_path=args.dataset, corpus=args.corpus,
            processed_root=args.processed_root, raw_root=args.raw_root,
            chunks_root=args.chunks_root,
            embeddings_root=args.embeddings_root, indexes_root=args.indexes_root,
            embedding_config_path=args.embedding_config,
            generation_config=generation, evaluation_config=evaluation,
            protocol_config=protocol, context_config=context,
            output_paths=tuple(args.output_path),
            require_live_credentials=args.require_live_credentials,
        )
        payload = report.to_dict()
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        payload = {"ready": False, "checks": [], "blockers": [str(error)], "warnings": []}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
