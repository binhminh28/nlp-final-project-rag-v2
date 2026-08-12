"""Generate durable answers from serialized authoritative ContextResult inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.chunking.prompt_config import load_project_dotenv
from rag_chunking.context.models import ContextResult
from rag_chunking.generation import (
    DeterministicFakeGenerationProvider,
    GenerationCache,
    GenerationConfig,
    GenerationInput,
    GenerationService,
    OpenRouterGenerationProvider,
    run_generation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate answers from authoritative ContextResult JSONL")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with query_id, question, context")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--provider", choices=("fake", "openrouter"), default="fake")
    parser.add_argument("--model", default="deterministic-fake-v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--generation-config-version", choices=("v1", "v2"), default="v1",
        help="v1 preserves legacy acceptance; v2 requires normal completion",
    )
    parser.add_argument(
        "--reasoning-effort", choices=("minimal", "low", "medium", "high"),
        help="Explicit OpenRouter reasoning.effort (v2 only)",
    )
    parser.add_argument("--stop", action="append", default=[], dest="stop_sequences")
    parser.add_argument(
        "--prepared-context-token-budget", type=int,
        help="Authoritative upstream ContextResult budget reference (v2 defaults to 4096)",
    )
    parser.add_argument("--context-window-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.5)
    parser.add_argument(
        "--diagnostics-output", type=Path,
        help="Non-canonical safe per-attempt diagnostic JSONL",
    )
    parser.add_argument(
        "--raw-diagnostics-output", type=Path,
        help="Non-canonical raw provider response directory; never contains request prompts",
    )
    return parser


def _load_inputs(path: Path, config: GenerationConfig) -> list[GenerationInput]:
    result: list[GenerationInput] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or set(value) != {"query_id", "question", "context"}:
                    raise ValueError("record must contain exactly query_id, question, and context")
                context = ContextResult.from_dict(value["context"])
                result.append(GenerationInput.create(value["query_id"], value["question"], context, config))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid generation input at {path}:{line_number}: {error}") from error
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = GenerationConfig(
            provider=args.provider, model=args.model, temperature=args.temperature,
            top_p=args.top_p, max_output_tokens=args.max_output_tokens, seed=args.seed,
            timeout_seconds=args.timeout_seconds, max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            context_window_tokens=args.context_window_tokens,
            schema_version=f"generation_config_{args.generation_config_version}",
            reasoning_effort=args.reasoning_effort,
            completion_integrity_policy=(
                "require_stop" if args.generation_config_version == "v2" else "allow_length"
            ),
            stop_sequences=tuple(args.stop_sequences),
            response_handling_contract=(
                "nonempty_text_require_stop_v2"
                if args.generation_config_version == "v2" else "nonempty_text_v1"
            ),
            prepared_context_token_budget=(
                args.prepared_context_token_budget or 4096
                if args.generation_config_version == "v2" else None
            ),
        )
        inputs = _load_inputs(args.input, config)
        if args.provider == "openrouter":
            load_project_dotenv()
            provider = OpenRouterGenerationProvider(
                diagnostics_output=args.diagnostics_output,
                raw_diagnostics_output=args.raw_diagnostics_output,
            )
        else:
            provider = DeterministicFakeGenerationProvider()
        service = GenerationService(config, provider, cache=GenerationCache(args.cache))
        result = run_generation(inputs, service, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    print(json.dumps({
        "run_fingerprint": result.run_fingerprint,
        "complete": result.complete,
        "answers": len(result.answers),
        "failures": len(result.failures),
        **result.stats,
    }, ensure_ascii=False, sort_keys=True))
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
