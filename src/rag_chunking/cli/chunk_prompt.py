"""CLI for cached LLM boundary planning with deterministic local chunking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag_chunking.chunking.prompt_based import PromptBasedChunker, PromptBasedChunkingConfig
from rag_chunking.chunking.prompt_client import (
    DEFAULT_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_PROVIDER,
    OpenRouterBoundaryPlanner,
    PlannerModelConfig,
)
from rag_chunking.chunking.prompt_config import load_project_dotenv
from rag_chunking.chunking.prompt_statistics import prompt_corpus_statistics
from rag_chunking.chunking.prompt_prompts import PROMPT_VERSION
from rag_chunking.chunking.prompt_validation import validate_prompt_based_chunks
from rag_chunking.chunking.prompt_writer import write_prompt_based_artifacts
from rag_chunking.data.writer import read_documents_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan semantic block boundaries with an LLM")
    parser.add_argument("--input", type=Path, required=True, help="Normalized documents JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Prompt-based artifact directory")
    parser.add_argument("--cache", type=Path, help="Persistent response cache (default: OUTPUT/cache)")
    parser.add_argument("--provider", default=os.environ.get("PROMPT_PLANNER_PROVIDER", DEFAULT_PROVIDER))
    parser.add_argument("--model", default=os.environ.get("PROMPT_PLANNER_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--base-url", default=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-response-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-chunk-tokens", type=int, default=512)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--planner-input-tokens", type=int, default=12000)
    parser.add_argument("--block-preview-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="Process only the first N documents")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass cache and replace valid entries")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--transport-retries", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        load_project_dotenv()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    args = build_parser().parse_args(argv)
    try:
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        if args.max_concurrency <= 0:
            raise ValueError("--max-concurrency must be positive")
        documents = read_documents_jsonl(args.input)
        if args.limit is not None:
            documents = documents[: args.limit]
        config = PromptBasedChunkingConfig(
            max_chunk_tokens=args.max_chunk_tokens,
            tokenizer_name=args.tokenizer,
            prompt_version=args.prompt_version,
            planner_input_tokens=args.planner_input_tokens,
            block_preview_tokens=args.block_preview_tokens,
            max_retries=args.max_retries,
        )
        model_config = PlannerModelConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_response_tokens=args.max_response_tokens,
            seed=None,
            timeout_seconds=args.timeout,
        )
        if args.provider != DEFAULT_PROVIDER:
            raise ValueError("The bundled live adapter supports --provider openrouter")
        cache_dir = args.cache or args.output / "cache"
        chunker = PromptBasedChunker(
            OpenRouterBoundaryPlanner(
                base_url=args.base_url, max_transport_retries=args.transport_retries
            ),
            cache_dir,
            config,
            model_config,
            force_refresh=args.force_refresh,
        )
        live_result = chunker.chunk_corpus_concurrent(
            documents, max_concurrency=args.max_concurrency
        )
        operational = chunker.metrics.operational_summary()
        if live_result.failures:
            args.output.mkdir(parents=True, exist_ok=True)
            failure_data = {
                "configuration": model_config.identity(),
                "max_concurrency": args.max_concurrency,
                "operational": operational,
                "failures": [
                    {
                        "document_path": failure.document_path,
                        "batch_index": failure.batch_index,
                        "block_start": failure.block_start,
                        "block_end": failure.block_end,
                        "cache_key": failure.cache_key,
                        "error_category": failure.error_category,
                        "error": failure.error,
                    }
                    for failure in live_result.failures
                ],
            }
            with (args.output / "planning_failures.json").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(failure_data, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
            for failure in live_result.failures:
                print(
                    f"ERROR: {failure.document_path} batch {failure.batch_index}: "
                    f"{failure.error_category}: {failure.error}"
                )
            print("Operational planner metrics: " + json.dumps(operational, sort_keys=True))
            return 1

        class _CacheOnlyPlanner:
            def plan(self, *values: object) -> str:
                raise RuntimeError("Validated cache unexpectedly missed during assembly")

        assembly_chunker = PromptBasedChunker(
            _CacheOnlyPlanner(), cache_dir, config, model_config
        )
        assembly = assembly_chunker.chunk_corpus_concurrent(
            documents, max_concurrency=args.max_concurrency
        )
        if assembly.failures:
            raise RuntimeError("Cache-only deterministic assembly encountered a missing job")
        chunks = assembly.chunks
        failed: set[str] = set()
        report = validate_prompt_based_chunks(
            documents, chunks, config, model_config, assembly_chunker.tokenizer, failed
        )
        if not report.valid:
            for error in report.errors:
                print(f"ERROR: {error}")
            return 1
        stats = prompt_corpus_statistics(documents, chunks, assembly_chunker.metrics)
        stats["failures"] = []
        stats["validation"] = {
            "valid": report.valid,
            "coverage_gaps": report.coverage_gaps,
            "unicode_issues": report.unicode_decoding_issues,
            "errors": report.errors[:100],
        }
        write_prompt_based_artifacts(
            chunks, documents, args.output, config, model_config, assembly_chunker.tokenizer, stats, str(args.input)
        )
        stale_failures = args.output / "planning_failures.json"
        if stale_failures.exists():
            stale_failures.unlink()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1

    token_stats = stats["tokens_per_chunk"]
    prompt_stats = stats["prompt"]
    print(f"Documents processed: {stats['documents']}")
    print(f"Documents failed: {stats['documents_failed']}")
    print(f"Chunks generated: {stats['chunks']}")
    print(f"Tokenizer: {assembly_chunker.tokenizer.name}")
    print(f"Planner: {model_config.provider}:{model_config.model}")
    print("Tokens/chunk: " + " ".join(f"{key}={token_stats[key]:.2f}" for key in ("min", "mean", "median", "max", "p25", "p75", "p95")))
    print(f"Cache hits: {prompt_stats['cache_hits']}; misses: {prompt_stats['cache_misses']}")
    print(f"Model calls: {prompt_stats['model_calls']}; retries: {prompt_stats['retries']}")
    print(f"Locally adjusted planner groups: {prompt_stats['planner_groups_locally_adjusted']}")
    print("Operational planner metrics: " + json.dumps(operational, sort_keys=True))
    print(f"Output path: {args.output}")
    return 1 if stats["documents_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
