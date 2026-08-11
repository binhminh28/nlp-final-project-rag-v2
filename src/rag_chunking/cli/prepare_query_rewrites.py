"""Populate and validate the retrieval query-rewrite cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chunking.chunking.prompt_config import load_project_dotenv
from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.evaluation.dataset import load_evaluation_dataset
from rag_chunking.tuning.rewrite import OpenRouterQueryRewriter, RewriteCache, RewriteConfig, prepare_rewrites


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare cached retrieval query rewrites")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--documents", type=Path, default=Path("data/processed/angular/documents.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/query-rewrite-cache"))
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash-0731:nitro")
    parser.add_argument("--reasoning-max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        load_project_dotenv()
        sources = {document.relative_path for document in read_documents_jsonl(args.documents)}
        dataset = load_evaluation_dataset(args.dataset, sources)
        config = RewriteConfig(model=args.model, reasoning_max_tokens=args.reasoning_max_tokens)
        provider = OpenRouterQueryRewriter()
        cache = RewriteCache(args.cache, config)
        records, stats = prepare_rewrites(
            [(record.query_id, record.query) for record in dataset.records], config, cache, provider, args.limit,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in records), encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps({"rewrite_config_fingerprint": config.fingerprint, "records": len(records), **stats}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
