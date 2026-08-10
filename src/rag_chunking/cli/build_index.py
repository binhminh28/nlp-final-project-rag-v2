"""Build a versioned local cosine vector index."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_chunking.embedding.index import build_local_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a local cosine index from embeddings")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_local_index(args.input, args.output)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    print(f"collection={manifest['collection_name']} vectors={manifest['vector_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
