"""Deterministic BM25 retrieval and dense-candidate reranking."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_chunking.chunking.models import Chunk
from rag_chunking.chunking.writer import read_chunks_jsonl
from rag_chunking.embedding.models import canonical_fingerprint


BM25_SCHEMA_VERSION = "bm25_lexical_index_v1"
TOKENIZER_VERSION = "unicode_words_code_symbols_v1"
_TOKEN = re.compile(r"[\w]+|(?:@|#|\$)[\w]+|===|!==|=>|\[\(|\)\]|[.():{}\[\]<>/+*-]", re.UNICODE)


def tokenize(value: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN.finditer(value)]


@dataclass(frozen=True, slots=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75
    tokenizer_version: str = TOKENIZER_VERSION
    schema_version: str = BM25_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("invalid BM25 parameters")

    def identity(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.identity())


class BM25Index:
    def __init__(self, chunks: list[Chunk], config: BM25Config = BM25Config(), chunk_config_fingerprint: str = "test") -> None:
        if not chunks or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("BM25 requires non-empty unique chunks")
        self.config = config
        self.chunk_config_fingerprint = chunk_config_fingerprint
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.order = sorted(self.chunks)
        self.lengths: dict[str, int] = {}
        postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for chunk_id in self.order:
            tokens = tokenize(self.chunks[chunk_id].text)
            self.lengths[chunk_id] = len(tokens)
            for token, frequency in sorted(Counter(tokens).items()):
                postings[token].append((chunk_id, frequency))
        self.postings = dict(postings)
        self.average_length = sum(self.lengths.values()) / len(self.lengths)
        identity = {
            "schema_version": BM25_SCHEMA_VERSION, "config_fingerprint": config.fingerprint,
            "strategy": chunks[0].strategy,
            "chunks": [{"chunk_id": item.chunk_id, "text_sha256": hashlib.sha256(item.text.encode()).hexdigest()} for item in sorted(chunks, key=lambda item: item.chunk_id)],
        }
        self.fingerprint = canonical_fingerprint(identity)

    @classmethod
    def from_path(cls, path: Path, config: BM25Config = BM25Config()) -> "BM25Index":
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        fingerprint = manifest.get("config_fingerprint") or manifest.get("configuration_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("chunk manifest has no configuration fingerprint")
        return cls(read_chunks_jsonl(path), config, fingerprint)

    def scores(self, query: str) -> dict[str, float]:
        result: dict[str, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))
        document_count = len(self.order)
        for token, query_frequency in query_terms.items():
            posting = self.postings.get(token, [])
            document_frequency = len(posting)
            if not document_frequency:
                continue
            idf = math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for chunk_id, frequency in posting:
                norm = 1.0 - self.config.b + self.config.b * self.lengths[chunk_id] / self.average_length
                result[chunk_id] += query_frequency * idf * (frequency * (self.config.k1 + 1.0)) / (frequency + self.config.k1 * norm)
        return dict(result)

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        scores = self.scores(query)
        ordered = sorted(self.order, key=lambda chunk_id: (-scores.get(chunk_id, 0.0), chunk_id))[:limit]
        return [self._hit(chunk_id, rank, scores.get(chunk_id, 0.0)) for rank, chunk_id in enumerate(ordered, 1)]

    def rerank(self, query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if len({hit.get("chunk_id") for hit in candidates}) != len(candidates):
            raise ValueError("reranker candidates must be unique")
        scores = self.scores(query)
        ordered = sorted(candidates, key=lambda hit: (-scores.get(hit["chunk_id"], 0.0), hit["chunk_id"]))[:limit]
        output = []
        for rank, hit in enumerate(ordered, 1):
            if hit["chunk_id"] not in self.chunks:
                raise ValueError("reranker candidate is absent from lexical index")
            value = dict(hit)
            value["rank"] = rank
            value["score"] = scores.get(hit["chunk_id"], 0.0)
            output.append(value)
        return output

    def _hit(self, chunk_id: str, rank: int, score: float) -> dict[str, Any]:
        chunk = self.chunks[chunk_id]
        value = chunk.to_dict()
        return {
            "rank": rank, "score": score, "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
            "source": chunk.source, "relative_path": chunk.relative_path, "strategy": chunk.strategy,
            "text": chunk.text, "metadata": chunk.metadata, "token_count": chunk.token_count,
            "character_count": len(chunk.text), "chunk_config_fingerprint": self.chunk_config_fingerprint,
        }


def load_dense_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            key = (value["query_id"], value["strategy"])
            if key in result:
                raise ValueError(f"duplicate dense row {key}")
            result[key] = value
    return result
