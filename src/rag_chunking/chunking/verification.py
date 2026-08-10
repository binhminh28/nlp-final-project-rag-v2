"""Read-only verification gate for any Unified Chunk artifact directory."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_chunking.data.models import NormalizedDocument

from .models import Chunk
from .statistics import _distribution, _percentile
from .tokenizer import TiktokenTokenizer
from .writer import configuration_fingerprint, read_chunks_jsonl


@dataclass(slots=True)
class ArtifactVerification:
    strategy: str | None = None
    documents: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify_artifact_directory(
    output_dir: Path, documents: list[NormalizedDocument] | None = None
) -> ArtifactVerification:
    """Load one strategy without branching and verify downstream core invariants."""

    result = ArtifactVerification()
    try:
        manifest = _read_object(output_dir / "manifest.json")
        statistics = _read_object(output_dir / "stats.json")
        chunks: list[Chunk] = read_chunks_jsonl(output_dir / "chunks.jsonl")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        result.errors.append(str(error))
        return result

    result.strategy = manifest.get("strategy") if isinstance(manifest.get("strategy"), str) else None
    result.documents = len(documents) if documents is not None else int(manifest.get("documents", 0))
    result.chunks = len(chunks)
    if result.strategy is None:
        result.errors.append("manifest has no valid strategy")
    if manifest.get("schema_version") != 1:
        result.errors.append("unsupported unified chunk schema_version")
    if manifest.get("chunks") != len(chunks) or statistics.get("chunks") != len(chunks):
        result.errors.append("manifest/stats chunk count does not match chunks.jsonl")
    if documents is not None:
        if manifest.get("documents") != len(documents) or statistics.get("documents") != len(documents):
            result.errors.append("manifest/stats document count does not match source corpus")
    elif manifest.get("documents") != statistics.get("documents"):
        result.errors.append("manifest and stats document counts disagree")

    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        result.errors.append("manifest has no canonical configuration object")
    elif manifest.get("config_fingerprint") != configuration_fingerprint(configuration):
        result.errors.append("manifest config_fingerprint does not match configuration")
    else:
        for key, value in configuration.items():
            if key in manifest and manifest[key] != value:
                result.errors.append(f"manifest {key} disagrees with canonical configuration")

    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        result.errors.append("duplicate chunk IDs")
    by_document: dict[str, list[Chunk]] = defaultdict(list)
    seen_documents: set[str] = set()
    current_document: str | None = None
    tokenizer_name = manifest.get("tokenizer")
    tokenizer: TiktokenTokenizer | None = None
    if isinstance(tokenizer_name, str) and tokenizer_name.startswith("tiktoken:"):
        try:
            tokenizer = TiktokenTokenizer(tokenizer_name.split(":", 1)[1])
        except ValueError as error:
            result.errors.append(f"invalid manifest tokenizer: {error}")
    else:
        result.errors.append("manifest has no canonical tokenizer")

    document_map = {document.doc_id: document for document in documents or []}
    for chunk in chunks:
        by_document[chunk.doc_id].append(chunk)
        if chunk.doc_id != current_document:
            if chunk.doc_id in seen_documents:
                result.errors.append("serialized chunks are not grouped in stable document order")
                break
            if current_document is not None:
                seen_documents.add(current_document)
            current_document = chunk.doc_id
        if chunk.strategy != result.strategy:
            result.errors.append(f"chunk {chunk.chunk_id} has inconsistent strategy")
        if chunk.tokenizer != tokenizer_name:
            result.errors.append(f"chunk {chunk.chunk_id} has inconsistent tokenizer")
        if tokenizer is not None and chunk.token_count != len(tokenizer.encode(chunk.text)):
            result.errors.append(f"chunk {chunk.chunk_id} has stale token_count")
        if "\ufffd" in chunk.text:
            source = document_map.get(chunk.doc_id)
            source_contains = source is not None and any("\ufffd" in block.text for block in source.blocks)
            if not source_contains:
                result.errors.append(f"chunk {chunk.chunk_id} generated U+FFFD")
        source = document_map.get(chunk.doc_id)
        if source is not None and (
            chunk.source != source.source
            or chunk.relative_path != source.relative_path
            or chunk.metadata.get("source_sha256") != source.source_sha256
        ):
            result.errors.append(f"chunk {chunk.chunk_id} has stale source provenance")

    for doc_id, document_chunks in by_document.items():
        if [chunk.chunk_index for chunk in document_chunks] != list(range(len(document_chunks))):
            result.errors.append(f"{doc_id} has non-contiguous chunk ordering")

    token_values = [chunk.token_count for chunk in chunks]
    expected_tokens = _distribution(token_values)
    expected_tokens.update(
        {
            "p25": _percentile(token_values, 0.25),
            "p75": _percentile(token_values, 0.75),
            "p95": _percentile(token_values, 0.95),
        }
    )
    if statistics.get("tokens_per_chunk") != expected_tokens:
        result.errors.append("stats token distribution does not match chunks.jsonl")
    return result
