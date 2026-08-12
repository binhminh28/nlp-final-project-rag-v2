"""Read-only canonical benchmark readiness checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from rag_chunking.chunking.writer import read_chunks_jsonl
from rag_chunking.context import ContextConfig
from rag_chunking.data.models import NORMALIZED_SCHEMA_VERSION
from rag_chunking.data.writer import read_documents_jsonl
from rag_chunking.embedding.config import load_embedding_config
from rag_chunking.embedding.models import (
    EMBEDDING_SCHEMA_VERSION, INDEX_SCHEMA_VERSION, canonical_fingerprint,
    index_identity,
)
from rag_chunking.evaluation.answer_models import EvaluationConfig
from rag_chunking.evaluation.qa_dataset import (
    QA_DATASET_SCHEMA_VERSION, is_team_qa_dataset, validate_canonical_qa_dataset,
)
from rag_chunking.evaluation.compatibility import audit_dataset_compatibility
from rag_chunking.generation import GenerationConfig
from rag_chunking.retrieval import RetrievalProtocolConfig


CANONICAL_STRATEGIES = ("fixed_size", "structure_aware", "prompt_based")


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkReadinessReport:
    ready: bool
    checks: tuple[ReadinessCheck, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checks": [item.to_dict() for item in self.checks],
            "blockers": list(self.blockers), "warnings": list(self.warnings),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(block.count(b"\n") for block in iter(lambda: stream.read(1024 * 1024), b""))


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _validate_index_payload(path: Path, manifest: dict[str, Any], expected_chunk_ids: set[str]) -> None:
    seen: set[str] = set()
    dimension = manifest["dimension"]
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                vector = entry["vector"]
                payload = entry["payload"]
                chunk_id = payload["chunk_id"]
                if not isinstance(vector, list) or len(vector) != dimension:
                    raise ValueError("vector dimension mismatch")
                if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
                    raise ValueError("vector contains invalid values")
                if payload.get("strategy") != manifest["strategy"]:
                    raise ValueError("payload strategy mismatch")
                expected_id = canonical_fingerprint({
                    "chunk_id": chunk_id,
                    "chunk_config_fingerprint": payload["chunk_config_fingerprint"],
                    "embedding_config_fingerprint": payload["embedding_config_fingerprint"],
                })
                if entry.get("index_id") != expected_id:
                    raise ValueError("index ID mismatch")
                if chunk_id in seen:
                    raise ValueError("duplicate indexed chunk ID")
                seen.add(chunk_id)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid index entry at {path}:{line_number}: {error}") from error
    if seen != expected_chunk_ids:
        raise ValueError("indexed chunk IDs do not match the canonical chunk artifact")


def _validate_strategy(
    strategy: str, *, corpus: str, documents_path: Path, document_ids: set[str],
    chunks_root: Path, embeddings_root: Path, indexes_root: Path,
    embedding_fingerprint: str,
) -> dict[str, Any]:
    chunk_dir = chunks_root / corpus / strategy
    chunk_manifest_path = chunk_dir / "manifest.json"
    chunk_manifest = _load_object(chunk_manifest_path)
    chunks = read_chunks_jsonl(chunk_dir / "chunks.jsonl")
    if chunk_manifest.get("strategy") != strategy or chunk_manifest.get("chunks") != len(chunks):
        raise ValueError("chunk strategy/count does not match manifest")
    if chunk_manifest.get("source_schema_version") != NORMALIZED_SCHEMA_VERSION:
        raise ValueError("chunk source schema is incompatible")
    if Path(chunk_manifest.get("source_input", "")).as_posix() != documents_path.as_posix():
        raise ValueError("chunk source_input does not identify the canonical processed corpus")
    if {item.doc_id for item in chunks} != document_ids:
        raise ValueError("chunk document coverage does not match the canonical corpus")
    chunk_ids = {item.chunk_id for item in chunks}
    if len(chunk_ids) != len(chunks):
        raise ValueError("chunk artifact contains duplicate IDs")

    embedding_dir = embeddings_root / corpus / strategy / embedding_fingerprint
    embedding_manifest = _load_object(embedding_dir / "manifest.json")
    if embedding_manifest.get("schema_version") != EMBEDDING_SCHEMA_VERSION or embedding_manifest.get("complete") is not True:
        raise ValueError("embedding artifact schema/commit is invalid")
    expected_embedding = {
        "corpus": corpus, "chunk_strategy": strategy,
        "chunk_config_fingerprint": chunk_manifest.get("config_fingerprint"),
        "embedding_config_fingerprint": embedding_fingerprint,
        "chunk_count": len(chunks), "build_scope": "full",
    }
    if any(embedding_manifest.get(key) != value for key, value in expected_embedding.items()):
        raise ValueError("chunk/embedding lineage is incompatible")
    if embedding_manifest.get("chunk_manifest_sha256") != _sha256(chunk_manifest_path):
        raise ValueError("embedding artifact does not bind the current chunk manifest")
    embeddings_path = embedding_dir / "embeddings.jsonl"
    if _line_count(embeddings_path) != len(chunks):
        raise ValueError("embedding record count does not match chunks")
    if _sha256(embeddings_path) != embedding_manifest.get("embedding_artifact_fingerprint"):
        raise ValueError("embedding artifact fingerprint mismatch")

    index_dir = indexes_root / corpus / strategy / embedding_fingerprint
    index_manifest = _load_object(index_dir / "manifest.json")
    identity = index_identity(
        corpus=corpus, strategy=strategy,
        chunk_config_fingerprint=chunk_manifest["config_fingerprint"],
        embedding_config_fingerprint=embedding_fingerprint,
        embedding_artifact_fingerprint=embedding_manifest["embedding_artifact_fingerprint"],
    )
    if index_manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError("index schema is incompatible")
    if index_manifest.get("index_fingerprint") != canonical_fingerprint(identity):
        raise ValueError("index fingerprint/lineage mismatch")
    if index_manifest.get("vector_count") != len(chunks):
        raise ValueError("index vector count does not match chunks")
    _validate_index_payload(index_dir / "index.jsonl", index_manifest, chunk_ids)
    return {
        "chunks": len(chunks),
        "chunk_config_fingerprint": chunk_manifest["config_fingerprint"],
        "embedding_config_fingerprint": embedding_fingerprint,
        "embedding_artifact_fingerprint": embedding_manifest["embedding_artifact_fingerprint"],
        "index_fingerprint": index_manifest["index_fingerprint"],
    }


def run_benchmark_preflight(
    *, dataset_path: Path, corpus: str = "angular",
    processed_root: Path = Path("data/processed"),
    raw_root: Path = Path("data/raw"),
    chunks_root: Path = Path("data/chunks"),
    embeddings_root: Path = Path("data/embeddings"),
    indexes_root: Path = Path("data/indexes"),
    embedding_config_path: Path = Path("configs/embedding.yaml"),
    generation_config: GenerationConfig,
    evaluation_config: EvaluationConfig,
    protocol_config: RetrievalProtocolConfig,
    context_config: ContextConfig,
    output_paths: tuple[Path, ...] = (),
    require_live_credentials: bool = False,
) -> BenchmarkReadinessReport:
    checks: list[ReadinessCheck] = []
    blockers: list[str] = []
    warnings: list[str] = []

    documents_path = processed_root / corpus / "documents.jsonl"
    try:
        processed_manifest = _load_object(processed_root / corpus / "manifest.json")
        documents = read_documents_jsonl(documents_path)
        by_id = {item.doc_id: item for item in documents}
        if len(by_id) != len(documents):
            raise ValueError("duplicate canonical document IDs")
        if processed_manifest.get("schema_version") != NORMALIZED_SCHEMA_VERSION:
            raise ValueError("processed corpus schema is incompatible")
        if processed_manifest.get("statistics", {}).get("documents") != len(documents):
            raise ValueError("processed corpus manifest count mismatch")
        corpus_fingerprint = _sha256(documents_path)
        checks.append(ReadinessCheck(
            "processed_corpus", "PASS", "Canonical processed corpus is valid.",
            {"documents": len(documents), "corpus_fingerprint": corpus_fingerprint},
        ))
        audit = processed_manifest.get("audit", {})
        if audit.get("unresolved_code_references") or audit.get("documents_with_warnings"):
            message = (
                "Processed corpus retains declared non-blocking audit warnings: "
                f"documents_with_warnings={audit.get('documents_with_warnings', 0)}, "
                f"unresolved_code_references={audit.get('unresolved_code_references', 0)}."
            )
            warnings.append(message)
            checks.append(ReadinessCheck("processed_corpus_audit", "WARNING", message, dict(audit)))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        blockers.append(f"Processed corpus invalid: {error}")
        checks.append(ReadinessCheck("processed_corpus", "BLOCKED", str(error), {}))
        by_id = {}

    try:
        embedding = load_embedding_config(embedding_config_path)
        checks.append(ReadinessCheck(
            "embedding_config", "PASS", "Embedding configuration is valid.",
            {"fingerprint": embedding.fingerprint, "model": embedding.model, "dimension": embedding.dimension},
        ))
    except (OSError, UnicodeError, ValueError) as error:
        blockers.append(f"Embedding configuration invalid: {error}")
        checks.append(ReadinessCheck("embedding_config", "BLOCKED", str(error), {}))
        embedding = None

    if by_id and embedding is not None:
        for strategy in CANONICAL_STRATEGIES:
            try:
                details = _validate_strategy(
                    strategy, corpus=corpus, documents_path=documents_path,
                    document_ids=set(by_id), chunks_root=chunks_root,
                    embeddings_root=embeddings_root, indexes_root=indexes_root,
                    embedding_fingerprint=embedding.fingerprint,
                )
                checks.append(ReadinessCheck(
                    f"strategy:{strategy}", "PASS",
                    f"{strategy} chunks, embeddings, and index are compatible.", details,
                ))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
                blockers.append(f"{strategy} artifacts invalid: {error}")
                checks.append(ReadinessCheck(f"strategy:{strategy}", "BLOCKED", str(error), {}))

    if not dataset_path.is_file():
        message = "Canonical production QA dataset not yet available."
        blockers.append(message)
        checks.append(ReadinessCheck(
            "canonical_qa_dataset", "BLOCKED", message,
            {"expected_path": dataset_path.as_posix(), "schema_version": QA_DATASET_SCHEMA_VERSION},
        ))
    elif by_id:
        try:
            if is_team_qa_dataset(dataset_path):
                compatibility = audit_dataset_compatibility(
                    dataset_path=dataset_path, documents=list(by_id.values()),
                    chunks_by_strategy={
                        strategy: read_chunks_jsonl(
                            chunks_root / corpus / strategy / "chunks.jsonl"
                        ) for strategy in CANONICAL_STRATEGIES
                    },
                    chunk_manifests={
                        strategy: _load_object(
                            chunks_root / corpus / strategy / "manifest.json"
                        ) for strategy in CANONICAL_STRATEGIES
                    },
                    raw_root=raw_root / corpus,
                )
                dataset = compatibility.dataset
                if not compatibility.passed:
                    raise ValueError(
                        "dataset compatibility gate failed: "
                        + "; ".join(compatibility.report["gate_reasons"])
                    )
                report_warnings = [
                    message
                    for item in compatibility.report["warnings"]
                    for message in item["messages"]
                ]
                details = {
                    "queries": len(dataset.records), "fingerprint": dataset.fingerprint,
                    "schema_version": dataset.schema_version,
                    "compatibility_fingerprint": compatibility.report["compatibility_fingerprint"],
                    "warnings": report_warnings,
                }
            else:
                dataset, report = validate_canonical_qa_dataset(dataset_path, by_id)
                if not report.valid:
                    raise ValueError("; ".join(report.errors))
                report_warnings = report.warnings
                details = {
                    "queries": len(dataset.records), "fingerprint": dataset.fingerprint,
                    "schema_version": dataset.schema_version, "warnings": report_warnings,
                }
            checks.append(ReadinessCheck(
                "canonical_qa_dataset", "PASS", "Canonical QA dataset is valid.",
                details,
            ))
            warnings.extend(report_warnings)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            blockers.append(f"Canonical QA dataset invalid: {error}")
            checks.append(ReadinessCheck("canonical_qa_dataset", "BLOCKED", str(error), {}))

    config_details = {
        "generation_config_fingerprint": generation_config.fingerprint,
        "answer_prompt_version": generation_config.prompt_template_version,
        "evaluation_config_fingerprint": evaluation_config.fingerprint,
        "protocol_config_fingerprint": protocol_config.fingerprint,
        "context_config_fingerprint": context_config.fingerprint,
    }
    checks.append(ReadinessCheck(
        "benchmark_configs", "PASS", "Protocol, context, generation, and evaluation configs are valid.",
        config_details,
    ))
    if require_live_credentials and generation_config.provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        message = "OPENROUTER_API_KEY is required for the requested live generation preflight."
        blockers.append(message)
        checks.append(ReadinessCheck("live_credentials", "BLOCKED", message, {}))
    else:
        checks.append(ReadinessCheck(
            "live_credentials", "PASS",
            "No live credential is required for this offline preflight." if not require_live_credentials else "Required live credential is configured.",
            {},
        ))
    for path in output_paths:
        manifest = path / "manifest.json"
        if manifest.exists():
            message = f"Committed output already exists and must match the requested identity: {path}"
            warnings.append(message)
            checks.append(ReadinessCheck("output_safety", "WARNING", message, {"path": path.as_posix()}))
        else:
            checks.append(ReadinessCheck(
                "output_safety", "PASS", "No committed output conflict detected.",
                {"path": path.as_posix()},
            ))
    return BenchmarkReadinessReport(not blockers, tuple(checks), tuple(blockers), tuple(warnings))
