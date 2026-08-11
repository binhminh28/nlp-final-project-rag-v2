"""Read-only validation of committed answer-generation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.generation.artifacts import GENERATION_RUN_SCHEMA_VERSION
from rag_chunking.generation.models import AnswerResult

from .answer_models import GenerationFailure
from .qa_dataset import QADataset


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line, parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid {path.name} at line {line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"invalid {path.name} at line {line_number}: record must be an object")
        values.append(value)
    return values


@dataclass(frozen=True, slots=True)
class CommittedGenerationRun:
    directory: Path
    strategy: str
    run_fingerprint: str
    committed_identity: str
    generation_config_fingerprint: str
    expected_query_ids: tuple[str, ...]
    answers: tuple[AnswerResult, ...]
    failures: tuple[GenerationFailure, ...]
    manifest: dict[str, Any]


def load_committed_generation_run(
    directory: Path, dataset: QADataset, *, strategy: str,
) -> CommittedGenerationRun:
    """Load a manifest-committed run without calling any upstream service."""

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("canonical evaluation requires a committed generation manifest")
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid committed generation manifest") from error
    if manifest.get("schema_version") != GENERATION_RUN_SCHEMA_VERSION:
        raise ValueError(f"incompatible generation manifest schema {manifest.get('schema_version')!r}")
    if manifest.get("complete") is not True:
        raise ValueError("generation manifest is not a valid complete commit marker")
    manifest_dataset = manifest.get("dataset_fingerprint")
    if manifest_dataset is not None and manifest_dataset != dataset.fingerprint:
        raise ValueError("generation manifest dataset fingerprint mismatch")
    manifest_strategy = manifest.get("strategy")
    if manifest_strategy is not None and manifest_strategy != strategy:
        raise ValueError("generation manifest strategy mismatch")
    generation_config_fingerprint = manifest.get("generation_config_fingerprint")
    if not isinstance(generation_config_fingerprint, str) or not generation_config_fingerprint:
        raise ValueError("generation manifest lacks a generation config fingerprint")
    requests = manifest.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("generation manifest requests must be a non-empty list")
    request_ids: list[str] = []
    query_by_id = {record.id: record for record in dataset.records}
    for request in requests:
        if not isinstance(request, dict) or set(request) != {"query_id", "question", "context_fingerprint"}:
            raise ValueError("generation manifest request has an invalid schema")
        query_id = request["query_id"]
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("generation request query_id must be non-empty")
        if query_id not in query_by_id:
            raise ValueError(f"unexpected generation query ID {query_id!r}")
        if request["question"] != query_by_id[query_id].question:
            raise ValueError(f"generation question mismatch for {query_id!r}")
        if not isinstance(request["context_fingerprint"], str) or not request["context_fingerprint"]:
            raise ValueError("generation request context fingerprint must be non-empty")
        request_ids.append(query_id)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("generation manifest contains duplicate query IDs")
    if request_ids != sorted(request_ids):
        raise ValueError("generation manifest requests are not in canonical query order")
    run_identity = {
        "schema_version": GENERATION_RUN_SCHEMA_VERSION,
        "generation_config_fingerprint": generation_config_fingerprint,
        "requests": requests,
    }
    run_fingerprint = canonical_fingerprint(run_identity)
    if manifest.get("run_fingerprint") != run_fingerprint:
        raise ValueError("generation run fingerprint does not match manifest identity")

    try:
        raw_answers = _read_jsonl(directory / "answers.jsonl")
        raw_failures = _read_jsonl(directory / "failures.jsonl")
        stats = _read_json(directory / "stats.json")
        answers = tuple(AnswerResult.from_dict(value) for value in raw_answers)
        failures = tuple(GenerationFailure.from_dict(value) for value in raw_failures)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("committed generation artifact failed validation") from error
    answer_ids = [answer.query_id for answer in answers]
    failure_ids = [failure.query_id for failure in failures]
    combined = answer_ids + failure_ids
    if len(combined) != len(set(combined)):
        raise ValueError("generation artifacts contain duplicate query IDs")
    extras = sorted(set(combined) - set(request_ids))
    if extras:
        raise ValueError(f"generation artifacts contain unexpected query IDs: {extras}")
    request_contexts = {item["query_id"]: item["context_fingerprint"] for item in requests}
    for answer in answers:
        if answer.strategy != strategy:
            raise ValueError(f"answer strategy mismatch for {answer.query_id!r}")
        if answer.dataset_fingerprint != dataset.fingerprint:
            raise ValueError(f"dataset fingerprint mismatch for {answer.query_id!r}")
        if answer.generation_config_fingerprint != generation_config_fingerprint:
            raise ValueError(f"generation config fingerprint mismatch for {answer.query_id!r}")
        if answer.context_fingerprint != request_contexts[answer.query_id]:
            raise ValueError(f"context lineage mismatch for {answer.query_id!r}")
    if not answers and manifest_dataset is None:
        raise ValueError("generation run has no verifiable dataset fingerprint")
    for failure in failures:
        if failure.generation_config_fingerprint != generation_config_fingerprint:
            raise ValueError(f"generation failure config mismatch for {failure.query_id!r}")
        if failure.context_fingerprint != request_contexts[failure.query_id]:
            raise ValueError(f"generation failure context mismatch for {failure.query_id!r}")
    if manifest.get("answer_count") != len(answers) or manifest.get("failure_count") != len(failures):
        raise ValueError("generation manifest artifact counts do not match")
    expected_stats = {
        "expected_queries": len(request_ids),
        "successful_queries": len(answers),
        "failed_queries": len(failures),
    }
    if any(stats.get(name) != value for name, value in expected_stats.items()):
        raise ValueError("generation stats coverage does not match committed artifacts")
    if stats.get("complete") is not True:
        raise ValueError("generation stats are not marked complete")
    committed_identity = canonical_fingerprint({
        "manifest_identity": run_identity,
        "run_fingerprint": run_fingerprint,
        "answer_result_fingerprints": [answer.result_fingerprint for answer in answers],
        "failures": [
            {
                "query_id": item.query_id, "error_type": item.error_type,
                "context_fingerprint": item.context_fingerprint,
                "generation_config_fingerprint": item.generation_config_fingerprint,
            }
            for item in failures
        ],
    })
    return CommittedGenerationRun(
        directory=directory, strategy=strategy, run_fingerprint=run_fingerprint,
        committed_identity=committed_identity,
        generation_config_fingerprint=generation_config_fingerprint,
        expected_query_ids=tuple(request_ids), answers=answers, failures=failures,
        manifest=manifest,
    )
