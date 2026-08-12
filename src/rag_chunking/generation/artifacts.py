"""Resumable generation runs with cache durability and manifest-last commitment."""

from __future__ import annotations

import fcntl
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_chunking.chunking.writer import serialize_json, write_artifact_set
from rag_chunking.embedding.models import canonical_fingerprint

from .models import (
    AnswerResult, GenerationInput, GenerationInputOverflowError,
    GenerationIntegrityError, GenerationProviderError,
)
from .service import GenerationService


GENERATION_RUN_SCHEMA_VERSION = "generation_run_v1"


@dataclass(frozen=True, slots=True)
class GenerationRunResult:
    output_directory: Path
    run_fingerprint: str
    answers: tuple[AnswerResult, ...]
    failures: tuple[dict[str, Any], ...]
    complete: bool
    stats: dict[str, Any]


def _jsonl(values: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for value in values
    )


def _identity(inputs: list[GenerationInput], service: GenerationService) -> dict[str, Any]:
    return {
        "schema_version": GENERATION_RUN_SCHEMA_VERSION,
        "generation_config_fingerprint": service.config.fingerprint,
        "requests": [
            {
                "query_id": item.query_id,
                "question": item.question,
                "context_fingerprint": item.context.context_fingerprint,
            }
            for item in inputs
        ],
    }


def _publish_partial(output: Path, answers: list[AnswerResult], failures: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    # A partial state deliberately has no manifest commit marker.
    write_artifact_set(output, {
        "answers.jsonl": _jsonl([
            answer.to_dict() for answer in sorted(answers, key=lambda item: item.query_id)
        ]),
        "failures.jsonl": _jsonl(sorted(failures, key=lambda item: item["query_id"])),
        "stats.json": serialize_json(stats),
    })


def run_generation(
    inputs: list[GenerationInput], service: GenerationService, output_directory: Path,
    *, max_concurrency: int = 1,
) -> GenerationRunResult:
    output_directory.mkdir(parents=True, exist_ok=True)
    lock_path = output_directory / ".generation-run.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                f"another generation process is writing {output_directory}"
            ) from error
        return _run_generation_locked(
            inputs, service, output_directory, max_concurrency=max_concurrency,
        )


def _run_generation_locked(
    inputs: list[GenerationInput], service: GenerationService, output_directory: Path,
    *, max_concurrency: int = 1,
) -> GenerationRunResult:
    if not inputs:
        raise ValueError("generation inputs must not be empty")
    if type(max_concurrency) is not int or max_concurrency <= 0:
        raise ValueError("max_concurrency must be a positive integer")
    if len({item.query_id for item in inputs}) != len(inputs):
        raise ValueError("generation inputs contain duplicate query IDs")
    for item in inputs:
        if item.generation_config_fingerprint != service.config.fingerprint:
            raise ValueError("generation input/config fingerprint mismatch")
    ordered = sorted(inputs, key=lambda item: item.query_id)
    identity = _identity(ordered, service)
    run_fingerprint = canonical_fingerprint(identity)
    existing_manifest = output_directory / "manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != run_fingerprint:
            raise ValueError("refusing to overwrite a committed generation run with a different identity")
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("committed generation manifest identity does not match requested run")
        if existing.get("complete") is not True:
            raise ValueError("generation manifest is not a valid commit marker")
        answer_path = output_directory / "answers.jsonl"
        stats_path = output_directory / "stats.json"
        try:
            answers = [
                AnswerResult.from_dict(json.loads(line))
                for line in answer_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            stored_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("committed generation artifact failed validation") from error
        if len(answers) != existing.get("answer_count") or len(answers) != len(ordered):
            raise ValueError("committed generation artifact count mismatch")
        if [answer.query_id for answer in answers] != [item.query_id for item in ordered]:
            raise ValueError("committed generation artifact query identity mismatch")
        if any(
            answer.generation_config_fingerprint != service.config.fingerprint
            or answer.context_fingerprint != item.context.context_fingerprint
            for answer, item in zip(answers, ordered)
        ):
            raise ValueError("committed generation artifact lineage mismatch")
        return GenerationRunResult(
            output_directory, run_fingerprint, tuple(answers), (), True, stored_stats
        )

    answers: list[AnswerResult] = []
    failures: list[dict[str, Any]] = []
    calls_before = service.provider.calls
    retries_before = service.provider.retries
    hits_before = service.cache_hits
    misses_before = service.cache_misses
    started = time.monotonic()

    def stats() -> dict[str, Any]:
        return {
            "expected_queries": len(ordered),
            "successful_queries": len(answers),
            "failed_queries": len(failures),
            "provider_calls": service.provider.calls - calls_before,
            "provider_retries": service.provider.retries - retries_before,
            "cache_hits": service.cache_hits - hits_before,
            "cache_misses": service.cache_misses - misses_before,
            "max_concurrency": max_concurrency,
            "elapsed_runtime_seconds": time.monotonic() - started,
            "complete": not failures and len(answers) == len(ordered),
        }

    def generate_one(item: GenerationInput) -> tuple[AnswerResult | None, dict[str, Any] | None]:
        try:
            return service.generate(item), None
        except (
            GenerationInputOverflowError, GenerationIntegrityError,
            GenerationProviderError, ValueError,
        ) as error:
            failure: dict[str, Any] = {
                "query_id": item.query_id,
                "context_fingerprint": item.context.context_fingerprint,
                "generation_config_fingerprint": service.config.fingerprint,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            if isinstance(error, GenerationProviderError):
                failure.update({
                    "retryable": error.retryable, "attempts": error.attempts,
                    "status_code": error.status_code,
                })
            if isinstance(error, GenerationIntegrityError):
                failure.update({
                    "finish_reason": error.finish_reason,
                    "output_tokens": error.output_tokens,
                    "visible_content_length": error.visible_content_length,
                })
            return None, failure

    try:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(generate_one, item): item for item in ordered}
            for future in as_completed(futures):
                answer, failure = future.result()
                if answer is not None:
                    answers.append(answer)
                if failure is not None:
                    failures.append(failure)
                _publish_partial(output_directory, answers, failures, stats())
    except BaseException:
        _publish_partial(output_directory, answers, failures, stats())
        raise

    answers.sort(key=lambda item: item.query_id)
    failures.sort(key=lambda item: item["query_id"])
    final_stats = stats()
    if not failures:
        manifest = {
            **identity,
            "run_fingerprint": run_fingerprint,
            "answer_count": len(answers),
            "failure_count": 0,
            "complete": True,
            "execution": {"max_concurrency": max_concurrency},
        }
        write_artifact_set(output_directory, {
            "answers.jsonl": _jsonl([answer.to_dict() for answer in answers]),
            "failures.jsonl": "",
            "stats.json": serialize_json(final_stats),
            "manifest.json": serialize_json(manifest),
        })
    return GenerationRunResult(
        output_directory, run_fingerprint, tuple(answers), tuple(failures),
        not failures, final_stats,
    )
