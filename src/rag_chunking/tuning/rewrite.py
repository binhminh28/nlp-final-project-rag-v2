"""Cached, fidelity-constrained retrieval query rewriting."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from rag_chunking.embedding.models import canonical_fingerprint
from rag_chunking.retrieval.models import normalize_query


REWRITE_SCHEMA_VERSION = "query_rewrite_v1"
REWRITE_CACHE_SCHEMA_VERSION = "query_rewrite_cache_v2"
REWRITE_PROMPT_VERSION = "retrieval_rewrite_v3"

SYSTEM_PROMPT = """Rewrite an Angular technical-documentation search query for retrieval.
Copy every API name, error code, code symbol, decorator, function-call token, and constraint
exactly as written. Do not add an API or code token absent from the original. Never convert
one Angular API style into another (for example, input() must not become @Input()).
Produce one concise search query, no longer than the original plus five words.
Do not answer the query. Do not introduce facts, technologies, or assumptions absent from it.
Return exactly one JSON object with one string field named rewrite."""


@dataclass(frozen=True, slots=True)
class RewriteConfig:
    provider: str = "openrouter"
    model: str = "deepseek/deepseek-v4-flash-0731:nitro"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    max_output_tokens: int = 512
    reasoning_max_tokens: int = 128
    max_empty_retries: int = 1
    prompt_version: str = REWRITE_PROMPT_VERSION
    schema_version: str = REWRITE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.provider or not self.model or not self.base_url:
            raise ValueError("rewrite provider identity must be non-empty")
        if self.max_output_tokens <= self.reasoning_max_tokens or self.reasoning_max_tokens < 0:
            raise ValueError("rewrite output budget must exceed reasoning budget")
        if self.max_empty_retries < 0 or self.temperature != 0:
            raise ValueError("rewrite uses positive output budget and temperature zero")

    def identity(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.identity())


class QueryRewriter(Protocol):
    calls: int
    input_tokens: int
    output_tokens: int

    def rewrite(self, query: str, config: RewriteConfig) -> str: ...


class RewriteFidelityError(ValueError):
    def __init__(self, message: str, proposed_rewrite: str) -> None:
        super().__init__(message)
        self.proposed_rewrite = proposed_rewrite


def parse_rewrite(value: str, original_query: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("rewrite response must be strict JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != {"rewrite"}:
        raise ValueError("rewrite response must contain only the rewrite field")
    rewritten = normalize_query(parsed["rewrite"])
    if len(rewritten) > 1000:
        raise ValueError("rewrite is unreasonably long")
    if rewritten == normalize_query(original_query):
        return rewritten
    validate_rewrite_fidelity(original_query, rewritten)
    return rewritten


def protected_tokens(query: str) -> set[str]:
    patterns = (
        r"\[\([^\]]+\)\]", r"@[A-Za-z_$][\w$]*", r"\bNG\d+\b",
        r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?\(\)",
        r"\b(?:[A-Z]{2,}[A-Za-z0-9_$]*|[A-Z][a-z0-9_$]+[A-Z][\w$]*|[a-z_$][\w$]*[A-Z][\w$]*)\b",
        r"\b[\w-]+\.(?:json|md|ts|html|css)\b",
    )
    return {match.group(0) for pattern in patterns for match in re.finditer(pattern, query)}


def validate_rewrite_fidelity(original_query: str, rewritten_query: str) -> None:
    original_tokens = protected_tokens(original_query)
    rewrite_tokens = protected_tokens(rewritten_query)
    missing = sorted(token for token in original_tokens if token not in rewritten_query)
    if missing:
        raise RewriteFidelityError(f"rewrite changed or removed protected query tokens: {missing}", rewritten_query)
    added = sorted(rewrite_tokens - original_tokens)
    if added:
        raise RewriteFidelityError(f"rewrite introduced protected query tokens: {added}", rewritten_query)
    if len(rewritten_query.split()) > len(normalize_query(original_query).split()) + 5:
        raise RewriteFidelityError("rewrite exceeds concise word-count fidelity bound", rewritten_query)


class OpenRouterQueryRewriter:
    def __init__(self, api_key: str | None = None, timeout_seconds: float = 60.0) -> None:
        configured = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._api_key = configured.strip() if configured else None
        self.timeout_seconds = timeout_seconds
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.latencies: list[float] = []

    def rewrite(self, query: str, config: RewriteConfig) -> str:
        if config.provider != "openrouter" or not self._api_key:
            raise ValueError("configured OpenRouter key is required for query rewriting")
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": normalize_query(query)},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if config.reasoning_max_tokens:
            payload["reasoning"] = {"max_tokens": config.reasoning_max_tokens, "exclude": True}
        request = urllib.request.Request(
            f"{config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        for attempt in range(config.max_empty_retries + 1):
            started = time.monotonic()
            self.calls += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"query rewrite failed with HTTP {error.code}") from error
            except (urllib.error.URLError, TimeoutError) as error:
                raise RuntimeError("query rewrite connection failed") from error
            finally:
                self.latencies.append(time.monotonic() - started)
            try:
                content = envelope["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise ValueError("rewrite response has no assistant content") from error
            usage = envelope.get("usage", {})
            if type(usage.get("prompt_tokens")) is int:
                self.input_tokens += usage["prompt_tokens"]
            if type(usage.get("completion_tokens")) is int:
                self.output_tokens += usage["completion_tokens"]
            if isinstance(content, str) and content.strip():
                return parse_rewrite(content, query)
            if attempt >= config.max_empty_retries:
                raise ValueError("rewrite response remained empty after bounded retry")
        raise RuntimeError("rewrite retry loop ended unexpectedly")


class RewriteCache:
    def __init__(self, directory: Path, config: RewriteConfig) -> None:
        self.directory = directory / config.fingerprint
        self.config = config

    def key(self, query: str) -> str:
        return canonical_fingerprint({
            "cache_schema_version": REWRITE_CACHE_SCHEMA_VERSION,
            "rewrite_config_fingerprint": self.config.fingerprint,
            "original_query": normalize_query(query),
        })

    def get_record(self, query: str) -> dict[str, str] | None:
        path = self.directory / f"{self.key(query)}.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "cache_schema_version": REWRITE_CACHE_SCHEMA_VERSION,
            "cache_key": self.key(query),
            "rewrite_config_fingerprint": self.config.fingerprint,
            "original_query_sha256": hashlib.sha256(normalize_query(query).encode()).hexdigest(),
        }
        if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
            raise ValueError(f"corrupt rewrite cache identity: {path}")
        rewrite = normalize_query(value.get("rewrite"))
        status = value.get("status")
        if status not in {"rewritten", "unchanged", "rejected_fidelity"}:
            raise ValueError(f"corrupt rewrite cache status: {path}")
        if status != "rejected_fidelity":
            validate_rewrite_fidelity(query, rewrite)
        elif rewrite != normalize_query(query):
            raise ValueError(f"rejected rewrite cache must fall back to original: {path}")
        proposed = value.get("proposed_rewrite")
        if not isinstance(proposed, str) or not proposed:
            raise ValueError(f"corrupt proposed rewrite: {path}")
        return {"rewrite": rewrite, "status": status, "proposed_rewrite": proposed}

    def get(self, query: str) -> str | None:
        record = self.get_record(query)
        return None if record is None else record["rewrite"]

    def put(self, query: str, rewrite: str, *, status: str, proposed_rewrite: str | None = None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        key = self.key(query)
        value = {
            "cache_schema_version": REWRITE_CACHE_SCHEMA_VERSION,
            "cache_key": key, "rewrite_config_fingerprint": self.config.fingerprint,
            "original_query_sha256": hashlib.sha256(normalize_query(query).encode()).hexdigest(),
            "original_query": normalize_query(query), "rewrite": normalize_query(rewrite),
            "status": status, "proposed_rewrite": normalize_query(proposed_rewrite or rewrite),
        }
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        handle, temporary = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / f"{key}.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def prepare_rewrites(
    queries: list[tuple[str, str]], config: RewriteConfig, cache: RewriteCache,
    provider: QueryRewriter, limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = queries[:limit] if limit is not None else queries
    records = []
    hits = misses = rejected = 0
    for query_id, query in selected:
        cached = cache.get_record(query)
        if cached is None:
            misses += 1
            try:
                rewrite = provider.rewrite(query, config)
                status = "unchanged" if rewrite == normalize_query(query) else "rewritten"
                proposed = rewrite
            except RewriteFidelityError as error:
                rewrite = normalize_query(query)
                proposed = error.proposed_rewrite
                status = "rejected_fidelity"
                rejected += 1
            cache.put(query, rewrite, status=status, proposed_rewrite=proposed)
        else:
            hits += 1
            rewrite = cached["rewrite"]
            proposed = cached["proposed_rewrite"]
            status = cached["status"]
            rejected += int(status == "rejected_fidelity")
        records.append({
            "query_id": query_id, "original_query": normalize_query(query), "rewritten_query": rewrite,
            "proposed_rewrite": proposed, "status": status,
            "rewrite_config_fingerprint": config.fingerprint,
        })
    return records, {
        "cache_hits": hits, "cache_misses": misses, "provider_calls": provider.calls,
        "input_tokens": provider.input_tokens, "output_tokens": provider.output_tokens,
        "fidelity_rejections": rejected,
    }
