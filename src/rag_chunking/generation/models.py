"""Immutable semantic contracts for answer generation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from rag_chunking.context.models import ContextResult
from rag_chunking.embedding.models import canonical_fingerprint


GENERATION_CONFIG_V1_SCHEMA_VERSION = "generation_config_v1"
GENERATION_CONFIG_SCHEMA_VERSION = "generation_config_v2"
GENERATION_INPUT_SCHEMA_VERSION = "generation_input_v1"
ANSWER_RESULT_SCHEMA_VERSION = "answer_result_v1"
ANSWER_PROMPT_VERSION = "answer_prompt_v1"
ANSWER_SYSTEM_PROMPT_VERSION = "answer_system_v1"
ANSWER_SYSTEM_PROMPT = (
    "Answer the question using only the supplied context. Remain faithful to the evidence. "
    "If the context is insufficient, state that clearly. Do not fabricate unsupported information."
)
CANONICAL_GENERATION_TOKENIZER = "tiktoken:cl100k_base"
TOKEN_ACCOUNTING_POLICY = "openai_chat_cl100k_base_3_per_message_1_priming_v1"


def _nonempty(value: object, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """All answer-affecting and bounded invocation semantics; credentials are runtime-only."""

    provider: str
    model: str
    temperature: float = 0.0
    top_p: float | None = None
    max_output_tokens: int = 512
    seed: int | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5
    response_format: str = "text"
    context_window_tokens: int = 8192
    tokenizer: str = CANONICAL_GENERATION_TOKENIZER
    token_accounting_policy: str = TOKEN_ACCOUNTING_POLICY
    prompt_template_version: str = ANSWER_PROMPT_VERSION
    system_prompt_version: str = ANSWER_SYSTEM_PROMPT_VERSION
    system_prompt: str = ANSWER_SYSTEM_PROMPT
    schema_version: str = GENERATION_CONFIG_V1_SCHEMA_VERSION
    reasoning_effort: str | None = None
    completion_integrity_policy: str = "allow_length"
    request_token_limit_parameter: str = "max_tokens"
    provider_routing: str = "openrouter_default"
    stop_sequences: tuple[str, ...] = ()
    response_handling_contract: str = "nonempty_text_v1"
    prepared_context_token_budget: int | None = None

    def __post_init__(self) -> None:
        for name in ("provider", "model", "system_prompt"):
            _nonempty(getattr(self, name), name)
        if self.schema_version not in {
            GENERATION_CONFIG_V1_SCHEMA_VERSION, GENERATION_CONFIG_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported generation config schema {self.schema_version!r}")
        _nonempty(self.prompt_template_version, "prompt_template_version")
        _nonempty(self.system_prompt_version, "system_prompt_version")
        if self.tokenizer != CANONICAL_GENERATION_TOKENIZER:
            raise ValueError(f"unsupported generation tokenizer {self.tokenizer!r}")
        if self.token_accounting_policy != TOKEN_ACCOUNTING_POLICY:
            raise ValueError(f"unsupported token accounting policy {self.token_accounting_policy!r}")
        if self.response_format != "text":
            raise ValueError("only plain text answer responses are supported")
        if type(self.temperature) not in (int, float) or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        if self.top_p is not None and (
            type(self.top_p) not in (int, float) or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be null or in (0, 1]")
        for name in ("max_output_tokens", "context_window_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_output_tokens > self.context_window_tokens:
            raise ValueError("max_output_tokens cannot exceed context_window_tokens")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be null or an integer")
        if type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if type(self.retry_backoff_seconds) not in (int, float) or not math.isfinite(self.retry_backoff_seconds) or self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be finite and non-negative")
        # JSON distinguishes 0 from 0.0 even though generation semantics do not.
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "retry_backoff_seconds", float(self.retry_backoff_seconds))
        if self.top_p is not None:
            object.__setattr__(self, "top_p", float(self.top_p))
        if self.reasoning_effort not in {None, "minimal", "low", "medium", "high"}:
            raise ValueError("reasoning_effort must be null, minimal, low, medium, or high")
        if self.completion_integrity_policy not in {"allow_length", "require_stop"}:
            raise ValueError("unsupported completion_integrity_policy")
        if self.request_token_limit_parameter != "max_tokens":
            raise ValueError("only the OpenRouter max_tokens request contract is supported")
        if self.provider_routing != "openrouter_default":
            raise ValueError("only default OpenRouter routing is supported")
        if isinstance(self.stop_sequences, list):
            object.__setattr__(self, "stop_sequences", tuple(self.stop_sequences))
        if (
            not isinstance(self.stop_sequences, tuple)
            or any(not isinstance(item, str) or not item for item in self.stop_sequences)
        ):
            raise ValueError("stop_sequences must contain only non-empty strings")
        if self.response_handling_contract not in {"nonempty_text_v1", "nonempty_text_require_stop_v2"}:
            raise ValueError("unsupported response_handling_contract")
        if self.prepared_context_token_budget is not None and (
            type(self.prepared_context_token_budget) is not int
            or self.prepared_context_token_budget <= 0
        ):
            raise ValueError("prepared_context_token_budget must be null or a positive integer")
        if self.schema_version == GENERATION_CONFIG_V1_SCHEMA_VERSION and any((
            self.reasoning_effort is not None,
            self.completion_integrity_policy != "allow_length",
            self.request_token_limit_parameter != "max_tokens",
            self.provider_routing != "openrouter_default",
            bool(self.stop_sequences),
            self.response_handling_contract != "nonempty_text_v1",
            self.prepared_context_token_budget is not None,
        )):
            raise ValueError("generation_config_v1 cannot use v2 generation semantics")
        if self.schema_version == GENERATION_CONFIG_SCHEMA_VERSION and (
            self.completion_integrity_policy != "require_stop"
            or self.response_handling_contract != "nonempty_text_require_stop_v2"
            or self.prepared_context_token_budget is None
        ):
            raise ValueError("generation_config_v2 requires explicit normal-completion integrity")

    def identity(self) -> dict[str, Any]:
        value = asdict(self)
        value["stop_sequences"] = list(self.stop_sequences)
        if self.schema_version == GENERATION_CONFIG_V1_SCHEMA_VERSION:
            for name in (
                "reasoning_effort", "completion_integrity_policy",
                "request_token_limit_parameter", "provider_routing", "stop_sequences",
                "response_handling_contract", "prepared_context_token_budget",
            ):
                value.pop(name)
        return value

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.identity())


@dataclass(frozen=True, slots=True)
class GenerationInput:
    """Gold-free handoff: exact question plus an authoritative ContextResult."""

    query_id: str
    question: str
    context: ContextResult
    generation_config_fingerprint: str
    schema_version: str = GENERATION_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.query_id, "query_id")
        _nonempty(self.question, "question")
        _nonempty(self.generation_config_fingerprint, "generation_config_fingerprint")
        if self.schema_version != GENERATION_INPUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported generation input schema {self.schema_version!r}")
        if not isinstance(self.context, ContextResult):
            raise ValueError("context must be a ContextResult")
        if self.query_id != self.context.query_id:
            raise ValueError("generation query_id does not match ContextResult")

    @classmethod
    def create(cls, query_id: str, question: str, context: ContextResult, config: GenerationConfig) -> "GenerationInput":
        return cls(query_id, question, context, config.fingerprint)


@dataclass(frozen=True, slots=True)
class InputTokenAccounting:
    context_tokens: int
    question_tokens: int
    system_instruction_tokens: int
    user_formatting_tokens: int
    chat_framing_tokens: int
    total_input_tokens: int
    policy: str = TOKEN_ACCOUNTING_POLICY
    provider_reported_input_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "context_tokens", "question_tokens", "system_instruction_tokens",
            "user_formatting_tokens", "chat_framing_tokens", "total_input_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.provider_reported_input_tokens is not None and (
            type(self.provider_reported_input_tokens) is not int or self.provider_reported_input_tokens < 0
        ):
            raise ValueError("provider_reported_input_tokens must be null or non-negative")
        if self.policy != TOKEN_ACCOUNTING_POLICY:
            raise ValueError(f"unsupported accounting policy {self.policy!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InputTokenAccounting":
        if not isinstance(value, dict):
            raise ValueError("input token accounting must be an object")
        allowed = {item.name for item in fields(cls)}
        if set(value) - allowed:
            raise ValueError("unknown input token accounting fields")
        return cls(**value)


def answer_result_fingerprint(
    *, prompt_fingerprint: str, generation_config_fingerprint: str,
    answer_text: str, status: str, finish_reason: str | None,
    provider: str, model: str, schema_version: str = ANSWER_RESULT_SCHEMA_VERSION,
) -> str:
    return canonical_fingerprint({
        "schema_version": schema_version,
        "prompt_fingerprint": prompt_fingerprint,
        "generation_config_fingerprint": generation_config_fingerprint,
        "answer_text": answer_text,
        "status": status,
        "finish_reason": finish_reason,
        "provider": provider,
        "model": model,
    })


@dataclass(frozen=True, slots=True)
class AnswerResult:
    query_id: str
    answer_text: str
    status: str
    context_fingerprint: str
    generation_config_fingerprint: str
    prompt_fingerprint: str
    result_fingerprint: str
    provider: str
    model: str
    input_tokens: InputTokenAccounting
    output_tokens: int | None
    finish_reason: str | None
    strategy: str
    context_config_fingerprint: str
    retrieval_config_fingerprint: str
    protocol_config_fingerprint: str
    embedding_config_fingerprint: str
    index_fingerprint: str
    dataset_fingerprint: str | None
    schema_version: str = ANSWER_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANSWER_RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported answer result schema {self.schema_version!r}")
        if self.status != "success":
            raise ValueError("successful AnswerResult requires status='success'")
        for name in (
            "query_id", "answer_text", "context_fingerprint", "generation_config_fingerprint",
            "prompt_fingerprint", "result_fingerprint", "provider", "model", "strategy",
            "context_config_fingerprint", "retrieval_config_fingerprint",
            "protocol_config_fingerprint", "embedding_config_fingerprint", "index_fingerprint",
        ):
            _nonempty(getattr(self, name), name)
        _nonempty(self.dataset_fingerprint, "dataset_fingerprint", optional=True)
        if not self.answer_text.strip():
            raise ValueError("answer_text must contain usable text")
        if not isinstance(self.input_tokens, InputTokenAccounting):
            raise ValueError("input_tokens must be InputTokenAccounting")
        if self.output_tokens is not None and (type(self.output_tokens) is not int or self.output_tokens < 0):
            raise ValueError("output_tokens must be null or non-negative")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise ValueError("finish_reason must be null or a string")
        expected = answer_result_fingerprint(
            prompt_fingerprint=self.prompt_fingerprint,
            generation_config_fingerprint=self.generation_config_fingerprint,
            answer_text=self.answer_text, status=self.status, finish_reason=self.finish_reason,
            provider=self.provider, model=self.model, schema_version=self.schema_version,
        )
        if self.result_fingerprint != expected:
            raise ValueError("result fingerprint does not match semantic answer contents")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["input_tokens"] = self.input_tokens.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnswerResult":
        if not isinstance(value, dict):
            raise ValueError("answer result must be an object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown answer result fields: {unknown}")
        data = dict(value)
        data["input_tokens"] = InputTokenAccounting.from_dict(data.get("input_tokens"))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class GenerationInputOverflowError(ValueError):
    query_id: str
    context_window_tokens: int
    input_tokens: int
    max_output_tokens: int
    context_tokens: int
    prompt_overhead_tokens: int
    context_fingerprint: str
    generation_config_fingerprint: str

    def __str__(self) -> str:
        return (
            f"generation input exceeds context window for {self.query_id}: "
            f"input={self.input_tokens} + output_allowance={self.max_output_tokens} > "
            f"window={self.context_window_tokens}; context={self.context_tokens}, "
            f"prompt_overhead={self.prompt_overhead_tokens}"
        )


@dataclass(frozen=True, slots=True)
class GenerationIntegrityError(ValueError):
    """A structurally valid response that did not complete normally."""

    query_id: str
    finish_reason: str | None
    output_tokens: int | None
    visible_content_length: int

    def __str__(self) -> str:
        return (
            f"generation did not complete normally for {self.query_id}: "
            f"finish_reason={self.finish_reason!r}, output_tokens={self.output_tokens}, "
            f"visible_content_length={self.visible_content_length}"
        )


class GenerationProviderError(RuntimeError):
    """Typed bounded provider failure safe for deterministic failure artifacts."""

    def __init__(self, message: str, *, retryable: bool, attempts: int = 1, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.attempts = attempts
        self.status_code = status_code
