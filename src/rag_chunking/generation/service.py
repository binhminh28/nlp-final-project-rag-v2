"""Generation orchestration over the authoritative ContextResult handoff."""

from __future__ import annotations

from dataclasses import replace

from .cache import GenerationCache
from .models import (
    AnswerResult,
    GenerationConfig,
    GenerationInput,
    GenerationIntegrityError,
    GenerationInputOverflowError,
    answer_result_fingerprint,
)
from .prompt import AnswerPrompt, AnswerPromptBuilder
from .provider import GenerationProvider, ProviderResponse


class GenerationService:
    """Build, validate, cache, and invoke generation without any retrieval dependency."""

    def __init__(
        self, config: GenerationConfig, provider: GenerationProvider,
        *, cache: GenerationCache | None = None, prompt_builder: AnswerPromptBuilder | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.cache = cache
        self.prompt_builder = prompt_builder or AnswerPromptBuilder(config)
        self.cache_hits = 0
        self.cache_misses = 0

    def construct_prompt(self, generation_input: GenerationInput) -> AnswerPrompt:
        expected_context_budget = self.config.prepared_context_token_budget
        if (
            expected_context_budget is not None
            and generation_input.context.context_token_budget != expected_context_budget
        ):
            raise ValueError(
                "prepared context token budget does not match generation configuration"
            )
        prompt = self.prompt_builder.build(generation_input)
        accounting = prompt.input_tokens
        if accounting.total_input_tokens + self.config.max_output_tokens > self.config.context_window_tokens:
            raise GenerationInputOverflowError(
                query_id=generation_input.query_id,
                context_window_tokens=self.config.context_window_tokens,
                input_tokens=accounting.total_input_tokens,
                max_output_tokens=self.config.max_output_tokens,
                context_tokens=accounting.context_tokens,
                prompt_overhead_tokens=accounting.total_input_tokens - accounting.context_tokens,
                context_fingerprint=generation_input.context.context_fingerprint,
                generation_config_fingerprint=self.config.fingerprint,
            )
        return prompt

    def generate(self, generation_input: GenerationInput) -> AnswerResult:
        prompt = self.construct_prompt(generation_input)
        if self.cache is not None:
            cached = self.cache.get(prompt.prompt_fingerprint, self.config.fingerprint)
            if cached is not None:
                self.cache_hits += 1
                return self._with_current_lineage(cached, generation_input)
            self.cache_misses += 1
        set_context = getattr(self.provider, "set_diagnostic_context", None)
        if callable(set_context):
            set_context(generation_input.query_id, prompt.prompt_fingerprint)
        response = self.provider.complete(prompt.messages, self.config)
        if self.config.completion_integrity_policy == "require_stop" and response.finish_reason != "stop":
            raise GenerationIntegrityError(
                query_id=generation_input.query_id,
                finish_reason=response.finish_reason,
                output_tokens=response.output_tokens,
                visible_content_length=len(response.text),
            )
        result = self._result(generation_input, prompt, response)
        if self.cache is not None:
            self.cache.put(result)
        return result

    def _result(
        self, generation_input: GenerationInput, prompt: AnswerPrompt, response: ProviderResponse,
    ) -> AnswerResult:
        context = generation_input.context
        accounting = replace(
            prompt.input_tokens, provider_reported_input_tokens=response.input_tokens
        )
        values = dict(
            query_id=generation_input.query_id,
            answer_text=response.text,
            status="success",
            context_fingerprint=context.context_fingerprint,
            generation_config_fingerprint=self.config.fingerprint,
            prompt_fingerprint=prompt.prompt_fingerprint,
            result_fingerprint="pending",
            provider=self.config.provider,
            model=self.config.model,
            input_tokens=accounting,
            output_tokens=response.output_tokens,
            finish_reason=response.finish_reason,
            strategy=context.strategy,
            context_config_fingerprint=context.context_config_fingerprint,
            retrieval_config_fingerprint=context.retrieval_config_fingerprint,
            protocol_config_fingerprint=context.protocol_config_fingerprint,
            embedding_config_fingerprint=context.embedding_config_fingerprint,
            index_fingerprint=context.index_fingerprint,
            dataset_fingerprint=context.dataset_fingerprint,
        )
        values["result_fingerprint"] = answer_result_fingerprint(
            prompt_fingerprint=prompt.prompt_fingerprint,
            generation_config_fingerprint=self.config.fingerprint,
            answer_text=response.text, status="success", finish_reason=response.finish_reason,
            provider=self.config.provider, model=self.config.model,
        )
        return AnswerResult(**values)

    @staticmethod
    def _with_current_lineage(cached: AnswerResult, generation_input: GenerationInput) -> AnswerResult:
        context = generation_input.context
        return replace(
            cached,
            query_id=generation_input.query_id,
            context_fingerprint=context.context_fingerprint,
            strategy=context.strategy,
            context_config_fingerprint=context.context_config_fingerprint,
            retrieval_config_fingerprint=context.retrieval_config_fingerprint,
            protocol_config_fingerprint=context.protocol_config_fingerprint,
            embedding_config_fingerprint=context.embedding_config_fingerprint,
            index_fingerprint=context.index_fingerprint,
            dataset_fingerprint=context.dataset_fingerprint,
        )
