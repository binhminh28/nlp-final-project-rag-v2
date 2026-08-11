"""Frozen, strategy-neutral answer prompt and deterministic token accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_chunking.chunking.tokenizer import TiktokenTokenizer
from rag_chunking.embedding.models import canonical_fingerprint

from .models import GenerationConfig, GenerationInput, InputTokenAccounting


USER_PREFIX = "Question:\n"
USER_MIDDLE = "\n\nContext:\n"
USER_SUFFIX = "\n\nAnswer:"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class AnswerPrompt:
    messages: tuple[ChatMessage, ChatMessage]
    prompt_fingerprint: str
    input_tokens: InputTokenAccounting

    def provider_messages(self) -> list[dict[str, str]]:
        return [message.to_dict() for message in self.messages]


class AnswerPromptBuilder:
    """Build only from exact question, exact rendered context, and frozen config."""

    _TOKENS_PER_MESSAGE = 3
    _ASSISTANT_PRIMING_TOKENS = 1

    def __init__(self, config: GenerationConfig, tokenizer: TiktokenTokenizer | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer or TiktokenTokenizer()
        if self.tokenizer.name != config.tokenizer:
            raise ValueError("tokenizer does not match GenerationConfig")

    def build(self, generation_input: GenerationInput) -> AnswerPrompt:
        if generation_input.generation_config_fingerprint != self.config.fingerprint:
            raise ValueError("generation input/config fingerprint mismatch")
        context = generation_input.context.rendered_context
        question = generation_input.question
        user_content = USER_PREFIX + question + USER_MIDDLE + context + USER_SUFFIX
        messages = (
            ChatMessage("system", self.config.system_prompt),
            ChatMessage("user", user_content),
        )
        context_tokens = len(self.tokenizer.encode(context))
        question_tokens = len(self.tokenizer.encode(question))
        system_tokens = len(self.tokenizer.encode(self.config.system_prompt))
        formatting_tokens = len(self.tokenizer.encode(USER_PREFIX + USER_MIDDLE + USER_SUFFIX))
        # Count exact role/content encodings plus a frozen OpenAI-compatible framing estimate.
        message_payload_tokens = sum(
            len(self.tokenizer.encode(message.role)) + len(self.tokenizer.encode(message.content))
            for message in messages
        )
        framing_tokens = len(messages) * self._TOKENS_PER_MESSAGE + self._ASSISTANT_PRIMING_TOKENS
        total = message_payload_tokens + framing_tokens
        accounting = InputTokenAccounting(
            context_tokens=context_tokens,
            question_tokens=question_tokens,
            system_instruction_tokens=system_tokens,
            user_formatting_tokens=formatting_tokens,
            chat_framing_tokens=framing_tokens + sum(len(self.tokenizer.encode(message.role)) for message in messages),
            total_input_tokens=total,
        )
        identity: dict[str, Any] = {
            "prompt_template_version": self.config.prompt_template_version,
            "messages": [message.to_dict() for message in messages],
        }
        return AnswerPrompt(messages, canonical_fingerprint(identity), accounting)
