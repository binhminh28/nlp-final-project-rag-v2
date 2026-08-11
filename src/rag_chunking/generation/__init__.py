"""Deterministic answer-request construction and durable generation artifacts."""

from .artifacts import GenerationRunResult, run_generation
from .cache import GenerationCache, GenerationCacheError
from .models import (
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM_PROMPT,
    AnswerResult,
    GenerationConfig,
    GenerationInput,
    GenerationInputOverflowError,
    GenerationProviderError,
    InputTokenAccounting,
)
from .prompt import AnswerPrompt, AnswerPromptBuilder
from .provider import (
    DeterministicFakeGenerationProvider,
    GenerationProvider,
    OpenRouterGenerationProvider,
    ProviderResponse,
)
from .service import GenerationService

__all__ = [
    "ANSWER_PROMPT_VERSION", "ANSWER_SYSTEM_PROMPT", "AnswerPrompt", "AnswerPromptBuilder",
    "AnswerResult", "DeterministicFakeGenerationProvider", "GenerationCache",
    "GenerationCacheError", "GenerationConfig", "GenerationInput",
    "GenerationInputOverflowError", "GenerationProvider", "GenerationProviderError",
    "GenerationRunResult", "GenerationService", "InputTokenAccounting",
    "OpenRouterGenerationProvider", "ProviderResponse", "run_generation",
]
