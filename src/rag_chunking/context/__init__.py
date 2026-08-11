"""Deterministic construction of generation-ready context from selected hits."""

from .builder import ContextBuilder, ContextOverflowError
from .models import (
    CONTEXT_CONFIG_SCHEMA_VERSION,
    CONTEXT_FORMAT_VERSION,
    CONTEXT_RESULT_SCHEMA_VERSION,
    ContextBuildInput,
    ContextConfig,
    ContextPiece,
    ContextResult,
)

__all__ = [
    "CONTEXT_CONFIG_SCHEMA_VERSION", "CONTEXT_FORMAT_VERSION",
    "CONTEXT_RESULT_SCHEMA_VERSION", "ContextBuildInput", "ContextBuilder",
    "ContextConfig", "ContextOverflowError", "ContextPiece", "ContextResult",
]
