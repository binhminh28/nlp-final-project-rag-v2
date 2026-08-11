"""Canonical dense-vector retrieval contracts, service, and budget policies."""

from .protocols import (
    SAME_TOKEN_BUDGET,
    SAME_TOP_K,
    ProtocolSelection,
    RetrievalProtocolConfig,
    apply_retrieval_protocol,
)

__all__ = [
    "SAME_TOKEN_BUDGET", "SAME_TOP_K", "ProtocolSelection",
    "RetrievalProtocolConfig", "apply_retrieval_protocol",
]

from .models import RetrievalConfig, RetrievalHit, RetrievalRequest, RetrievalResult
from .service import RetrievalService

__all__ += [
    "RetrievalConfig", "RetrievalHit", "RetrievalRequest", "RetrievalResult",
    "RetrievalService",
]
