"""Canonical dense-vector retrieval contracts and service."""

from .models import RetrievalConfig, RetrievalHit, RetrievalRequest, RetrievalResult
from .service import RetrievalService

__all__ = ["RetrievalConfig", "RetrievalHit", "RetrievalRequest", "RetrievalResult", "RetrievalService"]
