"""Canonical experiment configuration and identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rag_chunking.embedding.models import canonical_fingerprint


EXPERIMENT_SCHEMA_VERSION = "retrieval_tuning_experiment_v1"
EXPERIMENT_FAMILIES = frozenset({"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"})


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_id: str
    experiment_name: str
    experiment_family: str
    dataset_fingerprint: str
    retrieval_config_fingerprint: str
    embedding_config_fingerprint: str
    index_fingerprints: dict[str, str]
    candidate_depth: int
    ranking_method: str = "dense_cosine"
    query_transform: dict[str, Any] | None = None
    diversity: dict[str, Any] | None = None
    reranker: dict[str, Any] | None = None
    lexical_config: dict[str, Any] | None = None
    fusion_config: dict[str, Any] | None = None
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError("unsupported experiment schema")
        if self.experiment_family not in EXPERIMENT_FAMILIES:
            raise ValueError("unknown experiment family")
        if not self.experiment_id or not self.experiment_name:
            raise ValueError("experiment ID and name must be non-empty")
        if type(self.candidate_depth) is not int or self.candidate_depth <= 0:
            raise ValueError("candidate depth must be positive")
        if not self.index_fingerprints or any(not key or not value for key, value in self.index_fingerprints.items()):
            raise ValueError("index fingerprints must be non-empty")

    def identity(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.identity())

