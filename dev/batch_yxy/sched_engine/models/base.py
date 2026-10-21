"""Model scheduling metadata; importing this module never loads a model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..errors import CostExtractorUnavailableError

STATUS_VERIFIED = "verified"
STATUS_DECLARED = "declared"
FAMILY_TEXT_EMBEDDING = "text-embedding"
FAMILY_TEXT_RERANK = "text-rerank"
FAMILY_BEHAVIOR_SEQUENCE = "behavior-sequence"
FAMILY_SPARSE_REC = "sparse-rec"
COST_METRIC_TOKEN_LENGTH = "token_length"
COST_METRIC_SEQUENCE_LENGTH = "sequence_length"
COST_METRIC_NNZ = "nnz"
COST_METRIC_SAMPLE_COUNT = "sample_count"


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    family: str
    cost_metric: str
    feature_schema: tuple[str, ...]
    status: str = STATUS_DECLARED
    default_model_version: str = "default"
    verified_on: tuple[str, ...] = ()
    token_counter: Callable[[Any], int] | None = None
    repository: str | None = None
    task: str = "predict"
    output_kind: str = "scores"

    def extract_cost(self, payload: Any) -> int:
        if self.token_counter is None:
            raise CostExtractorUnavailableError(f"{self.model_id}: no bound cost counter")
        value = self.token_counter(payload)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("cost counter must return a positive integer")
        return value
