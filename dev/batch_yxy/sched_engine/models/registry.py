"""Fifteen target contracts. Declared metadata is not backend certification.

The one legacy verified label preserves the historical Ascend API contract;
it does not certify the new node runtime or NVIDIA execution.
"""
from dataclasses import replace
from typing import Callable, Any

from ..errors import ModelNotVerifiedError
from .base import (
    ModelProfile, STATUS_VERIFIED, FAMILY_TEXT_EMBEDDING, FAMILY_TEXT_RERANK,
    FAMILY_BEHAVIOR_SEQUENCE, FAMILY_SPARSE_REC, COST_METRIC_TOKEN_LENGTH,
    COST_METRIC_SEQUENCE_LENGTH, COST_METRIC_NNZ, COST_METRIC_SAMPLE_COUNT,
)


def _profile(name, family, metric, repo=None, task="predict", output="scores"):
    return ModelProfile(name, family, metric, (metric,), repository=repo,
                        task=task, output_kind=output)


QWEN3_EMBEDDING_0_6B = replace(
    _profile("Qwen3-Embedding-0.6B", FAMILY_TEXT_EMBEDDING, COST_METRIC_TOKEN_LENGTH,
             "Qwen/Qwen3-Embedding-0.6B", "embed", "vector"),
    status=STATUS_VERIFIED, default_model_version="0.6b",
    verified_on=("vllm-embed+ascend-910c",),
)
ALL_PROFILES = (
    QWEN3_EMBEDDING_0_6B,
    _profile("BGE", FAMILY_TEXT_EMBEDDING, COST_METRIC_TOKEN_LENGTH,
             "BAAI/bge-large-zh-v1.5", "embed", "vector"),
    _profile("Jina-Embedding", FAMILY_TEXT_EMBEDDING, COST_METRIC_TOKEN_LENGTH,
             "jinaai/jina-embeddings-v3", "embed", "vector"),
    _profile("Qwen3-Reranker", FAMILY_TEXT_RERANK, COST_METRIC_TOKEN_LENGTH,
             "Qwen/Qwen3-Reranker-0.6B", "rerank"),
    _profile("BGE-Reranker", FAMILY_TEXT_RERANK, COST_METRIC_TOKEN_LENGTH,
             "BAAI/bge-reranker-large", "rerank"),
    *(_profile(n, FAMILY_BEHAVIOR_SEQUENCE, COST_METRIC_SEQUENCE_LENGTH)
      for n in ("DIN", "DIEN", "BST")),
    *(_profile(n, FAMILY_SPARSE_REC, COST_METRIC_NNZ)
      for n in ("DeepFM", "DCN", "Wide & Deep", "DLRM")),
    *(_profile(n, FAMILY_SPARSE_REC, COST_METRIC_SAMPLE_COUNT, output="multi-head")
      for n in ("ESMM", "PLE", "MMoE")),
)
_REGISTRY = {p.model_id: p for p in ALL_PROFILES}
DIN = _REGISTRY["DIN"]


def get_profile(model_id: str) -> ModelProfile | None:
    return _REGISTRY.get(model_id) or next(
        (p for p in ALL_PROFILES if p.repository == model_id), None)


def all_profiles() -> tuple[ModelProfile, ...]:
    return ALL_PROFILES


def with_token_counter(model_id: str, counter: Callable[[Any], int]) -> ModelProfile:
    profile = get_profile(model_id)
    if profile is None or profile.status != STATUS_VERIFIED:
        raise ModelNotVerifiedError(f"{model_id}: no verified legacy profile")
    return replace(profile, token_counter=counter)
