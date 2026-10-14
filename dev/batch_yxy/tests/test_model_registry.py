"""模型画像注册表测试：15 个目标模型、状态诚实标注、计量契约。"""

import pytest

from sched_engine.errors import (
    CostExtractorUnavailableError,
    ModelNotVerifiedError,
)
from sched_engine.models import base as mb
from sched_engine.models.registry import (
    ALL_PROFILES,
    all_profiles,
    get_profile,
    with_token_counter,
)

EXPECTED = {
    # model_id: (family, cost_metric)
    "Qwen3-Embedding-0.6B": (mb.FAMILY_TEXT_EMBEDDING, mb.COST_METRIC_TOKEN_LENGTH),
    "BGE": (mb.FAMILY_TEXT_EMBEDDING, mb.COST_METRIC_TOKEN_LENGTH),
    "Jina-Embedding": (mb.FAMILY_TEXT_EMBEDDING, mb.COST_METRIC_TOKEN_LENGTH),
    "Qwen3-Reranker": (mb.FAMILY_TEXT_RERANK, mb.COST_METRIC_TOKEN_LENGTH),
    "BGE-Reranker": (mb.FAMILY_TEXT_RERANK, mb.COST_METRIC_TOKEN_LENGTH),
    "DIN": (mb.FAMILY_BEHAVIOR_SEQUENCE, mb.COST_METRIC_SEQUENCE_LENGTH),
    "DIEN": (mb.FAMILY_BEHAVIOR_SEQUENCE, mb.COST_METRIC_SEQUENCE_LENGTH),
    "BST": (mb.FAMILY_BEHAVIOR_SEQUENCE, mb.COST_METRIC_SEQUENCE_LENGTH),
    "DeepFM": (mb.FAMILY_SPARSE_REC, mb.COST_METRIC_NNZ),
    "DCN": (mb.FAMILY_SPARSE_REC, mb.COST_METRIC_NNZ),
    "ESMM": (mb.FAMILY_SPARSE_REC, mb.COST_METRIC_SAMPLE_COUNT),
    "Wide & Deep": (mb.FAMILY_SPARSE_REC, mb.COST_METRIC_NNZ),
    "DLRM": (mb.FAMILY_SPARSE_REC, mb.COST_METRIC_NNZ),
    "PLE": (mb.FAMILY_SPARSE_REC, mb.COST_METRIC_SAMPLE_COUNT),
    "MMoE": (mb.FAMILY_SPARSE_REC, mb.COST_METRIC_SAMPLE_COUNT),
}


def test_registry_has_exactly_15_target_models():
    assert len(ALL_PROFILES) == 15
    assert len({p.model_id for p in ALL_PROFILES}) == 15
    assert {p.model_id for p in ALL_PROFILES} == set(EXPECTED)


@pytest.mark.parametrize("model_id", sorted(EXPECTED))
def test_family_and_cost_metric_declarations(model_id):
    profile = get_profile(model_id)
    family, metric = EXPECTED[model_id]
    assert profile.family == family
    assert profile.cost_metric == metric
    assert profile.feature_schema, "画像必须声明特征字段"


def test_exactly_one_verified_model_and_honest_status():
    verified = [p for p in ALL_PROFILES if p.status == mb.STATUS_VERIFIED]
    assert [p.model_id for p in verified] == ["Qwen3-Embedding-0.6B"]
    assert verified[0].verified_on == ("vllm-embed+ascend-910c",)
    for profile in ALL_PROFILES:
        assert profile.status in (mb.STATUS_VERIFIED, mb.STATUS_DECLARED)
        if profile.status == mb.STATUS_DECLARED:
            assert profile.verified_on == ()
            assert profile.token_counter is None


def test_declared_model_extract_cost_raises():
    with pytest.raises(CostExtractorUnavailableError):
        get_profile("DIN").extract_cost({"sequence_length": 50})


def test_with_token_counter_only_for_verified():
    with pytest.raises(ModelNotVerifiedError):
        with_token_counter("DIN", lambda payload: 1)
    profile = with_token_counter(
        "Qwen3-Embedding-0.6B", lambda text: len(str(text).split())
    )
    assert profile.status == mb.STATUS_VERIFIED
    assert profile.extract_cost("a b c") == 3


def test_get_profile_unknown_returns_none():
    assert get_profile("no-such-model") is None
    assert len(all_profiles()) == 15
