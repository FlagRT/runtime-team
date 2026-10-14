"""设备画像注册表与预算钳制测试：6 类芯片、状态诚实标注、min 钳制与冲突。"""

import pytest

from dynamic_batching.errors import PolicyValidationError
from sched_engine.core.engine import clamp_policy
from sched_engine.devices import base as db
from sched_engine.devices.registry import (
    ALL_DEVICES,
    all_devices,
    get_device,
)
from sched_helpers import MODEL_ID, build_engine, make_engine_policy, make_sched

EXPECTED_VENDORS = {
    "ascend-910c": "华为昇腾",
    "kunlun": "昆仑芯",
    "cambricon": "寒武纪",
    "thead": "平头哥",
    "biren": "壁仞",
    "enflame": "燧原",
}


def test_registry_has_six_chip_families():
    assert len(ALL_DEVICES) == 6
    assert {d.device_id for d in ALL_DEVICES} == set(EXPECTED_VENDORS)
    assert {d.vendor for d in ALL_DEVICES} == set(EXPECTED_VENDORS.values())


def test_only_910c_verified_with_concrete_budgets():
    verified = [d for d in ALL_DEVICES if d.status == db.STATUS_VERIFIED]
    assert [d.device_id for d in verified] == ["ascend-910c"]
    dev = get_device("ascend-910c")
    assert dev.max_batch_size == 32
    assert dev.max_total_tokens == 32768
    assert dev.max_sequence_length == 4096
    assert dev.verified_with == ("Qwen3-Embedding-0.6B@vllm-embed",)


def test_pending_devices_have_no_budget_values():
    """待适配芯片：预算字段必须为空占位，不得因有接口就带出'可用'数值。"""
    for dev in ALL_DEVICES:
        if dev.device_id == "ascend-910c":
            continue
        assert dev.status == db.STATUS_PENDING
        assert dev.max_batch_size is None
        assert dev.max_total_tokens is None
        assert dev.max_sequence_length is None
        assert dev.verified_with == ()


def test_clamp_policy_takes_min_of_policy_and_device():
    dev = get_device("ascend-910c")
    policy = make_engine_policy(max_batch_size=64, max_total_tokens=65536)
    clamped = clamp_policy(policy, dev)
    assert clamped.max_batch_size == 32
    assert clamped.max_total_tokens == 32768
    # 设备字段 None（不设限）时不改变策略
    loose = db.DeviceProfile(
        device_id="loose", vendor="x", status=db.STATUS_VERIFIED,
        max_batch_size=None, max_total_tokens=None, max_sequence_length=None,
    )
    assert clamp_policy(policy, loose) is policy


def test_clamp_policy_conflict_raises_when_budget_below_max_bucket():
    dev = db.DeviceProfile(
        device_id="small", vendor="x", status=db.STATUS_VERIFIED,
        max_batch_size=8, max_total_tokens=100, max_sequence_length=512,
    )
    policy = make_engine_policy()  # 最大桶上界 512
    from sched_engine.errors import DeviceBudgetConflictError

    with pytest.raises(DeviceBudgetConflictError):
        clamp_policy(policy, dev)


def test_engine_rejects_pending_device():
    from sched_engine.errors import DeviceNotVerifiedError

    from dynamic_batching import FakeEmbedExecutor
    from sched_engine import SchedulerEngine, VllmEmbedFrameworkAdapter
    from sched_engine.models.registry import QWEN3_EMBEDDING_0_6B

    with pytest.raises(DeviceNotVerifiedError):
        SchedulerEngine(
            model_profiles={MODEL_ID: QWEN3_EMBEDDING_0_6B},
            framework_adapter=VllmEmbedFrameworkAdapter(FakeEmbedExecutor()),
            device_profile=get_device("kunlun"),
            policies=make_engine_policy(),
        )


def test_engine_clamps_policy_to_device_batch_limit():
    """策略批量 64 被设备画像 910C 钳制到 32：满批记录不超过 32。"""
    engine, _executor, _clock = build_engine(
        make_engine_policy(max_batch_size=64, max_total_tokens=65536, max_wait_ms=0.0)
    )
    engine.submit_many(
        make_sched(f"r{i:02d}", tokens=16, idx=i) for i in range(40)
    )
    report = engine.run()
    sizes = [rec.batch_size for rec in report.dispatch_records]
    assert max(sizes) <= 32
    assert sizes[0] == 32
    assert report.results.filled_count == 40


def test_invalid_policy_still_rejected_by_underlying_types():
    with pytest.raises(PolicyValidationError):
        make_engine_policy(bucket_upper_bounds=(512, 128))
