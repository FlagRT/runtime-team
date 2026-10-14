"""调度引擎端到端测试（可控时钟 + 模拟执行器，不依赖设备）。

场景：32 条不同 Token 长度请求 × 2 档优先级 × 3 种到达顺序，
覆盖入队、取队、分桶、组批、结果回填 5 个环节的校验；
含原路径/调度路径开关对照与可重复运行检查。
"""

import numpy as np
import pytest

from sched_helpers import (
    MODEL_ID,
    build_engine,
    make_engine_policy,
    make_sched,
    unique_text,
)
from sched_engine import PATH_LEGACY, PATH_SCHEDULED
from sched_engine.errors import EngineStateError

# 32 个不同 token 长度：覆盖 3 个桶（128/256/512 上界）与桶边界
# 128/129、256/257、511/512，最小 8、最大 512（= 策略最大已验证长度）。
LENGTHS = [
    8, 16, 24, 32, 48, 64, 96, 120, 127, 128, 129, 160, 192, 200, 240, 255,
    256, 257, 288, 320, 384, 448, 500, 511, 512, 72, 88, 144, 176, 272, 336, 400,
]
assert len(LENGTHS) == 32 and len(set(LENGTHS)) == 32


def arrival_inorder(i: int) -> int:
    """到达顺序 1：正序（按提交序递增）。"""
    return i * 1_000_000


def arrival_reverse(i: int) -> int:
    """到达顺序 2：逆序（晚提交的先到）。"""
    return (31 - i) * 1_000_000


def arrival_interleaved(i: int) -> int:
    """到达顺序 3：交错（i*17 mod 32，与 32 互素 ⇒ 到达时刻仍两两不同）。"""
    return ((i * 17) % 32) * 1_000_000


ARRIVAL_ORDERS = (arrival_inorder, arrival_reverse, arrival_interleaved)


def build_32_requests(arrival_fn) -> list:
    requests = []
    for i, tokens in enumerate(LENGTHS):
        requests.append(
            make_sched(
                f"r{i:02d}",
                arrival=arrival_fn(i),
                priority=1 if i % 2 else 0,  # 2 档优先级，各 16 条
                tokens=tokens,
                idx=i,
            )
        )
    return requests


def expected_dequeue_order(requests) -> list[str]:
    """取队规则：(-priority, arrival, 提交序) 全序。"""
    return [
        r.request_id
        for r in sorted(
            requests,
            key=lambda r: (-r.priority, r.arrival_mono_ns),
        )
    ]


@pytest.mark.parametrize("arrival_fn", ARRIVAL_ORDERS, ids=["inorder", "reverse", "interleaved"])
def test_32_requests_full_pipeline(arrival_fn):
    engine, executor, _clock = build_engine()
    requests = build_32_requests(arrival_fn)

    # 环节 1：入队（含兼容键隔离与状态查询）
    compat = engine.submit_many(requests)
    assert len(set(compat)) == 1  # 同模型/版本/profile → 单 lane
    status = engine.queue_status()
    lane = status[compat[0]]
    assert lane.depth == 32
    assert lane.by_priority == {1: 16, 0: 16}
    assert lane.oldest_arrival_mono_ns == min(r.arrival_mono_ns for r in requests)

    report = engine.run()

    # 环节 2：取队（确定序、可重复）
    assert report.dequeue_trace == expected_dequeue_order(requests)

    # 环节 3+4：分桶与组批（预算与桶一致性）
    records = report.dispatch_records
    assert records, "应至少产生一个批次"
    policy = make_engine_policy()
    union_ids: list[str] = []
    for rec in records:
        assert rec.batch_size <= policy.max_batch_size
        assert rec.total_tokens <= policy.max_total_tokens
        assert set(rec.priorities) <= {0, 1}
        assert len(set(rec.compatibility_key for _ in rec.request_ids)) == 1
        for rid, tokens in zip(rec.request_ids, rec.token_lengths):
            idx = int(rid[1:])
            assert tokens == LENGTHS[idx]
            assert policy.bucket_index(tokens) in rec.bucket_ids
        union_ids.extend(rec.request_ids)

    # 环节 5：结果回填 —— 恰好一次、无遗漏、无错位
    assert sorted(union_ids) == sorted(r.request_id for r in requests)
    assert len(union_ids) == len(set(union_ids))
    results = report.results.items()
    assert len(results) == 32
    for i, req in enumerate(requests):
        vec = results[req.request_id]
        # 向量由全文哈希确定性生成：ID ↔ 向量对应错一位即失败
        assert np.array_equal(vec, executor.vector(unique_text(i, LENGTHS[i])))
    report.results.assert_complete()

    # 5 环节计数齐全
    assert report.stage_counts["enqueued"] == 32
    assert report.stage_counts["dequeued"] == 32
    assert report.stage_counts["bucketed"] == 32
    assert report.stage_counts["completed"] == 32
    assert report.stage_counts["batched"] == len(records)
    assert report.stage_counts["executed"] == len(records)
    assert report.mode == PATH_SCHEDULED


@pytest.mark.parametrize("arrival_fn", ARRIVAL_ORDERS, ids=["inorder", "reverse", "interleaved"])
def test_dispatch_reasons_and_batch_limits(arrival_fn):
    """3 项组批约束在 32 请求场景下的体现：满批必发、尾批由等待/排空兜底。"""
    engine, _executor, _clock = build_engine()
    engine.submit_many(build_32_requests(arrival_fn))
    report = engine.run()
    reasons = {rec.dispatch_reason for rec in report.dispatch_records}
    assert "batch_limit" in reasons  # 桶内请求多于 max_batch_size=8 的桶必然满批发批
    sizes = [rec.batch_size for rec in report.dispatch_records]
    assert 1 <= min(sizes) <= max(sizes) <= 8


def test_repeatability_same_inputs_same_decisions():
    """相同输入（含可控时钟）重复运行 → 取队序与批次组成完全一致。"""
    traces = []
    compositions = []
    for _ in range(2):
        engine, _executor, _clock = build_engine()
        requests = build_32_requests(arrival_interleaved)
        engine.submit_many(requests)
        report = engine.run()
        traces.append(tuple(report.dequeue_trace))
        compositions.append(
            tuple(rec.request_ids for rec in report.dispatch_records)
        )
    assert traces[0] == traces[1]
    assert compositions[0] == compositions[1]


def test_path_switch_legacy_equals_scheduled_per_id():
    """原路径开关：同一执行器下 legacy 与 scheduled 每请求结果逐元素一致。"""
    from dynamic_batching import FakeEmbedExecutor

    requests = build_32_requests(arrival_interleaved)
    shared_executor = FakeEmbedExecutor(dim=1024)

    eng_legacy, _, _ = build_engine(executor=shared_executor, path_mode=PATH_LEGACY)
    eng_legacy.submit_many(requests)
    legacy_report = eng_legacy.run()
    assert legacy_report.mode == PATH_LEGACY
    assert legacy_report.dispatch_records == []  # 原路径不经组批
    assert legacy_report.results.filled_count == 32

    eng_sched, _, _ = build_engine(executor=shared_executor, path_mode=PATH_SCHEDULED)
    eng_sched.submit_many(requests)
    sched_report = eng_sched.run()
    assert sched_report.results.filled_count == 32

    for req in requests:
        assert np.array_equal(
            legacy_report.results.get(req.request_id),
            sched_report.results.get(req.request_id),
        )


def test_run_path_mode_override():
    engine, _executor, _clock = build_engine(path_mode=PATH_SCHEDULED)
    engine.submit_many(build_32_requests(arrival_inorder)[:5])
    report = engine.run(path_mode=PATH_LEGACY)
    assert report.mode == PATH_LEGACY


def test_engine_state_guards():
    engine, _executor, _clock = build_engine()
    with pytest.raises(EngineStateError):
        engine.run()  # 空队列拒绝运行
    engine.submit_many(build_32_requests(arrival_inorder)[:3])
    report = engine.run()
    assert report.results.filled_count == 3
    with pytest.raises(EngineStateError):
        engine.run()  # 单次会话不可重复 run
    with pytest.raises(EngineStateError):
        engine.submit(make_sched("extra"))  # run 后不可再入队


def test_engine_rejects_unknown_model_and_declared_model():
    from dynamic_batching.errors import UnknownPolicyError
    from sched_engine import SchedRequest
    from sched_engine.errors import ModelNotVerifiedError
    from sched_engine.models.registry import DIN

    engine, _executor, _clock = build_engine()
    ghost = SchedRequest(
        request_id="g0", model_id="ghost-model", payload="x", arrival_mono_ns=0
    )
    with pytest.raises(UnknownPolicyError):
        engine.submit(ghost)
    declared = SchedRequest(
        request_id="d0", model_id=DIN.model_id, payload="x", arrival_mono_ns=0
    )
    # 画像注册表外的引擎未登记 DIN，先以注册表内画像校验拒绝语义
    engine._profiles[DIN.model_id] = DIN
    with pytest.raises(ModelNotVerifiedError):
        engine.submit(declared)


def test_engine_uses_verified_profile_from_registry():
    from sched_engine.models.registry import get_profile

    engine, _executor, _clock = build_engine()
    assert get_profile(MODEL_ID).status == "verified"
    compat = engine.submit(make_sched("ok0"))
    assert compat == f"{MODEL_ID}|0.6b|default"
