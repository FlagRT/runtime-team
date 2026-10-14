"""优先级与三项组批硬约束的边界行为测试（决策核心层）。

覆盖：
- 同档优先级取队规则（到达序、提交序决胜）明确可测试；
- 优先级不绕过：兼容键隔离、max_batch_size、max_total_tokens、max_wait；
- 三项组批约束边界：批量恰好/超 1、token 预算恰好/超 1、等待窗口含边界。
"""

from dynamic_batching import BatchCoordinator
from dynamic_batching.types import BatchRequest
from helpers import make_policy, make_request


def req(rid, tokens, arrival=0, priority=0, **kw):
    return make_request(rid, tokens, arrival=arrival, priority=priority, **kw)


# ---------------- 同档取队规则 ----------------

def test_same_priority_fifo_by_arrival_then_submit_seq():
    co = BatchCoordinator(make_policy(max_batch_size=8))
    co.submit(req("late", 16, arrival=100))
    co.submit(req("early", 16, arrival=1))
    co.submit(req("mid-a", 16, arrival=50))
    co.submit(req("mid-b", 16, arrival=50))  # 同优先级同到达 → 提交序决胜
    plan = co.flush()[0]
    assert plan.request_ids == ("early", "mid-a", "mid-b", "late")


def test_high_priority_selected_first_within_bucket():
    co = BatchCoordinator(make_policy(max_batch_size=8))
    co.submit(req("low-early", 16, arrival=0, priority=0))
    co.submit(req("high-late", 16, arrival=100, priority=1))
    plan = co.flush()[0]
    assert plan.request_ids == ("high-late", "low-early")


def test_full_batch_scan_prefers_high_priority_bucket():
    policy = make_policy(bucket_upper_bounds=(16, 128), max_batch_size=2)
    co = BatchCoordinator(policy)
    for rid in ("low0", "low1"):  # 桶0：低优先级、先到、满批
        co.submit(req(rid, 8, arrival=0, priority=0))
    for rid in ("high0", "high1"):  # 桶1：高优先级、后到、满批
        co.submit(req(rid, 64, arrival=100, priority=1))
    plans = co.poll_ready(100)
    assert len(plans) == 2
    assert plans[0].request_ids == ("high0", "high1")  # 高优先级桶先派发
    assert plans[0].dispatch_reason == "batch_limit"
    assert plans[1].request_ids == ("low0", "low1")


# ---------------- 约束 1：max_batch_size 边界 ----------------

def test_batch_size_exact_and_plus_one_with_priority():
    co = BatchCoordinator(make_policy(max_batch_size=2, max_wait_ms=1000.0))
    co.submit(req("h0", 16, arrival=0, priority=1))
    co.submit(req("h1", 16, arrival=1, priority=1))
    co.submit(req("low0", 16, arrival=2, priority=0))
    plans = co.poll_ready(2)
    assert len(plans) == 1  # 优先级不能把第 3 条塞进已满批次
    assert plans[0].request_ids == ("h0", "h1")
    assert plans[0].dispatch_reason == "batch_limit"
    assert co.pending_count == 1
    plans = co.poll_ready(2 + 1_000_000_000)
    assert plans[0].request_ids == ("low0",)
    assert plans[0].dispatch_reason == "wait_limit"


# ---------------- 约束 2：max_total_tokens 边界 ----------------

def test_token_budget_exact_fit_and_plus_one():
    policy = make_policy(
        bucket_upper_bounds=(16,), max_batch_size=8, max_total_tokens=32,
        max_wait_ms=1000.0,
    )
    co = BatchCoordinator(policy)
    co.submit(req("a", 16, arrival=0, priority=1))
    co.submit(req("b", 16, arrival=1, priority=0))
    plans = co.poll_ready(2)
    assert plans[0].request_ids == ("a", "b")  # 16+16 == 32 恰好
    assert plans[0].dispatch_reason == "token_limit"
    assert plans[0].total_tokens == 32
    # 33 > 32：第三条不能进本批（哪怕高优先级）
    co2 = BatchCoordinator(policy)
    co2.submit(req("a", 16, arrival=0, priority=1))
    co2.submit(req("b", 16, arrival=1, priority=0))
    co2.submit(req("c", 16, arrival=2, priority=1))
    plans2 = co2.poll_ready(2)
    assert plans2[0].request_ids == ("a", "c")
    assert plans2[0].total_tokens == 32
    assert co2.pending_count == 1


# ---------------- 约束 3：max_wait 边界（含反饥饿） ----------------

def test_wait_window_boundary_inclusive_with_priority():
    co = BatchCoordinator(make_policy(max_wait_ms=50.0))
    co.submit(req("old-low", 16, arrival=10_000_000_000, priority=0))
    co.submit(req("new-high", 16, arrival=10_000_000_000 + 40_000_000, priority=9))
    # 最早请求到达 +50ms - 1：不派发（高优先级不提前、也不延迟）
    assert co.poll_ready(10_000_000_000 + 50_000_000 - 1) == []
    plans = co.poll_ready(10_000_000_000 + 50_000_000)
    assert len(plans) == 1
    assert plans[0].dispatch_reason == "wait_limit"
    assert "old-low" in plans[0].request_ids


def test_priority_does_not_extend_low_priority_wait():
    # 低优先级老请求到期即随批发出（不被高优先级晚到者延长等待）；
    # 高优先级可与其同批（预算允许时），次序在前。
    co = BatchCoordinator(make_policy(max_wait_ms=50.0))
    co.submit(req("old-low", 16, arrival=0, priority=0))
    co.submit(req("new-high", 16, arrival=90_000_000, priority=5))
    plans = co.poll_ready(50_000_000)
    assert len(plans) == 1
    assert plans[0].dispatch_reason == "wait_limit"
    assert plans[0].request_ids == ("new-high", "old-low")
    assert co.pending_count == 0


def test_low_priority_old_request_not_blocked_by_late_high_priority():
    # max_batch_size=1：满批扫描按优先级先发 new-high；同一次 poll 内
    # old-low 也随即发出（单条即满批 batch_limit）—— 高优先级不阻塞低优先级出队
    co = BatchCoordinator(make_policy(max_wait_ms=50.0, max_batch_size=1))
    co.submit(req("old-low", 16, arrival=0, priority=0))
    co.submit(req("new-high", 16, arrival=10_000_000, priority=5))
    plans = co.poll_ready(50_000_000)
    assert [p.request_ids for p in plans] == [("new-high",), ("old-low",)]
    assert co.pending_count == 0


def test_next_wakeup_ignores_priority():
    co = BatchCoordinator(make_policy(max_wait_ms=100.0))
    co.submit(req("low-old", 16, arrival=1_000, priority=0))
    co.submit(req("high-new", 16, arrival=80_000_000, priority=7))
    assert co.next_wakeup_mono_ns() == 1_000 + 100_000_000  # 由最早到达决定


# ---------------- 兼容键隔离不受优先级影响 ----------------

def test_priority_never_crosses_compatibility_key():
    co = BatchCoordinator(
        [make_policy(), make_policy(model_id="other-model", policy_version="v2")]
    )
    co.submit(req("a0", 16, arrival=0, priority=0, model_id="qwen3-embedding-0.6b"))
    co.submit(req("b0", 16, arrival=100, priority=99, model_id="other-model"))
    co.submit(req("b1", 16, arrival=101, priority=99, model_id="other-model"))
    plans = co.flush()
    assert len(plans) == 2
    by_key = {p.compatibility_key: p for p in plans}
    assert by_key["qwen3-embedding-0.6b|v1.0|default"].request_ids == ("a0",)
    assert by_key["other-model|v1.0|default"].request_ids == ("b0", "b1")


def test_batch_request_priority_validation():
    co = BatchCoordinator(make_policy())
    bad = BatchRequest(
        request_id="r",
        model_id="qwen3-embedding-0.6b",
        model_version="v1.0",
        input_token_count=16,
        payload_ref="p",
        arrival_mono_ns=0,
        priority="high",  # 非法类型
    )
    try:
        co.submit(bad)
    except Exception as exc:
        assert "priority" in str(exc)
    else:
        raise AssertionError("priority 非法类型应显式报错")
