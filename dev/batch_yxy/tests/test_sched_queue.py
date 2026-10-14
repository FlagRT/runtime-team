"""请求队列测试：入队 / 取队 / 状态查询 / 兼容键隔离 / 优先级规则。"""

import pytest

from dynamic_batching.errors import DuplicateRequestError, RequestValidationError
from sched_engine.core.queue import RequestQueue, UnknownLaneError
from sched_engine.core.request import SchedRequest


def mk(rid, arrival=0, priority=0):
    return SchedRequest(
        request_id=rid,
        model_id="m",
        payload=f"p-{rid}",
        arrival_mono_ns=arrival,
        priority=priority,
    )


def test_enqueue_and_lane_status_query():
    q = RequestQueue()
    q.enqueue(mk("a", arrival=10, priority=1), "compat-a")
    q.enqueue(mk("b", arrival=30, priority=0), "compat-a")
    q.enqueue(mk("c", arrival=20), "compat-b")
    assert q.depth == 3
    status = q.status()
    assert set(status) == {"compat-a", "compat-b"}
    lane_a = status["compat-a"]
    assert lane_a.depth == 2
    assert lane_a.by_priority == {1: 1, 0: 1}
    assert lane_a.head_request_id == "a"  # 优先级高者为队头
    assert lane_a.oldest_arrival_mono_ns == 10
    assert status["compat-b"].depth == 1


def test_dequeue_order_priority_then_arrival_then_submit_seq():
    q = RequestQueue()
    q.enqueue(mk("low-early", arrival=1, priority=0), "k")
    q.enqueue(mk("high-late", arrival=100, priority=5), "k")
    q.enqueue(mk("high-later", arrival=200, priority=5), "k")
    q.enqueue(mk("mid", arrival=50, priority=2), "k")
    q.enqueue(mk("low-late", arrival=300, priority=0), "k")
    out = q.dequeue("k")
    assert [r.request_id for r in out] == [
        "high-late",
        "high-later",
        "mid",
        "low-early",
        "low-late",
    ]


def test_same_priority_same_arrival_breaks_by_submit_seq():
    q = RequestQueue()
    for rid in ("first", "second", "third"):
        q.enqueue(mk(rid, arrival=42, priority=1), "k")
    assert [r.request_id for r in q.dequeue("k")] == ["first", "second", "third"]


def test_dequeue_isolated_per_lane_and_partial():
    q = RequestQueue()
    q.enqueue(mk("a1"), "k1")
    q.enqueue(mk("a2"), "k1")
    q.enqueue(mk("b1"), "k2")
    got = q.dequeue("k1", max_items=1)
    assert [r.request_id for r in got] == ["a1"]
    assert q.depth == 2
    assert q.lane_keys == ["k1", "k2"]


def test_dequeue_all_global_order_and_drain():
    q = RequestQueue()
    q.enqueue(mk("a-low", arrival=0), "k1")
    q.enqueue(mk("b-high", arrival=99, priority=1), "k2")
    q.enqueue(mk("c-mid", arrival=50), "k1")
    pairs = q.dequeue_all()
    assert [(key, r.request_id) for key, r in pairs] == [
        ("k2", "b-high"),
        ("k1", "a-low"),
        ("k1", "c-mid"),
    ]
    assert q.depth == 0
    assert q.status() == {}
    with pytest.raises(UnknownLaneError):
        q.dequeue("k1")


def test_duplicate_and_invalid_rejected():
    q = RequestQueue()
    q.enqueue(mk("a"), "k")
    with pytest.raises(DuplicateRequestError):
        q.enqueue(mk("a"), "other")
    with pytest.raises(RequestValidationError):
        q.enqueue(mk(""), "k")
    with pytest.raises(RequestValidationError):
        q.enqueue(mk("b"), "")


def test_unknown_lane_status_raises():
    q = RequestQueue()
    with pytest.raises(UnknownLaneError):
        q.lane_status("missing")
