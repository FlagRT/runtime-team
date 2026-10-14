"""结果回填存储测试：恰好一次、防漏、防重、防错位。"""

import pytest

from dynamic_batching.errors import ResultCountMismatchError
from sched_engine.errors import (
    DuplicateResultError,
    MissingResultError,
    UnexpectedResultError,
)
from sched_engine.core.results import ResultStore


def test_commit_and_get_roundtrip():
    store = ResultStore()
    store.declare(["r0", "r1"])
    store.commit("batch-1", ["r0", "r1"], ["v0", "v1"])
    assert store.get("r0") == "v0"
    assert store.get("r1") == "v1"
    assert store.batch_of("r0") == "batch-1"
    assert store.filled_count == 2
    assert store.missing() == []
    store.assert_complete()


def test_count_mismatch_rejected_atomic():
    store = ResultStore()
    store.declare(["r0", "r1"])
    with pytest.raises(ResultCountMismatchError):
        store.commit("batch-1", ["r0", "r1"], ["only-v0"])
    assert store.filled_count == 0  # 不部分写入
    assert store.missing() == ["r0", "r1"]


def test_duplicate_commit_rejected():
    store = ResultStore()
    store.declare(["r0"])
    store.commit("batch-1", ["r0"], ["v0"])
    with pytest.raises(DuplicateResultError):
        store.commit("batch-2", ["r0"], ["v0-again"])
    assert store.get("r0") == "v0"


def test_unknown_request_rejected():
    store = ResultStore()
    store.declare(["r0"])
    with pytest.raises(UnexpectedResultError):
        store.commit("batch-1", ["ghost"], ["v"])
    with pytest.raises(UnexpectedResultError):
        store.get("ghost")


def test_missing_results_raise_after_run():
    store = ResultStore()
    store.declare(["r0", "r1"])
    store.commit("batch-1", ["r0"], ["v0"])
    with pytest.raises(MissingResultError):
        store.assert_complete()


def test_duplicate_declare_rejected():
    store = ResultStore()
    store.declare(["r0"])
    with pytest.raises(DuplicateResultError):
        store.declare(["r0"])
