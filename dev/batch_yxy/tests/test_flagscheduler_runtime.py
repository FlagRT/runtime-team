"""Node contracts: bounded lifetime, concurrent ingress, mapping and failure."""
import asyncio
import copy
import json
from pathlib import Path
import threading

import pytest

from flagscheduler.backends import MockBackend
from flagscheduler.config import parse_config
from flagscheduler.contracts import Request, ExecutionOutput
from flagscheduler.runtime import NodeRuntime

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/flagscheduler.mock.json"


def config(**overrides):
    data = json.loads(EXAMPLE.read_text())
    data.update(overrides)
    return parse_config(data)


def run(scenario):
    return asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_live_requests_are_batched_and_stop_flushes():
    async def scenario():
        node = NodeRuntime(config())
        await node.start()
        futures = [node.submit(Request(f"r{i}", "qwen-embed", f"text {i}", priority=i % 2)) for i in range(16)]
        await node.stop()
        results = await asyncio.gather(*futures)
        assert all(r["status"] == "completed" for r in results)
        assert len({r["request_id"] for r in results}) == 16
        assert node.snapshot()["active"] == 0
        assert node.snapshot()["state"] == "closed"
        assert all(e["size"] <= 8 for e in node.snapshot()["recent_batches"])
        await node.stop()
        with pytest.raises(ValueError):
            node.submit(Request("late", "qwen-embed", "text"))
    run(scenario)


def test_timer_dispatches_incomplete_batch_without_flush():
    async def scenario():
        node = NodeRuntime(config())
        await node.start()
        result = await node.submit(Request("lonely", "qwen-embed", "text"))
        assert result["status"] == "completed"
        assert node.snapshot()["recent_batches"][0]["reason"] == "wait_limit"
        await node.stop()
    run(scenario)


def test_invalid_input_is_rejected_without_consuming_capacity():
    async def scenario():
        node = NodeRuntime(config(max_pending=1))
        await node.start()
        with pytest.raises(ValueError):
            node.submit(Request("too-long", "qwen-embed", "word " * 513))
        with pytest.raises(ValueError):
            node.submit(Request("unknown", "unknown", "text"))
        assert node.snapshot()["active"] == 0
        assert (await node.submit(Request("valid", "qwen-embed", "text")))["status"] == "completed"
        await node.stop()
    run(scenario)


def test_capacity_duplicate_and_history_are_bounded():
    async def scenario():
        node = NodeRuntime(config(max_pending=1, result_history=1))
        await node.start()
        first = node.submit(Request("one", "qwen-embed", "text"))
        with pytest.raises(ValueError, match="duplicate"):
            node.submit(Request("one", "qwen-embed", "text"))
        with pytest.raises(OverflowError):
            node.submit(Request("two", "qwen-embed", "text"))
        await first
        with pytest.raises(ValueError, match="duplicate"):
            node.submit(Request("one", "qwen-embed", "text"))
        await node.submit(Request("two", "qwen-embed", "text"))
        assert node.result("one") is None
        await node.submit(Request("one", "qwen-embed", "text"))
        assert len(node._history) == 1
        assert not node.planner._seen_ids  # No unbounded legacy ID retention.
        await node.stop()
    run(scenario)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown", "raise", "nan"])
def test_backend_failures_are_terminal_and_batch_atomic(fault):
    class Bad(MockBackend):
        def execute(self, inputs):
            if fault == "raise":
                raise RuntimeError("backend failed")
            if fault == "missing":
                return []
            if fault == "duplicate":
                return [ExecutionOutput(inputs[0].request_id, 1)] * 2
            if fault == "nan":
                return [ExecutionOutput(i.request_id, float("nan")) for i in inputs]
            return [ExecutionOutput("unsubmitted", 1)]
    async def scenario():
        node = NodeRuntime(config(), backend_factory=Bad)
        await node.start()
        futures = [node.submit(Request(str(i), "qwen-embed", "text")) for i in range(2)]
        await node.stop()
        assert all(r["status"] == "failed" for r in await asyncio.gather(*futures))
        assert node.failed == 2 and node.completed == 0
    run(scenario)


def test_reordered_outputs_map_by_id_and_bindings_never_mix():
    class Reverse(MockBackend):
        def execute(self, inputs):
            return [ExecutionOutput(i.request_id, i.payload) for i in reversed(inputs)]
    async def scenario():
        node = NodeRuntime(config(), backend_factory=Reverse)
        await node.start()
        requests = [Request("a", "qwen-embed", "first"), Request("b", "qwen-embed", "second"),
                    Request("c", "din", {"sequence_length": 3})]
        futures = [node.submit(r) for r in requests]
        await node.stop()
        assert [r["output"] for r in await asyncio.gather(*futures)] == [r.payload for r in requests]
        assert len({b["binding"] for b in node.snapshot()["recent_batches"]}) == 2
    run(scenario)


def test_ingress_is_not_blocked_by_device_execution():
    entered, release = threading.Event(), threading.Event()
    class Blocking(MockBackend):
        def execute(self, inputs):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test execution timeout")
            return super().execute(inputs)
    async def scenario():
        node = NodeRuntime(config(), backend_factory=Blocking)
        await node.start()
        first = node.submit(Request("first", "qwen-embed", "text"))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            second = node.submit(Request("second", "qwen-embed", "text"))
            assert node.snapshot()["active"] == 2
        finally:
            release.set()
        assert (await first)["status"] == "completed"
        assert (await second)["status"] == "completed"
        await node.stop()
    run(scenario)


def test_cancelled_client_future_does_not_cancel_inference():
    async def scenario():
        node = NodeRuntime(config())
        await node.start()
        future = node.submit(Request("disconnect", "qwen-embed", "text"))
        future.cancel()
        await node.stop()
        assert node.result("disconnect")["status"] == "completed"
    run(scenario)


def test_all_fifteen_contracts_use_same_runtime():
    from sched_engine.models.registry import all_profiles
    async def scenario():
        source = json.loads(EXAMPLE.read_text())
        template = source["bindings"][0]
        bindings, requests = [], []
        for profile in all_profiles():
            binding = copy.deepcopy(template)
            binding.update(id=profile.model_id, model=profile.repository or profile.model_id)
            bindings.append(binding)
            payload = "mock text" if profile.cost_metric == "token_length" else {profile.cost_metric: 3}
            requests.append(Request(profile.model_id, profile.model_id, payload))
        source["bindings"] = bindings
        node = NodeRuntime(parse_config(source))
        await node.start()
        futures = [node.submit(r) for r in requests]
        await node.stop()
        assert len(futures) == 15
        assert all(r["output"]["mock"] for r in await asyncio.gather(*futures))
    run(scenario)
