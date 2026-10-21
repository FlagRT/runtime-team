"""Persistent, single-node event loop; no model/framework assumptions.

One execution batch at a time provides a predictable initial resource model.
Blocking backends run in a worker thread so new requests can arrive while the
device executes. Planner mutations are confined to the event loop.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
import json
import time

from dynamic_batching.coordinator import BatchCoordinator
from dynamic_batching.errors import BatchCoordinatorError
from dynamic_batching.types import BatchPolicy, BatchRequest
from .backends import create_backend, load_plugin
from .config import NodeConfig
from .contracts import ExecutionInput, Request, positive_int


@dataclass
class _Pending:
    request: Request
    future: asyncio.Future
    arrival: int
    deadline: int | None


class NodeRuntime:
    """Application API for CLI, HTTP transport and future TUI.

    Future cancellation/disconnection does not cancel inference. At-most-once
    execution is per accepted ID in the active set + bounded result window,
    within this process only; this is not durable exactly-once delivery.
    """
    def __init__(self, config: NodeConfig, *, backend_factory=create_backend,
                 clock=time.monotonic_ns):
        self.config = config
        self.clock = clock
        self.bindings = {binding.id: binding for binding in config.bindings}
        policies = [BatchPolicy(model_id=b.id, **{**b.policy,
                    "bucket_upper_bounds": tuple(b.policy["bucket_upper_bounds"])}) for b in config.bindings]
        factory = BatchCoordinator if config.planner == "batch-coordinator" else load_plugin(config.planner, "flagscheduler.planners")
        planner_options = {"strict_duplicate_detection": False, "stats_maxlen": 256} if factory is BatchCoordinator else {}
        self.planner = factory(policies=policies, clock=clock, **planner_options)
        self.backends = {}
        try:
            for binding in config.bindings:
                self.backends[binding.id] = backend_factory(binding)
        except Exception:
            for backend in self.backends.values():
                backend.close()
            raise
        self._active: dict[str, _Pending] = {}
        self._history = OrderedDict()
        self._events = deque(maxlen=256)
        self._wake = asyncio.Event()
        self._task = None
        self._accepting = False
        self._closing = False
        self._closed = False
        self.completed = 0
        self.failed = 0

    async def start(self):
        if self._task is not None or self._closed:
            raise ValueError("runtime cannot be started twice")
        self._accepting = True
        self._task = asyncio.create_task(self._run(), name="flagscheduler-worker")

    def submit(self, request: Request) -> asyncio.Future:
        if not self._accepting:
            raise ValueError("node is not accepting requests")
        if request.binding not in self.bindings:
            raise ValueError(f"unknown binding: {request.binding}")
        if request.request_id in self._active or request.request_id in self._history:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        if len(self._active) >= self.config.max_pending:
            raise OverflowError("node queue capacity reached")
        # Counting is pure CPU preprocessing and must not initialize a model,
        # perform I/O, mutate execution state or use a different tokenizer.
        cost = positive_int(self.backends[request.binding].count_cost(request.payload), "request cost")
        now = self.clock()
        deadline = None if request.slo_ms is None else now + int(request.slo_ms * 1e6)
        binding = self.bindings[request.binding]
        try:
            self.planner.submit(BatchRequest(
                request_id=request.request_id, model_id=request.binding,
                model_version=binding.model_version, execution_profile=request.binding,
                input_token_count=cost, payload_ref=request.payload, arrival_mono_ns=now,
                priority=request.priority, deadline_mono_ns=deadline,
            ))
        except BatchCoordinatorError as exc:
            raise ValueError(str(exc)) from exc
        future = asyncio.get_running_loop().create_future()
        self._active[request.request_id] = _Pending(request, future, now, deadline)
        self._wake.set()
        return future

    def snapshot(self) -> dict:
        return {"schema_version": 1, "state": "accepting" if self._accepting else (
                    "closed" if self._closed else "draining"),
                "active": len(self._active), "queued": self.planner.pending_count,
                "max_pending": self.config.max_pending,
                "completed": self.completed, "failed": self.failed,
                "result_history": len(self._history), "planner": self.config.planner,
                "bindings": [{"id": b.id, "model": b.model, "backend": b.backend,
                              "model_version": b.model_version,
                              "scheduler": getattr(self.backends[b.id], "scheduler_info", None),
                              "verification": "mock-only" if b.backend == "mock" else "requires-device-validation"}
                             for b in self.bindings.values()],
                "recent_batches": list(self._events),
                "slo": "deadline metadata only; no SLO guarantee"}

    def result(self, request_id: str) -> dict | None:
        if request_id in self._active:
            return {"schema_version": 1, "request_id": request_id, "status": "pending"}
        return self._history.get(request_id)

    async def stop(self):
        if self._closed:
            return
        self._accepting = False
        self._closing = True
        self._wake.set()
        try:
            if self._task:
                await self._task
        finally:
            for backend in self.backends.values():
                await asyncio.to_thread(backend.close)
            self._closed = True

    def _finish(self, rid: str, *, value=None, error: Exception | None = None):
        pending = self._active.pop(rid)
        record = {"schema_version": 1, "request_id": rid, "binding": pending.request.binding,
                  "status": "failed" if error else "completed",
                  "elapsed_ms": (self.clock() - pending.arrival) / 1e6}
        if error:
            self.failed += 1
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        else:
            self.completed += 1
            record["output"] = value
        self._history[rid] = record
        while len(self._history) > self.config.result_history:
            self._history.popitem(last=False)
        # Deliver a structured error as a normal result so disconnects do not
        # leave unobserved future exceptions. Every accepted ID has a terminal state.
        if not pending.future.done():
            pending.future.set_result(record)

    async def _execute(self, plan):
        pending = [self._active[rid] for rid in plan.request_ids]
        binding_id = pending[0].request.binding
        if any(p.request.binding != binding_id for p in pending):
            raise ValueError("planner emitted incompatible bindings")
        inputs = [ExecutionInput(p.request.request_id, p.request.payload, p.request.priority,
                                 p.deadline, p.request.metadata) for p in pending]
        start = self.clock()
        try:
            outputs = await asyncio.to_thread(self.backends[binding_id].execute, inputs)
            mapped = {}
            expected = set(plan.request_ids)
            for output in outputs:
                if output.request_id not in expected or output.request_id in mapped:
                    raise ValueError("backend returned an unknown or duplicate result ID")
                mapped[output.request_id] = output.value
            if mapped.keys() != expected:
                raise ValueError("backend omitted result IDs")
            json.dumps(mapped, allow_nan=False)
            for rid in plan.request_ids:
                self._finish(rid, value=mapped[rid])
        except Exception as exc:
            # No implicit retries: a partially executed batch cannot safely be
            # replayed without an idempotency contract supplied by its backend.
            for rid in plan.request_ids:
                if rid in self._active:
                    self._finish(rid, error=exc)
        self._events.append({"batch_id": plan.batch_id, "binding": binding_id,
                             "request_ids": list(plan.request_ids),
                             "size": len(plan.request_ids), "total_cost": plan.total_tokens,
                             "reason": plan.dispatch_reason,
                             "execution_ms": (self.clock() - start) / 1e6})

    async def _run(self):
        try:
            while True:
                self._wake.clear()
                plans = self.planner.flush() if self._closing else self.planner.poll_ready(self.clock())
                for plan in plans:
                    await self._execute(plan)
                if self._closing and not self._active:
                    break
                if plans:
                    continue
                next_wakeup = self.planner.next_wakeup_mono_ns()
                timeout = None if next_wakeup is None else max((next_wakeup - self.clock()) / 1e9, 0.0001)
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout)
                except asyncio.TimeoutError:
                    pass
        except Exception as exc:
            self._accepting = False
            for rid in list(self._active):
                self._finish(rid, error=exc)
            raise
