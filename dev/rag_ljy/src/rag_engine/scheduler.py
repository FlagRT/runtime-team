"""Bounded priority stage queues with FIFO ties and waiting-time aging."""
from __future__ import annotations

import asyncio
import logging
import math
from time import monotonic
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any


PRIORITIES = {"high": 0, "normal": 1, "low": 2}
logger = logging.getLogger(__name__)


@dataclass
class _Job:
    operation: Callable[[], Any]
    result: asyncio.Future
    priority: str
    enqueued_at: float
    request_id: str | None
    dispatch_trace: list | None = None


class _PriorityQueue(asyncio.Queue):
    """Re-evaluate aging at dequeue, rather than freezing a heap key at arrival."""

    def __init__(self, capacity, aging_seconds, clock):
        super().__init__(maxsize=capacity)
        self.aging_seconds = aging_seconds
        self.clock = clock

    def effective_priority(self, job, now):
        promotions = int(max(0, now - job.enqueued_at) / self.aging_seconds)
        return max(0, PRIORITIES[job.priority] - promotions)

    def _put(self, job):
        # Aging starts when a slot is acquired, not while waiting for capacity.
        if job is not None:
            job.enqueued_at = self.clock()
        super()._put(job)

    def _get(self):
        now = self.clock()
        # deque order is the FIFO tie breaker; sentinels arrive only after drain.
        index = min(range(len(self._queue)), key=lambda i:
                    -1 if self._queue[i] is None else
                    self.effective_priority(self._queue[i], now))
        job = self._queue[index]
        if job is not None and job.dispatch_trace is not None:
            effective = self.effective_priority(job, now)
            job.dispatch_trace.append({
                "priority": job.priority, "effective_priority": ("high", "normal", "low")[effective],
                "wait_ms": max(0, now - job.enqueued_at) * 1000,
                "dispatched_monotonic": now,
                "waiting_candidates": len(self._queue),
                "overtook_lower_priority": sum(
                    self.effective_priority(earlier, now) > effective
                    for earlier in list(self._queue)[:index] if earlier is not None),
            })
        del self._queue[index]
        return job


class StageQueue:
    """Separate stage backpressure from the executor's internal work queue.

    Accepted requests wait for space instead of losing work between stages.
    Owners must drain request tasks before closing this queue. No batch-fill delay.
    """
    def __init__(self, name: str, executor: Executor, *, capacity: int, workers: int,
                 priority_aging_seconds: float = 30.0, clock=monotonic):
        if capacity < 1 or workers < 1:
            raise ValueError('Stage queue capacity and worker count must be positive')
        if not math.isfinite(priority_aging_seconds) or priority_aging_seconds <= 0:
            raise ValueError("Priority aging interval must be finite and positive")
        self.name = name
        self.capacity = capacity
        self.workers = workers
        self._executor = executor
        self._queue = _PriorityQueue(capacity, priority_aging_seconds, clock)
        self._running = 0
        self._closed = False
        self._consumers = [asyncio.create_task(self._consume(), name=f'rag-{name}-{i}')
                           for i in range(workers)]

    def snapshot(self):
        return {'capacity': self.capacity, 'queued': self._queue.qsize(),
                'running': self._running, 'workers': self.workers}

    async def run(self, operation, *, priority="normal", request_id=None, dispatch_trace=None):
        if self._closed:
            raise RuntimeError(f'{self.name} queue is closed')
        if priority not in PRIORITIES:
            raise ValueError(f"Unknown request priority: {priority}")
        result = asyncio.get_running_loop().create_future()
        await self._queue.put(_Job(operation, result, priority, 0.0, request_id, dispatch_trace))
        # HTTP disconnects must not cancel already accepted stage work.
        return await asyncio.shield(result)

    async def _consume(self):
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                now = self._queue.clock()
                logger.info("request_id=%s stage=%s event=dispatch priority=%s "
                            "effective_priority=%s queue_wait_ms=%.3f",
                            job.request_id, self.name, job.priority,
                            ("high", "normal", "low")[self._queue.effective_priority(job, now)],
                            (now - job.enqueued_at) * 1000)
                self._running += 1
                try:
                    value = await asyncio.get_running_loop().run_in_executor(
                        self._executor, job.operation)
                except Exception as error:
                    job.result.set_exception(error)
                else:
                    job.result.set_result(value)
                finally:
                    self._running -= 1
            finally:
                self._queue.task_done()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self._queue.join()
        for _ in self._consumers:
            await self._queue.put(None)
        await asyncio.gather(*self._consumers)
