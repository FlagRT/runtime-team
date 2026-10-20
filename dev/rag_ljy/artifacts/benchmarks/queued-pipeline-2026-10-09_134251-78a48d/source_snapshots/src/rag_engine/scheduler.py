"""Bounded FIFO stage queues; each consumer submits one job at a time."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any


@dataclass
class _Job:
    operation: Callable[[], Any]
    result: asyncio.Future


class StageQueue:
    """Separate stage backpressure from the executor's internal work queue.

    Accepted requests wait for space instead of losing work between stages.
    Owners must drain request tasks before closing this queue. No batch-fill delay.
    """
    def __init__(self, name: str, executor: Executor, *, capacity: int, workers: int):
        if capacity < 1 or workers < 1:
            raise ValueError('Stage queue capacity and worker count must be positive')
        self.name = name
        self.capacity = capacity
        self.workers = workers
        self._executor = executor
        self._queue = asyncio.Queue(maxsize=capacity)
        self._running = 0
        self._closed = False
        self._consumers = [asyncio.create_task(self._consume(), name=f'rag-{name}-{i}')
                           for i in range(workers)]

    def snapshot(self):
        return {'capacity': self.capacity, 'queued': self._queue.qsize(),
                'running': self._running, 'workers': self.workers}

    async def run(self, operation):
        if self._closed:
            raise RuntimeError(f'{self.name} queue is closed')
        result = asyncio.get_running_loop().create_future()
        await self._queue.put(_Job(operation, result))
        # HTTP disconnects must not cancel already accepted stage work.
        return await asyncio.shield(result)

    async def _consume(self):
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
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
