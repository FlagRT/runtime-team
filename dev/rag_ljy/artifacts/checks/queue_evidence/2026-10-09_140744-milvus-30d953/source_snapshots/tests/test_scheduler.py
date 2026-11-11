"""Exercise bounded queues and result routing with real worker threads."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from rag_engine.scheduler import StageQueue


def test_stage_backpressure_fifo_and_error_isolation():
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        order = []
        def operation(value):
            order.append(value)
            if value == 'first':
                entered.set()
                assert release.wait(5)
            if value == 'bad':
                raise ValueError('one job failed')
            return value
        with ThreadPoolExecutor(1) as executor:
            queue = StageQueue('test', executor, capacity=1, workers=1)
            first = asyncio.create_task(queue.run(lambda: operation('first')))
            assert await asyncio.to_thread(entered.wait, 5)
            second = asyncio.create_task(queue.run(lambda: operation('second')))
            await asyncio.sleep(0)
            third = asyncio.create_task(queue.run(lambda: operation('third')))
            try:
                await asyncio.sleep(0)
                assert queue.snapshot() == {'capacity': 1, 'queued': 1, 'running': 1, 'workers': 1}
                assert not third.done()
            finally:
                release.set()
            assert await asyncio.gather(first, second, third) == ['first', 'second', 'third']
            assert order == ['first', 'second', 'third']
            with pytest.raises(ValueError, match='one job failed'):
                await queue.run(lambda: operation('bad'))
            assert await queue.run(lambda: operation('later')) == 'later'
            await queue.close()
            await queue.close()
            assert all(t.done() for t in queue._consumers)
            with pytest.raises(RuntimeError, match='closed'):
                await queue.run(lambda: None)
    asyncio.run(scenario())


def test_parallel_stage_jobs_keep_individual_results():
    async def scenario():
        barrier = threading.Barrier(2)
        def operation(value):
            barrier.wait(timeout=5)
            return value
        with ThreadPoolExecutor(2) as executor:
            queue = StageQueue('embedding', executor, capacity=2, workers=2)
            try:
                values = await asyncio.gather(queue.run(lambda: operation('a')), queue.run(lambda: operation('b')))
                assert values == ['a', 'b']
            finally:
                await queue.close()
    asyncio.run(scenario())
