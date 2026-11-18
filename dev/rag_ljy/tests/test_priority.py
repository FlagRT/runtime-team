"""Priority acceptance checks: real queues/threads/HTTP, simulated models and memory."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from rag_engine.scheduler import StageQueue
from rag_engine.server import QueryRequest, create_app
from test_server import FakePipeline, ControlledMemoryMonitor
from test_generation_server import GenerationPipeline, generation_app


@pytest.mark.parametrize('mode', ['burst', 'staggered', 'continuous'])
def test_priority_order_fifo_ties_and_no_preemption(mode):
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        order = []
        def work(label):
            order.append(label)
            if label == 'running-low':
                entered.set()
                assert release.wait(5)
            return label
        with ThreadPoolExecutor(1) as executor:
            queue = StageQueue('test', executor, capacity=8, workers=1)
            first = asyncio.create_task(queue.run(lambda: work('running-low'), priority='low'))
            assert await asyncio.to_thread(entered.wait, 5)
            trace = []
            tasks = [first]
            arrivals = [('low', 'low'), ('normal', 'normal'), ('high-a', 'high'), ('high-b', 'high')]
            if mode == 'continuous':
                arrivals += [('low-later', 'low'), ('high-c', 'high')]
            try:
                for label, priority in arrivals:
                    tasks.append(asyncio.create_task(queue.run(lambda label=label: work(label), priority=priority, dispatch_trace=trace)))
                    await asyncio.sleep(0 if mode == 'burst' else .01)
                assert order == ['running-low']
            finally:
                release.set()
            assert await asyncio.gather(*tasks) == ['running-low'] + [v for v, _ in arrivals]
            expected = ['running-low', 'high-a', 'high-b']
            if mode == 'continuous':
                expected += ['high-c']
            expected += ['normal', 'low']
            if mode == 'continuous':
                expected += ['low-later']
            assert order == expected
            assert sum(d['overtook_lower_priority'] for d in trace) >= 4
            await queue.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('elapsed,expected', [(29, ['high', 'normal', 'low']),
                                               (30, ['high', 'low', 'normal']),
                                               (60, ['low', 'high', 'normal'])])
def test_aging_recomputed_at_dispatch(elapsed, expected):
    async def scenario():
        now = [0.0]
        entered, release = threading.Event(), threading.Event()
        order = []
        def block():
            entered.set()
            assert release.wait(5)
        with ThreadPoolExecutor(1) as executor:
            queue = StageQueue('test', executor, capacity=3, workers=1, clock=lambda: now[0])
            first = asyncio.create_task(queue.run(block))
            assert await asyncio.to_thread(entered.wait, 5)
            tasks = [first]
            try:
                tasks.append(asyncio.create_task(queue.run(lambda: order.append('low'), priority='low')))
                await asyncio.sleep(0)
                now[0] = elapsed
                for name in ('normal', 'high'):
                    tasks.append(asyncio.create_task(queue.run(lambda name=name: order.append(name), priority=name)))
                    await asyncio.sleep(0)
            finally:
                release.set()
            await asyncio.gather(*tasks)
            assert order == expected
            await queue.close()
    asyncio.run(scenario())


def test_priority_does_not_evict_or_expand_full_queue():
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        order = []
        def block():
            entered.set()
            assert release.wait(5)
        with ThreadPoolExecutor(1) as executor:
            queue = StageQueue('test', executor, capacity=1, workers=1)
            first = asyncio.create_task(queue.run(block))
            assert await asyncio.to_thread(entered.wait, 5)
            low = asyncio.create_task(queue.run(lambda: order.append('low'), priority='low'))
            await asyncio.sleep(0)
            high = asyncio.create_task(queue.run(lambda: order.append('high'), priority='high'))
            try:
                await asyncio.sleep(0)
                assert queue.snapshot()['queued'] == 1
                assert not high.done()
            finally:
                release.set()
            await asyncio.gather(first, low, high)
            assert order == ['low', 'high']
            await queue.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('priority', ['high', 'normal', 'low'])
def test_http_propagates_priority_through_all_four_stages(monkeypatch, priority):
    seen = []
    original = StageQueue.run
    async def record(self, operation, **kwargs):
        seen.append((self.name, kwargs['priority'], kwargs['request_id']))
        return await original(self, operation, **kwargs)
    monkeypatch.setattr(StageQueue, 'run', record)
    pipeline = GenerationPipeline()
    pipeline.finish_generation.set()
    with TestClient(generation_app(monkeypatch, pipeline, ControlledMemoryMonitor(100))) as client:
        response = client.post('/query', json={'query': 'example', 'generate': True,
                                               'max_new_tokens': 32, 'priority': priority})
        assert response.status_code == 200
        assert response.headers['x-request-priority'] == priority
        evidence = response.json()['scheduling']
        assert evidence['priority'] == priority
        assert set(evidence['stages']) == {'embedding','retrieval','rerank','generation'}
        assert all(len(records) == 1 for records in evidence['stages'].values())
        assert seen == [(stage, priority, response.headers['x-request-id']) for stage in
                        ('embedding', 'retrieval', 'rerank', 'generation')]
        assert client.get('/health').json()['scheduling']['aging_seconds'] == 30


def test_default_priority_and_invalid_http_values():
    pipeline = FakePipeline()
    with TestClient(create_app(pipeline_factory=lambda _: pipeline, memory_probe_enabled=False)) as client:
        for invalid in ('urgent', 'HIGH', 0, None):
            assert client.post('/query', json={'query': 'q', 'priority': invalid}).status_code == 422
        assert not pipeline.calls
        response = client.post('/query', json={'query': 'q'})
        assert response.status_code == 200
        assert response.headers['x-request-priority'] == 'normal'
    assert QueryRequest(query='q').priority == 'normal'


@pytest.mark.parametrize('used,reason', [(900, 'memory_pressure'), (700, 'request_budget')])
def test_high_priority_cannot_bypass_memory_admission(monkeypatch, used, reason):
    pipeline = GenerationPipeline()
    monitor = ControlledMemoryMonitor(used)
    with TestClient(generation_app(monkeypatch, pipeline, monitor)) as client:
        response = client.post('/query', json={'query': 'q', 'generate': True,
                                               'max_new_tokens': 32, 'priority': 'high'})
        assert response.status_code == 503
        assert response.headers['retry-after'] == '1'
        assert response.json()['detail']['reason'] == reason
        assert not pipeline.calls


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_aging_configuration(value):
    with pytest.raises(ValueError, match='aging'):
        create_app(priority_aging_seconds=value)


def test_http_requests_are_reordered_before_embedding_dispatch():
    class BlockingPipeline(FakePipeline):
        def embed(self, query, batch_size, timings):
            if query == 'first':
                self.started.set()
                assert self.release.wait(5)
            return super().embed(query, batch_size, timings)
    pipeline = BlockingPipeline()
    app = create_app(pipeline_factory=lambda _: pipeline, memory_probe_enabled=False)
    with TestClient(app) as client, ThreadPoolExecutor(4) as callers:
        async def wait_queued(count):
            for _ in range(500):
                if app.state.stage_queues['embedding'].snapshot()['queued'] == count:
                    return
                await asyncio.sleep(.005)
            raise AssertionError('request did not enter embedding queue')
        first = callers.submit(client.post, '/query', json={'query': 'first', 'priority': 'low'})
        pending = [first]
        try:
            assert pipeline.started.wait(5)
            for count, priority in enumerate(('low', 'normal', 'high'), 1):
                pending.append(callers.submit(client.post, '/query', json={'query': priority, 'priority': priority}))
                client.portal.call(wait_queued, count)
        finally:
            pipeline.release.set()
        responses = [f.result(5) for f in pending]
        assert all(r.status_code == 200 for r in responses)
        assert pipeline.calls == ['first', 'high', 'normal', 'low']
        assert [r.json()[0]['text'] for r in responses] == ['first', 'low', 'normal', 'high']
