"""Multi-request API with bounded embedding, retrieval, and reranking queues."""

from __future__ import annotations

import asyncio
import logging
import json
import math
from time import perf_counter_ns, monotonic
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from .query_runtime import create_pipeline, format_hits
from .retrieval import validate_execution_mode
from .scheduler import StageQueue
from .memory_probe import DeviceMemoryMonitor
from .admission import MemoryAdmissionConfig, MemoryAdmissionController


logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=32768)
    retriever_top_k: int = Field(default=50, ge=1, le=100)
    coarse_top_k: int = Field(default=30, ge=20, le=50)
    fine_top_k: int = Field(default=5, ge=3, le=5)
    embedding_batch_size: int = Field(default=8, ge=1, le=8)
    reranker_batch_size: int = Field(default=4, ge=1, le=50)


def create_app(
    device: str = "auto", pipeline_factory=None, *, execution_mode: str = "concurrent",
    max_inflight: int = 8, retrieval_workers: int = 4,
    embedding_workers: int = 2, stage_queue_capacity: int | None = None,
    memory_probe_enabled: bool = True, memory_probe_interval_s: float = 1.0,
    memory_probe_factory=None,
    memory_admission: MemoryAdmissionConfig | None = None,
    max_reranker_batch_size: int = 4,
    memory_estimator_factory=None,
) -> FastAPI:
    """Advance requests independently through bounded FIFO stage queues.

    Remote vLLM embedding calls use a separate pool. Local accelerator models
    remain on their initialization thread. Full stage queues apply backpressure
    within the overall admitted-request bound; new excess requests receive 503.
    """
    validate_execution_mode(execution_mode)
    if not 1 <= max_reranker_batch_size <= 50:
        raise ValueError("max_reranker_batch_size must be between 1 and 50")
    memory_admission = memory_admission or MemoryAdmissionConfig()
    if max_inflight < 0:
        raise ValueError("max_inflight must be nonnegative; 0 disables the count limit")
    if max_inflight == 0 and not memory_admission.enabled:
        raise ValueError("Disabling the request count limit requires memory admission")
    if stage_queue_capacity is None:
        stage_queue_capacity = max_inflight or 8
    if min(retrieval_workers, embedding_workers, stage_queue_capacity) < 1:
        raise ValueError("Worker counts and request/queue capacities must be positive")
    if not math.isfinite(memory_probe_interval_s) or memory_probe_interval_s <= 0:
        raise ValueError("Memory probe interval must be finite and positive")
    if memory_admission.enabled and not memory_probe_enabled:
        raise ValueError("Memory admission requires memory probing")
    factory = pipeline_factory or partial(create_pipeline, execution_mode=execution_mode)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-inference")
        retrieval_executor = ThreadPoolExecutor(
            max_workers=retrieval_workers, thread_name_prefix="rag-retrieval")
        embedding_executor = None
        app.state.stage_queues = {}
        app.state.ready = False
        app.state.tasks = set()
        app.state.executor = executor
        app.state.retrieval_executor = retrieval_executor
        app.state.memory_monitor = None
        app.state.cache_reclaim_task = None
        app.state.cache_reclaim_attempted = False
        app.state.memory_admission = MemoryAdmissionController(memory_admission)
        try:
            app.state.pipeline = await asyncio.get_running_loop().run_in_executor(
                executor, factory, device)
            app.state.memory_estimator = None
            app.state.estimator_fault = None
            app.state.memory_observations = {}
            if memory_admission.enabled and memory_admission.estimate_requests:
                from .memory_estimate import RequestMemoryEstimator
                estimator_factory = memory_estimator_factory or RequestMemoryEstimator.from_pipeline
                app.state.memory_estimator = await asyncio.get_running_loop().run_in_executor(
                    executor, estimator_factory, app.state.pipeline)
                reranker = getattr(app.state.pipeline, 'reranker', None)
                if reranker is not None:
                    await asyncio.get_running_loop().run_in_executor(executor, partial(
                        reranker.configure_memory_guard, app.state.memory_estimator,
                        memory_admission.safety_margin_mib))
            embedder = getattr(app.state.pipeline, "embedder", None)
            app.state.embedding_backend = getattr(embedder, "backend", "transformers")
            app.state.device = str(getattr(embedder, "device", device))
            remote_embedding = app.state.embedding_backend == "vllm"
            if remote_embedding:
                embedding_executor = ThreadPoolExecutor(
                    max_workers=embedding_workers, thread_name_prefix="rag-embedding-http")
            app.state.stage_queues = {
                "embedding": StageQueue("embedding", embedding_executor or executor,
                    capacity=stage_queue_capacity, workers=embedding_workers if remote_embedding else 1),
                "retrieval": StageQueue("retrieval", retrieval_executor,
                    capacity=stage_queue_capacity, workers=retrieval_workers),
                "rerank": StageQueue("rerank", executor,
                    capacity=stage_queue_capacity, workers=1),
            }
            # The remote embedding adapter's 'remote-vllm' label is not a device.
            # Probe the local rerank device without occupying its model worker.
            probe_device = str(getattr(getattr(app.state.pipeline, "reranker", None), "device", device))
            app.state.memory_monitor = DeviceMemoryMonitor(
                probe_device, enabled=memory_probe_enabled, interval_s=memory_probe_interval_s,
                probe=memory_probe_factory(probe_device) if memory_probe_factory is not None else None)
            def observe_memory(sample):
                # Latch real pressure on every sample, even with no HTTP traffic.
                pressure = app.state.memory_admission.snapshot(sample)
                if pressure.get('status') == 'ready':
                    app.state.cache_reclaim_attempted = False
                reranker = getattr(app.state.pipeline, 'reranker', None)
                if (app.state.ready and not app.state.tasks and pressure.get('reason') == 'memory_pressure'
                        and not app.state.cache_reclaim_attempted and callable(getattr(reranker, 'reclaim_unused_memory', None))):
                    app.state.cache_reclaim_attempted = True
                    if app.state.memory_estimator is not None:
                        app.state.memory_estimator.cache_snapshot = {'reclaimable_mib':0, 'sampled_monotonic':float('inf')}
                    app.state.cache_reclaim_task = asyncio.get_running_loop().run_in_executor(executor, reranker.reclaim_unused_memory)
                if sample.get('status') != 'ok' or not sample.get('stats'):
                    return
                stats = sample['stats']
                used = max(stats['used_mb'], stats['total_mb'] - stats['free_mb'])
                shared = len(app.state.tasks) > 1
                for observation in app.state.memory_observations.values():
                    observation['peak_device_used_mib'] = max(observation['peak_device_used_mib'], used)
                    observation['overlapped_other_requests'] |= shared
            app.state.memory_monitor.on_sample = observe_memory
            app.state.memory_monitor.start()
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            # Accepted requests keep ownership even if their HTTP waiter is cancelled.
            # Drain all stages before closing the pipeline or either executor.
            try:
                await asyncio.gather(*list(app.state.tasks), return_exceptions=True)
                if app.state.cache_reclaim_task is not None:
                    await app.state.cache_reclaim_task
                for queue in app.state.stage_queues.values():
                    await queue.close()
                if app.state.memory_monitor is not None:
                    await app.state.memory_monitor.close()
                close = getattr(getattr(app.state, "pipeline", None), "close", None)
                if close is not None:
                    await asyncio.get_running_loop().run_in_executor(executor, close)
            finally:
                if embedding_executor is not None:
                    embedding_executor.shutdown(wait=True, cancel_futures=True)
                retrieval_executor.shutdown(wait=True, cancel_futures=True)
                executor.shutdown(wait=True, cancel_futures=True)
                app.state.pipeline = None

    app = FastAPI(title="RAG retrieval and reranking", lifespan=lifespan)

    @app.get("/health")
    async def health():
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Models are not ready")
        active = len(app.state.tasks)
        memory = app.state.memory_monitor.snapshot()
        estimator_fault = app.state.estimator_fault or getattr(app.state.memory_estimator, 'fault', None)
        return {"status": "ready", "device": app.state.device, "busy": bool(active),
                "inflight": active, "max_inflight": max_inflight,
                "reranker_batch_limit": max_reranker_batch_size,
                "memory_estimation": {"enabled": app.state.memory_estimator is not None,
                    "version": getattr(app.state.memory_estimator, 'version', None),
                    "status": 'fault' if estimator_fault else ('analytical_uncalibrated' if app.state.memory_estimator else 'fixed_or_disabled'),
                    "fault": estimator_fault},
                "embedding_backend": app.state.embedding_backend,
                "retrieval_backend": getattr(getattr(app.state.pipeline, "store", None), "backend_name", "unknown"),
                "execution_mode": execution_mode,
                "device_memory": memory,
                "memory_admission": app.state.memory_admission.snapshot(memory),
                "stage_queues": {name: queue.snapshot()
                                 for name, queue in app.state.stage_queues.items()}}

    async def process(request, request_id):
        pipeline = app.state.pipeline
        timings = {}
        started = perf_counter_ns()

        async def stage(name, operation):
            queued = perf_counter_ns()

            def work():
                from .forward_guard import request_context
                context_token = request_context.set(request_id)
                begin = perf_counter_ns()
                timings[name + "_queue_ms"] = (begin - queued) / 1_000_000
                logger.info("request_id=%s stage=%s event=start monotonic_ns=%d queue_ms=%.3f",
                            request_id, name, begin, timings[name + "_queue_ms"])
                try:
                    return operation()
                finally:
                    logger.info("request_id=%s stage=%s event=end monotonic_ns=%d",
                                request_id, name, perf_counter_ns())
                    request_context.reset(context_token)

            return await app.state.stage_queues[name].run(work)

        status = "error"
        try:
            vector = await stage("embedding", partial(
                pipeline.embed, request.query, request.embedding_batch_size, timings))
            candidates = await stage("retrieval", partial(
                pipeline.search, request.query, vector, request.retriever_top_k,
                request.coarse_top_k, timings))
            hits = await stage("rerank", partial(
                pipeline.rank, request.query, candidates, request.fine_top_k,
                request.reranker_batch_size, timings))
            result = format_hits(hits)
            status = "ok"
            return result
        finally:
            timings["total_ms"] = (perf_counter_ns() - started) / 1_000_000
            logger.info("request_id=%s event=finished status=%s timings=%s",
                        request_id, status, json.dumps(timings, sort_keys=True))

    def plan_estimate(request, state, sample):
        estimator = app.state.memory_estimator
        if not hasattr(estimator, 'plan'):
            return estimator.estimate(request)
        credit = 0
        cache = getattr(estimator, 'cache_snapshot', {})
        if (not app.state.tasks and state.get('reserved_mib') == 0
                and cache.get('sampled_monotonic', float('inf')) <= sample.get('sample_started_monotonic', 0)):
            credit = min(cache.get('reclaimable_mib', 0), state.get('measured_used_mib', 0))
        return estimator.plan(request, state.get('available_after_reservations_mib', 0), cache_credit_mib=credit)

    @app.post('/memory/estimate')
    async def estimate_memory(request: QueryRequest):
        """Inspect an estimate without admitting, reserving, or executing a query."""
        if not app.state.ready or app.state.memory_estimator is None:
            raise HTTPException(status_code=503, detail={'reason': 'memory_estimator_unavailable'},
                                headers={'Retry-After': '1'})
        if request.reranker_batch_size > max_reranker_batch_size:
            raise HTTPException(status_code=422, detail='reranker_batch_size exceeds configured limit')
        try:
            sample = app.state.memory_monitor.snapshot()
            state = app.state.memory_admission.snapshot(sample)
            estimate = plan_estimate(request, state, sample)
        except Exception:
            logger.exception('event=memory_estimate_preview_failed')
            raise HTTPException(status_code=503, detail={'reason': 'memory_estimate_unavailable'},
                                headers={'Retry-After': '1'})
        return {'memory_estimate': estimate, 'device_memory': app.state.memory_monitor.snapshot(),
                'note': 'Preview only; no reservation or model execution. /query estimates again atomically before admission.'}

    @app.post("/query")
    async def query(request: QueryRequest, response: Response):
        request_id = uuid4().hex
        headers = {"X-Request-ID": request_id}
        if request.reranker_batch_size > max_reranker_batch_size:
            raise HTTPException(status_code=422, detail=f"reranker_batch_size exceeds configured limit {max_reranker_batch_size}", headers=headers)
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Service is not ready", headers=headers)
        # No await between capacity check and task registration: atomic on this loop.
        if max_inflight > 0 and len(app.state.tasks) >= max_inflight:
            raise HTTPException(status_code=503, detail="Request capacity is full; retry later",
                                headers={**headers, "Retry-After": "1"})
        admission = app.state.memory_admission
        estimate = None
        if memory_admission.enabled:
            sample = app.state.memory_monitor.snapshot()
            state = admission.snapshot(sample)
            if state['reason'] not in ('memory_available', 'request_budget'):
                reason = state['reason']
                logger.info('event=memory_admission decision=%s', json.dumps(admission.reject_unestimated(request_id, reason)))
                raise HTTPException(status_code=503, detail={'reason': reason, 'admission_snapshot': {'memory_admission':state, 'device_memory':sample}, 'decision_monotonic':monotonic()},
                                    headers={**headers, 'Retry-After': '1'})
            if app.state.estimator_fault or getattr(app.state.memory_estimator, 'fault', None):
                logger.info('event=memory_admission decision=%s', json.dumps(admission.reject_unestimated(request_id, 'memory_estimator_fault')))
                raise HTTPException(status_code=503, detail={"reason": "memory_estimator_fault"}, headers={**headers, "Retry-After": "1"})
            if memory_admission.estimate_requests:
                try:
                    estimate = plan_estimate(request, state, sample)
                    budget = estimate['estimated_incremental_mib']
                    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
                        raise ValueError('Invalid incremental memory estimate')
                except Exception:
                    logger.exception('event=memory_estimate_failed request_id=%s', request_id)
                    logger.info('event=memory_admission decision=%s', json.dumps(admission.reject_unestimated(request_id, 'memory_estimate_unavailable')))
                    raise HTTPException(status_code=503, detail={"reason": "memory_estimate_unavailable"}, headers={**headers, "Retry-After": "1"})
                logger.info('event=memory_estimate request_id=%s estimate=%s', request_id, json.dumps(estimate, sort_keys=True))
            sample = app.state.memory_monitor.snapshot()
            decision = admission.reserve(request_id, sample,
                request_budget_mib=estimate['estimated_incremental_mib'] if estimate else None,
                cache_credit_mib=estimate.get('cache_credit_mib', 0) if estimate else 0)
            logger.info("event=memory_admission decision=%s", json.dumps(decision, sort_keys=True))
            if not decision["accepted"]:
                raise HTTPException(status_code=503,
                    detail={"message": "Memory admission paused; retry later", "reason": decision["reason"], "memory_estimate": estimate, "admission_snapshot": {"memory_admission":admission.snapshot(sample), "device_memory":sample}, "decision_monotonic":monotonic()},
                    headers={**headers, "Retry-After": "1"})
            if estimate:
                response.headers['X-Memory-Estimate-MiB'] = str(estimate['estimated_incremental_mib'])
                if 'planned_reranker_batch_size' in estimate:
                    response.headers['X-Rerank-Planned-Batch'] = str(estimate['planned_reranker_batch_size'])
                    request = request.model_copy(update={'reranker_batch_size': estimate['planned_reranker_batch_size']})
                stats = sample['stats']
                used = max(stats['used_mb'], stats['total_mb'] - stats['free_mb'])
                app.state.memory_observations[request_id] = {
                    'baseline_device_used_mib': used, 'peak_device_used_mib': used,
                    'estimated_incremental_mib': estimate['estimated_incremental_mib'],
                    'overlapped_other_requests': bool(app.state.tasks)}
        response.headers["X-Request-ID"] = request_id
        try:
            task = asyncio.create_task(process(request, request_id), name="rag-" + request_id)
        except BaseException:
            admission.release(request_id)
            app.state.memory_observations.pop(request_id, None)
            raise
        app.state.tasks.add(task)
        logger.info("request_id=%s event=accepted inflight=%d", request_id, len(app.state.tasks))

        def finished(completed):
            app.state.tasks.discard(completed)
            admission.release(request_id)
            observation = app.state.memory_observations.pop(request_id, None)
            if observation:
                observed = max(0, observation['peak_device_used_mib'] - observation['baseline_device_used_mib'])
                observation['observed_device_increase_mib'] = observed
                observation['attribution'] = 'device-wide samples; shared when other requests overlap, not a per-process allocation counter'
                logger.info('event=memory_estimate_observation request_id=%s observation=%s', request_id, json.dumps(observation, sort_keys=True))
                if not observation['overlapped_other_requests'] and observed > observation['estimated_incremental_mib']:
                    app.state.estimator_fault = 'Observed device increase exceeded the request estimate; recalibration required'
            if memory_admission.enabled:
                logger.info("request_id=%s event=memory_budget_completed", request_id)
            if not completed.cancelled():
                error = completed.exception()
                if error is not None:
                    logger.error("request_id=%s query failed", request_id,
                                 exc_info=(type(error), error, error.__traceback__))

        task.add_done_callback(finished)
        try:
            return await asyncio.shield(task)
        except Exception as error:
            raise HTTPException(status_code=500, detail="Query failed; see server logs",
                                headers=headers) from error

    return app
