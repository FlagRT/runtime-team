"""Prove overlap with synchronization, not flaky elapsed-time thresholds."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from rag_engine.pipeline import RetrievalPipeline
from rag_engine.retrieval import hybrid_search, reciprocal_rank_fusion


def hit(identifier, score):
    return {"_id": identifier, "_score": score, "_source": {"text": identifier}}


class OverlapStore:
    def __init__(self):
        self.barrier = threading.Barrier(2)
        self.threads = []
        self.completed = set()
        self.lock = threading.Lock()

    def _search(self, channel, result):
        with self.lock:
            self.threads.append(threading.get_ident())
        # A sequential implementation cannot pass this barrier.
        self.barrier.wait(timeout=3)
        with self.lock:
            self.completed.add(channel)
        return result

    def bm25_search(self, _query, size):
        return self._search("bm25", [hit("shared", 2.0), hit("sparse", 1.0)])

    def dense_search(self, _vector, size, num_candidates):
        return self._search("dense", [hit("dense", 0.9), hit("shared", 0.8)])


def test_concurrent_search_overlaps_and_preserves_rrf_results(monkeypatch):
    import rag_engine.retrieval as retrieval

    store = OverlapStore()
    original = retrieval.reciprocal_rank_fusion

    def after_both(*args, **kwargs):
        assert store.completed == {"bm25", "dense"}
        return original(*args, **kwargs)

    monkeypatch.setattr(retrieval, "reciprocal_rank_fusion", after_both)
    timings = {}
    result = hybrid_search(
        store, "query", [1.0], execution_mode="concurrent", timings=timings,
    )
    expected = reciprocal_rank_fusion({
        "bm25": [hit("shared", 2.0), hit("sparse", 1.0)],
        "dense": [hit("dense", 0.9), hit("shared", 0.8)],
    })
    assert result == expected
    assert len(set(store.threads)) == 2
    assert threading.get_ident() not in store.threads
    assert set(timings) == {"bm25_ms", "dense_ms", "search_ms", "rrf_ms"}
    assert timings["search_ms"] >= max(timings["bm25_ms"], timings["dense_ms"])


def test_pipeline_reuses_search_pool_and_keeps_models_on_inference_thread():
    store = OverlapStore()
    model_threads = []

    class Embedder:
        def encode_queries(self, *_args, **_kwargs):
            model_threads.append(threading.get_ident())
            return [[1.0]]

    class Reranker:
        def rerank(self, _query, candidates, **_kwargs):
            assert store.completed == {"bm25", "dense"}
            model_threads.append(threading.get_ident())
            return candidates

    pipeline = RetrievalPipeline(store, Embedder(), Reranker())
    pool = pipeline._search_executor
    assert pipeline.execution_mode == "concurrent"
    try:
        first = pipeline.retrieve("first")
        store.completed.clear()
        second = pipeline.retrieve("second")
        assert first == second
        assert pipeline._search_executor is pool
        assert set(store.threads[:2]) == set(store.threads[2:])
        assert model_threads == [threading.get_ident()] * 4
    finally:
        pipeline.close()
    pipeline.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.retrieve("after shutdown")
    with pytest.raises(RuntimeError):
        pool.submit(lambda: None)


@pytest.mark.parametrize("failed_channel", ["bm25", "dense"])
def test_failure_waits_for_sibling_and_never_returns_partial_results(failed_channel):
    barrier = threading.Barrier(2)
    failed = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class FailingStore:
        def search(self, channel):
            barrier.wait(timeout=3)
            if channel == failed_channel:
                failed.set()
                raise RuntimeError(f"{channel} failed")
            assert release.wait(timeout=3)
            completed.set()
            return [hit("sibling", 1.0)]

        def bm25_search(self, *_args, **_kwargs):
            return self.search("bm25")

        def dense_search(self, *_args, **_kwargs):
            return self.search("dense")

    timings = {}
    with ThreadPoolExecutor(max_workers=1) as caller:
        pending = caller.submit(
            hybrid_search, FailingStore(), "query", [1.0],
            execution_mode="concurrent", timings=timings,
        )
        try:
            assert failed.wait(timeout=3)
            assert not pending.done()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match=f"{failed_channel} failed"):
            pending.result(timeout=3)
    assert completed.is_set()
    assert set(timings) == {"bm25_ms", "dense_ms", "search_ms"}


def test_invalid_mode_is_rejected_before_search():
    with pytest.raises(ValueError, match="execution_mode"):
        hybrid_search(None, "query", [1.0], execution_mode="invalid")
    with pytest.raises(ValueError, match="execution_mode"):
        RetrievalPipeline(None, None, None, execution_mode="invalid")


def test_second_submission_failure_drains_first_request():
    rejected = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class Store:
        def bm25_search(self, *_args, **_kwargs):
            assert release.wait(timeout=3)
            completed.set()
            return []

    with ThreadPoolExecutor(max_workers=1) as worker, ThreadPoolExecutor(max_workers=1) as caller:
        class RejectSecond:
            calls = 0

            def submit(self, fn):
                self.calls += 1
                if self.calls == 2:
                    rejected.set()
                    raise RuntimeError("second submission rejected")
                return worker.submit(fn)

        timings = {}
        pending = caller.submit(
            hybrid_search, Store(), "query", [1.0], execution_mode="concurrent",
            executor=RejectSecond(), timings=timings,
        )
        try:
            assert rejected.wait(timeout=3)
            assert not pending.done()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="submission rejected"):
            pending.result(timeout=3)
    assert completed.is_set()
    assert set(timings) == {"bm25_ms", "search_ms"}


def test_sequential_pipeline_has_no_search_workers():
    pipeline = RetrievalPipeline(None, None, None, execution_mode="sequential")
    assert pipeline._search_executor is None
    pipeline.close()
