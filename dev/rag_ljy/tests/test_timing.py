import pytest

from rag_engine.pipeline import RetrievalPipeline
from rag_engine.retrieval import hybrid_search
from rag_engine.timing import TIMING_FIELDS


class Clock:
    def __init__(self):
        self.ns = 0

    def now(self):
        return self.ns

    def advance(self, milliseconds):
        self.ns += milliseconds * 1_000_000


def make_runtime(monkeypatch, fail_dense=False):
    import rag_engine.retrieval as retrieval
    import rag_engine.timing as timing

    clock = Clock()
    monkeypatch.setattr(timing, "perf_counter_ns", clock.now)
    calls = []
    hit = {"_id": "chunk", "_score": 1.0, "_source": {"text": "document"}}

    class Store:
        def bm25_search(self, *_args, **_kwargs):
            calls.append("bm25-start")
            clock.advance(40)
            calls.append("bm25-end")
            return [hit]

        def dense_search(self, *_args, **_kwargs):
            calls.append("dense-start")
            clock.advance(60)
            calls.append("dense-end")
            if fail_dense:
                raise RuntimeError("dense failed")
            return [hit]

    class Embedder:
        def encode_queries(self, *_args, **_kwargs):
            clock.advance(10)
            return [[1.0, 0.0]]

    class Reranker:
        def rerank(self, _query, candidates, **_kwargs):
            clock.advance(90)
            return [{**candidate, "_rerank_score": 0.9} for candidate in candidates]

    original = retrieval.reciprocal_rank_fusion

    def rrf(*args, **kwargs):
        clock.advance(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(retrieval, "reciprocal_rank_fusion", rrf)
    store = Store()
    return RetrievalPipeline(store, Embedder(), Reranker(), execution_mode="sequential"), calls


def test_search_interval_is_sequential_sum_and_excludes_rrf(monkeypatch):
    pipeline, calls = make_runtime(monkeypatch)
    timings = {}
    result = hybrid_search(pipeline.store, "query", [1.0, 0.0], timings=timings)
    assert calls == ["bm25-start", "bm25-end", "dense-start", "dense-end"]
    assert timings == {"bm25_ms": 40.0, "dense_ms": 60.0, "search_ms": 100.0, "rrf_ms": 2.0}
    assert result[0]["_retrieval_ranks"] == {"bm25": 1, "dense": 1}


def test_pipeline_timings_are_complete_and_do_not_change_results(monkeypatch):
    pipeline, _ = make_runtime(monkeypatch)
    untimed = pipeline.retrieve("query")
    timings = {"stale": 999.0}
    timed = pipeline.retrieve("query", timings=timings)
    assert timed == untimed
    assert set(timings) == set(TIMING_FIELDS)
    assert timings == {
        "embedding_ms": 10.0, "bm25_ms": 40.0, "dense_ms": 60.0,
        "search_ms": 100.0, "rrf_ms": 2.0, "rerank_ms": 90.0, "total_ms": 202.0,
    }


def test_failed_stage_records_elapsed_time_and_does_not_run_later_stages(monkeypatch):
    pipeline, _ = make_runtime(monkeypatch, fail_dense=True)
    timings = {}
    with pytest.raises(RuntimeError, match="dense failed"):
        pipeline.retrieve("query", timings=timings)
    assert timings == {
        "embedding_ms": 10.0, "bm25_ms": 40.0, "dense_ms": 60.0,
        "search_ms": 100.0, "total_ms": 110.0,
    }


def test_no_clock_reads_when_timing_is_disabled(monkeypatch):
    pipeline, _ = make_runtime(monkeypatch)

    def forbidden():
        pytest.fail("untimed queries must not read the clock")

    monkeypatch.setattr("rag_engine.timing.perf_counter_ns", forbidden)
    assert pipeline.retrieve("query")
