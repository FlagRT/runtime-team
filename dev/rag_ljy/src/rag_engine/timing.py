"""Opt-in client-observed wall-clock timing, including database waits."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter_ns


TIMING_FIELDS = (
    "embedding_ms", "bm25_ms", "dense_ms", "search_ms",
    "rrf_ms", "rerank_ms", "total_ms",
)


@contextmanager
def measure_stage(
    timings: dict[str, float] | None, field: str
) -> Iterator[None]:
    """Record the actual interval, even on failure; never sum overlapping work.

    Model APIs return CPU lists, so their normal device-to-host transfer already
    waits for the outputs. These are Python-call timings, not NPU kernel timings.
    """
    if timings is None:
        yield
        return
    start = perf_counter_ns()
    try:
        yield
    finally:
        timings[field] = (perf_counter_ns() - start) / 1_000_000
