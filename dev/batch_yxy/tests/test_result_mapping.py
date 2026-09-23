import numpy as np
import pytest

from dynamic_batching import (
    DynamicBatchEmbedder,
    FakeEmbedExecutor,
    ResultCountMismatchError,
)


def mixed_items(n=10):
    return [(f"req-{i}", f"text {i} " + "word " * ((i * 7) % 40)) for i in range(n)]


def test_dynamic_returns_all_vectors_keyed_by_id():
    executor = FakeEmbedExecutor()
    embedder = DynamicBatchEmbedder(
        executor,
        make_policy_embedder(max_batch_size=4, max_wait_ms=5.0),
    )
    items = mixed_items()
    result = embedder.run(items)
    assert result.mode == "dynamic"
    assert set(result.vectors) == {rid for rid, _ in items}
    for rid, text in items:
        assert np.array_equal(result.vectors[rid], executor.vector(text))


def test_baseline_equals_dynamic_with_fake_executor():
    executor = FakeEmbedExecutor()
    items = mixed_items(20)
    baseline = DynamicBatchEmbedder(executor).run(items)
    dynamic = DynamicBatchEmbedder(
        executor, make_policy_embedder(max_batch_size=8, max_wait_ms=5.0)
    ).run(items)
    for rid, _ in items:
        assert np.allclose(baseline.vectors[rid], dynamic.vectors[rid], atol=0)


def test_result_count_mismatch_raises():
    class ShortExecutor(FakeEmbedExecutor):
        def embed(self, texts):
            return super().embed(texts)[:-1]

    embedder = DynamicBatchEmbedder(
        ShortExecutor(), make_policy_embedder(max_batch_size=4, max_wait_ms=5.0)
    )
    with pytest.raises(ResultCountMismatchError) as exc:
        embedder.run(mixed_items(6))
    assert exc.value.batch_id.startswith("batch-")


def test_timings_and_dispatch_records_populated():
    executor = FakeEmbedExecutor()
    embedder = DynamicBatchEmbedder(
        executor, make_policy_embedder(max_batch_size=4, max_wait_ms=5.0)
    )
    result = embedder.run(mixed_items(10))
    assert len(result.timings) == 10
    for timing in result.timings.values():
        assert timing.e2e_ms >= timing.queue_ms >= 0
    assert result.dispatch_records
    plan_ids = {r.batch_id for r in result.dispatch_records}
    assert set(result.batch_exec_ms) == plan_ids


def test_wakeup_drives_clock_advance_without_real_sleep():
    from helpers import ManualClock

    clock = ManualClock()
    advances = []

    def fake_sleep(seconds):
        advances.append(seconds)
        clock.advance_ms(seconds * 1000)

    executor = FakeEmbedExecutor()
    embedder = DynamicBatchEmbedder(
        executor,
        make_policy_embedder(max_batch_size=8, max_wait_ms=10.0),
        clock=clock,
        sleep=fake_sleep,
    )
    result = embedder.run(mixed_items(3))
    assert len(result.vectors) == 3
    assert advances and max(advances) > 0


def make_policy_embedder(max_batch_size, max_wait_ms):
    from helpers import make_policy

    return make_policy(
        bucket_upper_bounds=(256,),
        max_batch_size=max_batch_size,
        max_total_tokens=2048,
        max_wait_ms=max_wait_ms,
    )
