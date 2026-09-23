import pytest

from dynamic_batching import RequestTooLongError, padding_ratio
from dynamic_batching.coordinator import BatchCoordinator
from helpers import make_policy, make_request


def test_bucket_assignment_at_boundaries():
    co = BatchCoordinator(
        make_policy(bucket_upper_bounds=(32, 64, 128), max_batch_size=8)
    )
    for rid, tok in (("a", 1), ("b", 32), ("c", 33), ("d", 64), ("e", 65), ("f", 128)):
        co.submit(make_request(rid, tok))
    with pytest.raises(RequestTooLongError):
        co.submit(make_request("g", 129))
    plans = co.flush()
    by_bucket = {p.bucket_ids: p.request_ids for p in plans}
    assert by_bucket[(0,)] == ("a", "b")
    assert by_bucket[(1,)] == ("c", "d")
    assert by_bucket[(2,)] == ("e", "f")


def test_merge_off_keeps_buckets_separate():
    policy = make_policy(
        bucket_upper_bounds=(64, 1024), max_batch_size=4, max_total_tokens=4096,
        max_wait_ms=50.0,
    )
    co = BatchCoordinator(policy)
    for i in range(2):
        co.submit(make_request(f"s{i}", 16, arrival=i))
    for i in range(2):
        co.submit(make_request(f"l{i}", 512, arrival=10 + i))
    assert co.poll_ready(20) == []
    plans = co.flush()
    assert len(plans) == 2
    assert {p.bucket_ids for p in plans} == {(0,), (1,)}
    assert all(len(p.bucket_ids) == 1 for p in plans)


def test_adjacent_merge_on_combines_buckets():
    policy = make_policy(
        bucket_upper_bounds=(64, 1024),
        max_batch_size=4,
        max_total_tokens=4096,
        allow_adjacent_bucket_merge=True,
    )
    co = BatchCoordinator(policy)
    for i in range(2):
        co.submit(make_request(f"s{i}", 16, arrival=i))
    for i in range(2):
        co.submit(make_request(f"l{i}", 512, arrival=10 + i))
    plans = co.flush()
    assert len(plans) == 1
    plan = plans[0]
    assert plan.bucket_ids == (0, 1)
    assert plan.request_ids == ("s0", "s1", "l0", "l1")
    assert co.stats[-1].cross_bucket_merge is True


def test_adjacent_merge_still_respects_hard_budget():
    policy = make_policy(
        bucket_upper_bounds=(64, 1024),
        max_batch_size=8,
        max_total_tokens=1024,
        allow_adjacent_bucket_merge=True,
    )
    co = BatchCoordinator(policy)
    for i in range(2):
        co.submit(make_request(f"s{i}", 16, arrival=i))
    for i in range(2):
        co.submit(make_request(f"l{i}", 512, arrival=10 + i))
    plans = co.flush()
    assert len(plans) == 2
    first = plans[0]
    assert first.bucket_ids == (0, 1)
    assert first.token_lengths == (16, 16, 512)
    assert first.total_tokens == 544
    assert plans[1].request_ids == ("l1",)


def test_non_adjacent_buckets_never_merge():
    policy = make_policy(
        bucket_upper_bounds=(32, 64, 128),
        max_batch_size=8,
        max_total_tokens=4096,
        allow_adjacent_bucket_merge=True,
    )
    co = BatchCoordinator(policy)
    co.submit(make_request("short", 16, arrival=0))
    co.submit(make_request("long", 96, arrival=1))
    plans = co.flush()
    assert len(plans) == 2
    assert {p.bucket_ids for p in plans} == {(0,), (2,)}


def test_padding_ratio_blocks_cross_length_merge():
    policy = make_policy(
        bucket_upper_bounds=(32, 256),
        max_batch_size=8,
        max_total_tokens=4096,
        allow_adjacent_bucket_merge=True,
        max_padding_ratio=0.5,
    )
    co = BatchCoordinator(policy)
    co.submit(make_request("s0", 16, arrival=0))
    co.submit(make_request("s1", 16, arrival=1))
    co.submit(make_request("l0", 200, arrival=2))
    plans = co.flush()
    assert len(plans) == 2
    assert plans[0].request_ids == ("s0", "s1")
    assert plans[1].request_ids == ("l0",)

    same = BatchCoordinator(
        make_policy(
            bucket_upper_bounds=(32, 256),
            max_batch_size=8,
            max_total_tokens=4096,
            allow_adjacent_bucket_merge=True,
        )
    )
    same.submit(make_request("s0", 16, arrival=0))
    same.submit(make_request("s1", 16, arrival=1))
    same.submit(make_request("l0", 200, arrival=2))
    merged = same.flush()
    assert len(merged) == 1
    assert merged[0].bucket_ids == (0, 1)
    assert merged[0].padded_length == 200


def test_low_traffic_bucket_not_starved_by_busy_bucket():
    policy = make_policy(
        bucket_upper_bounds=(64, 1024), max_batch_size=4, max_total_tokens=4096,
        max_wait_ms=100.0,
    )
    co = BatchCoordinator(policy)
    co.submit(make_request("lonely", 16, arrival=0))
    for i in range(4):
        co.submit(make_request(f"l{i}", 512, arrival=10))
    plans = co.poll_ready(20)
    assert len(plans) == 1
    assert plans[0].dispatch_reason == "batch_limit"
    assert "lonely" not in plans[0].request_ids
    plans = co.poll_ready(100_000_000)
    assert len(plans) == 1
    assert plans[0].dispatch_reason == "wait_limit"
    assert plans[0].request_ids == ("lonely",)


def test_padded_length_and_padding_ratio_consistent():
    co = BatchCoordinator(make_policy(max_batch_size=8, bucket_upper_bounds=(128,)))
    for rid, tok in (("a", 16), ("b", 48), ("c", 33)):
        co.submit(make_request(rid, tok))
    plan = co.flush()[0]
    assert plan.padded_length == 48
    expected = (3 * 48 - 97) / (3 * 48)
    assert abs(padding_ratio(plan.token_lengths) - expected) < 1e-12
    record = co.stats[-1]
    assert record.padding_ratio == pytest.approx(expected)
    assert record.padded_length == 48
