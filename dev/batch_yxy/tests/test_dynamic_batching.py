from dynamic_batching import BatchCoordinator
from helpers import make_policy, make_request


def test_full_batch_dispatches_immediately_and_once():
    co = BatchCoordinator(make_policy(max_batch_size=4, max_wait_ms=50.0))
    for i in range(4):
        co.submit(make_request(f"r{i}", 16, arrival=i))
    plans = co.poll_ready(1_000)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.dispatch_reason == "batch_limit"
    assert plan.request_ids == ("r0", "r1", "r2", "r3")
    assert plan.token_lengths == (16, 16, 16, 16)
    assert plan.total_tokens == 64
    assert co.poll_ready(1_000) == []
    assert co.poll_ready(2_000) == []


def test_exceeding_batch_size_splits_and_tail_waits():
    co = BatchCoordinator(make_policy(max_batch_size=4, max_wait_ms=50.0))
    for i in range(5):
        co.submit(make_request(f"r{i}", 16, arrival=0))
    plans = co.poll_ready(0)
    assert len(plans) == 1
    assert plans[0].request_ids == ("r0", "r1", "r2", "r3")
    assert co.poll_ready(1_000) == []
    plans = co.poll_ready(50_000_000)
    assert len(plans) == 1
    assert plans[0].dispatch_reason == "wait_limit"
    assert plans[0].request_ids == ("r4",)


def test_token_budget_boundary_exact_and_exceed():
    policy = make_policy(
        bucket_upper_bounds=(16,), max_batch_size=8, max_total_tokens=32,
        max_wait_ms=1000.0,
    )
    co = BatchCoordinator(policy)
    for i in range(3):
        co.submit(make_request(f"r{i}", 16, arrival=0))
    plans = co.poll_ready(0)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.dispatch_reason == "token_limit"
    assert plan.token_lengths == (16, 16)
    assert plan.total_tokens == 32
    assert co.pending_count == 1
    assert co.poll_ready(1) == []


def test_wait_limit_boundary_inclusive():
    co = BatchCoordinator(make_policy(max_wait_ms=50.0))
    co.submit(make_request("r0", 16, arrival=10_000_000_000))
    assert co.poll_ready(10_000_000_000 + 50_000_000 - 1) == []
    plans = co.poll_ready(10_000_000_000 + 50_000_000)
    assert len(plans) == 1
    assert plans[0].dispatch_reason == "wait_limit"


def test_max_wait_zero_dispatches_immediately():
    co = BatchCoordinator(make_policy(max_wait_ms=0.0))
    co.submit(make_request("r0", 16, arrival=5))
    plans = co.poll_ready(5)
    assert len(plans) == 1
    assert plans[0].dispatch_reason == "wait_limit"


def test_next_wakeup_is_min_over_pending():
    policy = make_policy(max_batch_size=2, max_wait_ms=100.0)
    co = BatchCoordinator(policy)
    assert co.next_wakeup_mono_ns() is None
    co.submit(make_request("r0", 16, arrival=1_000))
    co.submit(make_request("r1", 16, arrival=2_000))
    co.submit(make_request("r2", 16, arrival=3_000))
    assert co.next_wakeup_mono_ns() == 1_000 + 100_000_000
    plans = co.poll_ready(1_000 + 100_000_000)
    assert len(plans) == 1
    assert plans[0].request_ids == ("r0", "r1")
    assert co.next_wakeup_mono_ns() == 3_000 + 100_000_000
    co.poll_ready(3_000 + 100_000_000)
    assert co.next_wakeup_mono_ns() is None


def test_single_request_via_flush():
    co = BatchCoordinator(make_policy(max_batch_size=8))
    co.submit(make_request("r0", 16))
    assert co.poll_ready(1) == []
    plans = co.flush()
    assert len(plans) == 1
    assert plans[0].dispatch_reason == "flush"
    assert co.flush() == []


def test_close_is_idempotent():
    co = BatchCoordinator(make_policy())
    co.submit(make_request("r0", 16))
    first = co.close()
    assert len(first) == 1 and first[0].dispatch_reason == "flush"
    assert co.close() == []
    assert co.flush() == []
    assert co.poll_ready(1) == []


def test_big_request_does_not_block_small_ones():
    policy = make_policy(
        bucket_upper_bounds=(1000,), max_batch_size=3, max_total_tokens=1000,
        max_wait_ms=50.0,
    )
    co = BatchCoordinator(policy)
    co.submit(make_request("s1", 10, arrival=0))
    co.submit(make_request("big", 1000, arrival=1))
    co.submit(make_request("s2", 10, arrival=2))
    co.submit(make_request("s3", 10, arrival=3))
    plans = co.poll_ready(4)
    assert len(plans) == 2
    assert plans[0].dispatch_reason == "batch_limit"
    assert plans[0].request_ids == ("s1", "s2", "s3")
    assert plans[1].dispatch_reason == "token_limit"
    assert plans[1].request_ids == ("big",)
    assert plans[1].total_tokens == 1000
    assert co.pending_count == 0


def test_seed_is_earliest_arrival_in_bucket():
    co = BatchCoordinator(make_policy(max_batch_size=3))
    for rid, arrival in (("late", 100), ("early", 1), ("mid", 50)):
        co.submit(make_request(rid, 16, arrival=arrival))
    plans = co.flush()
    assert plans[0].request_ids == ("early", "mid", "late")


def test_multiple_batches_per_poll_when_many_expired():
    co = BatchCoordinator(make_policy(max_batch_size=2, max_wait_ms=0.0))
    for i in range(4):
        co.submit(make_request(f"r{i}", 16, arrival=0))
    plans = co.poll_ready(0)
    assert len(plans) == 2
    assert plans[0].request_ids == ("r0", "r1")
    assert plans[1].request_ids == ("r2", "r3")


def test_plans_carry_budget_and_padding_fields():
    co = BatchCoordinator(make_policy(max_batch_size=4, max_total_tokens=1024))
    for i, tok in enumerate((16, 32, 48)):
        co.submit(make_request(f"r{i}", tok, arrival=i))
    plan = co.flush()[0]
    assert plan.total_tokens == 96
    assert plan.padded_length == 48
    assert plan.policy_version == "v1"
    assert plan.compatibility_key == "qwen3-embedding-0.6b|v1.0|default"
