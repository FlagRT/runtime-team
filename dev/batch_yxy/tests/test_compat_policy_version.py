from dynamic_batching import BatchCoordinator
from dynamic_batching.types import compatibility_key
from helpers import ManualClock, make_policy, make_request


def test_different_model_ids_never_share_batch():
    co = BatchCoordinator(
        [make_policy(), make_policy(model_id="other-embedding", policy_version="v2")]
    )
    co.submit(make_request("a0", 16, arrival=0))
    co.submit(make_request("a1", 16, arrival=1))
    co.submit(make_request("b0", 16, arrival=2, model_id="other-embedding", model_version="9.9"))
    co.submit(make_request("b1", 16, arrival=3, model_id="other-embedding", model_version="9.9"))
    plans = co.flush()
    assert len(plans) == 2
    keys = {p.compatibility_key for p in plans}
    assert keys == {
        "qwen3-embedding-0.6b|v1.0|default",
        "other-embedding|9.9|default",
    }
    for plan in plans:
        assert len(plan.request_ids) == 2


def test_model_version_difference_separates():
    co = BatchCoordinator(make_policy())
    co.submit(make_request("v1a", 16, arrival=0, model_version="1.0"))
    co.submit(make_request("v2a", 16, arrival=1, model_version="2.0"))
    plans = co.flush()
    assert len(plans) == 2
    assert {p.compatibility_key for p in plans} == {
        "qwen3-embedding-0.6b|1.0|default",
        "qwen3-embedding-0.6b|2.0|default",
    }


def test_execution_profile_difference_separates():
    co = BatchCoordinator(make_policy())
    co.submit(make_request("p0", 16, arrival=0, execution_profile="bf16"))
    co.submit(make_request("p1", 16, arrival=1, execution_profile="fp16"))
    co.submit(make_request("p2", 16, arrival=2))
    plans = co.flush()
    assert len(plans) == 3
    assert {p.compatibility_key for p in plans} == {
        "qwen3-embedding-0.6b|v1.0|bf16",
        "qwen3-embedding-0.6b|v1.0|fp16",
        "qwen3-embedding-0.6b|v1.0|default",
    }


def test_policy_version_pinned_for_queued_requests():
    clock = ManualClock()
    co = BatchCoordinator(make_policy(policy_version="v1", max_batch_size=4), clock=clock)
    co.submit(make_request("old", 16, arrival=clock.now))
    co.register_policy(make_policy(policy_version="v2", max_batch_size=4))
    co.submit(make_request("new", 16, arrival=clock.now))
    plans = co.flush()
    assert len(plans) == 2
    by_id = {p.request_ids[0]: p.policy_version for p in plans}
    assert by_id == {"old": "v1", "new": "v2"}


def test_same_version_reregistered_with_new_content_isolated():
    co = BatchCoordinator(make_policy(policy_version="v1", bucket_upper_bounds=(64,)))
    co.submit(make_request("r0", 16, arrival=0))
    co.register_policy(make_policy(policy_version="v1", bucket_upper_bounds=(128,)))
    co.submit(make_request("r1", 100, arrival=1))
    plans = co.flush()
    assert len(plans) == 2
    assert plans[0].request_ids == ("r0",)
    assert plans[1].request_ids == ("r1",)


def test_compatibility_key_function():
    req = make_request("r0", 16, execution_profile=None)
    assert compatibility_key(req) == "qwen3-embedding-0.6b|v1.0|default"
    req2 = make_request("r0", 16, execution_profile="npu-static")
    assert compatibility_key(req2) == "qwen3-embedding-0.6b|v1.0|npu-static"
