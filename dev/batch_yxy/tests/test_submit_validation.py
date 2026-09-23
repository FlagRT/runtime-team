import pytest

from dynamic_batching import (
    BatchCoordinator,
    ClosedCoordinatorError,
    DuplicateRequestError,
    RequestTooLongError,
    RequestValidationError,
    UnknownPolicyError,
)
from helpers import make_policy, make_request


def test_empty_request_id_rejected():
    co = BatchCoordinator(make_policy())
    with pytest.raises(RequestValidationError):
        co.submit(make_request("", 16))


def test_zero_or_negative_token_count_rejected():
    co = BatchCoordinator(make_policy())
    with pytest.raises(RequestValidationError):
        co.submit(make_request("r1", 0))
    with pytest.raises(RequestValidationError):
        co.submit(make_request("r1", -5))


def test_over_max_length_rejected_explicitly():
    co = BatchCoordinator(make_policy(bucket_upper_bounds=(128,)))
    with pytest.raises(RequestTooLongError) as exc:
        co.submit(make_request("r1", 129))
    assert "明确拒绝" in str(exc.value)


def test_unknown_model_rejected():
    co = BatchCoordinator(make_policy())
    with pytest.raises(UnknownPolicyError):
        co.submit(make_request("r1", 16, model_id="other-model"))


def test_bad_model_version_or_arrival_rejected():
    co = BatchCoordinator(make_policy())
    with pytest.raises(RequestValidationError):
        co.submit(make_request("r1", 16, model_version=""))
    with pytest.raises(RequestValidationError):
        co.submit(make_request("r1", 16, arrival_mono_ns=-1))


def test_duplicate_id_rejected_while_queued():
    co = BatchCoordinator(make_policy())
    co.submit(make_request("r1", 16))
    with pytest.raises(DuplicateRequestError):
        co.submit(make_request("r1", 16))


def test_duplicate_id_rejected_after_dispatch_by_default():
    co = BatchCoordinator(make_policy(max_wait_ms=0))
    co.submit(make_request("r1", 16))
    co.poll_ready(1_000_000)
    with pytest.raises(DuplicateRequestError):
        co.submit(make_request("r1", 16))


def test_non_strict_mode_allows_reuse_after_dispatch():
    co = BatchCoordinator(make_policy(max_wait_ms=0), strict_duplicate_detection=False)
    co.submit(make_request("r1", 16))
    co.poll_ready(1_000_000)
    co.submit(make_request("r1", 16))
    assert co.pending_count == 1


def test_submit_after_close_rejected():
    co = BatchCoordinator(make_policy())
    co.close()
    with pytest.raises(ClosedCoordinatorError):
        co.submit(make_request("r1", 16))


def test_valid_submit_then_flush():
    co = BatchCoordinator(make_policy())
    co.submit(make_request("r1", 16))
    assert co.pending_count == 1
    plans = co.flush()
    assert len(plans) == 1
    assert plans[0].request_ids == ("r1",)
    assert co.pending_count == 0
