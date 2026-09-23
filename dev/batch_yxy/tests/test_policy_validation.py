import pytest

from dynamic_batching import BatchPolicy, PolicyValidationError
from helpers import make_policy


def test_bounds_must_be_strictly_increasing():
    with pytest.raises(PolicyValidationError):
        make_policy(bucket_upper_bounds=(64, 64, 128))
    with pytest.raises(PolicyValidationError):
        make_policy(bucket_upper_bounds=(128, 64))


def test_bounds_must_be_nonempty_positive_ints():
    with pytest.raises(PolicyValidationError):
        make_policy(bucket_upper_bounds=())
    with pytest.raises(PolicyValidationError):
        make_policy(bucket_upper_bounds=(0, 128))
    with pytest.raises(PolicyValidationError):
        make_policy(bucket_upper_bounds=(64.5, 128))


def test_max_total_tokens_must_cover_max_length():
    with pytest.raises(PolicyValidationError):
        make_policy(bucket_upper_bounds=(128, 256), max_total_tokens=128)


def test_invalid_batch_size_wait_and_padding_ratio():
    with pytest.raises(PolicyValidationError):
        make_policy(max_batch_size=0)
    with pytest.raises(PolicyValidationError):
        make_policy(max_wait_ms=-1.0)
    with pytest.raises(PolicyValidationError):
        make_policy(max_padding_ratio=1.0)
    with pytest.raises(PolicyValidationError):
        make_policy(max_padding_ratio=-0.1)


def test_empty_model_id_or_version_rejected():
    with pytest.raises(PolicyValidationError):
        make_policy(model_id="")
    with pytest.raises(PolicyValidationError):
        make_policy(policy_version="")


def test_bucket_index_and_max_length():
    policy = make_policy(bucket_upper_bounds=(32, 64, 128))
    assert policy.max_length == 128
    assert policy.bucket_index(1) == 0
    assert policy.bucket_index(32) == 0
    assert policy.bucket_index(33) == 1
    assert policy.bucket_index(64) == 1
    assert policy.bucket_index(65) == 2
    assert policy.bucket_index(128) == 2
    assert policy.bucket_index(129) is None


def test_wait_window_ns():
    assert make_policy(max_wait_ms=50.0).wait_window_ns == 50_000_000
    assert make_policy(max_wait_ms=0).wait_window_ns == 0


def test_single_bucket_policy_valid():
    policy = make_policy(bucket_upper_bounds=(1024,), max_total_tokens=1024)
    assert policy.bucket_index(1024) == 0
    assert policy.bucket_index(1025) is None
