import pytest

from dynamic_batching import BatchPolicy, policy_from_config, policy_from_json_file


def test_full_config_parses(tmp_path):
    cfg = {
        "model_id": "Qwen3-Embedding-0.6B",
        "policy_version": "v1",
        "bucket_upper_bounds": [64, 128, 256],
        "max_batch_size": 32,
        "max_total_tokens": 8192,
        "max_wait_ms": 50.0,
        "allow_adjacent_bucket_merge": True,
        "max_padding_ratio": 0.3,
    }
    policy = policy_from_config(cfg)
    assert isinstance(policy, BatchPolicy)
    assert policy.bucket_upper_bounds == (64, 128, 256)
    assert policy.allow_adjacent_bucket_merge is True
    assert policy.max_padding_ratio == 0.3


def test_missing_required_key_raises():
    with pytest.raises(KeyError) as exc:
        policy_from_config({"model_id": "m", "policy_version": "v1"})
    assert "bucket_upper_bounds" in str(exc.value)


def test_invalid_config_surfaces_policy_validation():
    with pytest.raises(Exception):
        policy_from_config(
            {
                "model_id": "m",
                "policy_version": "v1",
                "bucket_upper_bounds": [128, 64],
                "max_batch_size": 32,
                "max_total_tokens": 8192,
                "max_wait_ms": 50.0,
            }
        )


def test_json_file_roundtrip(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        """
        {
          "model_id": "Qwen3-Embedding-0.6B",
          "policy_version": "v1-json",
          "bucket_upper_bounds": [64, 1024],
          "max_batch_size": 8,
          "max_total_tokens": 8192,
          "max_wait_ms": 10.0
        }
        """,
        encoding="utf-8",
    )
    policy = policy_from_json_file(path)
    assert policy.max_length == 1024
    assert policy.policy_version == "v1-json"
