"""Strict, versioned JSON configuration; no framework imports."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from sched_engine.models.registry import get_profile
from .contracts import positive_int, finite_number


@dataclass(frozen=True)
class BindingConfig:
    id: str
    model: str
    backend: str
    options: dict
    policy: dict
    model_version: str = "default"


@dataclass(frozen=True)
class NodeConfig:
    bindings: tuple[BindingConfig, ...]
    max_pending: int = 1024
    result_history: int = 1024
    planner: str = "batch-coordinator"
    schema_version: int = 1


def _keys(value: dict, allowed: set, context: str):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if value.keys() - allowed:
        raise ValueError(f"unknown {context} fields: {sorted(value.keys() - allowed)}")


def parse_config(data: dict) -> NodeConfig:
    _keys(data, {"schema_version", "bindings", "max_pending", "result_history", "planner"}, "config")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    bindings = data.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("bindings must be a nonempty array")
    parsed = []
    ids = set()
    from dynamic_batching.types import BatchPolicy
    for binding in bindings:
        _keys(binding, {"id", "model", "backend", "options", "policy", "model_version"}, "binding")
        if not {"id", "model", "backend", "policy"} <= binding.keys():
            raise ValueError("binding requires id, model, backend, policy")
        for field in ("id", "model", "backend", "model_version"):
            if not isinstance(binding.get(field, "default"), str) or not binding.get(field, "default"):
                raise ValueError(f"binding {field} must be a nonempty string")
        if "|" in binding["id"] or "|" in binding.get("model_version", ""):
            raise ValueError("binding id/model_version cannot contain compatibility separator '|'")
        if binding["id"] in ids:
            raise ValueError("duplicate binding id")
        ids.add(binding["id"])
        profile = get_profile(binding["model"])
        if profile is None:
            raise ValueError(f"unknown model contract: {binding['model']}")
        options = binding.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("backend options must be an object")
        if binding["backend"] == "vllm-embed" and profile.task != "embed":
            raise ValueError("vllm-embed only implements embedding; reranking requires a task adapter")
        if not isinstance(binding["policy"], dict):
            raise ValueError("policy must be an object")
        policy = dict(binding["policy"])
        _keys(policy, {"policy_version", "bucket_upper_bounds", "max_batch_size", "max_total_tokens",
                       "max_wait_ms", "allow_adjacent_bucket_merge", "max_padding_ratio"}, "policy")
        for key in ("max_batch_size", "max_total_tokens"):
            positive_int(policy.get(key), key)
        finite_number(policy.get("max_wait_ms"), "max_wait_ms")
        if "allow_adjacent_bucket_merge" in policy and not isinstance(policy["allow_adjacent_bucket_merge"], bool):
            raise ValueError("allow_adjacent_bucket_merge must be boolean")
        if policy.get("max_padding_ratio") is not None:
            ratio = finite_number(policy["max_padding_ratio"], "max_padding_ratio")
            if ratio >= 1:
                raise ValueError("max_padding_ratio must be < 1")
        bounds = policy.get("bucket_upper_bounds")
        if not isinstance(bounds, list) or not bounds:
            raise ValueError("bucket_upper_bounds must be a nonempty array")
        # Use binding ID as the coordinator's model key: two models, model
        # revisions or execution configurations can never share a policy lane.
        BatchPolicy(model_id=binding["id"], **{**policy, "bucket_upper_bounds": tuple(bounds)})
        parsed.append(BindingConfig(**{**binding, "options": options, "policy": policy}))
    planner = data.get("planner", "batch-coordinator")
    if not isinstance(planner, str) or not planner:
        raise ValueError("planner must name a plugin")
    return NodeConfig(tuple(parsed), positive_int(data.get("max_pending", 1024), "max_pending"),
                      positive_int(data.get("result_history", 1024), "result_history"), planner)


def load_config(path: str) -> NodeConfig:
    return parse_config(json.loads(Path(path).read_text(encoding="utf-8")))
