"""Execution plugins. Model payloads and output shapes stay in this layer."""
from __future__ import annotations

import hashlib
from importlib.metadata import entry_points
import json

from sched_engine.models.registry import get_profile
from flagscheduler.deployment import load_symbol
from .contracts import ExecutionOutput, positive_int


class MockBackend:
    """Deterministic protocol fixture, never presented as actual inference."""
    def __init__(self, binding):
        self.profile = get_profile(binding.model)

    def count_cost(self, payload):
        if self.profile.cost_metric == "token_length":
            if not isinstance(payload, str) or not payload.strip():
                raise ValueError("mock text input must be nonempty text")
            return len(payload.split())  # Mock only; never used for real models.
        if not isinstance(payload, dict):
            raise ValueError("mock structured input must be an object")
        return positive_int(payload.get(self.profile.cost_metric), self.profile.cost_metric)

    def execute(self, inputs):
        outputs = []
        for item in inputs:
            digest = hashlib.sha256(json.dumps(item.payload, sort_keys=True).encode()).digest()
            scores = [int.from_bytes(digest[i:i + 4], "big") / (2**32 - 1) for i in range(0, 32, 4)]
            value = scores if self.profile.task == "embed" else (
                {"ctr": scores[0], "cvr": scores[1]} if self.profile.output_kind == "multi-head" else scores[0])
            outputs.append(ExecutionOutput(item.request_id, {"mock": True, "output": value}))
        return outputs

    def close(self):
        pass


class VllmEmbeddingBackend:
    """Lazy in-process LLM.embed path; must run in a vLLM-capable Python.

    This controls outer batches. Internal scheduler selection comes from the
    saved environment catalog, or an explicit version-pinned llm_options entry.
    """
    def __init__(self, binding):
        from .deployment import plugin_environment
        from .catalog import SchedulerCatalog
        import os
        import sys
        execution_env = plugin_environment()
        for key in ("TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "PYTHONDONTWRITEBYTECODE"):
            os.environ[key] = execution_env[key]
        options = dict(binding.options)
        unknown = options.keys() - {"model_path", "llm_options", "expect_version", "scheduler_state_dir"}
        if unknown:
            raise ValueError(f"unknown vllm-embed options: {sorted(unknown)}")
        model_path = options.get("model_path")
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("vllm-embed requires explicit model_path")
        llm_options = dict(options.get("llm_options", {}))
        if "model" in llm_options:
            raise ValueError("use model_path rather than llm_options.model")
        if llm_options.get("async_scheduling") is not None and not isinstance(llm_options["async_scheduling"], bool):
            raise ValueError("async_scheduling must be boolean")
        mode = "async" if llm_options.get("async_scheduling") else "sync"
        catalog = SchedulerCatalog(sys.executable, mode=mode, state_dir=options.get("scheduler_state_dir"))
        if "scheduler_cls" in llm_options:
            selected_class = llm_options["scheduler_cls"]
            choice = next((e for e in catalog.view()["scripts"] if (
                e["default"] if selected_class is None else e["class_name"] == selected_class)), None)
            if choice is None:
                choice = {"id": "explicit", "class_name": selected_class, "default": False, "plugin_paths": []}
            exact_version = options.get("expect_version")
        else:
            choice = catalog.current()
            exact_version = options.get("expect_version", choice["expect_version"])
            if not choice["default"]:
                llm_options["scheduler_cls"] = choice["class_name"]
        llm_options["async_scheduling"] = mode == "async"
        if llm_options.get("scheduler_cls") is not None:
            from importlib.metadata import version
            if exact_version != version("vllm"):
                raise ValueError("custom scheduler requires exact expect_version")
        for path in reversed(choice.get("plugin_paths", [])):
            if path not in sys.path:
                sys.path.insert(0, path)
        os.environ["PYTHONPATH"] = plugin_environment(choice.get("plugin_paths", []))["PYTHONPATH"]
        self.scheduler_info = {**choice, "current": True,
                               "labels": (["Default"] if choice["default"] else []) + ["Currently"],
                               "mode": mode, "selection_scope": "binding configuration at startup"}
        from vllm import LLM
        from dynamic_batching.vllm_embed_adapter import VllmEmbedExecutor
        self.llm = LLM(model=model_path, **llm_options)
        self.executor = VllmEmbedExecutor(self.llm)
        self.tokenizer = self.llm.get_tokenizer()

    def count_cost(self, payload):
        if not isinstance(payload, str) or not payload.strip():
            raise ValueError("embedding payload must be nonempty text")
        count = len(self.tokenizer.encode(payload))
        if count > self.executor.max_length:
            raise ValueError("input exceeds model length; no implicit truncation")
        return count

    def execute(self, inputs):
        vectors = self.executor.embed([item.payload for item in inputs])
        if len(vectors) != len(inputs):
            raise ValueError("vLLM returned a different number of outputs")
        return [ExecutionOutput(item.request_id, vector.tolist())
                for item, vector in zip(inputs, vectors)]

    def close(self):
        # vLLM owns worker teardown at process exit. Do not kill global workers.
        pass


def load_plugin(name: str, group: str):
    if name.startswith("python:"):
        return load_symbol(name[len("python:"):])
    found = list(entry_points(group=group, name=name))
    if len(found) != 1:
        raise ValueError(f"expected exactly one {group} entry point named {name!r}")
    return found[0].load()


def create_backend(binding):
    builtins = {"mock": MockBackend, "vllm-embed": VllmEmbeddingBackend}
    factory = builtins.get(binding.backend)
    if factory is None:
        factory = load_plugin(binding.backend, "flagscheduler.backends")
    backend = factory(binding)
    for method in ("count_cost", "execute", "close"):
        if not callable(getattr(backend, method, None)):
            raise ValueError(f"backend missing {method}")
    return backend
