"""Opt-in, instance-local Qwen3 eager inference adapter; diagnostic only.

No global ATen registration, training, compile, vLLM fused ops or automatic
execution retry. Use exclusively owned model instances on a single thread.
"""
from collections import Counter
import importlib
import inspect
import threading
import types

import torch

from pytorch_eager_adapter import _common_reason


class QwenScopedGems:
    """Wrap exact known module classes, preserving native bound methods.

    Synchronizes target launches for diagnostic completion evidence, not speed.
    Unknown classes/overridden forwards are not silently treated as equivalent.
    Do not use concurrently with other registration/patching mechanisms.
    """

    def __init__(self, model, enabled=True, allowed_ops=()):
        self.model = model
        self.enabled = enabled
        self.allowed_ops = frozenset(allowed_ops)
        if not self.allowed_ops <= {"silu", "rms_norm"}:
            raise ValueError("Unknown operator in allowlist")
        self.counts = Counter()
        self.routes = {}
        self._saved = []
        self._entered = False
        self._owner = None

    def _record(self, name, op, backend, reason, stage, x):
        self.counts[(op, backend, reason, stage)] += 1
        key = (op, backend, reason, stage, tuple(x.shape), str(x.dtype))
        if len(self.routes) < 64 and key not in self.routes:
            self.routes[key] = dict(module=name, op=op, backend=backend,
                                    reason=reason, stage=stage, shape=list(x.shape),
                                    dtype=str(x.dtype), device=str(x.device))

    def summary(self):
        return {"counts": [dict(op=k[0], backend=k[1], reason=k[2], stage=k[3], count=v)
                           for k, v in sorted(self.counts.items())],
                "representative_routes": list(self.routes.values())}

    def _reason(self, module, op, x):
        if not self.enabled:
            return "disabled"
        if op not in self.allowed_ops:
            return "operator_not_enabled"
        if module.training or torch.is_grad_enabled():
            return "inference_only"
        weight = module.weight if op == "rms_norm" else None
        reason = _common_reason(x, weight)
        if reason:
            return reason
        if op == "rms_norm":
            if x.ndim < 1 or weight.shape != (x.shape[-1],):
                return "weight_shape"
            if weight.device != x.device or weight.dtype != x.dtype or not weight.is_contiguous():
                return "weight_layout_or_dtype"
            if not isinstance(module.variance_epsilon, (int, float)) or module.variance_epsilon <= 0:
                return "eps"
        return None

    def _invoke(self, name, op, module, original, signature, args, kwargs):
        if threading.get_ident() != self._owner:
            raise RuntimeError("Model adapter is single-thread diagnostic only")
        bound = signature.bind(*args, **kwargs)  # preserve argument errors
        bound.apply_defaults()
        x = next(iter(bound.arguments.values()))
        if type(x) is not torch.Tensor:
            return original(*args, **kwargs)
        reason = self._reason(module, op, x)
        gems = None
        if reason is None:
            try:
                gems = importlib.import_module("flag_gems")
            except ModuleNotFoundError as exc:
                if exc.name != "flag_gems":
                    raise
                reason = "missing_flag_gems"
        if reason is not None:
            self._record(name, op, "native", reason, "selected", x)
            return original(*args, **kwargs)
        self._record(name, op, "flaggems", "supported", "selected", x)
        try:
            torch.npu.synchronize(x.device)
            if op == "silu":
                y = gems.silu(x)
            else:
                y = gems.rms_norm(x, (x.shape[-1],), module.weight, module.variance_epsilon)
            torch.npu.synchronize(x.device)
            if y.shape != x.shape or y.dtype != x.dtype or y.device != x.device:
                raise RuntimeError("Target output violates shape/dtype/device contract")
            self._record(name, op, "flaggems", "supported", "completed", x)
            return y
        except Exception as exc:
            self._record(name, op, "flaggems", type(exc).__name__, "error", x)
            raise  # Never retry after possible writes/device submission.

    def __enter__(self):
        from transformers.activations import SiLUActivation
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        if self._entered:
            raise RuntimeError("Adapter cannot be nested or re-entered")
        mapping = {Qwen3RMSNorm: "rms_norm", SiLUActivation: "silu"}
        targets = [(n, m, mapping[type(m)]) for n, m in self.model.named_modules()
                   if type(m) in mapping]
        if not targets:
            raise ValueError("No exact supported Qwen3/SiLU module classes found")
        # Validate all targets before making the first mutation.
        if any("forward" in m.__dict__ for _, m, _ in targets):
            raise RuntimeError("Existing instance forward override; refusing conflicting patch")
        self._owner = threading.get_ident()
        self._entered = True
        try:
            for name, module, op in targets:
                original = module.forward
                signature = inspect.signature(original)
                if len(signature.parameters) != 1:
                    raise RuntimeError("Unexpected native forward signature")

                def wrapped(this, *args, _name=name, _op=op, _original=original,
                            _signature=signature, **kwargs):
                    return self._invoke(_name, _op, this, _original, _signature, args, kwargs)

                bound = types.MethodType(wrapped, module)
                self._saved.append((module, bound))
                module.forward = bound
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for module, installed in reversed(self._saved):
            if module.__dict__.get("forward") is installed:
                delattr(module, "forward")
        self._saved.clear()
        self._entered = False
        self._owner = None
