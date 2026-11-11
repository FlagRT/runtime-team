"""Experimental scoped PyTorch eager integration, independent of vLLM.

This probe uses the existing FlagGems kernels. It is not a process-global ATen
registration or a production framework package. Compile/export are out of scope.
"""
from collections import deque
from dataclasses import dataclass
import importlib
import inspect
import threading

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode


@dataclass(frozen=True)
class Route:
    op: str
    backend: str
    reason: str


_local = threading.local()
_SIGNATURES = {F.silu: inspect.signature(F.silu),
               F.rms_norm: inspect.signature(F.rms_norm)}


def _load_gems():
    return importlib.import_module("flag_gems")


def _common_reason(x, weight=None):
    if type(x) is not torch.Tensor:
        return "tensor_subclass"
    if weight is not None and type(weight) not in (torch.Tensor, torch.nn.Parameter):
        return "tensor_subclass"
    if torch.is_grad_enabled() and (x.requires_grad or
                                  (weight is not None and weight.requires_grad)):
        return "autograd"
    if x.device.type != "npu":
        return "device"
    if x.dtype not in (torch.float16, torch.bfloat16):
        return "dtype"
    if x.layout != torch.strided or not x.is_contiguous():
        return "layout"
    if not x.numel():
        return "empty"
    return None


class PreferGems(TorchFunctionMode):
    """Opt-in inference scope for F.silu / F.rms_norm and their nn modules.

    Unsupported calls use the original framework function on the SAME device.
    Once a kernel starts, any exception propagates: no execution-time retries.
    Route records contain no tensors and are bounded; 'selected' records an
    attempt, not a successful device completion.
    NPU accelerated calls currently synchronize before and after launch: the
    installed backend combination showed intermittent native-output differences
    without an intermediate barrier. This is a diagnostic correctness path,
    not a throughput implementation; stream-safe async interop remains open.
    """

    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.routes = deque(maxlen=256)

    def __enter__(self):
        if getattr(_local, "active", False):
            raise RuntimeError("Nested PreferGems scopes are not supported")
        result = super().__enter__()
        _local.active = True
        return result

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            _local.active = False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in _SIGNATURES:
            return func(*args, **kwargs)
        # Preserve Python argument validation instead of hiding malformed calls.
        bound = _SIGNATURES[func].bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        x = p["input"]
        weight = p.get("weight")
        reason = "disabled" if not self.enabled else _common_reason(x, weight)
        if reason is None and func is F.silu and p["inplace"]:
            reason = "inplace"
        if reason is None and func is F.rms_norm:
            shape = p["normalized_shape"]
            # Conservative fast path: one trailing normalized dimension.
            if (not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 1
                    or x.ndim == 0 or shape[0] != x.shape[-1]):
                reason = "normalized_shape"
            elif weight is None:
                reason = "missing_weight"
            elif (weight.shape != tuple(shape) or weight.device != x.device
                  or weight.dtype != x.dtype or not weight.is_contiguous()):
                reason = "weight"
        gems = None
        if reason is None:
            try:
                gems = _load_gems()
            except ModuleNotFoundError as exc:
                if exc.name != "flag_gems":
                    raise  # Broken dependency is not an unsupported input.
                reason = "missing_flag_gems"
        if reason is not None:
            self.routes.append(Route(func.__name__, "native", reason))
            # TorchFunctionMode removes this mode during the handler call.
            return func(*args, **kwargs)

        self.routes.append(Route(func.__name__, "flaggems", "selected"))
        try:
            if x.device.type == "npu":
                torch.npu.synchronize(x.device)
            if func is F.silu:
                result = gems.silu(x)
            else:
                eps = p["eps"] if p["eps"] is not None else torch.finfo(x.dtype).eps
                result = gems.rms_norm(x, p["normalized_shape"], weight, eps)
            if x.device.type == "npu":
                torch.npu.synchronize(x.device)
            return result
        except Exception as exc:
            self.routes.append(Route(func.__name__, "flaggems", f"error:{type(exc).__name__}"))
            raise
