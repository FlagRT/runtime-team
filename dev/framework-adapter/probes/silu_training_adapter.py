"""Experimental F.silu bridge: per-call admission, first-order autograd only.

Dedicated single-thread eager worker only. Temporary FlagGems registration is
process-global; the lock coordinates THIS adapter, not arbitrary other workers.
Environment identity must be measured by the launcher, not copied from a policy.
No compile/export, tensor subclasses, autocast, higher-order grad or fused ops.
No numerical admission is implied by enabling experimental=True.
"""
from collections import Counter, deque
import inspect
import threading

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from silu_policy import InputSpec


_lock = threading.Lock()
_active = threading.local()
_signature = inspect.signature(F.silu)


class SiluAdapter(TorchFunctionMode):
    def __init__(self, policy, environment, runtime, gems):
        super().__init__()
        self.policy, self.environment = policy, environment
        self.runtime, self.gems = runtime, gems
        self.events = deque(maxlen=128)
        self.counts = Counter()
        self.entered = False
        self.failed = False
        self.owner = None

    def record(self, backend, stage, reason):
        self.counts[(backend, stage, reason)] += 1
        self.events.append(dict(backend=backend, stage=stage, reason=reason))

    def check(self, backward_worker=False):
        # Accelerator autograd runs callbacks on its own engine thread. The
        # owning scope must still be alive; concurrent registrations stay locked.
        if not self.entered or (not backward_worker and threading.get_ident() != self.owner):
            raise RuntimeError("Forward and backward must remain in the owning adapter scope")
        if self.failed:
            raise RuntimeError("Adapter failed; abort this training step, do not update parameters")

    def __enter__(self):
        if self.entered or getattr(_active, "scope", False):
            raise RuntimeError("Nested/re-entered adapter is unsupported")
        if self.failed:
            raise RuntimeError("Failed adapter cannot be reused")
        result = super().__enter__()
        self.entered, self.owner = True, threading.get_ident()
        _active.scope = True
        return result

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.failed = True
        try:
            return super().__exit__(kind, value, traceback)
        finally:
            self.entered = False
            _active.scope = False

    def target(self, stage, x, grad=None):
        self.check(backward_worker=(stage == "backward"))
        if not _lock.acquire(blocking=False):
            self.failed = True
            raise RuntimeError("Concurrent registration rejected")
        ops = ("silu", "silu_backward")
        before = {}
        try:
            before = {op: torch._C._dispatch_dump_table("aten::" + op) for op in ops}
            self.runtime.synchronize(x.device.index)
            with self.gems.use_gems(include=list(ops)):
                if stage == "forward":
                    out = torch.ops.aten.silu.default(x)
                else:
                    out = torch.ops.aten.silu_backward.default(grad, x)
                self.runtime.synchronize(x.device.index)
            if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
                raise RuntimeError("Target output contract violation")
            self.record("flaggems", stage, "completed")
            return out
        except BaseException:
            self.failed = True
            self.record("flaggems", stage, "error_no_retry")
            raise
        finally:
            try:
                restored = all(torch._C._dispatch_dump_table("aten::" + op) == table
                               for op, table in before.items())
                if not restored:
                    self.failed = True
                    raise RuntimeError("Dispatcher not restored; terminate isolated worker")
            finally:
                _lock.release()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is not F.silu:
            return func(*args, **kwargs)
        self.check()
        bound = _signature.bind(*args, **kwargs)
        bound.apply_defaults()
        x = bound.arguments["input"]
        if type(x) is not torch.Tensor:
            self.record("native", "forward", "tensor_subclass")
            return func(*args, **kwargs)
        spec = InputSpec(x.device.type, str(x.dtype), tuple(x.shape), x.is_contiguous(),
                         x.layout == torch.strided, True, bound.arguments["inplace"],
                         torch.is_autocast_enabled(x.device.type))
        reason = self.policy.reason(self.environment, spec)
        if reason:
            self.record("native", "forward", reason)
            return func(*args, **kwargs)
        return _Silu.apply(x, self)


class _Silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, adapter):
        ctx.adapter = adapter
        ctx.save_for_backward(x)
        return adapter.target("forward", x)

    @staticmethod
    def backward(ctx, grad):
        adapter = ctx.adapter
        adapter.check(backward_worker=True)
        if torch.is_grad_enabled():
            adapter.failed = True
            raise RuntimeError("Higher-order gradients are not supported")
        x, = ctx.saved_tensors
        # Grad layout may differ from forward; do not silently extend admission.
        if not grad.is_contiguous() or grad.dtype != x.dtype or grad.device != x.device:
            adapter.failed = True
            raise RuntimeError("Unadmitted gradient; abort step without native retry")
        return adapter.target("backward", x, grad), None
