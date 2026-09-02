"""flagos_boot.py — vllm-venv 引导模块(重建自阶段4执行记录挂点清单,2026-09-02 适配 py312)。

由 site-packages/flagos_torchfl.pth 加载(内容单行: import flagos_boot)。
在任意 python 进程 site 初始化时执行,先于业务代码,保证:
  - import torch_fl(挂载 torch.npu shim / torch_npu 别名 / flagos 后端)
  - torch.device("npu"/"cann") → flagos 别名(直构 + 工厂 kwargs + 子线程)
  - torch.npu 补齐: empty_cache / mem_get_info / max_memory_allocated / stream / set_device(int 兼容)
  - torch_npu 补齐: _inductor / NPUGraph / _C(get_raw_stream=0) / version / __file__ / 常用转发
  - Thread.start 子线程继承 torch function mode 栈(thread-local 搬运)
  - aten::detach_ no-op、aten::scatter_ 系列 CPU fallback、aten::split_with_sizes CPU fallback
  - torch.accelerator.synchronize/empty_cache → torch.npu 重定向

仅含 torch_fl main(2026-09-02 快照)尚未覆盖的挂点;已内置的(cuda 别名 mode、
factory wrap、torch.npu 基础 shim、torch_npu 别名)不再重复。
"""
import ctypes
import os
import sys
import threading
import types
from contextlib import contextmanager

# torch 导入时会自动加载 torch.backends entry points(flagcx 等 vendor 后端),
# 在 flagcx 构建/加载失败时会毒化进程(见阶段4执行记录避坑)。显式禁用,
# flagcx 由 vllm_fl 按需 import(FLAGCX_PATH 触发)。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_fl  # noqa: F401  — 先于一切:挂 shim + flagos 后端

_PU = torch._C._get_privateuse1_backend_name()  # "flagos"
flagos = torch_fl.flagos


def _log(*a):
    print("[flagos_boot]", *a, file=sys.stderr)


# ---------------------------------------------------------------------------
# 1) torch.device("npu"/"cann") → flagos 别名
#    torch_fl main 只别名 cuda(_remap);npu/cann 走这里。
#    直构 torch.device 由包装类处理;工厂 kwargs / .to() 由 mode 处理。
# ---------------------------------------------------------------------------
def _remap_npu(dev):
    if isinstance(dev, str):
        if dev in ("npu", "cann"):
            return _PU
        if dev.startswith("npu:"):
            return f"{_PU}:{dev[4:]}"
        if dev.startswith("cann:"):
            return f"{_PU}:{dev[5:]}"
        return dev
    if isinstance(dev, torch.device) and dev.type in ("npu", "cann"):
        return torch.device(_PU, dev.index if dev.index is not None else 0)
    return dev


_cur_device = torch.device  # torch_fl 已包装的 device(cuda→flagos)


class _NpuDeviceMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _cur_device)


class device(metaclass=_NpuDeviceMeta):
    def __new__(cls, *args, **kwargs):
        if args:
            args = (_remap_npu(args[0]),) + args[1:]
        elif "device" in kwargs:
            kwargs = {**kwargs, "device": _remap_npu(kwargs["device"])}
        return _cur_device(*args, **kwargs)


torch.device = device


from torch.overrides import TorchFunctionMode  # noqa: E402


class _NpuAliasMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        dev = kwargs.get("device")
        if dev is not None:
            r = _remap_npu(dev)
            if r is not dev:
                kwargs = {**kwargs, "device": r}
        elif args and func is torch.Tensor.to and len(args) > 1:
            r = _remap_npu(args[1])
            if r is not args[1]:
                args = (args[0], r) + args[2:]
        return func(*args, **kwargs)


_MODE = _NpuAliasMode()
torch._C._push_on_torch_function_stack(_MODE)


# ---------------------------------------------------------------------------
# 2) Thread.start 子线程继承 mode 栈(torch function mode 是 thread-local,
#    vllm 的 execute_model 在后台线程运行)
# ---------------------------------------------------------------------------
def _install_thread_patch():
    _orig_start = threading.Thread.start

    def _start(self, *a, **kw):
        target = self._target
        if target is not None:
            try:
                parent_modes = list(torch.overrides._get_current_function_mode_stack())
            except Exception:  # noqa: BLE001
                parent_modes = []
            if parent_modes:

                def _wrapped(*ta, **tkw):
                    try:
                        for m in parent_modes:
                            torch._C._push_on_torch_function_stack(m)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        return target(*ta, **tkw)
                    finally:
                        try:
                            for _ in parent_modes:
                                torch._C._pop_torch_function_stack()
                        except Exception:  # noqa: BLE001
                            pass

                self._target = _wrapped
        return _orig_start(self, *a, **kw)

    threading.Thread.start = _start


_install_thread_patch()


# ---------------------------------------------------------------------------
# 3) torch.npu 补齐
# ---------------------------------------------------------------------------
if not hasattr(torch.npu, "empty_cache"):
    torch.npu.empty_cache = flagos.empty_cache


def _mem_get_info(device=None):
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtGetMemInfo.restype = ctypes.c_int32
    free = ctypes.c_size_t(0)
    total = ctypes.c_size_t(0)
    rc = lib.aclrtGetMemInfo(ctypes.c_int32(1), ctypes.byref(free), ctypes.byref(total))
    if rc != 0:
        raise RuntimeError(f"aclrtGetMemInfo failed rc={rc}")
    return free.value, total.value


if not hasattr(torch.npu, "mem_get_info"):
    torch.npu.mem_get_info = _mem_get_info


def _max_memory_allocated(device=None):
    return flagos.memory_stats(device)["peak_allocated_bytes"]


if not hasattr(torch.npu, "max_memory_allocated"):
    torch.npu.max_memory_allocated = _max_memory_allocated


@contextmanager
def _npu_stream(stream=None):
    yield


if not hasattr(torch.npu, "stream"):
    torch.npu.stream = _npu_stream


# set_device 收 torch.device → int 转换(main 分支仍要求 int)
_orig_set_device = flagos.set_device


def _set_device(device):
    if isinstance(device, torch.device):
        device = device.index if device.index is not None else 0
    elif isinstance(device, str):
        device = int(device.split(":")[-1])
    return _orig_set_device(device)


flagos.set_device = _set_device
torch.npu.set_device = _set_device


# ---------------------------------------------------------------------------
# 4) torch_npu 补齐(worker.py:396 import torch_npu._inductor、triton PCH 等)
# ---------------------------------------------------------------------------
tn = sys.modules["torch_npu"]
tn.empty_cache = flagos.empty_cache
tn.mem_get_info = _mem_get_info
tn.synchronize = flagos.synchronize
tn.set_device = _set_device
tn.current_device = flagos.current_device
tn.device_count = flagos.device_count
tn.Stream = flagos.Stream
tn.Event = flagos.Event
tn.current_stream = flagos.current_stream
tn.max_memory_allocated = _max_memory_allocated

tn.version = types.ModuleType("torch_npu.version")
tn.version.__version__ = "0.0.0-flagos-shim"
tn.version.git_version = "0.0.0-flagos-shim"

_c = types.ModuleType("torch_npu._C")
_c.get_raw_stream = lambda device=None: 0  # 默认流 handle 0
tn._C = _c

_inductor = types.ModuleType("torch_npu._inductor")
sys.modules["torch_npu._inductor"] = _inductor
tn._inductor = _inductor


class NPUGraph:  # 占位;ascend 平台已禁用 graph capture(vllm_fl platform)
    def __init__(self, *a, **k):
        raise NotImplementedError("NPUGraph is not supported on the flagos/ascend backend")


tn.NPUGraph = NPUGraph

# __file__ 指向真实 stub 包路径(flagcx 构建路径逻辑 / triton_ascend PCH 均会 dirname)
_stub_tn_init = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_npu", "__init__.py")
tn.__file__ = _stub_tn_init if os.path.exists(_stub_tn_init) else os.path.join(
    os.path.dirname(torch_fl.__file__), "__init__.py"
)


# ---------------------------------------------------------------------------
# 5) aten::detach_ no-op(torch.tensor C++ 内部直调,无 PrivateUse1 注册)
# ---------------------------------------------------------------------------
try:
    torch.library.impl("aten::detach_", "PrivateUse1", lambda self: self)
except Exception as e:  # noqa: BLE001
    _log("detach_ register failed:", e)


# ---------------------------------------------------------------------------
# 6) aten::scatter_ 系列 CPU fallback(多 prompt decode 批量路径)
# ---------------------------------------------------------------------------
def _scatter_inplace():
    def impl(self, dim, index, src=None, value=None, reduce=None):
        self_cpu = self.detach().cpu()
        idx_cpu = index.cpu()
        if src is not None:
            out = self_cpu.scatter_(dim, idx_cpu, src.cpu(), reduce=reduce) if reduce else self_cpu.scatter_(dim, idx_cpu, src.cpu())
        else:
            out = self_cpu.scatter_(dim, idx_cpu, value, reduce=reduce) if reduce else self_cpu.scatter_(dim, idx_cpu, value)
        self.copy_(out)
        return self
    return impl


def _scatter_outplace():
    def impl(self, dim, index, src=None, value=None, reduce=None):
        self_cpu = self.detach().cpu()
        idx_cpu = index.cpu()
        if src is not None:
            out = self_cpu.scatter(dim, idx_cpu, src.cpu(), reduce=reduce) if reduce else self_cpu.scatter(dim, idx_cpu, src.cpu())
        else:
            out = self_cpu.scatter(dim, idx_cpu, value, reduce=reduce) if reduce else self_cpu.scatter(dim, idx_cpu, value)
        return out.to(self.device)
    return impl


_SCATTER_IMPLS = {
    "aten::scatter_.src": _scatter_inplace(),
    "aten::scatter_.value": _scatter_inplace(),
    "aten::scatter_.reduce": _scatter_inplace(),
    "aten::scatter_.value_reduce": _scatter_inplace(),
    "aten::scatter.src": _scatter_outplace(),
    "aten::scatter.value": _scatter_outplace(),
    "aten::scatter.reduce": _scatter_outplace(),
    "aten::scatter.value_reduce": _scatter_outplace(),
}
for _op, _fn in _SCATTER_IMPLS.items():
    try:
        torch.library.impl(_op, "PrivateUse1", _fn)
    except Exception as e:  # noqa: BLE001
        _log(f"scatter register failed ({_op}):", e)


# ---------------------------------------------------------------------------
# 7) aten::split_with_sizes CPU fallback(模型加载)
# ---------------------------------------------------------------------------
def _split_with_sizes(self, split_sizes, dim=0):
    parts = self.detach().cpu().split_with_sizes(list(split_sizes), dim)
    return tuple(p.to(self.device) for p in parts)


try:
    torch.library.impl("aten::split_with_sizes", "PrivateUse1", _split_with_sizes)
except Exception as e:  # noqa: BLE001
    _log("split_with_sizes register failed:", e)


# ---------------------------------------------------------------------------
# 8) torch.accelerator 重定向(收尾清理 / empty_cache)
# ---------------------------------------------------------------------------
def _acc_synchronize(device=None):
    return flagos.synchronize()


try:
    torch.accelerator.synchronize = _acc_synchronize
    torch.accelerator.empty_cache = flagos.empty_cache
except Exception as e:  # noqa: BLE001
    _log("accelerator redirect failed:", e)

_log("boot done. device:", _PU, "| modes on stack:", len(torch.overrides._get_current_function_mode_stack()))
