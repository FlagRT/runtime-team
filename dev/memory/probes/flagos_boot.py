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
import importlib.machinery
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
# 1b) factory wrap(不依赖 mode —— mode 是 thread-local,triton benchmark /
#     autotune 等非 Thread.start 线程无 mode 栈,device='npu' 会直闯 C++
#     解析失败;旧机挂点 12 同款)
# ---------------------------------------------------------------------------
def _install_factory_wrap():
    _orig = {n: getattr(torch, n) for n in (
        "empty", "zeros", "ones", "full", "randn", "rand", "randint",
        "randperm", "arange", "linspace", "eye", "tensor", "scalar_tensor",
        "empty_like", "zeros_like", "ones_like", "full_like", "randn_like",
        "rand_like", "randint_like",
    ) if hasattr(torch, n)}

    def _make(orig):
        def _w(*a, **k):
            if "device" in k and k["device"] is not None:
                r = _remap_npu(k["device"])
                if r is not k["device"]:
                    k = {**k, "device": r}
            return orig(*a, **k)
        _w.__name__ = getattr(orig, "__name__", "wrapped_factory")
        return _w

    for n, fn in _orig.items():
        try:
            setattr(torch, n, _make(fn))
        except Exception as e:  # noqa: BLE001
            _log(f"factory wrap {n} failed:", e)


_install_factory_wrap()


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


# vllm_fl PlatformFL.get_current_memory_usage 调 reset_peak_memory_stats + max_memory_allocated
def _reset_peak_memory_stats(device=None):
    return flagos.reset_peak_memory_stats(device)


if not hasattr(torch.npu, "reset_peak_memory_stats"):
    torch.npu.reset_peak_memory_stats = _reset_peak_memory_stats


# vllm attention 路径(triton_reshape_and_cache_flash)经 torch.cuda.get_device_capability
# 调 flagos.get_device_properties(torch.device 对象);torch_fl main 只收 int/None →
# _memory_reserved(torch.device) 抛 "cannot be interpreted as an integer"。包一层转 int。
_orig_get_device_properties = flagos.get_device_properties


def _get_device_properties(device=None):
    # 注意:不能用 isinstance(device, torch.device) —— boot 的 device 包装类
    # metaclass __instancecheck__ 会递归;用类型名判定。
    _tn = type(device).__name__
    if _tn == "device":
        device = device.index if device.index is not None else 0
    elif _tn == "str":
        device = int(device.split(":")[-1]) if ":" in device else 0
    return _orig_get_device_properties(device)


torch.npu.get_device_properties = _get_device_properties
flagos.get_device_properties = _get_device_properties
# torch_fl/__init__.py:919 已把 torch.cuda.get_device_properties 绑定到 flagos
# 原函数(早于本 boot 段执行),须重新绑定到 int 兼容 wrapper。
if hasattr(torch, "cuda"):
    torch.cuda.get_device_properties = _get_device_properties


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
sys.modules["torch_npu.version"] = tn.version

_c = types.ModuleType("torch_npu._C")
_c.get_raw_stream = lambda device=None: 0  # 默认流 handle 0
# triton_ascend backend_register.get_current_stream 会取 _npu_getCurrentRawStream[NoWait]
# (triton_ascend PCH/launch 路径);flagos 下统一回默认流 handle 0。
_c._npu_getCurrentRawStream = lambda device=None: 0
_c._npu_getCurrentRawStreamNoWait = lambda device=None: 0
tn._C = _c
# import torch_npu._C 会命中 sys.modules 而不再查文件系统(stub 包无真 _C 子模块)
sys.modules["torch_npu._C"] = _c

_inductor = types.ModuleType("torch_npu._inductor")
sys.modules["torch_npu._inductor"] = _inductor
tn._inductor = _inductor


class NPUGraph:  # 占位;ascend 平台已禁用 graph capture(vllm_fl platform)
    def __init__(self, *a, **k):
        raise NotImplementedError("NPUGraph is not supported on the flagos/ascend backend")


tn.NPUGraph = NPUGraph
# vllm_fl/compilation/graph.py:47 访问的是 torch.npu.NPUGraph(torch_fl 挂的 shim),
# 与 sys.modules["torch_npu"] 别名是不同对象 —— 补齐须挂到 torch.npu 侧。
torch.npu.NPUGraph = NPUGraph

# __file__ 指向真实 stub 包路径(flagcx 构建路径逻辑 / triton_ascend PCH 均会 dirname)
_stub_tn_init = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_npu", "__init__.py")
tn.__file__ = _stub_tn_init if os.path.exists(_stub_tn_init) else os.path.join(
    os.path.dirname(torch_fl.__file__), "__init__.py"
)

# torch_fl 注册 shim 时 __spec__ 只有 origin="torch_fl_shim"、无 submodule_search_locations;
# triton_ascend 编译 npu_utils 时 backend_register._get_package_dir("torch_npu") 走 find_spec,
# 拿不到包目录会退回 dirname(origin) 解析到 cwd(/tmp) → 找不到 torch_npu 头文件。
# 磁盘上存在真实 stub 包(拷自 vllm-ascend 镜像的头文件)时,把 __spec__ 指向它。
if os.path.exists(_stub_tn_init):
    tn.__spec__ = importlib.machinery.ModuleSpec(
        name="torch_npu",
        loader=None,
        origin=_stub_tn_init,
        is_package=True,
    )
    tn.__spec__.submodule_search_locations = [os.path.dirname(_stub_tn_init)]
    # shim 由 types.ModuleType 构造无 __path__,import 子模块会报 "is not a package";
    # 补 __path__ 让 sys.modules 直注册的 _C/version/_inductor 之外的路由可诊断。
    tn.__path__ = [os.path.dirname(_stub_tn_init)]


# ---------------------------------------------------------------------------
# 5) aten::detach_ no-op(torch.tensor C++ 内部直调,无 PrivateUse1 注册)
# ---------------------------------------------------------------------------
try:
    torch.library.impl("aten::detach_", "PrivateUse1", lambda self: self)
except Exception as e:  # noqa: BLE001
    _log("detach_ register failed:", e)


# ---------------------------------------------------------------------------
# 5b) flag_gems.device 与 torch_fl 后端名对齐(lift_fresh 等 ops 按
#     x.device.type != flag_gems.device 判定后端;flag_gems 由 GEMS_VENDOR
#     探测得 'npu',而 torch_fl 的 PrivateUse1 注册名是 'flagos' —— 不一致会
#     raise。只改模块属性,不动 DeviceDetector 探测逻辑。)
# ---------------------------------------------------------------------------
try:
    import flag_gems  # noqa: E402

    _fg_dev = getattr(flag_gems, "device", None)
    if isinstance(_fg_dev, str) and _fg_dev != _PU:
        flag_gems.device = _PU
        _log(f"flag_gems.device '{_fg_dev}' -> '{_PU}'")
except Exception as e:  # noqa: BLE001
    _log("flag_gems.device align skipped:", e)


# ---------------------------------------------------------------------------
# 5c) aten::_has_compatible_shallow_copy_type 宽松注册(PrivateUse1 key)
#     torch_fl 已在 CatchAll 注册宽松版(cpu/PrivateUse1 互拷放行),但实测
#     EngineCore(spawn)进程内 CatchAll 被 composite 默认实现盖回 → vllm
#     device_loading_context 里 p.data = p.data.to(flagos) 抛
#     "variable.set_data ... incompatible tensor type"(set_data 底层查此 op)。
#     用更具体的 PrivateUse1 key 再注册一次:dispatch 优先级高于 composite,
#     任何进程生效。CPU 源张量发起时 key 是 CPU?——实测两端都走到该 op,
#     且 CPU 侧由 CatchAll 宽松版兜底(torch_fl register.cc),此处补 PrivateUse1。
# ---------------------------------------------------------------------------
try:
    # 宽松版:CPU/PrivateUse1(flagos) 之间一律允许浅拷贝(数据搬移合法,
    # 底层 storage 由 set_data 在各自 allocator 上管理)
    def _compat_flagos(self, from_):
        return True

    for _k in ("PrivateUse1", "CPU"):
        try:
            torch.library.impl(
                "aten::_has_compatible_shallow_copy_type", _k
            )(_compat_flagos)
        except Exception as _e:  # noqa: BLE001
            _log(f"compat op relax [{_k}] failed:", _e)
    _log("compat op relaxed for PrivateUse1+CPU (EngineCore set_data fix)")
except Exception as e:  # noqa: BLE001
    _log("compat op relax failed:", e)


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
