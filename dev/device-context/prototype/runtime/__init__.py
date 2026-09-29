#!/usr/bin/env python3
"""
统一运行时 API（runtime/__init__.py）

用户视角的全部入口。**同一份代码切换后端只改 use() 一行**：

    import runtime
    runtime.use("ascend")            # 或 "kunlun"（昆仑芯 P800）/ "cambricon"（寒武纪 MLU590）
                                     # 三家均为**厂商官方 torch 插件**路线（torch_npu / torch.cuda 兼容层 / torch_mlu）
    runtime.set_device(0)
    s = runtime.create_stream()
    fe = runtime.translate_error(exc, location="op:matmul")
    r = runtime.recover_device(0, mode="real")   # -> dict: {ordinal, mode, recovered, state, detail, ...}
    ok = r["recovered"]

    # 内存句柄（工作包 B）：只做句柄语义，不做池化策略
    h = runtime.allocate(4 * 1024 * 1024)        # -> 统一句柄 dict（不含厂商指针）
    runtime.free(h)                              # 重复释放会如实报错

    # 设备上下文生命周期（工作包 C）：能力未声明时如实报错
    ctx = runtime.context_create(0)
    runtime.context_count()
    runtime.context_destroy(ctx)                 # 只接受本层创建的句柄

对应职责子层：设备上下文 + 多流 Stream + 错误码翻译 + 状态恢复。
架构位置：上承算子层/编译层与模型转换器，下接多机多卡分布式训练与推理。
"""

from .api.errors import (
    DISPOSITION,
    ErrorCategory,
    FlagosError,
    translate_via_backend,
)
from .api.stream import Event, Stream
from .backends.base import (
    CONTEXT_HANDLE_KEYS,
    DEVICE_STATE_TOKENS,
    MEMORY_HANDLE_KEYS,
    RuntimeBackend,
)
from .backends import registry
from .backends.registry import (
    BackendNotFound,
    available,
    current,
    current_name,
    discover,
    get,
    register,
    set_current,
    use,
)

__all__ = [
    # 后端选择
    "use", "set_current", "current", "get", "available", "discover", "register",
    "current_name", "BackendNotFound", "RuntimeBackend", "registry",
    # 错误分级
    "FlagosError", "ErrorCategory", "DISPOSITION", "translate_via_backend",
    "Stream", "Event",
    # 设备 / 流（转发到当前后端）
    "device_count", "set_device", "memory_stats",
    "create_stream", "create_event", "current_stream", "synchronize",
    "probe_device", "recover_device", "translate_error", "device_state",
    # 内存句柄与生命周期（工作包 B）
    "allocate", "free", "memory_handle_count", "MEMORY_HANDLE_KEYS",
    # 设备上下文生命周期（工作包 C）
    "context_create", "context_destroy", "context_count", "context_query",
    "CONTEXT_HANDLE_KEYS",
    "DEVICE_STATE_TOKENS",
    # 审计（`.native` 逃生舱 / 退化路径）
    "native_accesses", "degradations",
]

__version__ = "0.1.0"


# ─────────────── 转发：全部调用当前后端 ───────────────

def device_count() -> int:
    return current().device_count()


def set_device(ordinal: int) -> None:
    current().set_device(ordinal)


def memory_stats(ordinal: int = 0) -> dict:
    """设备内存统计（规范三键 `total_mb` / `used_mb` / `free_mb`；
    声明了 `memory_alloc_stat` 的后端另给 `allocated_mb`）。"""
    return current().memory_stats(ordinal)


def create_stream() -> Stream:
    """创建统一 Stream 对象（包装后端原生流）。

    2026-09-29（工作包 C）：同时登记"该流创建于哪个本层上下文"，
    以便上下文销毁后在使用点**如实拦截**（厂商侧是静默的，见 backend.check_stream_usable）。
    """
    b = current()
    native = b.create_stream()
    b.note_stream_created(native)
    return Stream(b, native)


def create_event() -> Event:
    """创建统一 Event 对象（包装后端原生事件）。"""
    b = current()
    return Event(b, b.create_event())


def current_stream():
    return current().current_stream()


def synchronize(ordinal: int = 0, timeout_ms=None) -> None:
    current().synchronize(ordinal, timeout_ms=timeout_ms)


def translate_error(exc: BaseException, location: str = "") -> FlagosError:
    return translate_via_backend(exc, current(), location=location)


def probe_device(ordinal: int = 0) -> bool:
    return current().probe_device(ordinal)


def recover_device(ordinal: int = 0, mode: str = "probe", reason: str = "") -> dict:
    return current().recover_device(ordinal, mode=mode, reason=reason)


def device_state(ordinal: int = 0):
    return current().device_state(ordinal)


# ─────────────── 内存句柄与生命周期（工作包 B）───────────────

def allocate(size_bytes: int, ordinal: int = 0) -> dict:
    """申请设备内存，返回**统一内存句柄**（公共字段见 `MEMORY_HANDLE_KEYS`）。

    只提供句柄语义（申请/释放），**不含池化、碎片、峰值、扩容策略** —— 那些属显存方向。
    """
    return current().allocate(size_bytes, ordinal=ordinal)


def free(handle) -> None:
    """释放 `allocate()` 返回的句柄。**重复释放/非本层句柄会报错**（不静默）。"""
    current().free(handle)


def memory_handle_count() -> int:
    """当前在世的内存句柄数（泄漏判据的取数入口）。"""
    return current().memory_handle_count()


# ─────────────── 设备上下文生命周期（工作包 C）───────────────

def context_create(ordinal: int = 0) -> dict:
    """新建设备上下文，返回统一句柄；后端未声明 `context_lifecycle` 时如实报错。"""
    return current().context_create(ordinal=ordinal)


def context_destroy(handle) -> None:
    """销毁本层创建的设备上下文句柄；**对非本层句柄一律拒绝**。"""
    current().context_destroy(handle)


def context_query() -> dict:
    """查询**此刻进程实际生效的设备上下文**（**只读**，不改状态）。

    与 `context_count()` 的分工：后者答"**本层造了几个**"，本接口答"**此刻在哪个上下文上**"
    —— 对**框架自建**的上下文同样有意义（P800 就是这种情况：上下文由 XPytorch 自建，
    本层一个都不造，但上层仍需要知道"我在哪个上下文上、它归谁管"）。

    固定键：`queryable / present / ordinal / flags / managed_by / reason`。
    `managed_by` 取 `"unified"`（本层创建）/ `"external"`（厂商或框架自建）/ `None`。
    """
    return current().context_query()


def context_count() -> int:
    """本层当前在世的设备上下文数（不含进程默认上下文）。"""
    return current().context_count()


# ─────────────── 审计（`.native` 逃生舱 / 退化路径）───────────────

def native_accesses() -> dict:
    """`.native` 逃生舱取用审计：`{"total": n, "by_kind": {...}}`。

    取用即视为绑定该厂商；本计数用于事后定位"可移植性被破坏"的位置。
    """
    return current().native_accesses()


def degradations() -> dict:
    """能力缺失导致的**退化路径**审计（如 `record_stream` 不可用 → 保守同步）。"""
    return current().degradations()


def _auto_discover():
    """导入时自动发现已安装后端（失败静默，用户可显式 use() 触发）。"""
    try:
        discover()
    except Exception:
        pass


_auto_discover()
