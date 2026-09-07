#!/usr/bin/env python3
"""
统一运行时 API（runtime/__init__.py）

用户视角的全部入口。**同一份代码切换后端只改 use() 一行**：

    import runtime
    runtime.use("ascend")            # 或 "kunlun"
    runtime.set_device(0)
    s = runtime.create_stream()
    fe = runtime.translate_error(exc, location="op:matmul")
    ok = runtime.recover_device(0, mode="real")

对应职责子层：设备上下文 + 多流 Stream + 错误码翻译 + 状态恢复。
架构位置：上承算子层/编译层与模型转换器，下接多机多卡分布式训练与推理。
"""

from .api.errors import (
    DISPOSITION,
    ErrorCategory,
    FlagosError,
    translate_via_backend,
)
from .backends.base import RuntimeBackend
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
    # 设备 / 流（转发到当前后端）
    "device_count", "set_device", "memory_stats",
    "create_stream", "create_event", "current_stream", "synchronize",
    "probe_device", "recover_device", "translate_error", "device_state",
]

__version__ = "0.1.0"


# ─────────────── 转发：全部调用当前后端 ───────────────

def device_count() -> int:
    return current().device_count()


def set_device(ordinal: int) -> None:
    current().set_device(ordinal)


def memory_stats(ordinal: int = 0) -> dict:
    return current().memory_stats(ordinal)


def create_stream():
    return current().create_stream()


def create_event():
    return current().create_event()


def current_stream():
    return current().current_stream()


def synchronize(ordinal: int = 0, timeout_ms=None) -> None:
    current().synchronize(ordinal, timeout_ms=timeout_ms)


def translate_error(exc: BaseException, location: str = "") -> FlagosError:
    return translate_via_backend(exc, current(), location=location)


def probe_device(ordinal: int = 0) -> bool:
    return current().probe_device(ordinal)


def recover_device(ordinal: int = 0, mode: str = "probe", reason: str = "") -> bool:
    return current().recover_device(ordinal, mode=mode, reason=reason)


def device_state(ordinal: int = 0):
    return current().device_state(ordinal)


def _auto_discover():
    """导入时自动发现已安装后端（失败静默，用户可显式 use() 触发）。"""
    try:
        discover()
    except Exception:
        pass


_auto_discover()
