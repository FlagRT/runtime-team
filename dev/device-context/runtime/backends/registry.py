#!/usr/bin/env python3
"""
Backend 注册表（runtime/backends/registry.py）

对应周计划 W2 任务 1/2：整体框架的注册与分发机制。

职责：
  - register(backend)：注册一个后端实现
  - use(name)：切换当前后端（用户唯一入口）
  - current() / get(name)：取回后端实例
  - discover()：自动发现已安装的后端插件（vendor 目录扫描）

设计要点：
  1. 后端为**单例**：同一进程内同一后端的实现只实例化一次（设备状态需保持）。
  2. 自动发现失败不影响显式注册（降级可用）。
  3. 切换后端会重置"当前设备"绑定语义——由后端 set_device 决定，不做隐式假设。
"""

import importlib
import logging
from typing import Dict, List, Optional

from .base import RuntimeBackend

logger = logging.getLogger(__name__)

#: 已注册后端（name → 已实例化的 backend）
_REGISTRY: Dict[str, RuntimeBackend] = {}

#: 当前生效后端名
_CURRENT: Optional[str] = None

#: 自动发现时扫描的 vendor 模块（新增厂商只需在此登记或提供同名子包）
_KNOWN_BACKENDS = ("ascend", "kunlun")


class BackendNotFound(RuntimeError):
    """请求了未注册/未发现的后端。"""


def register(backend: RuntimeBackend, make_current: bool = False) -> RuntimeBackend:
    """注册一个后端实现。

    Args:
        backend: RuntimeBackend 的实例
        make_current: 是否同时设为当前后端

    Returns:
        注册后的 backend（便于链式使用）
    """
    if not isinstance(backend, RuntimeBackend):
        raise TypeError(f"backend 必须是 RuntimeBackend 子类实例，收到 {type(backend)}")
    if not backend.name:
        raise ValueError("backend.name 不能为空")
    _REGISTRY[backend.name] = backend
    if make_current or _CURRENT is None:
        set_current(backend.name)
    logger.debug("registered backend: %s", backend.info())
    return backend


def discover(names=_KNOWN_BACKENDS, verbose: bool = False) -> List[str]:
    """自动发现并注册后端插件（扫描 backends/<name>/backend.py）。

    vendor 插件目录模式：每家厂商一个子目录，
    目录内提供 `BACKEND` 或 `build()` 工厂。发现失败（如缺依赖）仅告警，不中断。

    Returns:
        成功注册的后端名列表
    """
    loaded = []
    for name in names:
        if name in _REGISTRY:
            loaded.append(name)
            continue
        try:
            mod = importlib.import_module(f".{name}.backend", package=__package__)
        except Exception as e:  # 缺依赖/未实现 → 跳过，不影响其他后端
            if verbose:
                logger.warning("backend '%s' 未加载: %s", name, e)
            continue
        factory = getattr(mod, "build", None) or getattr(mod, "BACKEND", None)
        if factory is None:
            if verbose:
                logger.warning("backend '%s' 缺少 build()/BACKEND 工厂", name)
            continue
        backend = factory() if callable(factory) else factory
        register(backend, make_current=False)
        loaded.append(name)
    return loaded


def set_current(name: str) -> RuntimeBackend:
    """切换当前后端；未注册则先尝试发现。"""
    global _CURRENT
    if name not in _REGISTRY:
        discover(names=(name,), verbose=False)
    if name not in _REGISTRY:
        raise BackendNotFound(
            f"后端 '{name}' 未注册。已注册: {sorted(_REGISTRY) or '（空）'}"
        )
    _CURRENT = name
    return _REGISTRY[name]


def use(name: str) -> RuntimeBackend:
    """用户入口：选择后端（等价于 set_current，语义更明确）。"""
    return set_current(name)


def current() -> RuntimeBackend:
    """取当前后端；未选择时抛错（不做隐式默认，避免"跑在错的卡上"）。"""
    if _CURRENT is None:
        raise BackendNotFound(
            "尚未选择后端，请先调用 runtime.use('<name>') 或 runtime.set_current()"
        )
    return _REGISTRY[_CURRENT]


def get(name: str) -> RuntimeBackend:
    """按名取后端（不切换当前后端）。"""
    if name not in _REGISTRY:
        discover(names=(name,), verbose=False)
    if name not in _REGISTRY:
        raise BackendNotFound(f"后端 '{name}' 未注册")
    return _REGISTRY[name]


def available() -> List[str]:
    """返回当前已注册的后端名（已排序）。"""
    return sorted(_REGISTRY)


def current_name() -> Optional[str]:
    return _CURRENT


def clear() -> None:
    """清空注册表（仅供测试）。"""
    global _CURRENT
    _REGISTRY.clear()
    _CURRENT = None
