#!/usr/bin/env python3
"""FlagOS 后端（torch_fl / flagos 设备后端适配）。

适用环境：锁定训练镜像（flagrt/ascend-operator-runtime-comm），其设备后端为
torch_fl 的 flagos（镜像明确禁止 torch_npu 与 Torch-FL 运行时共存）。

设计原则与其他后端一致：**不重复实现已验证能力，只做适配**。
- 错误码翻译复用 conformance/errors.py（与昇腾后端同源）
- 设备状态/恢复复用 conformance/device_state.py（能力受限时声明不支持）
- 流/事件直接返回框架原生对象（方法面已与统一封装对齐：
  wait_event / wait_stream / synchronize / record / query / wait）
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, Optional

from ...api.errors import DISPOSITION, ErrorCategory, FlagosError
from ..base import RuntimeBackend

# conformance 目录（已有资产所在）
_CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "ascend_regression" / "conformance"
)


class FlagosEventAdapter:
    """torch_fl(flagos) Event 的统一语义适配（与 NpuEventAdapter 同构）。

    补齐两点与统一事件契约的语义缺口：
      - E3：未 record 事件 query() 误报完成 → recorded 跟踪修正（返回 False）
      - E2v2：主机有界等待 wait_host（query 轮询，永不永久阻塞）
    """

    def __init__(self, *args, **kwargs):
        self._ev = None
        self._args, self._kwargs = args, kwargs
        self._recorded = False

    def _ensure(self):
        if self._ev is None:
            m = _current_flagos_module()
            self._ev = m.Event(*self._args, **self._kwargs)
        return self._ev

    def record(self, stream=None):
        r = self._ensure().record(stream)
        self._recorded = True
        return r

    def wait(self, stream=None):
        return self._ensure().wait(stream)

    def synchronize(self):
        r = self._ensure().synchronize()
        self._recorded = True
        return r

    def query(self):
        if not self._recorded:
            return False
        return self._ensure().query()

    def wait_host(self, timeout_ms=None):
        deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000.0
        while not self.query():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.002)
        return True

    def elapsed_time(self, end_event):
        return self._ensure().elapsed_time(getattr(end_event, "_ev", end_event))

    def __getattr__(self, item):
        return getattr(self._ensure(), item)


_CURRENT: dict = {}


def _current_flagos_module():
    """返回已加载的 torch.flagos 模块（由 FlagosBackend 在加载时注入）。"""
    mod = _CURRENT.get("mod")
    if mod is None:
        import torch_fl  # noqa: F401
        import torch
        mod = torch.flagos
        _CURRENT["mod"] = mod
    return mod


class FlagosBackend(RuntimeBackend):
    """基于 torch_fl(flagos) 的后端实现。"""

    name = "flagos"
    device_type = "flagos"   # 设备串前缀：flagos:0

    def __init__(self) -> None:
        self._torch = None
        self._mod = None          # torch.flagos
        self._errors_mod = None
        self._loaded = False

    # ───────────── 延迟加载 ─────────────
    def _load(self) -> None:
        if self._loaded:
            return
        # 镜像约束：必须先 import torch_fl，再 import torch（否则 PrivateUse1 被占用）
        import torch_fl  # noqa: F401
        import torch

        self._torch = torch
        self._mod = torch.flagos
        _CURRENT["mod"] = self._mod
        if hasattr(self._mod, "init"):
            try:
                self._mod.init()
            except Exception:
                pass
        self._loaded = True

    @property
    def torch(self):
        self._load()
        return self._torch

    @property
    def mod(self):
        self._load()
        return self._mod

    def _load_errors(self):
        if self._errors_mod is None:
            import importlib.util
            import sys

            sys.path.insert(0, str(_CONFORMANCE_DIR))
            spec = importlib.util.spec_from_file_location(
                "dc_conformance_errors", _CONFORMANCE_DIR / "errors.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._errors_mod = mod
        return self._errors_mod

    # ───────────── 设备 ─────────────
    def device_count(self) -> int:
        return int(self.mod.device_count())

    def set_device(self, ordinal: int) -> None:
        m = self.mod
        if hasattr(m, "set_device"):
            m.set_device(ordinal)
        else:
            self.torch.flagos.device(ordinal)

    def memory_stats(self, ordinal: int) -> dict:
        raw = self.mod.memory_stats() or {}
        # 归一化：同时给出统一字段与框架原字段
        total = raw.get("total_bytes") or raw.get("total_mb", 0) * 1024 * 1024
        alloc = raw.get("allocated_bytes", raw.get("allocated_mb", 0) * 1024 * 1024)
        out = dict(raw)
        if total:
            out["total_mb"] = int(total / 1024 / 1024)
        out["used_mb"] = int(alloc / 1024 / 1024)
        return out

    def probe_device(self, ordinal: int) -> bool:
        try:
            m = self.mod
            if hasattr(m, "is_available") and not m.is_available():
                return False
            dev = f"flagos:{ordinal}"
            x = self.torch.zeros(2, 2, device=dev)
            return float(x.sum().item()) == 0.0
        except Exception:
            return False

    # ───────────── 流 / 事件 ─────────────
    def create_stream(self):
        return self.mod.Stream()

    def create_event(self):
        # 统一语义适配（未 record 不误报完成 + 主机有界等待）
        return FlagosEventAdapter()

    def current_stream(self):
        return self.mod.current_stream()

    def stream_context(self, native_stream):
        return self.mod.stream(native_stream)

    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        # flagos 原生同步为阻塞式；有界语义由上层 pyACL 路径（如可用）补充
        self.mod.synchronize()

    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        native_stream.synchronize()

    def wait_event_host(self, native_event, timeout_ms: int) -> bool:
        deadline = time.time() + (timeout_ms / 1000.0)
        while time.time() < deadline:
            try:
                if native_event.query():
                    return True
            except Exception:
                return False
            time.sleep(0.001)
        return False

    # ───────────── 错误翻译 ─────────────
    _INT_TO_CATEGORY = {
        1: ErrorCategory.L1_RESOURCE,
        2: ErrorCategory.L2_PARAM,
        3: ErrorCategory.L3_EXECUTION,
        4: ErrorCategory.L4_FATAL,
    }

    def _to_unified(self, fe) -> FlagosError:
        """后端负责把历史类型（IntEnum 分级）转成统一 FlagosError。"""
        cat = getattr(fe, "category", None)
        if not isinstance(cat, ErrorCategory):
            try:
                cat = self._INT_TO_CATEGORY.get(int(cat), ErrorCategory.L3_EXECUTION)
            except Exception:
                cat = ErrorCategory.L3_EXECUTION
        return FlagosError(
            category=cat,
            backend=self.name,
            raw=str(getattr(fe, "raw", fe)),
            message=getattr(fe, "message", str(fe)),
            mapped=bool(getattr(fe, "mapped", False)),
            graded_by=getattr(fe, "graded_by", "unknown"),
            code=getattr(fe, "code", None),
            location=getattr(fe, "location", ""),
        )

    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        errors = self._load_errors()
        fe = errors.translate_error(exc, location=location)
        return self._to_unified(fe)

    # ───────────── 恢复 ─────────────
    def recover_device(self, ordinal: int, mode: str = "probe",
                       **kwargs: Any) -> dict:
        """flagos 后端当前仅提供 probe 级恢复（real 重建需设备级原语，暂不支持）。"""
        ok = self.probe_device(ordinal)
        return {
            "ordinal": ordinal,
            "mode": mode,
            "recovered": bool(ok),
            "detail": "flagos 后端：probe 级探活恢复（real/hybrid 暂未支持）",
        }

    # ───────────── 能力声明 ─────────────
    def supports(self, capability: str) -> bool:
        table = {
            "stream_priority": False,
            "device_rebuild_real": False,
            "device_rebuild_hybrid": False,
            "device_rebuild_probe": True,
            "error_code_map": True,
            "bounded_sync": False,          # 原生同步为阻塞式
            "ipc_event": False,
        }
        return table.get(capability, False)

    def info(self) -> dict:
        self._load()
        return {
            "name": self.name,
            "framework": "torch_fl(flagos)",
            "torch": self._torch.__version__,
            "device_count": self.device_count(),
            "supports": {k: self.supports(k) for k in (
                "stream_priority", "device_rebuild_real", "device_rebuild_probe",
                "error_code_map", "bounded_sync")},
        }


def build() -> FlagosBackend:
    return FlagosBackend()
