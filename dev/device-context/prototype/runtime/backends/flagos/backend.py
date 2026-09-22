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
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ...api.errors import DISPOSITION, ErrorCategory, FlagosError
from ..base import RuntimeBackend

# conformance 目录（已有资产所在）
_CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "conformance"


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
        self._device_state = None  # 共享的四态机（标准 import，见 _load_device_state）
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
        graded_by = getattr(fe, "graded_by", "unknown")
        cat = getattr(fe, "category", None)
        if not isinstance(cat, ErrorCategory):
            try:
                cat = self._INT_TO_CATEGORY.get(int(cat), ErrorCategory.L3_EXECUTION)
            except Exception:
                cat = ErrorCategory.L3_EXECUTION
        return FlagosError(
            category=cat,
            root_cause=getattr(fe, "root_cause", str(fe)),
            location=getattr(fe, "location", "") or "",
            error_code=getattr(fe, "error_code", None),
            mapped=bool(getattr(fe, "mapped", False)),
            graded_by=graded_by,
            is_grade_confident=bool(
                getattr(fe, "is_grade_confident", graded_by == "code_map")),
            recovery_decision=getattr(fe, "recovery_decision", {}) or {},
            backend=self.name,
        )

    # 复用 conformance/errors.py 的 ACL 码表（与 ascend 同源），样例码同 ascend
    SAMPLE_CODED_ERROR = "device reset failed, error code is 507015"

    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        errors = self._load_errors()
        fe = errors.translate_error(exc, location=location)
        u = self._to_unified(fe)
        # 2026-09-22：与 ascend/kunlun 对齐，后端侧回填 backend 名（此前缺失）
        u.backend = self.name
        return u

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
    #: 设备四态查询（2026-09-22 补实现，第 7 个跨后端缺陷）
    def _load_device_state(self):
        """按需加载 `conformance/device_state` 资产（进程内设备四态机）。

        ⚠️ **必须用标准 `import`（共享 `sys.modules`），不能用 importlib 独立模块名加载**
        —— 与 `kunlun` / `cambricon` 同一纪律。`device_state` 是**有状态单例**
        （模块级 `_ensure(ordinal)` 持有每台设备的四态、转换事件与订阅者）；
        若像本类的 `_load_errors()` 那样用 `spec_from_file_location("dc_xxx", ...)` 加载，
        会得到**两份状态机** —— conformance/上层设置的状态后端查不到，反之亦然。
        （`errors` 是无状态纯函数，两份无所谓；但那正是「第 4 个跨后端框架缺陷」的成因，
        **不应效仿**。）
        """
        if self._device_state is None:
            sys.path.insert(0, str(_CONFORMANCE_DIR))
            import device_state as _device_state
            self._device_state = _device_state
        return self._device_state

    def device_state(self, ordinal: int):
        """查询设备四态：`available` / `degraded` / `isolated` / `destroyed`。

        **2026-09-22 补实现（第 7 个跨后端缺陷）**：本类原先在 `_capabilities` 里
        **声明了 `device_state` 却没有对应方法** —— 与 2026-09-20 在 `kunlun` 上发现的
        「声明与实现不符」是**同一类**问题。处置口径与当时一致：**补实现，不是删声明**
        （四态机是芯片无关的共享资产，本后端复用它与另三家同一份实现、同一套语义）。

        暴露路径：`backend_offline_check.py --backend flagos` 第 [6] 组
        （此前该工具只为 cambricon 内置 stub，flagos 从未被自检过 ⇒ 该缺陷一直未被拦到）。

        边界（如实标注）：本后端只声明 `recovery_probe`；四态**转换**由上层/监控方向驱动，
        本方法只负责**查询**，恢复执行走 `recover_device(mode="probe")`。
        """
        return self._load_device_state().query_device_state(ordinal)

    # 能力声明：与 ascend 后端同一套键名（此前键名不一致，已统一）
    #: 能力**全集**（已知能力名，`info()["supports"]` 按此逐项 True/False 呈现；
    #: 与 kunlun / cambricon 同一份清单 —— 清单本身是"已知能力"，不代表本后端支持）
    _CAPABILITY_KEYS = (
        "device", "memory", "stream", "event", "bounded_sync",
        "error_map", "recovery_probe", "recovery_real",
        "device_state", "graph_capture", "stream_priority", "multidevice",
    )

    #: 本后端**声明支持**的能力（不支持/未验证的一律不写进来 —— 如实声明，不伪造）
    _capabilities = {
        "device", "memory", "stream", "event",
        "error_map",            # 复用 conformance/errors.py 错误码映射
        "recovery_probe",       # probe 级恢复
        "device_state",         # 2026-09-22 补实现后声明成立
        "multidevice",
        # 不支持：bounded_sync（原生同步为阻塞式）/ recovery_real / stream_priority
    }

    def supports(self, capability: str) -> bool:
        """能力查询（键名与 ascend 后端一致，便于上层统一判断）。"""
        return capability in getattr(self, "_capabilities", set())

    def info(self) -> dict:
        self._load()
        return {
            "name": self.name,
            "framework": "torch_fl(flagos)",
            "torch": self._torch.__version__,
            "device_count": self.device_count(),
            # 2026-09-22 修（同一次自检暴露的第二处）：原先这里手写了一份键名清单
            # （"stream_priority" / "device_rebuild_real" / "device_rebuild_probe" /
            #   "error_code_map" / "bounded_sync"），与 `_capabilities` 里的键名
            # （"stream_priority" / "recovery_real" / "recovery_probe" / "error_map" / …）
            # **对不上** ⇒ `info()["supports"]` 对已声明能力恒报 False，读者会得出
            # 完全相反的结论。
            # 改为与 kunlun / cambricon **同款**：按 `_CAPABILITY_KEYS` 全集逐项呈现，
            # 清单不再是手写的第二份真相 ⇒ 从结构上杜绝再次漂移。
            "supports": {k: self.supports(k) for k in self._CAPABILITY_KEYS},
        }


def build() -> FlagosBackend:
    return FlagosBackend()
