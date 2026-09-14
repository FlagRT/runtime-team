#!/usr/bin/env python3
"""昆仑芯 P800 后端（XPytorch / torch.cuda 适配）。

⚠️ **命名空间说明（务必先读）**
  昆仑芯 P800 这栈上设备 API **走 `torch.cuda`，不是 `torch.xpu`**。四条独立证据：
    1. 编译标志：`torch.__config__.show()` → `USE_CUDA=ON, ..., USE_XPU=OFF`
    2. 运行时：`torch.xpu.is_available() == False`（`Torch not compiled with XPU enabled`）
    3. 官方佐证：FlagTree xpu3.6 单测 `third_party/xpu/python/test/unit/conftest.py`
       的 `--device` **默认值就是 `'cuda'`**
    4. 厂商运行口径：选卡用 `CUDA_VISIBLE_DEVICES`（实测 `=2` → 1 卡）
  机制：XPytorch（`/env/xpytorch-*.run`）+ `torch_xray` 符号重写，把 CUDA 命名空间调用
  重定向到 P800；启动时可见 `XCCL .../libbkcl.so loaded` 与 `SYMBOL_REWRITE torch success`。

  因此本后端：
    - `name = "kunlun"`        ← 厂商标识（我们注册表里的键）
    - `device_type = "cuda"`   ← 真实可用的设备串前缀（`torch.device("cuda:0")` 有效）
  **两者不同名是刻意为之**：后端名按厂商，设备串按命名空间。
  `kunlun` 与未来的 `nvidia` 共用 `torch.cuda`，故厂商判别不能用命名空间，
  需用厂商特征（`torch_xray`/`torch_xmlir` 模块、`/proc/kunlun/`、`xpu-smi`、通信库）。

**运行前置**（缺一不可）
  - `conda activate python310_torch29_cuda`（镜像默认 `python3` 是 conda base 3.13，**不是**目标环境）
  - `export XPU_EVENT_KL3_ENABLE=1`（官方手册要求）

**已知上游缺陷（已对外提交，见接入方案 §7.4）**
  - `torch.cuda.Stream.priority_range()` 触发 PyTorch `INTERNAL ASSERT FAILED
    at c10/cuda/CUDAStream.h:188`（稳定复现）→ 本后端 `stream_priority_range()` 直接返回 None，
    **不得透传该调用**
  - 厂商错误码不透出到 Python 异常（仅退出钩子偶见 `error code= 101`）→
    `translate_error` 的 `code_map` 路径**不可达**，只能走 `message_hint`

接入定义（接口约定 §2）：新增一家芯片 = 实现本接口 + 跑通 conformance 13 例 + 6 例。
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ...api.errors import FlagosError
from ..base import RuntimeBackend

logger = logging.getLogger(__name__)

# conformance 目录（既有资产所在）
_CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "conformance"


class KunlunEventAdapter:
    """`torch.cuda.Event` 的统一语义适配（与 NpuEventAdapter / FlagosEventAdapter 同构）。

    补齐两点与统一事件契约的语义缺口：
      - **E3：未 record 的 Event 调 `query()` 原生返回 `True`（误报"已完成"）** ——
        实测昆仑芯如此；用 `recorded` 标志修正为 `False`。
      - **E2v2：主机侧有界等待 `wait_host(timeout_ms)`** —— 原生 Event 无此方法，
        用 `query()` 轮询实现，**永不永久阻塞**。
    """

    def __init__(self, *args, **kwargs):
        self._ev = None
        self._args, self._kwargs = args, kwargs
        self._recorded = False

    def _ensure(self):
        if self._ev is None:
            import torch
            self._ev = torch.cuda.Event(*self._args, **self._kwargs)
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
        # E3 修正：未 record 不得报"已完成"
        if not self._recorded:
            return False
        return self._ensure().query()

    def wait_host(self, timeout_ms: Optional[int] = None) -> bool:
        """主机侧**有界**等待：完成返回 True，超时返回 False（不永久阻塞）。"""
        deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000.0
        while not self.query():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.001)
        return True

    def elapsed_time(self, end_event):
        return self._ensure().elapsed_time(getattr(end_event, "_ev", end_event))

    def __getattr__(self, item):
        return getattr(self._ensure(), item)


class KunlunBackend(RuntimeBackend):
    """基于 XPytorch（`torch.cuda`）的昆仑芯 P800 后端。"""

    name = "kunlun"
    device_type = "cuda"   # 见模块 docstring：设备串按命名空间，不按厂商

    #: 规范能力键全集（用于 info() 的自洽报告；键名与 ascend/flagos 对齐）
    _CAPABILITY_KEYS = (
        "device", "memory", "stream", "event", "bounded_sync",
        "error_map", "recovery_probe", "recovery_real",
        "device_state", "graph_capture", "stream_priority", "multidevice",
    )

    #: 本后端**声明支持**的能力（不支持的一律不写进来，不伪造）
    _capabilities = {
        "device",
        "memory",
        "stream",
        "event",
        "bounded_sync",      # 主机侧等待真有界；流同步为"超时上报"语义，见 synchronize_stream
        "recovery_probe",    # 探针级恢复
        "device_state",      # 四态机
        "multidevice",       # 单机 8 卡
        # ── 以下**不支持**，故不声明 ──
        # "error_map"       : 无厂商错误码（Python 层不可得）→ 只有 message_hint 分级
        # "recovery_real"   : 无设备级重置/重建原语（实测 torch.cuda 只有内存统计类 reset*）
        # "stream_priority" : priority_range() 触发 PyTorch INTERNAL ASSERT（上游缺陷）
        # "graph_capture"   : 未验证
    }

    #: 分级来源可达性（如实标注：code_map 路径在昆仑芯不可达）
    _grading_paths = {"code_map": False, "message_hint": True}

    def __init__(self) -> None:
        self._torch = None
        self._errors_mod = None
        self._loaded = False

    # ───────────── 延迟加载 ─────────────
    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        self._torch = torch
        if not os.environ.get("XPU_EVENT_KL3_ENABLE"):
            logger.warning(
                "环境变量 XPU_EVENT_KL3_ENABLE 未设置；官方手册要求测试前 "
                "export XPU_EVENT_KL3_ENABLE=1，否则部分路径行为未定义"
            )
        self._loaded = True

    @property
    def torch(self):
        self._load()
        return self._torch

    def _load_errors(self):
        """复用 conformance/errors.py 的统一翻译骨架（含厂商中立的 message 规则）。

        注意：其中的 `ACL_ERR_TO_CATEGORY`（108 条昇腾错误码）在昆仑芯**永不命中**，
        实际只用 `_MESSAGE_HINTS` 消息规则 —— 故 `graded_by` 只会是
        `message_hint` 或 `default`，`mapped` 恒为 False。已如实反映在
        `_grading_paths` 与 `info()` 中。
        """
        if self._errors_mod is None:
            sys.path.insert(0, str(_CONFORMANCE_DIR))
            spec = importlib.util.spec_from_file_location(
                "dc_conformance_errors", _CONFORMANCE_DIR / "errors.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._errors_mod = mod
        return self._errors_mod

    # ───────────── 设备 ─────────────
    def device_count(self) -> int:
        return int(self.torch.cuda.device_count())

    def set_device(self, ordinal: int) -> None:
        self.torch.cuda.set_device(ordinal)

    def memory_stats(self, ordinal: int) -> dict:
        """归一化为 {total_mb, used_mb, free_mb}。

        ⚠️ 实测：昆仑芯 `torch.cuda.memory_stats()` **返回空 dict**，不可用；
        必须走 `mem_get_info()`（实测 `(103045660672, 103079215104)` 准确）+ `memory_allocated()`。
        """
        torch = self.torch
        prev = torch.cuda.current_device()
        if prev != ordinal:
            torch.cuda.set_device(ordinal)
        try:
            free, total = torch.cuda.mem_get_info(ordinal)
            used = total - free
        finally:
            if prev != ordinal:
                try:
                    torch.cuda.set_device(prev)
                except Exception:
                    pass
        return {
            "total_mb": int(total / 1024 / 1024),
            "used_mb": int(used / 1024 / 1024),
            "free_mb": int(free / 1024 / 1024),
        }

    def probe_device(self, ordinal: int) -> bool:
        """轻量探活：分配 2x2 零张量并求和，不干扰业务。"""
        try:
            torch = self.torch
            if not torch.cuda.is_available():
                return False
            prev = torch.cuda.current_device()
            if prev != ordinal:
                torch.cuda.set_device(ordinal)
            try:
                x = torch.zeros(2, 2, device=f"cuda:{ordinal}")
                return float(x.sum().item()) == 0.0
            finally:
                if prev != ordinal:
                    try:
                        torch.cuda.set_device(prev)
                    except Exception:
                        pass
        except Exception:
            return False

    # ───────────── 流 / 事件 ─────────────
    def create_stream(self):
        return self.torch.cuda.Stream()

    def create_event(self):
        # 统一语义适配：未 record 不误报完成（E3）+ 主机侧有界等待（E2v2）
        return KunlunEventAdapter()

    def current_stream(self):
        return self.torch.cuda.current_stream()

    def stream_context(self, native_stream):
        return self.torch.cuda.stream(native_stream)

    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        """设备同步。timeout_ms 为空 → 原生阻塞同步；非空 → 轮询上报超时。"""
        torch = self.torch
        if timeout_ms is None:
            torch.cuda.synchronize()
            return
        self._bounded_wait(lambda: torch.cuda.current_stream().query(), timeout_ms, "device")

    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        """有界等待指定流；超时抛 `TimeoutError`。

        ⚠️ **语义边界（如实标注）**：实测昆仑芯 `Stream.synchronize()` 签名为
        `(self) -> None`，**无 timeout 参数**，且无中断原语。因此这里的「有界」是
        **超时上报**语义 —— 超时抛 `TimeoutError` 让上层得以降级/记录，
        但**不保证中断底层已提交的执行**。
        规范当前只写了「超时抛 TimeoutError」，未区分「真中断」与「超时上报」，
        已作为修订建议提交（接入方案 §7.3）。
        """
        if timeout_ms is None:
            native_stream.synchronize()
            return
        self._bounded_wait(native_stream.query, timeout_ms, "stream")

    def _bounded_wait(self, query_fn, timeout_ms: int, what: str) -> None:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            try:
                if query_fn():
                    return
            except Exception:
                # 查询本身失败 → 不视为超时，交由上层错误翻译处理
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"kunlun: {what} synchronize timeout after {timeout_ms} ms"
                    "（超时上报语义，不保证中断底层执行）"
                )
            time.sleep(0.001)

    def wait_event_host(self, native_event, timeout_ms: int) -> bool:
        """主机侧有界等待事件；完成 True / 超时 False（永不永久阻塞）。"""
        if hasattr(native_event, "wait_host"):
            return bool(native_event.wait_host(timeout_ms))
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                if native_event.query():
                    return True
            except Exception:
                return False
            time.sleep(0.001)
        return False

    # ───────────── 可选能力 ─────────────
    def stream_priority_range(self):
        """**不支持，返回 None。**

        ⚠️ 实测调用 `torch.cuda.Stream.priority_range()` 会触发 PyTorch 自身的
        `INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDAStream.h":188`
        (`greatest_priority <= -1 ... Unexpected CUDA stream priority range`) ——
        属上游（XPytorch 后端上报了非法优先级区间），**不是本层可修项**。
        因此此处**主动拦截，绝不透传**，避免触发 C++ 断言。
        （注：`Stream.priority` 单值属性可读，实测为 0；但区间不可得。）
        """
        return None

    # ───────────── 错误翻译 ─────────────
    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        errors = self._load_errors()
        fe = errors.translate_error(exc, location=location)
        graded_by = getattr(fe, "graded_by", "default")
        # 如实降级：昆仑芯无错误码 → code_map 不可达，只有 message_hint / default 可信
        if graded_by == "code_map":
            graded_by = "message_hint_unexpected"
        return FlagosError(
            category=getattr(fe, "category", None) or _l3(),
            root_cause=getattr(fe, "root_cause", f"{type(exc).__name__}: {exc}"),
            location=getattr(fe, "location", "") or location,
            error_code=getattr(fe, "error_code", None),
            mapped=bool(getattr(fe, "mapped", False)),
            graded_by=graded_by,
            # 无厂商码可依据 → 除"命中消息规则"外均不标为高置信
            is_grade_confident=(graded_by == "message_hint"),
            recovery_decision=getattr(fe, "recovery_decision", {}) or {},
            backend=self.name,
        )

    # ───────────── 恢复 ─────────────
    def recover_device(self, ordinal: int, mode: str = "probe",
                       reason: str = "", **kwargs: Any) -> dict:
        """设备重建。统一返回 dict，`recovered` 语义 = **设备当前可用**。

        ⚠️ 实测：`torch.cuda` 上**没有设备级重置/重建原语** ——
        `reset*` 系列全是内存统计类（`reset_peak_memory_stats` 等），
        无 `reset_device` / context 重建。故 **`real` 模式不支持**，
        一律以探活结果判定设备是否可用，并在 `detail` 中如实说明。
        """
        alive = self.probe_device(ordinal)
        if mode not in ("probe", "hybrid"):
            return {
                "ordinal": ordinal, "mode": mode, "recovered": alive,
                "detail": ("昆仑芯无设备级重置/重建原语（torch.cuda 仅内存统计类 reset*，"
                           "无 reset_device/context 重建）→ real 模式不支持；"
                           "已按探活结果判定，如需 real 需上游提供原语"),
            }
        return {
            "ordinal": ordinal, "mode": mode, "recovered": alive,
            "detail": f"probe 级探活：设备当前{'可用' if alive else '不可用'}",
        }

    # ───────────── 能力声明 ─────────────
    def supports(self, capability: str) -> bool:
        return capability in self._capabilities

    def info(self) -> dict:
        self._load()
        return {
            "name": self.name,
            "device_type": self.device_type,
            "framework": "XPytorch (torch.cuda 兼容层) + torch_xray 符号重写",
            "torch": self._torch.__version__,
            "torch_build": {"USE_CUDA": True, "USE_XPU": False},
            "device_count": self.device_count(),
            "capabilities": sorted(self._capabilities),
            # 与 _capabilities 同一套键名（自洽，不重复出现 A 键声明/B 键查询的问题）
            "supports": {k: self.supports(k) for k in self._CAPABILITY_KEYS},
            "error_grading": dict(self._grading_paths),
            "bounded_sync_scope": "主机侧等待（event.wait_host）真有界；流同步为超时上报语义",
            "vendor_discriminator": [
                "torch_xray / torch_xmlir 模块存在",
                "/proc/kunlun 存在",
                "xpu-smi 存在",
                "通信库 libbkcl.so（XCCL/BKCL）",
            ],
            "known_upstream_defects": [
                "Stream.priority_range() → PyTorch INTERNAL ASSERT (c10/cuda/CUDAStream.h:188)",
                "厂商错误码不透出到 Python 异常（仅退出钩子偶见 error code=101）",
            ],
        }


def _l3():
    from ...api.errors import ErrorCategory
    return ErrorCategory.L3_EXECUTION


def build() -> KunlunBackend:
    return KunlunBackend()
