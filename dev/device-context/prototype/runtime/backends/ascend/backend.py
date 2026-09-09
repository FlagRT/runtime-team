#!/usr/bin/env python3
"""
昇腾（Ascend）后端实现（runtime/backends/ascend/backend.py）

对应周计划 W2 任务 3：把 910C 阶段已交付的资产注册进统一框架。

关键原则：**不重复实现，直接复用已验证模块**——
  - 错误码翻译  → 复用 conformance/errors.py（108 条映射 + F5 可观测）
  - 状态恢复    → 复用 conformance/recovery.py（R1-R5 + rebuild_mode 真实重建）
  - 设备状态    → 复用 conformance/device_state.py（四态机）
本文件只做"接口适配"：把已有能力包装成 RuntimeBackend 的标准形态。

底层：torch_npu（昇腾原生 PyTorch 扩展），本层不触碰算子分发。
"""

from pathlib import Path
from typing import Optional

from ...api.errors import ErrorCategory, FlagosError
from ..base import RuntimeBackend

# conformance 目录（已有资产所在）：device-context/benchmarks/ascend_regression/conformance
_CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "conformance"

# conformance 模块用 IntEnum 分级（L1=1..L4=4），统一层用字符串枚举 —— 转换表
_INT_TO_CATEGORY = {
    1: ErrorCategory.L1_RESOURCE,
    2: ErrorCategory.L2_PARAM,
    3: ErrorCategory.L3_EXECUTION,
    4: ErrorCategory.L4_FATAL,
}


class AscendBackend(RuntimeBackend):
    """昇腾后端：torch_npu + 已有 conformance 资产。"""

    name = "ascend"
    device_type = "npu"

    # 已具备的能力（conformance 会据此生成 stub-skip 报告）
    _capabilities = {
        "device", "memory", "stream", "event",
        "sync_timeout",            # pyACL synchronize_*_with_timeout（历史键名）
        "bounded_sync",           # 统一键名：有界同步（与 flagos 对齐）
        "error_map",               # 108 条 ACL 错误码映射
        "recovery_probe", "recovery_real",  # 探针重试 + 真实重建
        "device_state",            # 四态机
        "graph_capture",           # torch.npu.graph
        "stream_priority",         # least=7 / greatest=0
        "multidevice",
    }

    def __init__(self):
        self._torch = None
        self._acl = None
        self._errors = None
        self._recovery = None
        self._device_state = None
        self._conformance_loaded = False

    # ─────────────── 延迟加载（避免在无 torch_npu 环境 import 失败）───────────────

    @property
    def torch(self):
        if self._torch is None:
            import torch
            import torch_npu  # noqa: F401  触发 npu 设备注册
            self._torch = torch
        return self._torch

    @property
    def acl(self):
        """pyACL；不可用时返回 None（有界同步降级）。"""
        if self._acl is None:
            try:
                import acl
                acl.init()
                self._acl = acl
            except Exception:
                self._acl = False  # 标记不可用，避免重复尝试
        return self._acl or None

    def _load_conformance(self):
        """按需导入已有 conformance 模块（错误码/恢复/状态）。"""
        if self._conformance_loaded:
            return
        import sys
        sys.path.insert(0, str(_CONFORMANCE_DIR))
        import errors as _errors
        import recovery as _recovery
        import device_state as _device_state
        self._errors = _errors
        self._recovery = _recovery
        self._device_state = _device_state
        self._conformance_loaded = True

    # ─────────────── 设备（职责 D2）──────────────

    def device_count(self) -> int:
        return self.torch.npu.device_count()

    def set_device(self, ordinal: int) -> None:
        self.torch.npu.set_device(ordinal)

    # ─────────────── 内存（职责 D3）──────────────

    def memory_stats(self, ordinal: int) -> dict:
        torch = self.torch
        prev = torch.npu.current_device()
        if prev != ordinal:
            torch.npu.set_device(ordinal)
        try:
            free, total = torch.npu.mem_get_info()
        except Exception:
            # 老版本 torch_npu 可能没有 mem_get_info
            total = torch.npu.get_device_properties(ordinal).total_memory
            free = total - torch.npu.memory_allocated(ordinal)
        finally:
            if prev != ordinal:
                torch.npu.set_device(prev)
        return {
            "total_mb": int(total / 1024 / 1024),
            "used_mb": int((total - free) / 1024 / 1024),
            "free_mb": int(free / 1024 / 1024),
        }

    # ─────────────── 执行 / 多流 Stream（职责 D4/D5）──────────────

    def create_stream(self):
        return self.torch.npu.Stream()

    def create_event(self):
        """优先复用已有 NpuEventAdapter（补 wait_host + 未 record query 修正），
        保证与 910C 阶段的事件语义契约完全一致；不可用时退回原生 Event。"""
        self._load_conformance()
        try:
            from npu_events import NpuEventAdapter
            return NpuEventAdapter()
        except Exception:
            return self.torch.npu.Event()

    def current_stream(self):
        return self.torch.npu.current_stream()

    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        """设备同步。timeout_ms 非空时使用 pyACL 有界等待（超时抛 TimeoutError）。"""
        if timeout_ms is None:
            self.torch.npu.synchronize()
            return
        acl = self.acl
        if acl is None:
            # 无 pyACL 时降级为普通同步（调用方需知悉：不再是"有界"的）
            self.torch.npu.synchronize()
            return
        acl.rt.set_device(ordinal)
        rc = acl.rt.synchronize_device_with_timeout(timeout_ms)
        if rc != 0:
            raise TimeoutError(
                f"设备 {ordinal} 同步超时（{timeout_ms}ms），pyACL rc={rc}"
            )

    # ─────────────── 多流 Stream 支撑（供 api/stream.py 封装）───────────────

    def stream_context(self, native_stream):
        return self.torch.npu.stream(native_stream)

    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        """有界等待指定流（pyACL synchronize_stream_with_timeout）。

        注意：需要流的底层句柄（torch Stream 的 .npu_stream 属性），
        且任务与同步必须在同一流上才会触发超时（同步空流会立即返回）。
        """
        acl = self.acl
        if acl is None:
            native_stream.synchronize()
            return
        handle = getattr(native_stream, "npu_stream", None)
        if handle is None:
            native_stream.synchronize()
            return
        rc = acl.rt.synchronize_stream_with_timeout(handle, timeout_ms)
        if rc != 0:
            raise TimeoutError(f"流同步超时（{timeout_ms}ms），pyACL rc={rc}")

    def wait_event_host(self, native_event, timeout_ms: int) -> bool:
        """主机侧有界等待：轮询 query，避免无限阻塞。"""
        import time
        deadline = time.time() + timeout_ms / 1000.0
        while True:
            try:
                if native_event.query():
                    return True
            except Exception:
                # 未 record 的事件 query 语义由契约定义；此处按"未完成"处理
                pass
            if time.time() >= deadline:
                return False
            time.sleep(0.001)

    # ─────────────── 错误码翻译（职责 D10）──────────────

    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        """翻译为**统一** FlagosError。

        注意：conformance 模块的历史 FlagosError 使用 IntEnum 分级（L1=1..L4=4）
        且不含 disposition/retryable 等统一语义字段。为保证对外类型一致，
        这里统一转换为 runtime.api.errors.FlagosError。
        """
        self._load_conformance()
        fe = self._errors.translate_error(exc, location=location)
        return self._to_unified(fe)

    @staticmethod
    def _to_unified(fe) -> FlagosError:
        """历史 FlagosError → 统一 FlagosError（IntEnum → 统一枚举）。"""
        cat = getattr(fe, "category", None)
        if isinstance(cat, int) and not isinstance(cat, ErrorCategory):
            cat = _INT_TO_CATEGORY.get(int(cat), ErrorCategory.L3_EXECUTION)
        graded_by = getattr(fe, "graded_by", "default")
        return FlagosError(
            category=cat if isinstance(cat, ErrorCategory) else ErrorCategory.L3_EXECUTION,
            root_cause=getattr(fe, "root_cause", str(fe)),
            location=getattr(fe, "location", "") or "",
            error_code=getattr(fe, "error_code", None),
            mapped=bool(getattr(fe, "mapped", False)),
            graded_by=graded_by,
            is_grade_confident=bool(
                getattr(fe, "is_grade_confident", graded_by == "code_map")
            ),
            recovery_decision=getattr(fe, "recovery_decision", {}) or {},
        )

    # ─────────────── 状态恢复（职责 D11）──────────────

    def probe_device(self, ordinal: int) -> bool:
        self._load_conformance()
        return self._recovery.probe_device(ordinal, device=self.device_type)

    def recover_device(self, ordinal: int, mode: str = "probe",
                       reason: str = "") -> dict:
        """设备重建。mode: probe / real / hybrid（与 recovery.rebuild_mode 一致）。

        统一返回 dict（底层 recovery 返回 bool，此处包装，
        与 flagos 后端及接口约定保持一致）。
        """
        self._load_conformance()
        ok = self._recovery.recover_device(
            ordinal,
            reason=reason or f"runtime: rebuild({mode})",
            device=self.device_type,
            rebuild_mode=mode,
        )
        return {
            "ordinal": ordinal,
            "mode": mode,
            "recovered": bool(ok),
            "detail": f"ascend 后端：recovery.rebuild_mode={mode}",
        }

    # ─────────────── 可选能力 ───────────────

    def stream_priority_range(self):
        acl = self.acl
        if acl is None:
            return None
        try:
            return acl.rt.device_get_stream_priority_range()
        except Exception:
            return None

    def device_state(self, ordinal: int):
        """查询设备四态（AVAILABLE/DEGRADED/ISOLATED/DESTROYED）。"""
        self._load_conformance()
        return self._device_state.query_device_state(ordinal)


def build() -> AscendBackend:
    """注册表自动发现使用的工厂函数。"""
    return AscendBackend()
