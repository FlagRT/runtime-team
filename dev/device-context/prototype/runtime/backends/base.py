#!/usr/bin/env python3
"""
Backend 插件接口规范 v0.1（runtime/backends/base.py）

对应职责（本方向负责子层）：
  - 设备上下文：设备句柄与生命周期、内存句柄与生命周期、执行句柄
  - 多流 Stream：Stream / Event 抽象与同步语义
  - 错误码翻译：厂商错误码 → L1-L4 统一分级
  - 设备状态恢复：四态监控 + 五段式恢复

设计原则：
  1. 厂商已有原生 PyTorch 扩展（torch_npu / 昆仑芯 SDK）负责算子分发，
     本层**不重复实现算子分发**，只统一"设备抽象与流语义"。
  2. 新增一家芯片 = 实现本接口 + 跑通 conformance。这是"统一接口"的可验收定义。
  3. v0.1 为原型期，允许破坏性变更（每季度评审一次）。

参考：vendor 插件目录模式（backends/<vendor>/），
      但本层位于其上层——设备抽象层而非算子 dispatch 层。
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..api.errors import FlagosError


class RuntimeBackend(ABC):
    """厂商后端插件接口 v0.1。

    实现者需提供：name / device_type 两个类属性，以及下列全部抽象方法。
    """

    #: 后端名，用户通过 runtime.use(name) 选择（如 "ascend" / "kunlun"）
    name: str = ""
    #: 用户可见的设备串前缀（如 "npu" / "xpu"）
    device_type: str = ""

    # ───────────────────────── 设备（职责 D2）─────────────────────────

    @abstractmethod
    def device_count(self) -> int:
        """可用设备数量。"""

    @abstractmethod
    def set_device(self, ordinal: int) -> None:
        """绑定当前进程/线程的默认设备。"""

    # ───────────────────────── 内存（职责 D3）─────────────────────────

    @abstractmethod
    def memory_stats(self, ordinal: int) -> dict:
        """返回 {"total_mb": int, "used_mb": int, "free_mb": int}。"""

    # ───────────────────── 执行 / 多流 Stream（职责 D4/D5）──────────────────────

    @abstractmethod
    def create_stream(self):
        """创建并返回该后端的 Stream 对象。"""

    @abstractmethod
    def create_event(self):
        """创建并返回该后端的 Event 对象。"""

    @abstractmethod
    def current_stream(self):
        """返回当前默认流。"""

    @abstractmethod
    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        """设备同步。timeout_ms 非空时应为有界等待（超时抛 TimeoutError），
        这是长驻服务避免整体 hang 死的关键能力。"""

    # ── 多流 Stream 支撑（供 api/stream.py 封装使用）──

    @abstractmethod
    def stream_context(self, native_stream):
        """返回切换到该流的上下文管理器（with 使用）。"""

    @abstractmethod
    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        """有界等待指定流；超时抛 TimeoutError。"""

    @abstractmethod
    def wait_event_host(self, native_event, timeout_ms: int) -> bool:
        """主机侧有界等待事件；返回 True 表示已完成，超时返回 False。"""

    # ───────────────────────── 错误码翻译（职责 D10）─────────────────────────

    @abstractmethod
    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        """厂商错误 → 统一 FlagosError（含 mapped / graded_by 可观测字段）。"""

    # ───────────────────────── 状态恢复（职责 D11）─────────────────────────

    @abstractmethod
    def probe_device(self, ordinal: int) -> bool:
        """轻量探活：区分可继续与需重建。健康设备应返回 True。"""

    @abstractmethod
    def recover_device(self, ordinal: int, mode: str = "probe",
                       reason: str = "") -> dict:
        """设备重建，统一返回 dict：{ordinal, mode, recovered, detail}。

        2026-09-09 统一：此前 ascend 返回 bool、flagos 返回 dict，
        同一接口跨后端返回类型不一致，上层无法统一处理（已按 dict 归一）。
        """
        """设备重建。

        mode:
          - "probe"  探针重试近似（默认保底，进程内安全）
          - "real"   真实重建（CANN 官方序列：destroyEvent→destroyStream→
                     destroyContext→aclrtResetDevice→setDevice→重建）
          - "hybrid" 先探针（快路径），失败后真实重建

        注意：real 模式会重置当前进程默认上下文；多进程共享设备时其他进程不受影响，
        但本进程需重新 set_device。生产默认启用前需多卡多进程压力测试。
        """

    # ───────────────────────── 可选能力（默认不支持）──────────────────────────

    def stream_priority_range(self):
        """流优先级范围 (least, greatest)；不支持返回 None。"""
        return None

    def supports(self, capability: str) -> bool:
        """能力查询，便于 conformance 做 stub-skip 报告。"""
        return capability in getattr(self, "_capabilities", set())

    # ───────────────────────── 元信息 ─────────────────────────

    def info(self) -> dict:
        return {
            "name": self.name,
            "device_type": self.device_type,
            "capabilities": sorted(getattr(self, "_capabilities", set())),
        }
