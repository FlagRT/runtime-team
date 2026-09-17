#!/usr/bin/env python3
"""
统一运行时 API · 错误分级（runtime/api/errors.py）

对应职责：错误码翻译（厂商错误码 → 统一分级 + 归因位置 + 保留 root cause）。

设计要点（继承自 910C 阶段的 D10 成果）：
  - 分级直接决定处置策略，是运行时层最重要的"对外语义"
  - L1 资源 → 可重试；L2 参数 → 上抛调用方；L3 执行 → 同上下文重放；L4 致命 → 设备恢复
  - mapped / graded_by 是 F5 可观测字段：区分"确定分级"与"保守兜底"，
    上层不能把兜底 L3 当确定结论使用
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ErrorCategory(Enum):
    """统一错误分级。分级语义与处置策略见下。"""
    L1_RESOURCE = "L1_RESOURCE"      # 资源类（OOM 等）→ 可重试
    L2_PARAM = "L2_PARAM"            # 参数/契约违反 → 上抛调用方（重试无意义）
    L3_EXECUTION = "L3_EXECUTION"    # 执行期失败 → 同上下文重放
    L4_FATAL = "L4_FATAL"            # 硬件/致命 → 设备状态恢复


# 分级 → 处置策略（供上层查询，避免各处硬编码 if-else）
DISPOSITION = {
    ErrorCategory.L1_RESOURCE: "retry",
    ErrorCategory.L2_PARAM: "raise",
    ErrorCategory.L3_EXECUTION: "replay",
    ErrorCategory.L4_FATAL: "device_recovery",
}


@dataclass
class FlagosError:
    """统一错误对象：厂商错误经后端翻译后的标准形态。

    Attributes:
        category: 统一分级（L1-L4）
        root_cause: 原始错误信息（保留，便于定位）
        location: 归因位置（如 "stream:0/op:matmul"）
        error_code: 厂商原始错误码（若有）
        mapped: 是否命中映射表（False 表示按规则/兜底推断）
        graded_by: 分级来源 —— code_map（码表命中）/ message_hint（消息规则）/ default（兜底）
        is_grade_confident: 分级是否可信（兜底时为 False）
        disposition: 由 category 推导的处置策略
        backend: 产生该错误的后端名（如 "ascend"）
    """
    category: ErrorCategory
    root_cause: str
    location: str = ""
    error_code: Optional[int] = None
    mapped: bool = False
    graded_by: str = "default"
    is_grade_confident: bool = False
    backend: str = ""
    recovery_decision: dict = field(default_factory=dict)

    @property
    def disposition(self) -> str:
        return DISPOSITION[self.category]

    @property
    def retryable(self) -> bool:
        return self.category == ErrorCategory.L1_RESOURCE

    @property
    def replayable(self) -> bool:
        return self.category in (ErrorCategory.L1_RESOURCE, ErrorCategory.L3_EXECUTION)

    def __str__(self) -> str:
        code = f" code={self.error_code}" if self.error_code is not None else ""
        return (f"[{self.category.value}]{code} @ {self.location or '-'} :: "
                f"{self.root_cause[:120]}")


def translate_via_backend(exc: BaseException, backend, location: str = "") -> FlagosError:
    """统一入口：把异常交给具体后端翻译，并补齐后端名与处置信息。"""
    fe = backend.translate_error(exc, location=location)
    fe.backend = backend.name
    return fe
