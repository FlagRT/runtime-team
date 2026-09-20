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

#: conformance 模块用 IntEnum 分级（L1=1..L4=4），统一层用字符串枚举 —— 数值转换表
_INT_TO_CATEGORY = {
    1: ErrorCategory.L1_RESOURCE,
    2: ErrorCategory.L2_PARAM,
    3: ErrorCategory.L3_EXECUTION,
    4: ErrorCategory.L4_FATAL,
}


def coerce_category(value) -> Optional["ErrorCategory"]:
    """把「任意来源的分级值」归一为统一 `ErrorCategory`；无法识别时返回 None。

    接受的输入：统一枚举成员 / 整数（含 IntEnum，如 conformance 的 L1–L4）/ 名称字符串。

    为什么需要它：`conformance/errors.py` 会被后端以 importlib **动态加载为独立模块**
    （见 `backends/kunlun/backend.py::_load_errors`），于是同一语义存在两份枚举类对象。
    即使取值相同（L2_PARAM 都是 2），`dc_conformance_errors.ErrorCategory.L2_PARAM`
    与 `runtime.api.errors.ErrorCategory.L2_PARAM` 也**不相等** ⇒ `DISPOSITION[cat]`
    会 KeyError。2026-09-20 由昆仑芯 P800 推理腿首次暴露（ascend 因自身做了 IntEnum
    转换而未暴露）。
    """
    if isinstance(value, ErrorCategory):
        return value
    if isinstance(value, int) and not isinstance(value, bool):     # 含 IntEnum
        return _INT_TO_CATEGORY.get(int(value))
    name = getattr(value, "name", None)
    if not isinstance(name, str):
        name = value if isinstance(value, str) else None
    if isinstance(name, str):
        members = ErrorCategory.__members__
        return members.get(name) or members.get(name.upper())
    return None


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
    """统一入口：把异常交给具体后端翻译，并补齐后端名与处置信息。

    2026-09-20（昆仑芯 P800 推理腿暴露）：新增**归一化兜底**。
    后端可能返回"动态加载的 conformance 模块"里的 FlagosError / ErrorCategory
    （`kunlun._load_errors()` 用 importlib 以独立模块名加载 `conformance/errors.py`，
    于是同一语义存在**两份枚举类对象**）。即使值相同（L2_PARAM 都是 2），两者也不相等，
    一旦透传到上层，`fe.disposition` 会因 `DISPOSITION[cat]` 查表失败而 KeyError。
    此处统一重建为 api 层对象，保证对外类型一致（对已自行转换的 ascend 是幂等操作）。
    """
    fe = backend.translate_error(exc, location=location)
    fe = normalize_error(fe, default_backend=backend.name)
    fe.backend = backend.name
    return fe


def normalize_error(fe, default_backend: str = "") -> FlagosError:
    """任意来源的错误对象 → 统一 `FlagosError`（幂等：已是统一对象则原样返回）。"""
    cat = coerce_category(getattr(fe, "category", None))
    if isinstance(fe, FlagosError) and cat is getattr(fe, "category", None):
        return fe
    return FlagosError(
        category=cat or ErrorCategory.L3_EXECUTION,      # 分级不可识别 → 保守兜底 L3
        root_cause=getattr(fe, "root_cause", str(fe)),
        location=getattr(fe, "location", "") or "",
        error_code=getattr(fe, "error_code", None),
        mapped=bool(getattr(fe, "mapped", False)),
        graded_by=getattr(fe, "graded_by", "default"),
        is_grade_confident=bool(getattr(fe, "is_grade_confident", False)),
        backend=getattr(fe, "backend", "") or default_backend,
        recovery_decision=getattr(fe, "recovery_decision", {}) or {},
    )
