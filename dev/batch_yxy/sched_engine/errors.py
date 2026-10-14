"""调度引擎错误类型（与 dynamic_batching.errors 同族的显式报错风格，不静默吞并）。"""

from __future__ import annotations


class SchedEngineError(Exception):
    """调度引擎错误基类。"""


class ModelNotVerifiedError(SchedEngineError):
    """使用尚未验证的模型画像（declared 状态仅登记契约，不可执行）。"""


class CostExtractorUnavailableError(SchedEngineError):
    """模型画像未绑定成本计量器，且框架适配层无同源计数器可用。"""


class DeviceNotVerifiedError(SchedEngineError):
    """使用尚未适配验证的设备画像（pending 状态仅占位，不可执行）。"""


class DeviceBudgetConflictError(SchedEngineError):
    """设备能力与策略预算冲突（如设备 token 上限低于策略最大桶上界）。"""


class DuplicateResultError(SchedEngineError):
    """同一 request_id 被重复申报或重复回填。"""


class UnexpectedResultError(SchedEngineError):
    """回填了未申报请求的结果。"""


class MissingResultError(SchedEngineError):
    """运行结束后仍有请求未获得结果（禁止漏处理）。"""


class FrameworkExecutionError(SchedEngineError):
    """推理框架执行错误（经适配层 map_error 统一包装）。"""

    def __init__(self, framework: str, cause: BaseException) -> None:
        super().__init__(f"framework {framework} execution failed: {cause!r}")
        self.framework = framework
        self.cause = cause


class EngineStateError(SchedEngineError):
    """引擎生命周期状态非法（如空队列运行、重复运行）。"""
