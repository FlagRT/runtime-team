"""调度核心：请求队列、优先级取队、结果回填与引擎编排。

本层不依赖 vLLM、模型专用输入类型或芯片 SDK；组批决策复用
dynamic_batching.BatchCoordinator（分桶/预算/封口），不重写。
"""

from .engine import (
    PATH_LEGACY,
    PATH_SCHEDULED,
    EngineRunReport,
    SchedulerEngine,
    clamp_policy,
)
from .queue import QueueLaneStatus, RequestQueue, UnknownLaneError
from .request import SchedRequest
from .results import ResultStore

__all__ = [
    "SchedulerEngine",
    "EngineRunReport",
    "PATH_LEGACY",
    "PATH_SCHEDULED",
    "clamp_policy",
    "SchedRequest",
    "RequestQueue",
    "QueueLaneStatus",
    "UnknownLaneError",
    "ResultStore",
]
