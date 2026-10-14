"""与模型、推理框架、芯片解耦的调度引擎（四层架构）。

四层职责：
1. sched_engine.core        调度核心：请求队列、优先级取队、分桶组批（复用
   dynamic_batching.BatchCoordinator）、批次预算、发批、请求→结果恰好一次映射；
   不依赖 vLLM、模型专用输入类型或芯片 SDK；
2. sched_engine.models      模型画像：调度特征/兼容字段/成本计量契约 + 15 模型注册表
   （Qwen3-Embedding-0.6B verified，其余 declared）；
3. sched_engine.frameworks  推理框架适配器：submit_batch/get_results/map_error
   统一接口；vllm-embed 已接入（复用 VllmEmbedExecutor），其他框架仅契约；
4. sched_engine.devices     设备能力画像：批量/Shape/资源预算芯片差异隔离；
   ascend-910c verified，其余 5 家 pending。

接入层级说明：本引擎的组批是外层请求分组（每个批次计划对应一次框架调用）；
推理引擎内部（vLLM V1 连续组批/抢占）的实际执行批次不由本引擎控制或宣称。
"""

from .core import (
    PATH_LEGACY,
    PATH_SCHEDULED,
    EngineRunReport,
    QueueLaneStatus,
    RequestQueue,
    ResultStore,
    SchedRequest,
    SchedulerEngine,
    clamp_policy,
)
from .devices import (
    ALL_DEVICES,
    DeviceProfile,
    all_devices,
    get_device,
)
from .errors import (
    CostExtractorUnavailableError,
    DeviceBudgetConflictError,
    DeviceNotVerifiedError,
    DuplicateResultError,
    EngineStateError,
    FrameworkExecutionError,
    MissingResultError,
    ModelNotVerifiedError,
    SchedEngineError,
    UnexpectedResultError,
)
from .frameworks import (
    DECLARED_FRAMEWORKS,
    VERIFIED_FRAMEWORKS,
    BatchHandle,
    FrameworkAdapter,
    VllmEmbedFrameworkAdapter,
)
from .models import (
    ALL_PROFILES,
    ModelProfile,
    all_profiles,
    get_profile,
)

__all__ = [
    # core
    "SchedulerEngine",
    "EngineRunReport",
    "SchedRequest",
    "RequestQueue",
    "QueueLaneStatus",
    "ResultStore",
    "clamp_policy",
    "PATH_LEGACY",
    "PATH_SCHEDULED",
    # models
    "ModelProfile",
    "ALL_PROFILES",
    "all_profiles",
    "get_profile",
    # frameworks
    "FrameworkAdapter",
    "BatchHandle",
    "VllmEmbedFrameworkAdapter",
    "VERIFIED_FRAMEWORKS",
    "DECLARED_FRAMEWORKS",
    # devices
    "DeviceProfile",
    "ALL_DEVICES",
    "all_devices",
    "get_device",
    # errors
    "SchedEngineError",
    "ModelNotVerifiedError",
    "CostExtractorUnavailableError",
    "DeviceNotVerifiedError",
    "DeviceBudgetConflictError",
    "DuplicateResultError",
    "UnexpectedResultError",
    "MissingResultError",
    "FrameworkExecutionError",
    "EngineStateError",
]
