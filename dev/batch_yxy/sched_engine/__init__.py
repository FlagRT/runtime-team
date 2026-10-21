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

Public exports are lazy: discovery/CLI do not import inference dependencies.
"""
from importlib import import_module

_EXPORTS = {
    'PATH_LEGACY': ('.core', 'PATH_LEGACY'),
    'PATH_SCHEDULED': ('.core', 'PATH_SCHEDULED'),
    'EngineRunReport': ('.core', 'EngineRunReport'),
    'QueueLaneStatus': ('.core', 'QueueLaneStatus'),
    'RequestQueue': ('.core', 'RequestQueue'),
    'ResultStore': ('.core', 'ResultStore'),
    'SchedRequest': ('.core', 'SchedRequest'),
    'SchedulerEngine': ('.core', 'SchedulerEngine'),
    'clamp_policy': ('.core', 'clamp_policy'),
    'ALL_DEVICES': ('.devices', 'ALL_DEVICES'),
    'DeviceProfile': ('.devices', 'DeviceProfile'),
    'all_devices': ('.devices', 'all_devices'),
    'get_device': ('.devices', 'get_device'),
    'CostExtractorUnavailableError': ('.errors', 'CostExtractorUnavailableError'),
    'DeviceBudgetConflictError': ('.errors', 'DeviceBudgetConflictError'),
    'DeviceNotVerifiedError': ('.errors', 'DeviceNotVerifiedError'),
    'DuplicateResultError': ('.errors', 'DuplicateResultError'),
    'EngineStateError': ('.errors', 'EngineStateError'),
    'FrameworkExecutionError': ('.errors', 'FrameworkExecutionError'),
    'MissingResultError': ('.errors', 'MissingResultError'),
    'ModelNotVerifiedError': ('.errors', 'ModelNotVerifiedError'),
    'SchedEngineError': ('.errors', 'SchedEngineError'),
    'UnexpectedResultError': ('.errors', 'UnexpectedResultError'),
    'DECLARED_FRAMEWORKS': ('.frameworks', 'DECLARED_FRAMEWORKS'),
    'VERIFIED_FRAMEWORKS': ('.frameworks', 'VERIFIED_FRAMEWORKS'),
    'BatchHandle': ('.frameworks', 'BatchHandle'),
    'FrameworkAdapter': ('.frameworks', 'FrameworkAdapter'),
    'VllmEmbedFrameworkAdapter': ('.frameworks', 'VllmEmbedFrameworkAdapter'),
    'ALL_PROFILES': ('.models', 'ALL_PROFILES'),
    'ModelProfile': ('.models', 'ModelProfile'),
    'all_profiles': ('.models', 'all_profiles'),
    'get_profile': ('.models', 'get_profile'),
}

__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
