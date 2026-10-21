"""动态组批与长度感知分桶模块（Qwen3-Embedding + vLLM 最小接入）。

核心（与 vLLM 解耦）：BatchCoordinator / BatchRequest / BatchPolicy / BatchPlan。
执行适配：executors.FakeEmbedExecutor（离线）、vllm_embed_adapter.VllmEmbedExecutor
与 DynamicBatchEmbedder（A/B 开关）。

Public exports are lazy: discovery/CLI do not import inference dependencies.
"""
from importlib import import_module

_EXPORTS = {
    'policy_from_config': ('.config', 'policy_from_config'),
    'policy_from_json_file': ('.config', 'policy_from_json_file'),
    'DISPATCH_BATCH_LIMIT': ('.coordinator', 'DISPATCH_BATCH_LIMIT'),
    'DISPATCH_FLUSH': ('.coordinator', 'DISPATCH_FLUSH'),
    'DISPATCH_TOKEN_LIMIT': ('.coordinator', 'DISPATCH_TOKEN_LIMIT'),
    'DISPATCH_WAIT_LIMIT': ('.coordinator', 'DISPATCH_WAIT_LIMIT'),
    'BatchCoordinator': ('.coordinator', 'BatchCoordinator'),
    'BatchCoordinatorError': ('.errors', 'BatchCoordinatorError'),
    'ClosedCoordinatorError': ('.errors', 'ClosedCoordinatorError'),
    'DuplicateRequestError': ('.errors', 'DuplicateRequestError'),
    'PolicyValidationError': ('.errors', 'PolicyValidationError'),
    'RequestTooLongError': ('.errors', 'RequestTooLongError'),
    'RequestValidationError': ('.errors', 'RequestValidationError'),
    'ResultCountMismatchError': ('.errors', 'ResultCountMismatchError'),
    'UnknownPolicyError': ('.errors', 'UnknownPolicyError'),
    'EmbeddingExecutor': ('.executors', 'EmbeddingExecutor'),
    'FakeEmbedExecutor': ('.executors', 'FakeEmbedExecutor'),
    'BatchDispatchRecord': ('.observability', 'BatchDispatchRecord'),
    'padding_ratio': ('.observability', 'padding_ratio'),
    'summarize_records': ('.observability', 'summarize_records'),
    'BatchPlan': ('.types', 'BatchPlan'),
    'BatchPolicy': ('.types', 'BatchPolicy'),
    'BatchRequest': ('.types', 'BatchRequest'),
    'compatibility_key': ('.types', 'compatibility_key'),
    'MODE_BASELINE': ('.vllm_embed_adapter', 'MODE_BASELINE'),
    'MODE_DYNAMIC': ('.vllm_embed_adapter', 'MODE_DYNAMIC'),
    'DynamicBatchEmbedder': ('.vllm_embed_adapter', 'DynamicBatchEmbedder'),
    'EmbedRunResult': ('.vllm_embed_adapter', 'EmbedRunResult'),
    'RequestTiming': ('.vllm_embed_adapter', 'RequestTiming'),
    'VllmEmbedExecutor': ('.vllm_embed_adapter', 'VllmEmbedExecutor'),
}

__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
