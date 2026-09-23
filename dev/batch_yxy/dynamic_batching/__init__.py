"""动态组批与长度感知分桶模块（Qwen3-Embedding + vLLM 最小接入）。

核心（与 vLLM 解耦）：BatchCoordinator / BatchRequest / BatchPolicy / BatchPlan。
执行适配：executors.FakeEmbedExecutor（离线）、vllm_embed_adapter.VllmEmbedExecutor
与 DynamicBatchEmbedder（A/B 开关）。
"""

from .config import policy_from_config, policy_from_json_file
from .coordinator import (
    DISPATCH_BATCH_LIMIT,
    DISPATCH_FLUSH,
    DISPATCH_TOKEN_LIMIT,
    DISPATCH_WAIT_LIMIT,
    BatchCoordinator,
)
from .errors import (
    BatchCoordinatorError,
    ClosedCoordinatorError,
    DuplicateRequestError,
    PolicyValidationError,
    RequestTooLongError,
    RequestValidationError,
    ResultCountMismatchError,
    UnknownPolicyError,
)
from .executors import EmbeddingExecutor, FakeEmbedExecutor
from .observability import BatchDispatchRecord, padding_ratio, summarize_records
from .types import (
    BatchPlan,
    BatchPolicy,
    BatchRequest,
    compatibility_key,
)
from .vllm_embed_adapter import (
    MODE_BASELINE,
    MODE_DYNAMIC,
    DynamicBatchEmbedder,
    EmbedRunResult,
    RequestTiming,
    VllmEmbedExecutor,
)

__all__ = [
    "BatchCoordinator",
    "BatchRequest",
    "BatchPolicy",
    "BatchPlan",
    "compatibility_key",
    "policy_from_config",
    "policy_from_json_file",
    "padding_ratio",
    "BatchDispatchRecord",
    "summarize_records",
    "EmbeddingExecutor",
    "FakeEmbedExecutor",
    "VllmEmbedExecutor",
    "DynamicBatchEmbedder",
    "EmbedRunResult",
    "RequestTiming",
    "MODE_BASELINE",
    "MODE_DYNAMIC",
    "DISPATCH_BATCH_LIMIT",
    "DISPATCH_TOKEN_LIMIT",
    "DISPATCH_WAIT_LIMIT",
    "DISPATCH_FLUSH",
    "BatchCoordinatorError",
    "PolicyValidationError",
    "RequestValidationError",
    "DuplicateRequestError",
    "UnknownPolicyError",
    "RequestTooLongError",
    "ClosedCoordinatorError",
    "ResultCountMismatchError",
]
