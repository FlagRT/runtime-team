"""推理框架适配层统一契约：提交批次、获取结果、错误映射。

执行语义契约（与 dynamic_batching 结果映射契约一致）：
- submit_batch(payloads) 提交一个批次的执行输入；
- get_results(handle) 返回与提交 payloads 等长且顺序一致的结果列表；
- map_error(exc) 将框架原生异常映射为引擎统一错误。

扩展契约（仅定义，不创建未经验证的实现）：
- 异步化：submit_batch 立即返回句柄、get_results 轮询/阻塞到完成；
- 流式/事件回调、按请求粒度的错误码；
- 其他推理框架（SGLang / TensorRT-LLM / MindIE / 自研运行时）接入时：
  实现本协议 + 提供 token_count 同源计数器（若成本计量为 token），
  在真实芯片上完成链路验证后方可标记 verified 并登记 VERIFIED_FRAMEWORKS。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..errors import FrameworkExecutionError

STATUS_VERIFIED = "verified"
STATUS_DECLARED = "declared"

VERIFIED_FRAMEWORKS: tuple[str, ...] = ("vllm-embed",)
DECLARED_FRAMEWORKS: tuple[str, ...] = ("sglang", "tensorrt-llm", "mindie")


@dataclass(frozen=True)
class BatchHandle:
    """批次句柄：适配层私有解释执行输入，引擎不感知框架细节。"""

    batch_id: str
    payloads: tuple[Any, ...]


class FrameworkAdapter(Protocol):
    """统一框架适配协议。token_count 为可选同源计数器（成本计量用）。"""

    name: str
    status: str

    def submit_batch(self, payloads: Sequence[Any]) -> BatchHandle: ...

    def get_results(self, handle: BatchHandle) -> list[Any]: ...

    def map_error(self, exc: BaseException) -> FrameworkExecutionError: ...


__all__ = [
    "BatchHandle",
    "FrameworkAdapter",
    "FrameworkExecutionError",
    "STATUS_VERIFIED",
    "STATUS_DECLARED",
    "VERIFIED_FRAMEWORKS",
    "DECLARED_FRAMEWORKS",
]
