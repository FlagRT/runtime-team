"""vLLM Embedding 框架适配器：包装既有 EmbeddingExecutor 协议实现。

复用 dynamic_batching.vllm_embed_adapter.VllmEmbedExecutor（NPU 真路径）与
executors.FakeEmbedExecutor（离线测试路径），不重复实现执行逻辑。

控制边界：本适配层只承接外层组批（每个 BatchHandle 对应一次 LLM.embed）；
vLLM V1 引擎内部仍自行调度（连续组批/抢占），内部实际执行批次不由本层
控制或宣称。token_count 透传执行器的同源计数器（与 LLM.embed 入口同一
tokenizer，按 max_model_len 截断）。
"""

from __future__ import annotations

import itertools
from typing import Any, Sequence

from dynamic_batching.errors import ResultCountMismatchError

from ..errors import CostExtractorUnavailableError, FrameworkExecutionError
from .base import STATUS_VERIFIED, BatchHandle


class VllmEmbedFrameworkAdapter:
    """适配 EmbeddingExecutor 协议（embed(texts) -> list[vector]）。

    executor 可为 VllmEmbedExecutor（vLLM LLM.embed）或任何满足协议的
    执行器（如 FakeEmbedExecutor）；同源 token_count 透传（若存在）。
    """

    name = "vllm-embed"
    status = STATUS_VERIFIED

    def __init__(self, executor: Any):
        self._executor = executor
        self._handle_seq = itertools.count(1)
        self.call_count = 0

    @property
    def executor(self) -> Any:
        return self._executor

    def token_count(self, payload: Any) -> int:
        """同源成本计量：与执行入口使用同一 tokenizer/计量规则。"""
        counter = getattr(self._executor, "token_count", None)
        if not callable(counter):
            raise CostExtractorUnavailableError(
                f"executor {type(self._executor).__name__} 无同源 token_count"
            )
        return int(counter(payload))

    def submit_batch(self, payloads: Sequence[Any]) -> BatchHandle:
        return BatchHandle(
            batch_id=f"fw-{next(self._handle_seq):06d}",
            payloads=tuple(payloads),
        )

    def get_results(self, handle: BatchHandle) -> list[Any]:
        self.call_count += 1
        outputs = self._executor.embed(list(handle.payloads))
        if len(outputs) != len(handle.payloads):
            raise ResultCountMismatchError(
                handle.batch_id, len(handle.payloads), len(outputs)
            )
        return outputs

    def map_error(self, exc: BaseException) -> FrameworkExecutionError:
        return FrameworkExecutionError(self.name, exc)
