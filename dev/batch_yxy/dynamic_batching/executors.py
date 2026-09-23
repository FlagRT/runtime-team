"""执行适配层：假执行器（离线/测试）与执行器协议。"""

from __future__ import annotations

import hashlib
import time
from typing import Callable, Protocol, Sequence

import numpy as np


class EmbeddingExecutor(Protocol):
    """执行器协议：payload_ref 解释为文本；返回与输入等长、顺序一致的向量列表。"""

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]: ...


class FakeEmbedExecutor:
    """离线假执行器，不依赖设备。

    向量由文本哈希确定性生成（同文本同向量）；成本模型可模拟设备执行：
    cost = fixed_cost_ms + per_token_ms * tokens，其中 tokens 在
    padding_aware=True 时按 N*max(len) 计（定长 Padding 路径），否则按
    sum(len) 计。用于 T0–T3 功能测试与无设备时的可复现对照。
    """

    def __init__(
        self,
        *,
        dim: int = 1024,
        fixed_cost_ms: float = 0.0,
        per_token_ms: float = 0.0,
        padding_aware: bool = True,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.dim = dim
        self.fixed_cost_ms = fixed_cost_ms
        self.per_token_ms = per_token_ms
        self.padding_aware = padding_aware
        self._sleep = sleep
        self.call_count = 0

    def token_count(self, text: str) -> int:
        return len(text.split())

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        lengths = [len(t.split()) for t in texts]
        if self.fixed_cost_ms or self.per_token_ms:
            tokens = (
                len(texts) * max(lengths) if self.padding_aware else sum(lengths)
            )
            cost_ms = self.fixed_cost_ms + self.per_token_ms * tokens
            if cost_ms > 0:
                self._sleep(cost_ms / 1000.0)
        self.call_count += 1
        return [self.vector(t) for t in texts]

    def vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(
            hashlib.md5(text.encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        return vec
