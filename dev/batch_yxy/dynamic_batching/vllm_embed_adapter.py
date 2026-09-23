"""vLLM Qwen3-Embedding 最小接入适配（实现文档第 5 节路径 2）。

控制边界：BatchCoordinator 只做外层请求分组，每个 BatchPlan 对应一次
LLM.embed 调用；vLLM V1 内部仍会自行调度（连续组批/抢占），本模块不宣称
控制实际设备批次。结果映射依赖 LLM.embed "返回顺序与输入顺序一致" 的契约，
适配层再校验数量并按 request_id 逐项回填。

A/B 开关：DynamicBatchEmbedder(policy=None) 即原路径基线（单次 LLM.embed
全量直通，行为与直接调用 LLM.embed 等价），作为对照与回退方式。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .coordinator import BatchCoordinator
from .errors import ResultCountMismatchError
from .observability import BatchDispatchRecord
from .types import BatchPolicy, BatchRequest

MODE_BASELINE = "baseline"
MODE_DYNAMIC = "dynamic"


class VllmEmbedExecutor:
    """vLLM 离线 LLM.embed 执行适配层。payload_ref=str 文本。

    token_count 与 vLLM 入口使用同一 tokenizer（add_special_tokens 默认），
    并按 max_model_len 截断，避免为了分桶引入另一套分词规则。
    """

    def __init__(self, llm, *, token_count_max_length: int | None = None):
        self._llm = llm
        self._tokenizer = llm.get_tokenizer()
        if token_count_max_length is not None:
            self._max_length = int(token_count_max_length)
        else:
            self._max_length = (
                int(getattr(llm.model_config, "max_model_len", 0) or 0) or 32768
            )

    @property
    def max_length(self) -> int:
        return self._max_length

    def token_count(self, text: str) -> int:
        ids = self._tokenizer.encode(text)
        return min(len(ids), self._max_length)

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        outputs = self._llm.embed(list(texts), use_tqdm=False)
        vectors = [np.asarray(o.outputs.embedding, dtype=np.float32) for o in outputs]
        if len(vectors) != len(texts):
            raise ResultCountMismatchError("direct-call", len(texts), len(vectors))
        return vectors


@dataclass
class RequestTiming:
    request_id: str
    arrival_mono_ns: int
    dispatched_mono_ns: int
    completed_mono_ns: int

    @property
    def e2e_ms(self) -> float:
        return (self.completed_mono_ns - self.arrival_mono_ns) / 1e6

    @property
    def queue_ms(self) -> float:
        return (self.dispatched_mono_ns - self.arrival_mono_ns) / 1e6

    @property
    def exec_ms(self) -> float:
        return (self.completed_mono_ns - self.dispatched_mono_ns) / 1e6


@dataclass
class EmbedRunResult:
    mode: str
    vectors: dict[str, np.ndarray]
    timings: dict[str, RequestTiming] = field(default_factory=dict)
    dispatch_records: list[BatchDispatchRecord] = field(default_factory=list)
    batch_exec_ms: dict[str, float] = field(default_factory=dict)
    wall_ms: float = 0.0


class DynamicBatchEmbedder:
    """A/B 开关封装。

    baseline：单次 executor.embed(全部文本) 直通（原路径基线）。
    dynamic：提交给 BatchCoordinator，按 BatchPlan 逐批执行并按 ID 回填向量。
    """

    def __init__(
        self,
        executor,
        policy: BatchPolicy | None = None,
        *,
        model_id: str | None = None,
        model_version: str = "default",
        execution_profile: str | None = None,
        coordinator: BatchCoordinator | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if policy is None and coordinator is None:
            self.mode = MODE_BASELINE
            self._coordinator: BatchCoordinator | None = None
        else:
            self.mode = MODE_DYNAMIC
            self._coordinator = coordinator or BatchCoordinator([policy])
        self._executor = executor
        if model_id is not None:
            self._model_id = model_id
        elif policy is not None:
            self._model_id = policy.model_id
        else:
            self._model_id = "default-model"
        self._model_version = model_version
        self._execution_profile = execution_profile
        self._clock = clock
        self._sleep = sleep

    def _token_count(self, text: str) -> int:
        counter = getattr(self._executor, "token_count", None)
        if counter is not None:
            return int(counter(text))
        return len(text.split())

    def run(
        self,
        items: Sequence[tuple[str, str]],
        *,
        arrival_mono_ns: Sequence[int] | None = None,
    ) -> EmbedRunResult:
        """执行 (request_id, text) 序列，返回按 ID 映射的向量与时延数据。"""
        if arrival_mono_ns is not None and len(arrival_mono_ns) != len(items):
            raise ValueError("arrival_mono_ns 长度必须与 items 一致")
        start = self._clock()
        if self.mode == MODE_BASELINE:
            return self._run_baseline(items, start)
        return self._run_dynamic(items, arrival_mono_ns, start)

    def _run_baseline(
        self, items: Sequence[tuple[str, str]], start: int
    ) -> EmbedRunResult:
        texts = [text for _, text in items]
        t0 = self._clock()
        vectors = self._executor.embed(texts)
        t1 = self._clock()
        if len(vectors) != len(items):
            raise ResultCountMismatchError("baseline", len(items), len(vectors))
        result = EmbedRunResult(mode=MODE_BASELINE, vectors={})
        for (rid, _), vec in zip(items, vectors):
            result.vectors[rid] = vec
            result.timings[rid] = RequestTiming(rid, start, t0, t1)
        result.wall_ms = (self._clock() - start) / 1e6
        return result

    def _run_dynamic(
        self,
        items: Sequence[tuple[str, str]],
        arrival_mono_ns: Sequence[int] | None,
        start: int,
    ) -> EmbedRunResult:
        coordinator = self._coordinator
        assert coordinator is not None
        now = start
        arrivals: dict[str, int] = {}
        for i, (rid, text) in enumerate(items):
            arrival = arrival_mono_ns[i] if arrival_mono_ns is not None else now
            arrivals[rid] = arrival
            coordinator.submit(
                BatchRequest(
                    request_id=rid,
                    model_id=self._model_id,
                    model_version=self._model_version,
                    input_token_count=self._token_count(text),
                    payload_ref=text,
                    arrival_mono_ns=arrival,
                    execution_profile=self._execution_profile,
                )
            )
        result = EmbedRunResult(mode=MODE_DYNAMIC, vectors={})
        while coordinator.pending_count > 0:
            now = self._clock()
            plans = coordinator.poll_ready(now)
            if not plans:
                wakeup = coordinator.next_wakeup_mono_ns()
                if wakeup is None:
                    break
                delay_s = max(0.0, (wakeup - self._clock()) / 1e9)
                if delay_s > 0:
                    self._sleep(delay_s)
                continue
            for plan in plans:
                t0 = self._clock()
                outputs = self._executor.embed(plan.payload_refs)
                t1 = self._clock()
                if len(outputs) != len(plan.request_ids):
                    raise ResultCountMismatchError(
                        plan.batch_id, len(plan.request_ids), len(outputs)
                    )
                for rid, vec in zip(plan.request_ids, outputs):
                    result.vectors[rid] = vec
                result.batch_exec_ms[plan.batch_id] = (t1 - t0) / 1e6
                for rid in plan.request_ids:
                    result.timings[rid] = RequestTiming(
                        rid, arrivals[rid], plan.created_mono_ns, t1
                    )
        missing = [rid for rid, _ in items if rid not in result.vectors]
        if missing:
            raise RuntimeError(f"内部错误：以下请求未获得向量: {missing[:5]}")
        result.dispatch_records = list(coordinator.stats)
        result.wall_ms = (self._clock() - start) / 1e6
        return result
