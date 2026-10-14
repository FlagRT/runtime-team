"""SchedulerEngine：四层架构编排（调度核心 ← 模型画像 / 框架适配器 / 设备画像）。

分层与数据流：

    SchedRequest ─submit→ RequestQueue（兼容键 lane + 优先级取队）
        │ 取队 dequeue_all
        ▼
    BatchCoordinator（复用：分桶、批次预算、封口派发）──→ BatchPlan
        │ 每批一次
        ▼
    FrameworkAdapter（submit_batch / get_results / map_error）
        │ 结果与 plan.request_ids 位置对齐
        ▼
    ResultStore（declare → commit 恰好一次，按 request_id 回填）

控制边界（与 dynamic_batching.vllm_embed_adapter 一致）：本引擎的"组批"
指外层请求分组（每个 BatchPlan 对应框架适配层一次调用）；推理引擎内部
（如 vLLM V1 的连续组批/抢占）自行形成的实际执行批次不由本层控制或宣称。

路径开关：
- PATH_LEGACY（原路径）：单次全量直通框架适配器，行为等价于直接调用
  executor 全量执行，作为对照与回退开关；
- PATH_SCHEDULED（新调度路径）：队列 → 优先级取队 → 分桶组批 → 执行 →
  按 ID 回填。

设备画像在策略生效前做预算钳制（min(策略, 设备)，冲突显式失败）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from dynamic_batching.coordinator import BatchCoordinator
from dynamic_batching.errors import (
    RequestTooLongError,
    UnknownPolicyError,
)
from dynamic_batching.observability import BatchDispatchRecord
from dynamic_batching.types import (
    BatchPlan,
    BatchPolicy,
    BatchRequest,
    compatibility_key,
)

from ..devices.base import DeviceProfile, STATUS_VERIFIED
from ..errors import (
    CostExtractorUnavailableError,
    DeviceBudgetConflictError,
    DeviceNotVerifiedError,
    EngineStateError,
    FrameworkExecutionError,
    ModelNotVerifiedError,
    SchedEngineError,
)
from ..frameworks.base import FrameworkAdapter
from ..models.base import ModelProfile
from .queue import QueueLaneStatus, RequestQueue
from .request import SchedRequest
from .results import ResultStore

PATH_LEGACY = "legacy"
PATH_SCHEDULED = "scheduled"
_PATH_MODES = (PATH_LEGACY, PATH_SCHEDULED)


@dataclass
class EngineRunReport:
    """一次 run 的报告：5 个环节的观测数据。"""

    mode: str
    results: ResultStore
    dequeue_trace: list[str] = field(default_factory=list)
    dispatch_records: list[BatchDispatchRecord] = field(default_factory=list)
    batch_exec_ms: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    wall_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


def clamp_policy(policy: BatchPolicy, device: DeviceProfile) -> BatchPolicy:
    """设备能力钳制：max_batch_size / max_total_tokens 取 min(策略, 设备)。

    设备字段为 None 表示该项不设芯片侧上限。钳制后仍须满足
    max_total_tokens >= 最大桶上界（BatchPolicy 不变式），否则显式报错
    （不允许静默降低桶上界放水）。
    """
    batch_size = (
        policy.max_batch_size
        if device.max_batch_size is None
        else min(policy.max_batch_size, device.max_batch_size)
    )
    total_tokens = (
        policy.max_total_tokens
        if device.max_total_tokens is None
        else min(policy.max_total_tokens, device.max_total_tokens)
    )
    if total_tokens < policy.max_length:
        raise DeviceBudgetConflictError(
            f"设备 {device.device_id} 的 token 预算上限 {total_tokens} 低于策略 "
            f"{policy.policy_version} 最大桶上界 {policy.max_length}，拒绝运行"
        )
    if (
        batch_size == policy.max_batch_size
        and total_tokens == policy.max_total_tokens
    ):
        return policy
    return BatchPolicy(
        model_id=policy.model_id,
        policy_version=policy.policy_version,
        bucket_upper_bounds=policy.bucket_upper_bounds,
        max_batch_size=batch_size,
        max_total_tokens=total_tokens,
        max_wait_ms=policy.max_wait_ms,
        allow_adjacent_bucket_merge=policy.allow_adjacent_bucket_merge,
        max_padding_ratio=policy.max_padding_ratio,
    )


class SchedulerEngine:
    """单次会话引擎：submit* 入队 → run() 执行 → EngineRunReport。

    单次会话语义：run() 只能调用一次（结果存储与队列已排空），保证报告
    可重复、无残留状态；重复运行请新建引擎。
    """

    def __init__(
        self,
        *,
        model_profiles: Mapping[str, ModelProfile],
        framework_adapter: FrameworkAdapter,
        device_profile: DeviceProfile,
        policies: BatchPolicy
        | Mapping[str, BatchPolicy]
        | Iterable[BatchPolicy]
        | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        path_mode: str = PATH_SCHEDULED,
    ) -> None:
        if device_profile.status != STATUS_VERIFIED:
            raise DeviceNotVerifiedError(
                f"设备 {device_profile.device_id} 画像状态为 {device_profile.status}，"
                "仅 verified 设备可运行（待适配芯片不得因有接口而标记支持）"
            )
        if path_mode not in _PATH_MODES:
            raise SchedEngineError(f"未知路径模式: {path_mode}（可选 {_PATH_MODES}）")
        if policies is None:
            self._policies: dict[str, BatchPolicy] = {}
        elif isinstance(policies, BatchPolicy):
            self._policies = {policies.model_id: clamp_policy(policies, device_profile)}
        elif isinstance(policies, Mapping):
            self._policies = {
                mid: clamp_policy(p, device_profile)
                for mid, p in policies.items()
            }
        else:
            self._policies = {
                p.model_id: clamp_policy(p, device_profile) for p in policies
            }
        self._profiles = dict(model_profiles)
        self._adapter = framework_adapter
        self._device = device_profile
        self._clock = clock
        self._sleep = sleep
        self.path_mode = path_mode
        self._queue = RequestQueue()
        self._batch_requests: dict[str, BatchRequest] = {}
        self._submit_order: list[str] = []
        self._stage_counts = {
            "enqueued": 0,
            "dequeued": 0,
            "bucketed": 0,
            "batched": 0,
            "executed": 0,
            "completed": 0,
        }
        self._ran = False

    @property
    def device(self) -> DeviceProfile:
        return self._device

    @property
    def framework_name(self) -> str:
        return self._adapter.name

    def register_policy(self, policy: BatchPolicy) -> None:
        """注册/替换模型策略（立即做设备钳制）。"""
        self._policies[policy.model_id] = clamp_policy(policy, self._device)

    # ------------------------------------------------------------------
    # 环节 1：入队（模型画像 → 兼容键 + 成本计量）
    # ------------------------------------------------------------------
    def submit(
        self,
        request: SchedRequest,
        *,
        model_version: str | None = None,
        execution_profile: str | None = None,
    ) -> str:
        """入队并返回兼容键。画像未验证/无策略/超长/无成本计量器均显式报错。"""
        if self._ran:
            raise EngineStateError("engine 已 run，单次会话不可重复入队")
        profile = self._profiles.get(request.model_id)
        if profile is None:
            raise UnknownPolicyError(f"模型未注册画像: {request.model_id}")
        if profile.status != "verified":
            raise ModelNotVerifiedError(
                f"模型 {request.model_id} 画像状态为 {profile.status}，"
                "仅 verified 模型可入队"
            )
        policy = self._policies.get(request.model_id)
        if policy is None:
            raise UnknownPolicyError(f"模型未注册组批策略: {request.model_id}")
        cost = self._extract_cost(request, profile)
        if cost > policy.max_length:
            raise RequestTooLongError(
                f"[{request.request_id}] 成本计量 {cost} 超出策略 "
                f"{policy.policy_version} 最大已验证长度 {policy.max_length}"
            )
        batch_request = BatchRequest(
            request_id=request.request_id,
            model_id=request.model_id,
            model_version=model_version or profile.default_model_version,
            input_token_count=cost,
            payload_ref=request.payload,
            arrival_mono_ns=request.arrival_mono_ns,
            execution_profile=execution_profile,
            deadline_mono_ns=request.deadline_mono_ns,
            priority=request.priority,
        )
        compat = compatibility_key(batch_request)
        self._queue.enqueue(request, compat)
        self._batch_requests[request.request_id] = batch_request
        self._submit_order.append(request.request_id)
        self._stage_counts["enqueued"] += 1
        return compat

    def submit_many(
        self,
        requests: Sequence[SchedRequest],
        **kwargs: Any,
    ) -> list[str]:
        return [self.submit(r, **kwargs) for r in requests]

    def queue_status(self) -> dict[str, QueueLaneStatus]:
        """队列状态查询（各兼容键 lane 深度/优先级分布/队头/最早到达）。"""
        return self._queue.status()

    def _extract_cost(self, request: SchedRequest, profile: ModelProfile) -> int:
        """成本计量优先用框架适配层同源计数器（与执行入口同一分词/计量规则）。"""
        adapter_counter = getattr(self._adapter, "token_count", None)
        if callable(adapter_counter):
            return int(adapter_counter(request.payload))
        if profile.token_counter is not None:
            return int(profile.token_counter(request.payload))
        raise CostExtractorUnavailableError(
            f"模型 {request.model_id} 无可用成本计量器"
            f"（框架 {self._adapter.name} 无同源 token_count，画像也未绑定 token_counter）"
        )

    # ------------------------------------------------------------------
    # 环节 2–5：取队 → 分桶组批 → 执行 → 回填
    # ------------------------------------------------------------------
    def run(self, *, path_mode: str | None = None) -> EngineRunReport:
        mode = path_mode or self.path_mode
        if mode not in _PATH_MODES:
            raise SchedEngineError(f"未知路径模式: {mode}（可选 {_PATH_MODES}）")
        if self._ran:
            raise EngineStateError("engine 为单次会话设计，run 不可重复调用")
        if self._queue.depth == 0:
            raise EngineStateError("队列为空，拒绝 run")
        self._ran = True
        start = self._clock()

        ordered = self._queue.dequeue_all()  # 环节 2：取队（确定序）
        dequeue_trace = [r.request_id for _, r in ordered]
        self._stage_counts["dequeued"] = len(ordered)

        store = ResultStore()
        store.declare(self._submit_order)
        if mode == PATH_LEGACY:
            report = self._run_legacy(store, start)
        else:
            report = self._run_scheduled(ordered, store, start)
        report.dequeue_trace = dequeue_trace
        report.stage_counts = dict(self._stage_counts)
        store.assert_complete()  # 防漏处理：任何未回填请求显式报错
        report.wall_ms = (self._clock() - start) / 1e6
        return report

    # -- 原路径（对照/回退开关） --
    def _run_legacy(self, store: ResultStore, start: int) -> EngineRunReport:
        payloads = [self._batch_requests[rid].payload_ref for rid in self._submit_order]
        t0 = self._clock()
        handle = self._adapter.submit_batch(payloads)
        outputs = self._safe_get_results(handle)
        t1 = self._clock()
        store.commit("legacy-direct", self._submit_order, outputs)
        self._stage_counts["executed"] += 1
        self._stage_counts["completed"] += len(self._submit_order)
        report = EngineRunReport(mode=PATH_LEGACY, results=store)
        report.batch_exec_ms["legacy-direct"] = (t1 - t0) / 1e6
        report.notes.append(
            "原路径：单次全量直通框架适配器，未经队列/分桶/组批（A/B 对照基线）"
        )
        return report

    # -- 新调度路径 --
    def _run_scheduled(
        self,
        ordered: Sequence[tuple[str, SchedRequest]],
        store: ResultStore,
        start: int,
    ) -> EngineRunReport:
        coordinator = BatchCoordinator(
            list(self._policies.values()), clock=self._clock
        )
        for _compat, req in ordered:
            # 环节 3：分桶（按已钳制策略入桶；重复/超长在协调器侧兜底校验）
            coordinator.submit(self._batch_requests[req.request_id])
            self._stage_counts["bucketed"] += 1

        batch_exec_ms: dict[str, float] = {}
        while coordinator.pending_count > 0:
            now = self._clock()
            plans: list[BatchPlan] = coordinator.poll_ready(now)  # 环节 4：组批发批
            if not plans:
                wakeup = coordinator.next_wakeup_mono_ns()
                if wakeup is None:
                    break
                delay_s = max(0.0, (wakeup - self._clock()) / 1e9)
                if delay_s > 0:
                    self._sleep(delay_s)
                continue
            for plan in plans:
                self._stage_counts["batched"] += 1
                t0 = self._clock()
                handle = self._adapter.submit_batch(plan.payload_refs)
                outputs = self._safe_get_results(handle)
                t1 = self._clock()
                # 环节 5：按 request_id 恰好一次回填（数量/身份/重复校验）
                store.commit(plan.batch_id, plan.request_ids, outputs)
                self._stage_counts["executed"] += 1
                self._stage_counts["completed"] += len(plan.request_ids)
                batch_exec_ms[plan.batch_id] = (t1 - t0) / 1e6

        report = EngineRunReport(mode=PATH_SCHEDULED, results=store)
        report.dispatch_records = list(coordinator.stats)
        report.batch_exec_ms = batch_exec_ms
        report.notes.append(
            f"调度路径：device={self._device.device_id} "
            f"framework={self._adapter.name} policies="
            f"{sorted(self._policies)}"
        )
        return report

    def _safe_get_results(self, handle: Any) -> list:
        try:
            return self._adapter.get_results(handle)
        except SchedEngineError:
            raise
        except BaseException as exc:  # noqa: BLE001 - 统一映射后重抛
            raise self._adapter.map_error(exc) from exc
