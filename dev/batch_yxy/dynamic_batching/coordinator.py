"""BatchCoordinator：动态组批与长度感知分桶决策核心（与 vLLM 解耦）。

决策规则（对齐实现文档第 4 节）：
- 兼容键（模型ID|模型版本|执行profile）是硬约束，长度桶是优化分组；
- 组批以兼容键+策略版本分组，桶内以最早到达请求为种子贪心装入，
  同时候选扫描有界（scan_window）；
- 硬预算：batch_size <= max_batch_size 且 sum(tokens) <= max_total_tokens
  且单请求长度 <= 策略最大已验证长度（submit 时校验）；
- 大请求放不进当前批次时跳过并保留给下一批次，不阻塞其他可入批请求；
- 封口（派发）触发：batch_limit（批量打满）/ token_limit（token 预算打满）
  立即派发；wait_limit（组内最早请求达到 max_wait_ms）；flush（显式排空）。
  执行资源触发由调用方自行调用 flush/poll_ready 表达；
- 相邻桶合并仅在 allow_adjacent_bucket_merge 开启时发生，且仍受全部硬预算
  与 max_padding_ratio 约束，跨桶信息记录在 bucket_ids 与统计中。

本类不做线程同步，调用方需串行调用（事件循环/单线程）。
"""

from __future__ import annotations

import bisect
import itertools
import logging
import time
from collections import deque
from typing import Callable, Iterable, Mapping

from .errors import (
    ClosedCoordinatorError,
    DuplicateRequestError,
    RequestTooLongError,
    RequestValidationError,
    UnknownPolicyError,
)
from .observability import BatchDispatchRecord, padding_ratio
from .types import BatchPlan, BatchPolicy, BatchRequest, compatibility_key

logger = logging.getLogger(__name__)

DISPATCH_BATCH_LIMIT = "batch_limit"
DISPATCH_TOKEN_LIMIT = "token_limit"
DISPATCH_WAIT_LIMIT = "wait_limit"
DISPATCH_FLUSH = "flush"

_MS_TO_NS = 1_000_000


class _Entry:
    __slots__ = ("request", "policy", "bucket")

    def __init__(self, request: BatchRequest, policy: BatchPolicy, bucket: int):
        self.request = request
        self.policy = policy
        self.bucket = bucket


class _Group:
    __slots__ = ("compat", "policy", "buckets")

    def __init__(self, compat: str, policy: BatchPolicy):
        self.compat = compat
        self.policy = policy
        self.buckets: dict[int, list[_Entry]] = {}

    def push(self, entry: _Entry) -> None:
        queue = self.buckets.setdefault(entry.bucket, [])
        bisect.insort(
            queue, entry, key=lambda e: e.request.arrival_mono_ns
        )


class _BatchDraft:
    __slots__ = ("selected", "total", "max_len", "buckets_used")

    def __init__(self) -> None:
        self.selected: list[_Entry] = []
        self.total = 0
        self.max_len = 0
        self.buckets_used: set[int] = set()


class BatchCoordinator:
    """可注入单调时钟的组批协调器。

    clock 仅用于 flush/close 取当前时间；poll_ready 的时间由参数显式传入，
    便于测试注入假时钟。strict_duplicate_detection=True 时，已派发历史中的
    request_id 再次提交也会报重复（内存随之增长；长期运行可关闭后自行保证
    ID 唯一性）。
    """

    def __init__(
        self,
        policies: Mapping[str, BatchPolicy] | Iterable[BatchPolicy] | None = None,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        scan_window: int = 256,
        stats_maxlen: int = 8192,
        strict_duplicate_detection: bool = True,
    ) -> None:
        if policies is None:
            self._policies: dict[str, BatchPolicy] = {}
        elif isinstance(policies, BatchPolicy):
            self._policies = {policies.model_id: policies}
        elif isinstance(policies, Mapping):
            self._policies = dict(policies)
        else:
            self._policies = {p.model_id: p for p in policies}
        self._clock = clock
        self._scan_window = max(1, int(scan_window))
        self._strict_duplicate_detection = strict_duplicate_detection
        self.stats: deque[BatchDispatchRecord] = deque(maxlen=stats_maxlen)
        self._groups: dict[tuple[str, str], _Group] = {}
        self._queued_ids: set[str] = set()
        self._seen_ids: set[str] = set()
        self._batch_seq = itertools.count(1)
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def pending_count(self) -> int:
        return len(self._queued_ids)

    def get_policy(self, model_id: str) -> BatchPolicy | None:
        return self._policies.get(model_id)

    def register_policy(self, policy: BatchPolicy) -> None:
        """注册/替换模型策略。已排队请求保留提交时的策略快照，不受影响。"""
        self._policies[policy.model_id] = policy

    def submit(self, request: BatchRequest) -> None:
        """提交请求。非法输入（重复 ID/空输入/超长/未知模型等）显式报错。"""
        if self._closed:
            raise ClosedCoordinatorError("coordinator 已 close，拒绝 submit")
        if not isinstance(request, BatchRequest):
            raise RequestValidationError(f"期望 BatchRequest，得到 {type(request)!r}")
        rid = request.request_id
        if not isinstance(rid, str) or not rid:
            raise RequestValidationError("request_id 必须为非空字符串")
        if rid in self._queued_ids or (
            self._strict_duplicate_detection and rid in self._seen_ids
        ):
            raise DuplicateRequestError(f"request_id 重复: {rid}")
        if not isinstance(request.model_version, str) or not request.model_version:
            raise RequestValidationError(f"[{rid}] model_version 必须为非空字符串")
        tokens = request.input_token_count
        if (
            not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or tokens < 1
        ):
            raise RequestValidationError(
                f"[{rid}] input_token_count 必须为正整数（空输入请在入口拒绝）: {tokens}"
            )
        if (
            not isinstance(request.arrival_mono_ns, int)
            or request.arrival_mono_ns < 0
        ):
            raise RequestValidationError(f"[{rid}] arrival_mono_ns 必须为非负整数")
        policy = self._policies.get(request.model_id)
        if policy is None:
            raise UnknownPolicyError(f"模型未注册策略: {request.model_id}")
        bucket = policy.bucket_index(tokens)
        if bucket is None:
            raise RequestTooLongError(
                f"[{rid}] input_token_count={tokens} 超出策略 {policy.policy_version} "
                f"最大已验证长度 {policy.max_length}，明确拒绝"
            )

        compat = compatibility_key(request)
        key = (compat, policy.policy_version)
        group = self._groups.get(key)
        if group is not None and group.policy != policy:
            n = 2
            while (compat, f"{policy.policy_version}#{n}") in self._groups:
                n += 1
            key = (compat, f"{policy.policy_version}#{n}")
            group = None
        if group is None:
            group = _Group(compat, policy)
            self._groups[key] = group
        group.push(_Entry(request, policy, bucket))
        self._queued_ids.add(rid)
        self._seen_ids.add(rid)

    def poll_ready(self, now_mono_ns: int) -> list[BatchPlan]:
        """检查所有分组，返回当前应派发的批次（并从候选队列移除）。

        可在新请求到达、执行资源可用或定时器到期时调用；相同状态下重复
        调用不会重复发出请求（已派发条目即被移除）。
        """
        plans: list[BatchPlan] = []
        for key in list(self._groups.keys()):
            while True:
                group = self._groups.get(key)
                if group is None:
                    break
                plan = self._try_dispatch(group, key, now_mono_ns)
                if plan is None:
                    break
                plans.append(plan)
        return plans

    def next_wakeup_mono_ns(self) -> int | None:
        """所有待排队列中最早请求的 wait_limit 到期时间；无排队返回 None。

        max_wait_ms 是合批等待上限而非固定睡眠时间：打满预算的批次在
        poll_ready 时立即派发，无需等到该时间点。
        """
        wakeup = None
        for group in self._groups.values():
            for queue in group.buckets.values():
                if queue:
                    candidate = (
                        queue[0].request.arrival_mono_ns
                        + group.policy.wait_window_ns
                    )
                    if wakeup is None or candidate < wakeup:
                        wakeup = candidate
        return wakeup

    def flush(self) -> list[BatchPlan]:
        """排空所有候选（进程关闭/测试用）：未满批也发出，仍受硬预算约束。"""
        now = self._clock()
        plans: list[BatchPlan] = []
        for key in list(self._groups.keys()):
            group = self._groups.get(key)
            if group is None:
                continue
            policy = group.policy
            for bucket in sorted(group.buckets):
                while group.buckets.get(bucket):
                    draft = self._fill(group, policy, bucket)
                    if not draft.selected:
                        break
                    plans.append(self._emit(group, key, policy, draft, DISPATCH_FLUSH, now))
        return plans

    def close(self) -> list[BatchPlan]:
        """关闭协调器：先 flush 排空剩余请求（reason=flush），再拒绝后续 submit。

        幂等：重复 close 返回空列表；close 后 flush/poll_ready 均返回空。
        """
        if self._closed:
            return []
        self._closed = True
        return self.flush()

    def _try_dispatch(
        self, group: _Group, key: tuple[str, str], now_mono_ns: int
    ) -> BatchPlan | None:
        policy = group.policy
        active = {b: q for b, q in group.buckets.items() if q}
        if not active:
            del self._groups[key]
            return None
        ordered = sorted(
            active, key=lambda b: active[b][0].request.arrival_mono_ns
        )
        for bucket in ordered:
            draft = self._fill(group, policy, bucket)
            if len(draft.selected) >= policy.max_batch_size:
                return self._emit(group, key, policy, draft, DISPATCH_BATCH_LIMIT, now_mono_ns)
            if draft.total >= policy.max_total_tokens:
                return self._emit(group, key, policy, draft, DISPATCH_TOKEN_LIMIT, now_mono_ns)
        head_bucket = ordered[0]
        head = active[head_bucket][0]
        if now_mono_ns - head.request.arrival_mono_ns >= policy.wait_window_ns:
            draft = self._fill(group, policy, head_bucket)
            return self._emit(group, key, policy, draft, DISPATCH_WAIT_LIMIT, now_mono_ns)
        return None

    def _fill(
        self, group: _Group, policy: BatchPolicy, seed_bucket: int
    ) -> _BatchDraft:
        draft = _BatchDraft()
        scan_order = [seed_bucket]
        if policy.allow_adjacent_bucket_merge:
            scan_order.append(seed_bucket + 1)
            scan_order.append(seed_bucket - 1)
        for bucket in scan_order:
            queue = group.buckets.get(bucket)
            if not queue:
                continue
            scanned = 0
            for entry in queue:
                if len(draft.selected) >= policy.max_batch_size:
                    break
                if scanned >= self._scan_window:
                    break
                scanned += 1
                self._try_add(draft, policy, entry, bucket)
            if len(draft.selected) >= policy.max_batch_size:
                break
        return draft

    @staticmethod
    def _try_add(
        draft: _BatchDraft, policy: BatchPolicy, entry: _Entry, bucket: int
    ) -> bool:
        tokens = entry.request.input_token_count
        size = len(draft.selected)
        new_total = draft.total + tokens
        if size >= policy.max_batch_size or new_total > policy.max_total_tokens:
            return False
        new_max = max(draft.max_len, tokens)
        if policy.max_padding_ratio is not None:
            denom = new_max * (size + 1)
            if denom > 0 and (denom - new_total) / denom > policy.max_padding_ratio:
                return False
        draft.selected.append(entry)
        draft.total = new_total
        draft.max_len = new_max
        draft.buckets_used.add(bucket)
        return True

    def _emit(
        self,
        group: _Group,
        key: tuple[str, str],
        policy: BatchPolicy,
        draft: _BatchDraft,
        reason: str,
        now_mono_ns: int,
    ) -> BatchPlan:
        for entry in draft.selected:
            queue = group.buckets.get(entry.bucket)
            if queue is not None:
                try:
                    queue.remove(entry)
                except ValueError:
                    pass
                if not queue:
                    del group.buckets[entry.bucket]
            self._queued_ids.discard(entry.request.request_id)
        if not any(group.buckets.values()):
            self._groups.pop(key, None)

        requests = [e.request for e in draft.selected]
        lengths = tuple(r.input_token_count for r in requests)
        bucket_ids = tuple(sorted(draft.buckets_used))
        batch_id = f"batch-{next(self._batch_seq):06d}"
        plan = BatchPlan(
            batch_id=batch_id,
            policy_version=policy.policy_version,
            compatibility_key=group.compat,
            bucket_ids=bucket_ids,
            request_ids=tuple(r.request_id for r in requests),
            payload_refs=tuple(r.payload_ref for r in requests),
            token_lengths=lengths,
            total_tokens=draft.total,
            padded_length=draft.max_len,
            dispatch_reason=reason,
            created_mono_ns=now_mono_ns,
        )
        record = BatchDispatchRecord(
            batch_id=batch_id,
            policy_version=policy.policy_version,
            compatibility_key=group.compat,
            bucket_ids=bucket_ids,
            request_ids=plan.request_ids,
            token_lengths=lengths,
            batch_size=len(requests),
            total_tokens=draft.total,
            max_batch_size=policy.max_batch_size,
            max_total_tokens=policy.max_total_tokens,
            head_wait_ns=now_mono_ns - requests[0].arrival_mono_ns,
            dispatch_reason=reason,
            cross_bucket_merge=len(bucket_ids) > 1,
            padded_length=draft.max_len,
            padding_ratio=padding_ratio(lengths),
            created_mono_ns=now_mono_ns,
        )
        self.stats.append(record)
        logger.info(
            "dispatch %s reason=%s compat=%s buckets=%s size=%d tokens=%d/%d "
            "padded=%d padding_ratio=%.3f cross_bucket=%s",
            batch_id,
            reason,
            group.compat,
            bucket_ids,
            len(requests),
            draft.total,
            policy.max_total_tokens,
            draft.max_len,
            record.padding_ratio,
            record.cross_bucket_merge,
        )
        return plan
