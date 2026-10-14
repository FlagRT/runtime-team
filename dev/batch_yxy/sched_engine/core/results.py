"""结果回填存储：按 request_id 恰好一次回填，防漏处理、防重复返回、防错位。

错位防护：commit 以 plan.request_ids 与 outputs 的位置对齐契约为基础
（框架适配层保证返回顺序与提交顺序一致），并在回填前校验数量一致、
身份已申报、未重复回填。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from dynamic_batching.errors import ResultCountMismatchError

from ..errors import (
    DuplicateResultError,
    MissingResultError,
    UnexpectedResultError,
)


class ResultStore:
    """单次会话的结果存储：先 declare 全部预期 ID，再按批次 commit。"""

    def __init__(self) -> None:
        self._expected: set[str] = set()
        self._results: dict[str, Any] = {}
        self._batch_of: dict[str, str] = {}

    @property
    def expected_count(self) -> int:
        return len(self._expected)

    @property
    def filled_count(self) -> int:
        return len(self._results)

    def declare(self, request_ids: Iterable[str]) -> None:
        for rid in request_ids:
            if not isinstance(rid, str) or not rid:
                raise UnexpectedResultError(f"非法 request_id: {rid!r}")
            if rid in self._expected:
                raise DuplicateResultError(f"request_id 重复申报: {rid}")
            self._expected.add(rid)

    def commit(
        self, batch_id: str, request_ids: Sequence[str], outputs: Sequence[Any]
    ) -> None:
        """按批次回填。任何不一致显式报错，不部分写入。"""
        if len(outputs) != len(request_ids):
            raise ResultCountMismatchError(batch_id, len(request_ids), len(outputs))
        for rid in request_ids:
            if rid not in self._expected:
                raise UnexpectedResultError(f"[{batch_id}] 未申报的请求结果: {rid}")
            if rid in self._results:
                raise DuplicateResultError(f"[{batch_id}] 请求重复回填: {rid}")
        for rid, out in zip(request_ids, outputs):
            self._results[rid] = out
            self._batch_of[rid] = batch_id

    def get(self, request_id: str) -> Any:
        if request_id not in self._expected:
            raise UnexpectedResultError(f"未申报的请求: {request_id}")
        if request_id not in self._results:
            raise MissingResultError(f"请求尚未回填: {request_id}")
        return self._results[request_id]

    def try_get(self, request_id: str) -> Any:
        return self._results.get(request_id)

    def missing(self) -> list[str]:
        return sorted(self._expected - self._results.keys())

    def assert_complete(self) -> None:
        missing = self.missing()
        if missing:
            raise MissingResultError(
                f"运行结束后仍有 {len(missing)} 个请求未回填: {missing[:5]}..."
            )

    def batch_of(self, request_id: str) -> str:
        if request_id not in self._batch_of:
            raise UnexpectedResultError(f"未回填的请求: {request_id}")
        return self._batch_of[request_id]

    def items(self) -> dict[str, Any]:
        return dict(self._results)
