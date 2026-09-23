"""可观测数据：Padding 比例、每批派发记录与汇总。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


def padding_ratio(token_lengths: Sequence[int]) -> float:
    """(N*max - sum) / max(N*max, 1)。

    批次长度差异指标；仅在执行端确实按最大长度定长 Padding 时代表实际
    Padding 开销。
    """
    lengths = list(token_lengths)
    if not lengths:
        return 0.0
    denom = len(lengths) * max(lengths)
    return (denom - sum(lengths)) / max(denom, 1)


@dataclass(frozen=True)
class BatchDispatchRecord:
    """每批派发记录（第 4.3 节最小可观测集）。"""

    batch_id: str
    policy_version: str
    compatibility_key: str
    bucket_ids: tuple[int, ...]
    request_ids: tuple[str, ...]
    token_lengths: tuple[int, ...]
    batch_size: int
    total_tokens: int
    max_batch_size: int
    max_total_tokens: int
    head_wait_ns: int
    dispatch_reason: str
    cross_bucket_merge: bool
    padded_length: int
    padding_ratio: float
    created_mono_ns: int


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, int(round(pct / 100.0 * (len(sorted_values) - 1)))))
    return float(sorted_values[idx])


def summarize_records(records: Iterable[BatchDispatchRecord]) -> dict:
    """批量分布/封口原因/Padding/等待时间的汇总，供对照测试报告使用。"""
    records = list(records)
    if not records:
        return {"num_batches": 0}
    sizes = sorted(r.batch_size for r in records)
    waits = sorted(r.head_wait_ns / 1e6 for r in records)
    ratios = sorted(r.padding_ratio for r in records)
    return {
        "num_batches": len(records),
        "num_requests": sum(sizes),
        "batch_size": {
            "min": sizes[0],
            "p50": _percentile(sizes, 50),
            "p95": _percentile(sizes, 95),
            "max": sizes[-1],
            "distribution": dict(Counter(sizes)),
        },
        "dispatch_reasons": dict(Counter(r.dispatch_reason for r in records)),
        "cross_bucket_batches": sum(1 for r in records if r.cross_bucket_merge),
        "head_wait_ms": {
            "p50": _percentile(waits, 50),
            "p95": _percentile(waits, 95),
            "max": waits[-1],
        },
        "padding_ratio": {
            "avg": sum(ratios) / len(ratios),
            "p50": _percentile(ratios, 50),
            "p95": _percentile(ratios, 95),
            "max": ratios[-1],
        },
    }
