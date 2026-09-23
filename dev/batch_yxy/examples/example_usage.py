"""外部调用示例：组批核心 + 执行适配层（假执行器演示，无设备依赖）。

运行：.venv/bin/python examples/example_usage.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamic_batching import (
    BatchCoordinator,
    BatchPolicy,
    BatchRequest,
    DynamicBatchEmbedder,
    FakeEmbedExecutor,
    summarize_records,
)

POLICY = BatchPolicy(
    model_id="Qwen3-Embedding-0.6B",
    policy_version="v1-demo",
    bucket_upper_bounds=(64, 256, 1024),
    max_batch_size=4,
    max_total_tokens=2048,
    max_wait_ms=10.0,
    allow_adjacent_bucket_merge=False,
)

TEXTS = [
    ("r-0001", "退款流程 " * 5),
    ("r-0002", "天气 " * 12),
    ("r-0003", "长文本 " * 120),
    ("r-0004", "短句"),
    ("r-0005", "中等长度 " * 30),
]


def demo_coordinator_direct() -> None:
    print("== 直接使用 BatchCoordinator ==")
    now = time.monotonic_ns()
    co = BatchCoordinator(POLICY)
    for i, (rid, text) in enumerate(TEXTS):
        co.submit(
            BatchRequest(
                request_id=rid,
                model_id=POLICY.model_id,
                model_version="0.6b",
                input_token_count=len(text.split()),
                payload_ref=text,
                arrival_mono_ns=now + i * 1_000_000,
                execution_profile="demo",
            )
        )
    while co.pending_count:
        plans = co.poll_ready(time.monotonic_ns())
        for plan in plans:
            print(
                f"{plan.batch_id} reason={plan.dispatch_reason} "
                f"buckets={plan.bucket_ids} requests={plan.request_ids} "
                f"tokens={plan.token_lengths} total={plan.total_tokens} "
                f"padded={plan.padded_length}"
            )
        if not plans:
            plans = co.flush()
            for plan in plans:
                print(
                    f"{plan.batch_id} reason={plan.dispatch_reason} "
                    f"buckets={plan.bucket_ids} requests={plan.request_ids}"
                )
    print("汇总:", summarize_records(co.stats))


def demo_embedder_ab() -> None:
    print("\n== DynamicBatchEmbedder A/B 开关（假执行器） ==")
    executor = FakeEmbedExecutor(dim=1024)
    baseline = DynamicBatchEmbedder(executor).run(TEXTS)
    dynamic = DynamicBatchEmbedder(executor, POLICY, model_version="0.6b").run(TEXTS)
    print(f"baseline 向量数={len(baseline.vectors)}  wall={baseline.wall_ms:.1f}ms")
    print(f"dynamic  向量数={len(dynamic.vectors)}  wall={dynamic.wall_ms:.1f}ms")
    same = all(
        (baseline.vectors[rid] == dynamic.vectors[rid]).all() for rid, _ in TEXTS
    )
    print(f"两种模式向量逐元素一致: {same}")


if __name__ == "__main__":
    demo_coordinator_direct()
    demo_embedder_ab()
