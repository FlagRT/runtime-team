"""调度引擎离线演示：可控时钟 + 模拟执行器（无设备依赖，可重复运行）。

场景与本轮验收对齐：32 条不同 Token 长度请求 × 2 档优先级 × 3 种到达
顺序，覆盖入队/取队/分桶/组批/结果回填 5 个环节；另含原路径对照。

运行（容器内）：
    source /workspace/dev/batch_yxy/.venv/bin/activate
    python /workspace/dev/batch_yxy/examples/sched_engine_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamic_batching import FakeEmbedExecutor


def main() -> None:
    from dynamic_batching import BatchPolicy
    from sched_engine import (
        PATH_LEGACY,
        PATH_SCHEDULED,
        SchedRequest,
        SchedulerEngine,
        VllmEmbedFrameworkAdapter,
    )
    from sched_engine.devices.registry import ASCEND_910C
    from sched_engine.models.registry import QWEN3_EMBEDDING_0_6B

    model_id = "Qwen3-Embedding-0.6B"
    policy = BatchPolicy(
        model_id=model_id,
        policy_version="se-demo-910c",
        bucket_upper_bounds=(128, 256, 512),
        max_batch_size=8,
        max_total_tokens=4096,
        max_wait_ms=50.0,
    )

    lengths = [
        8, 16, 24, 32, 48, 64, 96, 120, 127, 128, 129, 160, 192, 200, 240, 255,
        256, 257, 288, 320, 384, 448, 500, 511, 512, 72, 88, 144, 176, 272, 336, 400,
    ]

    class ManualClock:
        def __init__(self):
            self.now = 1_000_000_000

        def __call__(self):
            return self.now

        def advance_ms(self, ms):
            self.now += int(ms * 1e6)

    def build(arrival_fn, executor, path_mode):
        clock = ManualClock()

        def fake_sleep(seconds):
            clock.advance_ms(seconds * 1000)

        engine = SchedulerEngine(
            model_profiles={model_id: QWEN3_EMBEDDING_0_6B},
            framework_adapter=VllmEmbedFrameworkAdapter(executor),
            device_profile=ASCEND_910C,
            policies=policy,
            clock=clock,
            sleep=fake_sleep,
            path_mode=path_mode,
        )
        requests = []
        for i, tokens in enumerate(lengths):
            text = " ".join([f"req{i}"] + ["w"] * (tokens - 1))
            requests.append(
                SchedRequest(
                    request_id=f"r{i:02d}",
                    model_id=model_id,
                    payload=text,
                    arrival_mono_ns=arrival_fn(i),
                    priority=1 if i % 2 else 0,
                )
            )
        return engine, requests

    orders = {
        "inorder": lambda i: i * 1_000_000,
        "reverse": lambda i: (31 - i) * 1_000_000,
        "interleaved": lambda i: ((i * 17) % 32) * 1_000_000,
    }

    for name, arrival_fn in orders.items():
        executor = FakeEmbedExecutor(dim=1024)
        engine, requests = build(arrival_fn, executor, PATH_SCHEDULED)
        engine.submit_many(requests)
        report = engine.run()
        reasons = {}
        for rec in report.dispatch_records:
            reasons[rec.dispatch_reason] = reasons.get(rec.dispatch_reason, 0) + 1
        vectors = report.results.items()
        ok = all(
            (vectors[r.request_id] == executor.vector(r.payload)).all()
            for r in requests
        )
        print(
            f"[{name:>11}] mode={report.mode} 批次={len(report.dispatch_records)} "
            f"封口={reasons} 结果数={len(vectors)} 逐ID向量一致={ok} "
            f"wall={report.wall_ms:.1f}ms"
        )

    # 原路径对照（同一执行器）
    executor = FakeEmbedExecutor(dim=1024)
    engine, requests = build(orders["interleaved"], executor, PATH_LEGACY)
    engine.submit_many(requests)
    legacy = engine.run()
    print(
        f"[     legacy] mode={legacy.mode} 原路径单次直通 "
        f"结果数={legacy.results.filled_count}"
    )


if __name__ == "__main__":
    main()
