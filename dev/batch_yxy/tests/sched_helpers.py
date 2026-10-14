"""sched_engine 测试公共构件：可控时钟、确定文本、引擎构建。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamic_batching import BatchPolicy, FakeEmbedExecutor
from helpers import ManualClock
from sched_engine import (
    PATH_SCHEDULED,
    SchedRequest,
    SchedulerEngine,
    VllmEmbedFrameworkAdapter,
)
from sched_engine.devices.registry import ASCEND_910C
from sched_engine.models.registry import QWEN3_EMBEDDING_0_6B

MODEL_ID = "Qwen3-Embedding-0.6B"


def unique_text(idx: int, tokens: int) -> str:
    """空白分词 token 数恰为 tokens、全文唯一（首词带序号）的文本。

    FakeEmbedExecutor.token_count = len(text.split())，向量由全文哈希确定
    生成 —— 文本唯一 ⇒ 向量唯一，可检测结果错位。
    """
    if tokens < 1:
        raise ValueError("tokens 必须 >= 1")
    return " ".join([f"req{idx}"] + ["w"] * (tokens - 1))


def make_sched(
    rid: str,
    *,
    arrival: int = 0,
    priority: int = 0,
    tokens: int = 8,
    idx: int = 0,
) -> SchedRequest:
    return SchedRequest(
        request_id=rid,
        model_id=MODEL_ID,
        payload=unique_text(idx, tokens),
        arrival_mono_ns=arrival,
        priority=priority,
    )


def make_engine_policy(**overrides):
    defaults = dict(
        model_id=MODEL_ID,
        policy_version="se-v1",
        bucket_upper_bounds=(128, 256, 512),
        max_batch_size=8,
        max_total_tokens=4096,
        max_wait_ms=50.0,
    )
    defaults.update(overrides)
    return BatchPolicy(**defaults)


def build_engine(
    policy=None,
    *,
    clock: ManualClock | None = None,
    executor: FakeEmbedExecutor | None = None,
    path_mode: str = PATH_SCHEDULED,
) -> tuple[SchedulerEngine, FakeEmbedExecutor, ManualClock]:
    clock = clock or ManualClock()
    executor = executor or FakeEmbedExecutor(dim=1024)
    adapter = VllmEmbedFrameworkAdapter(executor)

    def fake_sleep(seconds: float) -> None:
        clock.advance_ms(seconds * 1000)

    engine = SchedulerEngine(
        model_profiles={MODEL_ID: QWEN3_EMBEDDING_0_6B},
        framework_adapter=adapter,
        device_profile=ASCEND_910C,
        policies=policy or make_engine_policy(),
        clock=clock,
        sleep=fake_sleep,
        path_mode=path_mode,
    )
    return engine, executor, clock
