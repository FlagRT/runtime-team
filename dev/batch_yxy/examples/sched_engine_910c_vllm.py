"""910C 联调脚本：sched_engine + vLLM Qwen3-Embedding-0.6B（单卡 davinci0）。

本脚本提供 910C 真机验证路径，本轮未执行（无空闲卡）。有空闲卡时运行：

    # 宿主机：确认 davinci0 空闲后进容器
    npu-smi info
    docker exec -it flagos-batch-yxy-dev-910c bash
    source /workspace/dev/batch_yxy/.venv/bin/activate
    python /workspace/dev/batch_yxy/examples/sched_engine_910c_vllm.py

验收口径（与 docs/note_调度引擎四层架构说明.md 一致）：
- 调度映射正确性：32 条请求均得到且只得到对应结果（数量/身份/顺序校验）；
- 数值一致性：与平台跨调用漂移分开记录 —— vllm-ascend 0.20.2rc1 pooling
  存在跨调用非确定缺陷（见 probes/qwen3_embed_nondeterminism_probe.py），
  baseline 与 scheduled 的向量逐元素对比仅作参考，不作为映射正确性依据。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_PATH = "/root/models/Qwen3-Embedding-0.6B"  # 容器内本地权重路径（按需修改）


def main() -> None:
    from vllm import LLM

    from dynamic_batching import BatchPolicy, VllmEmbedExecutor
    from sched_engine import (
        PATH_LEGACY,
        PATH_SCHEDULED,
        SchedRequest,
        SchedulerEngine,
        VllmEmbedFrameworkAdapter,
    )
    from sched_engine.devices.registry import ASCEND_910C
    from sched_engine.models.registry import QWEN3_EMBEDDING_0_6B

    llm = LLM(
        model=MODEL_PATH,
        task="embed",
        max_model_len=4096,
        dtype="float16",
        enforce_eager=True,  # 与既有 bench 口径一致
    )
    executor = VllmEmbedExecutor(llm)  # 同源 tokenizer 计量（截断到 max_model_len）
    adapter = VllmEmbedFrameworkAdapter(executor)

    model_id = "Qwen3-Embedding-0.6B"
    policy = BatchPolicy(
        model_id=model_id,
        policy_version="se-910c-v1",
        bucket_upper_bounds=(128, 256, 512),
        max_batch_size=8,
        max_total_tokens=4096,
        max_wait_ms=50.0,
    )

    lengths = [
        8, 16, 24, 32, 48, 64, 96, 120, 127, 128, 129, 160, 192, 200, 240, 255,
        256, 257, 288, 320, 384, 448, 500, 511, 512, 72, 88, 144, 176, 272, 336, 400,
    ]

    def build_requests(arrival_fn):
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
        return requests

    def run(path_mode, requests):
        engine = SchedulerEngine(
            model_profiles={model_id: QWEN3_EMBEDDING_0_6B},
            framework_adapter=adapter,
            device_profile=ASCEND_910C,
            policies=policy,
            path_mode=path_mode,
        )
        engine.submit_many(requests)
        return engine.run()

    orders = {
        "inorder": lambda i: i * 1_000_000,
        "reverse": lambda i: (31 - i) * 1_000_000,
        "interleaved": lambda i: ((i * 17) % 32) * 1_000_000,
    }

    for name, arrival_fn in orders.items():
        report = run(PATH_SCHEDULED, build_requests(arrival_fn))
        records = report.dispatch_records
        reasons = {}
        for rec in records:
            reasons[rec.dispatch_reason] = reasons.get(rec.dispatch_reason, 0) + 1
        ids = [rid for rec in records for rid in rec.request_ids]
        # 映射正确性：恰好一次、无遗漏
        assert len(ids) == len(set(ids)) == 32, "每条请求恰好出现在一个批次"
        assert report.results.filled_count == 32
        dims = {report.results.get(rid).shape for rid in ids}
        print(
            f"[{name:>11}] mode={report.mode} 批次={len(records)} 封口={reasons} "
            f"结果数={report.results.filled_count} 向量维度={dims} "
            f"wall={report.wall_ms:.1f}ms"
        )

    # 原路径对照 + 数值一致性参考（受平台跨调用漂移影响，单独记录不替代映射结论）
    import numpy as np

    requests = build_requests(orders["interleaved"])
    legacy = run(PATH_LEGACY, requests)
    scheduled = run(PATH_SCHEDULED, requests)
    same, cos = 0, []
    for req in requests:
        a = legacy.results.get(req.request_id)
        b = scheduled.results.get(req.request_id)
        if np.array_equal(a, b):
            same += 1
        cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    print(
        f"[  数值参考] legacy vs scheduled 逐元素一致 {same}/32（平台已知跨调用"
        f"漂移，仅供参考）；cosine min={min(cos):.4f} mean={sum(cos) / len(cos):.4f}"
    )


if __name__ == "__main__":
    main()
