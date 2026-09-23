"""NPU 最小功能联调：动态组批路径的数量/顺序/ID 映射/维度校验（不断言数值一致性）。

运行（容器内，非 dev/batch_yxy cwd 或已规避 vllm 命名空间遮蔽）：
  .venv/bin/python examples/npu_functional_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dynamic_batching import BatchPolicy, DynamicBatchEmbedder, VllmEmbedExecutor

sys.path.remove(str(Path(__file__).resolve().parents[1]))

MODEL = "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B"


def main() -> int:
    from vllm import LLM

    llm = LLM(model=MODEL, max_model_len=4096, enforce_eager=True,
              enable_prefix_caching=False)
    executor = VllmEmbedExecutor(llm)
    tok = llm.get_tokenizer()

    rng = np.random.default_rng(11)
    words = [f"w{j}" for j in range(1024)]
    items = []
    for i in range(24):
        n = int(rng.integers(16, 65)) if rng.random() < 0.5 else int(rng.integers(512, 2049))
        text = " ".join(words[(i * 37 + j) % 1024] for j in range(n))
        while len(tok.encode(text)) > 2048:
            text = " ".join(text.split()[: int(len(text.split()) * 0.9)])
        items.append((f"req-{i:03d}", text))

    policy = BatchPolicy(
        model_id="Qwen3-Embedding-0.6B",
        policy_version="v1-npu-check",
        bucket_upper_bounds=(64, 128, 256, 512, 1024, 2048),
        max_batch_size=8,
        max_total_tokens=8192,
        max_wait_ms=20.0,
    )
    baseline = DynamicBatchEmbedder(executor).run(items)
    dynamic = DynamicBatchEmbedder(executor, policy, model_version="0.6b").run(items)

    assert set(baseline.vectors) == {rid for rid, _ in items}
    assert set(dynamic.vectors) == {rid for rid, _ in items}
    for vec in dynamic.vectors.values():
        assert vec.shape == (1024,), f"维度异常: {vec.shape}"
        assert bool(np.isfinite(vec).all()), "向量含非有限值"
    assert len(dynamic.timings) == len(items)
    assert all(len(r.request_ids) <= policy.max_batch_size for r in dynamic.dispatch_records)
    assert all(r.total_tokens <= policy.max_total_tokens for r in dynamic.dispatch_records)

    n_batches = len(dynamic.dispatch_records)
    sizes = [r.batch_size for r in dynamic.dispatch_records]
    reasons = {r.dispatch_reason for r in dynamic.dispatch_records}
    cross = sum(1 for r in dynamic.dispatch_records if r.cross_bucket_merge)
    info_cos = [
        float(np.dot(baseline.vectors[rid], dynamic.vectors[rid])
              / (np.linalg.norm(baseline.vectors[rid]) * np.linalg.norm(dynamic.vectors[rid])))
        for rid, _ in items
    ]
    print(f"CHECK PASS: 请求 {len(items)} 条全部返回，维度 1024，ID 一一对应")
    print(f"动态批次数={n_batches} 批量分布={sizes} 封口原因={reasons} 跨桶批={cross}")
    print(f"baseline vs dynamic 余弦（信息性，受平台非确定影响不作验收）: "
          f"min={min(info_cos):.4f} mean={float(np.mean(info_cos)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
