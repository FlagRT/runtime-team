"""P800 冒烟：vllm-kunlun 0.11.0 加载 Qwen3-Embedding-0.6B 并跑 embed。"""
import sys
import time

sys.path.insert(0, "/work/FlagScheduler")

import numpy as np

from dynamic_batching.vllm_compat import pooling_llm_options


def main() -> None:
    opts = pooling_llm_options()
    print(f"[smoke] pooling 选项: {opts}", flush=True)

    t0 = time.time()
    from vllm import LLM

    llm = LLM(
        model="/work/models/qwen3-embed-0.6b-xpu",
        max_model_len=4096,
        dtype="float16",
        enforce_eager=True,
        **opts,
    )
    print(f"[smoke] 模型加载完成: {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    out = llm.embed(["昆仑芯 P800 冒烟测试", "hello world", "dynamic batching"])
    dt = time.time() - t1
    vecs = [np.asarray(o.outputs.embedding) for o in out]
    v0, v1 = vecs[0], vecs[1]
    cos = float(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    print(
        f"[smoke] embed OK: 3条 / {dt * 1000:.0f}ms, 维度={vecs[0].shape}, "
        f"norm={np.linalg.norm(v0):.4f}, cos(中,英)={cos:.4f}",
        flush=True,
    )
    print("[smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
