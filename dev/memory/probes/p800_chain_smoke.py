#!/usr/bin/env python
"""P800 推理链路冒烟测试 (memory 子方向 / 重建版)

对应 910c "推理插件接入-阶段4 已验证链路" 的最小复刻, 即 vllm-plugin-FL
kunlunxin_example/README.md 的单卡测试 (flag_gems 算子路径).

用法:
    source /root/miniconda/bin/activate python310_torch29_cuda
    CUDA_VISIBLE_DEVICES=1 python /workspace/dev/memory/probes/p800_chain_smoke.py [模型路径]
默认模型: /workspace/models/Qwen2.5-1.5B-Instruct
"""
import os
import sys
import time

# ---- 必须在 import vllm/flag_gems 之前设置 (kunlunxin_example README) ----
os.environ.setdefault("VLLM_FL_PLATFORM", "kunlunxin")
os.environ.setdefault("VLLM_FL_PREFER", "flagos")
os.environ.setdefault("USE_FLAGGEMS", "1")
os.environ.setdefault("GEMS_VENDOR", "kunlunxin")
os.environ.setdefault("KLX_USE_AUTOTUNE", "0")  # 关 autotune, 避免首请求慢路径

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/Qwen2.5-1.5B-Instruct"

if __name__ == "__main__":
    print(f"[smoke] model = {MODEL}")
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        max_num_batched_tokens=16384,
        max_num_seqs=256,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
    )
    print(f"[smoke] 模型加载耗时: {time.time() - t0:.1f}s")

    prompts = [
        "Hello, my name is",
        "What is FlagOS?",
        "请用一句话介绍显存管理。",
    ]
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    dt = time.time() - t0
    for out in outputs:
        print(f"  prompt: {out.prompt[:40]!r} -> {out.outputs[0].text[:60]!r}")
    print(f"[smoke] 3 请求生成耗时: {dt:.1f}s")

    # 长一点的第二轮, 确认 KV cache 增长后仍正常
    sampling_params2 = SamplingParams(max_tokens=64, temperature=0.0)
    t0 = time.time()
    outputs = llm.generate([prompts[1]] * 4, sampling_params2)
    dt = time.time() - t0
    print(f"[smoke] 4x64-token 请求耗时: {dt:.1f}s")
    print("[smoke] 推理链路 OK")
