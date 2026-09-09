#!/usr/bin/env python3
"""Qwen3-4B offline 推理闭环探针(910C torch_fl 栈,已冻结路线——见 probes/README.md)。

旧机阶段4 单卡/TP 验证脚本未入库,本脚本按 TP 验证记录矩阵重建:
  - 4 prompts 与旧机一致(Hello my name / France / 2+2 / Python)
  - TP 由环境变量 VLLM_FL_TP 控制(旧机语义, 默认 1)
  - ASCEND_RT_VISIBLE_DEVICES 单卡/多卡可见(TP 语义), 容器内跑

用法(容器内):
  ASCEND_RT_VISIBLE_DEVICES=0 VLLM_FL_TP=1 python qwen3_offline_tp.py
  ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_FL_TP=2 python qwen3_offline_tp.py
输出: 平台/vendor、加载耗时、每 prompt 生成文本(与 TP=1 逐字对比用)。
"""
import os
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402


def main() -> None:
    tp = int(os.environ.get("VLLM_FL_TP", "1"))
    model = os.environ.get("MODEL_PATH", "/workspace/models/Qwen3-4B")
    print("=" * 70)
    print(f"Qwen3-4B offline inference | TP={tp} | device={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")
    print(f"platform: {current_platform} ({type(current_platform).__name__})")
    print(f"model: {model}")
    print("=" * 70)

    t0 = time.time()
    llm = LLM(model=model, tensor_parallel_size=tp,
              max_num_batched_tokens=16384, max_num_seqs=256,
              enforce_eager=True)
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "2+2=",
        "Python is a",
    ]
    sampling_params = SamplingParams(max_tokens=30, temperature=0.0)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"[gen ] {time.time() - t0:.1f}s", flush=True)

    for output in outputs:
        print(f"PROMPT: {output.prompt!r}")
        print(f"OUTPUT: {output.outputs[0].text!r}", flush=True)

    del llm
    print("\nREASONING OK (exit clean)")


if __name__ == "__main__":
    main()
