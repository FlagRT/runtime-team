#!/usr/bin/env python3
"""最小化诊断:单 prompt + 短生成,逐步打印,定位 prefill 卡点。

用法(容器内):
  ASCEND_RT_VISIBLE_DEVICES=0 python qwen3_mini_probe.py
"""
import os
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402


def main() -> None:
    print("=" * 60, flush=True)
    print(f"Qwen3-4B mini probe | device={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}", flush=True)
    print(f"platform: {type(current_platform).__name__}", flush=True)

    t0 = time.time()
    llm = LLM(model="/workspace/models/Qwen3-4B",
              max_num_batched_tokens=4096, max_num_seqs=16,
              enforce_eager=True)
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    # 单 prompt,极短生成
    sampling_params = SamplingParams(max_tokens=4, temperature=0.0)
    t0 = time.time()
    print("[gen] start...", flush=True)
    outputs = llm.generate(["Hello, my name is"], sampling_params)
    print(f"[gen] {time.time()-t0:.1f}s", flush=True)
    for output in outputs:
        print(f"PROMPT: {output.prompt!r}", flush=True)
        print(f"OUTPUT: {output.outputs[0].text!r}", flush=True)

    del llm
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
