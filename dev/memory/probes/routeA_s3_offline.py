#!/usr/bin/env python
"""S3 端到端推理（Route A / P800）— offline inference, 基于官方示例

用法（容器内）:
    source /root/miniconda/bin/activate python310_torch29_cuda
    CUDA_VISIBLE_DEVICES=1 VLLM_PLUGINS=fl VLLM_FL_PLATFORM=kunlunxin \
    VLLM_FL_PREFER=flagos|vendor USE_FLAGGEMS=1 GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0 \
    python -u /workspace/dev/memory/probes/routeA_s3_offline.py

记录: 平台/vendor 选择、加载耗时、显存、生成耗时与文本
"""
import os
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
os.environ.setdefault("VLLM_FL_PLATFORM", "kunlunxin")
os.environ.setdefault("VLLM_FL_PREFER", "flagos|vendor")
os.environ.setdefault("USE_FLAGGEMS", "1")
os.environ.setdefault("GEMS_VENDOR", "kunlunxin")
os.environ.setdefault("KLX_USE_AUTOTUNE", "0")

import torch
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


def main():
    print("=" * 70)
    print("S3 offline inference (Route A)")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"VLLM_FL_PREFER      = {os.environ.get('VLLM_FL_PREFER')}")
    print("=" * 70)

    MODEL = os.environ.get("S3_MODEL", "/workspace/models/Qwen3-4B")
    print(f"model: {MODEL}")
    try:
        import vllm_fl
        print(f"plugin: vllm_fl @ {vllm_fl.__file__}")
    except Exception as e:
        print(f"plugin import err: {e}")

    t0 = time.time()
    enforce_eager = os.environ.get("S3_ENFORCE_EAGER", "0") == "1"
    print(f"enforce_eager: {enforce_eager}")
    llm = LLM(model=MODEL, max_num_batched_tokens=16384, max_num_seqs=2048,
              enforce_eager=enforce_eager)
    t_load = time.time() - t0
    print(f"\n[加载耗时] {t_load:.1f}s")

    free0, total0 = torch.cuda.mem_get_info()
    print(f"[加载后显存] free={free0/1e9:.2f}GB total={total0/1e9:.2f}GB used={(total0-free0)/1e9:.2f}GB")

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "量子计算的基本原理是什么？请简要说明。",
    ]
    sampling_params = SamplingParams(max_tokens=64, temperature=0.0)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t_gen = time.time() - t0

    for output in outputs:
        print(f"\nPrompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")

    n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\n[生成耗时] {t_gen:.2f}s, 共 {n_tokens} tokens, 平均 {n_tokens/t_gen:.1f} tok/s (3请求共享批处理)")

    del llm
    torch.cuda.empty_cache()
    print("\n S3 offline inference 完成")


if __name__ == "__main__":
    main()
