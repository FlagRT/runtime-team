#!/usr/bin/env python
"""S4 KV 卸载传输冒烟（Route A / P800 vllm 0.13 官方）— 验证 gpu->cpu store + cpu->gpu load 真实发生

背景: routeA_s4_kv_host_offload.py 已证明 OffloadingConnector/CPUOffloadingSpec 配置与初始化可用,
但短序列日志(默认 INFO)看不到传输证据。本探针:
  1. 开 DEBUG 日志抓 offload 行 ("offloading N blocks" / "hit N offloaded tokens");
  2. 同一长 prompt 跑两遍: 第 2 遍应命中 CPU 块(manager.lookup 命中)并跳过重算 ——
     即真实执行 cpu->gpu swap_blocks 加载(厂商 attention 后端 KV 布局兼容性的关键验证)。

用法（0.13 官方容器 flagos-official-moe-recheck 内）:
    source /root/miniconda/bin/activate python310_torch29_cuda
    CUDA_VISIBLE_DEVICES=3 VLLM_PLUGINS=fl VLLM_FL_PLATFORM=kunlunxin VLLM_FL_PREFER=flagos|vendor \
    USE_FLAGGEMS=1 GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0 DO_NOT_TRACK=1 \
    S4_NUM_CPU_BLOCKS=910 VLLM_LOGGING_LEVEL=DEBUG \
    python -u /workspace/dev/memory/probes/routeA_s4_kv_offload_xfer.py
"""
import os
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
os.environ.setdefault("VLLM_FL_PLATFORM", "kunlunxin")
os.environ.setdefault("VLLM_FL_PREFER", "flagos|vendor")
os.environ.setdefault("USE_FLAGGEMS", "1")
os.environ.setdefault("GEMS_VENDOR", "kunlunxin")
os.environ.setdefault("KLX_USE_AUTOTUNE", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")


def main():
    print("=" * 70)
    print("S4 KV offload transfer smoke (Route A / P800, vllm 0.13 official)")
    print("=" * 70)
    model = os.environ.get("S4_MODEL", "/workspace/models/Qwen3-4B")
    num_cpu_blocks = int(os.environ.get("S4_NUM_CPU_BLOCKS", "910"))
    print(f"model={model} num_cpu_blocks={num_cpu_blocks}")

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kvt = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"num_cpu_blocks": num_cpu_blocks},
    )
    llm = LLM(model=model, max_num_batched_tokens=16384, max_num_seqs=2048,
              enforce_eager=True, gpu_memory_utilization=0.9,
              kv_transfer_config=kvt)

    # 同一长 prompt 跑两遍: 第一遍算+store, 第二遍应 load 命中
    prompt = ("The capital of France is Paris. Germany is Berlin. "
              "Japan is Tokyo. Brazil is Brasilia. " * 60)  # ~720 tokens
    sp1 = SamplingParams(max_tokens=8, temperature=0.0)
    sp2 = SamplingParams(max_tokens=16, temperature=0.0)

    t0 = time.time()
    out1 = llm.generate([prompt], sp1)
    t1 = time.time() - t0
    print(f"\n[run1] {t1:.2f}s -> {out1[0].outputs[0].text[:60]!r}")

    t0 = time.time()
    out2 = llm.generate([prompt], sp2)
    t2 = time.time() - t0
    print(f"[run2] {t2:.2f}s -> {out2[0].outputs[0].text[:60]!r}")
    print(f"[对比] run2/run1 耗时 = {t2/t1:.2f}x (若 CPU 块命中, run2 应显著快于 run1)")

    del llm
    print("\n S4 transfer smoke 完成")


if __name__ == "__main__":
    main()
