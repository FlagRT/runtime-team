#!/usr/bin/env python
"""S4 KV 卸载到 Host（Route A / P800 vllm 0.13 官方镜像）— 官方 OffloadingConnector 原生路径

用途: 分层缓存原型第一步 —— 跑通 vLLM 0.13 官方 KV cache CPU 卸载（native 后端）:
  - 验证 --kv-offloading-size 的接线疑点: 0.13 的 _post_init_kv_transfer_config 写入
    kv_connector_extra_config 时 num_cpu_blocks 固定为 0, 且 kv_bytes_per_rank 全树无消费点
    (config/vllm.py:495-500), CPUOffloadingSpec 对 num_cpu_blocks=0 直接 raise (cpu.py:24-29)。
    → 本探针绕过该死路, 直接构造 KVTransferConfig 显式给 num_cpu_blocks。
  - 记录: GPU KV 容量变化(日志 Maximum concurrency / KV cache size)、生成质量与吞吐、CPU 侧占用。

用法（0.13 官方容器 flagos-official-moe-recheck 内）:
    source /root/miniconda/bin/activate python310_torch29_cuda
    CUDA_VISIBLE_DEVICES=3 VLLM_PLUGINS=fl VLLM_FL_PLATFORM=kunlunxin VLLM_FL_PREFER=flagos|vendor \
    USE_FLAGGEMS=1 GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0 DO_NOT_TRACK=1 \
    S4_MODEL=/workspace/models/Qwen3-4B S4_NUM_CPU_BLOCKS=910 \
    python -u /workspace/dev/memory/probes/routeA_s4_kv_host_offload.py

S4_NUM_CPU_BLOCKS: 0 = 不启用卸载(基线对照); >0 = 启用 OffloadingConnector 的 CPU 块数
  (Qwen3-4B: 2(K+V) x 36层 x 8kv_heads x 128head_dim x 16block x 2B = 2.25MiB/块;
   910 块 ≈ 2GiB, 1820 块 ≈ 4GiB)
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


def main():
    print("=" * 70)
    print("S4 KV offload to Host (Route A / P800, vllm 0.13 official)")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("=" * 70)

    model = os.environ.get("S4_MODEL", "/workspace/models/Qwen3-4B")
    num_cpu_blocks = int(os.environ.get("S4_NUM_CPU_BLOCKS", "0"))
    gpu_mem_util = float(os.environ.get("S4_GPU_MEM_UTIL", "0.9"))
    print(f"model={model} num_cpu_blocks={num_cpu_blocks} gpu_mem_util={gpu_mem_util}")

    from vllm import LLM, SamplingParams

    kwargs = dict(model=model, max_num_batched_tokens=16384, max_num_seqs=2048,
                  enforce_eager=True, gpu_memory_utilization=gpu_mem_util)
    if num_cpu_blocks > 0:
        from vllm.config import KVTransferConfig
        kvt = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"num_cpu_blocks": num_cpu_blocks},
        )
        kwargs["kv_transfer_config"] = kvt
        print(f"KVTransferConfig: connector=OffloadingConnector num_cpu_blocks={num_cpu_blocks}")

    t0 = time.time()
    llm = LLM(**kwargs)
    print(f"[加载耗时] {time.time()-t0:.1f}s")

    free0, total0 = torch.cuda.mem_get_info()
    print(f"[加载后显存] free={free0/1e9:.2f}GB total={total0/1e9:.2f}GB "
          f"used={(total0-free0)/1e9:.2f}GB")

    prompts = [
        "The capital of France is",
        "Hello, my name is",
        "量子计算的基本原理是什么？请简要说明。",
        # 长序列: 逼出 KV 卸载(前缀共享, 触发 store/load 事件)
        "Write a short story about a robot learning to paint. " * 12,
    ]
    sp = SamplingParams(max_tokens=64, temperature=0.0)

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    t_gen = time.time() - t0
    for o in outputs:
        print(f"\nPrompt: {o.prompt[:60]!r}...")
        print(f"Generated: {o.outputs[0].text[:200]!r}")
    n = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\n[生成耗时] {t_gen:.2f}s, {n} tokens, {n/t_gen:.1f} tok/s")

    # 尽力读取块表配置（0.13 V1 结构, 属性缺失时静默跳过, 块数以日志为准）
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        print(f"[块表] num_gpu_blocks={cc.num_gpu_blocks} num_cpu_blocks={cc.num_cpu_blocks}")
    except Exception as e:
        print(f"[块表] 读取失败(以日志为准): {e}")

    del llm
    torch.cuda.empty_cache()
    print("\n S4 完成")


if __name__ == "__main__":
    main()
