#!/usr/bin/env python3
"""S4 KV 卸载到 Host —— 910C 昇腾版(vllm 0.20.2 + vllm-plugin-FL)

P800 版(routeA_s4_kv_host_offload.py / routeA_s4_kv_offload_xfer.py)基于 vllm 0.13
官方昆仑芯镜像;本脚本把同一目标搬到 910C 昇腾栈。

⚠️ 2026-09-03 实测结论(见《routeA-S4-KV卸载Host-910C尝试-20260903.md》):
  vllm 0.20.2 官方 native OffloadingConnector 在昇腾栈**当前不可用**,硬阻塞两处:
    A. `v1/kv_offload/cpu/spec.py:84` get_handlers() 平台门 ——
       `Exception: CPU Offloading is currently only supported on CUDA-alike GPUs`
       (PlatformFL device_name=npu → is_cuda_alike()==False;P800 xpytorch 为 True 故放行)
    B. `vllm._C` 是 CUDA 构建、本镜像无 libcudart → `ops.swap_blocks_batch` 不存在,
       传输 handler(cpu_gpu.py:315)必失败。
  本探针遂降级为「阻塞路径特征化」工具:验证配置 API、复现平台门、定位下游缺口。
  另注:0.20.2 的 extra_config 键是 `cpu_bytes_to_use`(字节),不再是 0.13 的 `num_cpu_blocks`。

  待昇腾 CPU-offload handlers 适配落地后,可解开 S4_FORCE_CUDA_ALIKE 并接回原目标:
    1. connector + KVTransferConfig 初始化;2. GPU/CPU 块数与 KV cache size;
    3. 同一长 prompt 两遍 → store/load 命中、run2<run1、两遍输出逐字一致。

  已知(当前 910C 可用推理栈):
    - 默认 TRITON_ATTN 在 unified_attention 卡死 → 需 VLLM_FL_USE_FLAGGEMS_ATTN=1;
    - 首个 attention 初始化极慢 + 算子链在 flag_gems triton kernel 上偏慢(~17s/tok),
      本探针只验卸载链路,不以生成质量为通过条件。
    - 上面两处硬阻塞是 vllm 层的(平台门 + CUDA 扩展),与设备栈无关,昇腾任何组合都会撞。

用法(容器 flagos-fl-dev-910c 内,venv312 已 pth 自动引导 flagos_boot):
    ASCEND_RT_VISIBLE_DEVICES=0 VLLM_FL_USE_FLAGGEMS_ATTN=1 DO_NOT_TRACK=1 \
    S4_NUM_CPU_BLOCKS=1024 S4_MAX_TOKENS=4 VLLM_LOGGING_LEVEL=DEBUG \
    /root/vllm-venv312/bin/python -u dev/memory/probes/routeA_s4_kv_host_offload_910c.py

S4_NUM_CPU_BLOCKS: 0 = 不启用卸载(基线对照);>0 = OffloadingConnector 的 CPU 块数
S4_MAX_TOKENS:     每 prompt 生成 token 数(默认 4,昇腾慢路径下别调大)
S4_XFER:           1(默认)= 末尾对同一长 prompt 再跑一遍,验 store→load 命中
"""
import os
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch  # noqa: E402


def _mem_free_total():
    try:
        free, total = torch.npu.mem_get_info()
        return free / 1e9, total / 1e9
    except Exception as e:  # noqa: BLE001
        return None, None


def _block_table(llm):
    for path in ("llm_engine", "engine"):
        eng = getattr(llm, path, None)
        if eng is None:
            continue
        cfg = getattr(eng, "vllm_config", None)
        cc = getattr(cfg, "cache_config", None) if cfg else None
        if cc is not None:
            return getattr(cc, "num_gpu_blocks", None), getattr(cc, "num_cpu_blocks", None)
    return None, None


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    model = os.environ.get("S4_MODEL", "/workspace/models/Qwen3-4B")
    # vllm 0.20.2 CPUOffloadingSpec 期望 cpu_bytes_to_use(字节),不再是 0.13 的 num_cpu_blocks
    cpu_bytes_gb = float(os.environ.get("S4_CPU_BYTES_GB", "4"))
    cpu_bytes = int(cpu_bytes_gb * 1024**3) if cpu_bytes_gb > 0 else 0
    connector = os.environ.get("S4_CONNECTOR", "OffloadingConnector")
    force_cuda_alike = os.environ.get("S4_FORCE_CUDA_ALIKE", "0") == "1"
    max_tokens = int(os.environ.get("S4_MAX_TOKENS", "4"))
    gpu_mem_util = float(os.environ.get("S4_GPU_MEM_UTIL", "0.9"))
    do_xfer = os.environ.get("S4_XFER", "1") == "1"

    if force_cuda_alike:
        # 探针开关:强行绕过 CPUOffloadingSpec.get_handlers() 的 is_cuda_alike 硬门,
        # 用于定位「即使过门,下游 swap_blocks / cuda stream 是否可用」
        from vllm.platforms.interface import Platform
        try:
            from vllm.platforms import current_platform
            type(current_platform).is_cuda_alike = lambda self: True  # noqa: E731
        except Exception as e:  # noqa: BLE001
            print(f"[warn] force_cuda_alike patch on instance failed: {e}", flush=True)
        Platform.is_cuda_alike = lambda self: True  # noqa: E731
        print("[cfg] S4_FORCE_CUDA_ALIKE=1 -> is_cuda_alike() patched to True", flush=True)

    print("=" * 72, flush=True)
    print("S4 KV offload to Host  |  Route A / 910C Ascend  |  vllm", end=" ", flush=True)
    import vllm
    print(vllm.__version__, flush=True)
    print(f"platform         : {type(current_platform).__name__}", flush=True)
    print(f"device           : {os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}", flush=True)
    print(f"flaggems_attn    : {os.environ.get('VLLM_FL_USE_FLAGGEMS_ATTN')}", flush=True)
    print(f"model            : {model}", flush=True)
    print(f"connector        : {connector}  cpu_bytes_to_use={cpu_bytes_gb}GB  (0=baseline)", flush=True)
    print(f"max_tokens       : {max_tokens}   xfer_recheck: {do_xfer}", flush=True)
    print("=" * 72, flush=True)

    kwargs = dict(model=model, max_num_batched_tokens=4096, max_num_seqs=16,
                  enforce_eager=True, gpu_memory_utilization=gpu_mem_util)
    if cpu_bytes > 0:
        from vllm.config import KVTransferConfig
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector=connector,
            kv_role="kv_both",
            kv_connector_extra_config={"cpu_bytes_to_use": cpu_bytes},
        )
        print(f"[cfg] KVTransferConfig: {connector} kv_both "
              f"cpu_bytes_to_use={cpu_bytes} ({cpu_bytes_gb}GB)", flush=True)

    t0 = time.time()
    llm = LLM(**kwargs)
    print(f"[load] engine ready in {time.time() - t0:.1f}s", flush=True)

    f, t = _mem_free_total()
    if f is not None:
        print(f"[mem ] after load: free={f:.2f}GB total={t:.2f}GB used={t - f:.2f}GB", flush=True)
    g, c = _block_table(llm)
    print(f"[blk ] num_gpu_blocks={g}  num_cpu_blocks={c}", flush=True)

    short_p = "The capital of France is"
    long_p = ("The quick brown fox jumps over the lazy dog. "
              "Paris is the capital of France; Berlin of Germany; Tokyo of Japan. " * 8)
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    print("\n[gen ] pass over [short, long] ...", flush=True)
    t0 = time.time()
    outs = llm.generate([short_p, long_p], sp)
    dt = time.time() - t0
    for o in outs:
        print(f"  prompt={o.prompt[:50]!r}... -> {o.outputs[0].text[:120]!r}", flush=True)
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[gen ] {dt:.2f}s  {ntok} tok  {ntok / dt:.2f} tok/s", flush=True)

    if do_xfer:
        print("\n[xfer] re-run the SAME long prompt (expect CPU-block hit, run2 < run1) ...",
              flush=True)
        t0 = time.time()
        r1 = llm.generate([long_p], SamplingParams(max_tokens=max_tokens, temperature=0.0))
        d1 = time.time() - t0
        t0 = time.time()
        r2 = llm.generate([long_p], SamplingParams(max_tokens=max_tokens, temperature=0.0))
        d2 = time.time() - t0
        txt1, txt2 = r1[0].outputs[0].text, r2[0].outputs[0].text
        print(f"[xfer] run1 {d1:.2f}s -> {txt1[:80]!r}", flush=True)
        print(f"[xfer] run2 {d2:.2f}s -> {txt2[:80]!r}", flush=True)
        print(f"[xfer] deterministic across runs: {txt1 == txt2}", flush=True)
        print(f"[xfer] run2/run1 = {d2 / d1:.2f}x  ({'HIT: run2 faster' if d2 < d1 else 'no speedup'})",
              flush=True)

    g, c = _block_table(llm)
    print(f"\n[blk ] final num_gpu_blocks={g}  num_cpu_blocks={c}", flush=True)
    del llm
    torch.npu.empty_cache() if hasattr(torch.npu, "empty_cache") else None
    print("S4 (910C) done", flush=True)


if __name__ == "__main__":
    main()
