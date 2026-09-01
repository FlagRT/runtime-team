#!/usr/bin/env python
"""S3 纯 MoE 生成质量 A/B（Route A / P800 新线镜像）— expert GEMM 厂商内核 vs 纯 torch 参考

用途: 定位纯 MoE eager 生成退化（首 token 正确后退化为重复/乱码）的责任层。
方法: 同一模型/同一 prompt/同一 greedy 采样，仅切换 fused_experts_impl 实现:
  - vendor   : 厂商 xtorch_ops 内核（vllm_fl patch_fused_moe 默认注入，_klx_fused_experts）
  - reference: 纯 torch 参考实现（batch-gather + bmm + index_add，本文件内联）
若 vendor 乱码、reference 正常 → expert GEMM 厂商内核（或其输入预处理 gen_block_statistic/
moe_pre_sorted）为退化源; 若两者均乱码 → 责任层在 attention/KV/路由/采样等上游。

用法（新线容器 flagos-newline-moe 内）:
    source /root/miniconda/bin/activate python310_torch29_cuda
    CUDA_VISIBLE_DEVICES=2 VLLM_PLUGINS=fl VLLM_FL_PLATFORM=kunlunxin VLLM_FL_PREFER=flagos \
    USE_FLAGGEMS=1 GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0 DO_NOT_TRACK=1 \
    S3_MODEL=/models/Qwen3-30B-A3B S3_ENFORCE_EAGER=1 S3_MOE_IMPL=both \
    python -u /tmp/routeA_s3_moe_ab.py

S3_MOE_IMPL: vendor | reference | both（默认 vendor，与 routeA_s3_offline 基线一致）

⚠️ 重要: vllm 0.20.2 的 EngineCore 是独立子进程, 主进程 monkeypatch 不会传播到模型运行侧。
   reference 模式必须先在容器内把 probes/ref_moe_impl.py 的 ref_fused_experts_impl 注入到
   /workspace/vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/kunlunxin/impl/fused_moe/fused_moe.py
   的 fused_experts_impl 函数体最前面（改前备份 .orig_bak）。本文件内联 _ref_fused_experts_impl
   仅作独立数值参考用（单进程对比），EngineCore 日志中出现 [REF-MOE] 标记才算注入生效。
"""
import os
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
os.environ.setdefault("VLLM_FL_PLATFORM", "kunlunxin")
os.environ.setdefault("VLLM_FL_PREFER", "flagos")  # 新线口径（旧 flagos|vendor 触发 moe_align_block_size 6/7 参崩溃）
os.environ.setdefault("USE_FLAGGEMS", "1")
os.environ.setdefault("GEMS_VENDOR", "kunlunxin")
os.environ.setdefault("KLX_USE_AUTOTUNE", "0")

import torch


# ---------------------------------------------------------------------------
# 纯 torch 参考 fused_experts_impl（签名对齐 vllm_fl.ops.fused_moe.fused_moe.fused_experts_impl）
# 仅支持非量化 bf16/fp16 + silu；与厂商内核同样在 moe_post 阶段应用路由权重（scatter-sum）
# ---------------------------------------------------------------------------
def _ref_fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale=None, w2_scale=None, w1_zp=None, w2_zp=None,
    a1_scale=None, a2_scale=None, block_shape=None,
    w1_bias=None, w2_bias=None,
) -> torch.Tensor:
    if use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16:
        raise NotImplementedError("A/B 探针参考实现不支持量化路径")
    if apply_router_weight_on_input:
        raise NotImplementedError("A/B 探针参考实现不支持 apply_router_weight_on_input=True")
    if activation != "silu":
        raise NotImplementedError(f"A/B 探针参考实现仅支持 silu, got {activation}")

    num_tokens, hidden_dim = hidden_states.shape
    topk = topk_ids.shape[1]
    ffn_hd = w1.shape[1] // 2
    dev = hidden_states.device

    flat_ids = topk_ids.reshape(-1).long()                    # [M*topk] 全局 expert id
    if expert_map is not None:
        local_ids = expert_map[flat_ids]
        valid = local_ids >= 0
        use_ids = local_ids.clamp(min=0)
    else:
        use_ids = flat_ids
        valid = torch.ones_like(flat_ids, dtype=torch.bool)

    x_flat = hidden_states.repeat_interleave(topk, dim=0)     # [M*topk, H]
    w1e = w1[use_ids]                                         # [M*topk, 2F, H] (中间维 2F, 末维 H)
    gate_up = torch.einsum("mi,mji->mj", x_flat, w1e)         # [M*topk, 2F]
    inter = torch.nn.functional.silu(gate_up[:, :ffn_hd]) * gate_up[:, ffn_hd:]
    w2e = w2[use_ids]                                         # [M*topk, H, F] (中间维 H, 末维 F)
    y = torch.einsum("mj,mji->mi", inter, w2e)                # [M*topk, H]
    y = y * topk_weights.reshape(-1, 1)
    y = y * valid.unsqueeze(-1).to(y.dtype)

    out = torch.zeros(num_tokens, hidden_dim, dtype=hidden_states.dtype, device=dev)
    tok_idx = torch.arange(num_tokens, device=dev).repeat_interleave(topk)
    out.index_add_(0, tok_idx, y)
    return out


# ---------------------------------------------------------------------------
def _run(mode: str, model: str, enforce_eager: bool):
    import vllm  # noqa: F401  触发插件加载（patch_fused_moe 注入厂商实现）
    import vllm_fl.ops.fused_moe.fused_moe as fm_lib

    if mode == "vendor":
        impl = fm_lib.fused_experts_impl
        print(f"[{mode}] fused_experts_impl = {impl.__module__}.{impl.__name__}")
    elif mode == "reference":
        # EngineCore 子进程不继承主进程 monkeypatch —— 仅提示, 实际注入须改插件源码
        # （见文件头 ⚠️ 说明; 容器内注入后此模式才有效）
        print(f"[{mode}] 提示: 主进程 monkeypatch 对 EngineCore 子进程无效, "
              f"需容器内注入 probes/ref_moe_impl.py（EngineCore 日志见 [REF-MOE] 标记）")

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=model, max_num_batched_tokens=16384, max_num_seqs=2048,
              enforce_eager=enforce_eager)
    print(f"[{mode}] 加载耗时 {time.time()-t0:.1f}s")

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "量子计算的基本原理是什么？请简要说明。",
    ]
    sp = SamplingParams(max_tokens=64, temperature=0.0)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    t_gen = time.time() - t0
    for o in outputs:
        print(f"[{mode}] Prompt: {o.prompt!r}")
        print(f"[{mode}] Generated: {o.outputs[0].text!r}")
    n = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"[{mode}] 生成 {n} tokens / {t_gen:.2f}s = {n/t_gen:.1f} tok/s")
    del llm
    torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("S3 pure-MoE quality A/B (Route A / P800 newline)")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"VLLM_FL_PREFER      = {os.environ.get('VLLM_FL_PREFER')}")
    print("=" * 70)
    model = os.environ.get("S3_MODEL", "/models/Qwen3-30B-A3B")
    enforce_eager = os.environ.get("S3_ENFORCE_EAGER", "1") == "1"
    impl = os.environ.get("S3_MOE_IMPL", "vendor")
    modes = ["vendor", "reference"] if impl == "both" else [impl]
    print(f"model={model} enforce_eager={enforce_eager} modes={modes}")
    for m in modes:
        print(f"\n----- run mode: {m} -----")
        _run(m, model, enforce_eager)
    print("\n A/B 完成")


if __name__ == "__main__":
    main()
