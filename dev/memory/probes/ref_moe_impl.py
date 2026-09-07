"""纯 torch 参考 fused_experts_impl —— 纯 MoE 质量 A/B 用（新线容器注入版）

签名对齐 vllm_fl.ops.fused_moe.fused_moe.fused_experts_impl / 厂商 wrapper。
仅支持非量化 bf16/fp16 + silu; 与厂商内核同样在最终阶段应用路由权重 (scatter-sum)。

实现形态: numpy 排序索引（CPU 侧） + 逐 expert 2D mm（GPU 侧, 与稠密推理 QKV 投影同路径）。
规避的坑（实测 2026-08-22→09-01, 均为 EngineCore 上下文内 flag_gems triton 内核问题）:
  - batch-gather einsum/bmm: flag_gems bmm autotuner ZeroDivisionError（CUDA event 计时返 0）
  - mask.nonzero: flag_gems nonzero triton 内核 error 719 launch failure
  - batch-gather w1[use_ids] 全量展开: profile_run(16384 tokens) 下 OOM(768GiB)
CPU 侧 numpy 做 argsort/bincount 计算每个 expert 的 token 索引, GPU 侧只调
gather / 2D matmul / index_add_ —— 全部是引擎内已验证路径。

注入方式（EngineCore 是子进程, 主进程 monkeypatch 不生效, 须改插件源码）:
  在容器内把本文件的 ref_fused_experts_impl 注入到
  /workspace/vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/kunlunxin/impl/fused_moe/fused_moe.py
  的 fused_experts_impl 函数体最前面（return _ref_moe(...) 直通），
  容器内改前先 cp 备份 .orig_bak，测完恢复。
首次调用打印 [REF-MOE] 标记, 用于在 EngineCore 日志中确认真实生效。
"""
import numpy as np
import torch

_printed = False


def _marker():
    global _printed
    if not _printed:
        print("[REF-MOE] fused_experts -> pure-torch reference impl (numpy-sort + per-expert mm)", flush=True)
        _printed = True


def ref_fused_experts_impl(
    hidden_states, w1, w2, topk_weights, topk_ids,
    inplace=False, activation="silu", apply_router_weight_on_input=False,
    use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
    use_int4_w4a16=False, per_channel_quant=False, global_num_experts=-1,
    expert_map=None, w1_scale=None, w2_scale=None, w1_zp=None, w2_zp=None,
    a1_scale=None, a2_scale=None, block_shape=None, w1_bias=None, w2_bias=None,
):
    _marker()
    if use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16:
        raise NotImplementedError("REF-MOE: quantized path not supported in A/B probe")
    if apply_router_weight_on_input:
        raise NotImplementedError("REF-MOE: apply_router_weight_on_input=True not supported")
    if activation != "silu":
        raise NotImplementedError(f"REF-MOE: only silu supported, got {activation}")

    num_tokens, hidden_dim = hidden_states.shape
    topk = topk_ids.shape[1]
    ffn_hd = w1.shape[1] // 2
    dev = hidden_states.device
    E = w1.shape[0]  # 本地 expert 数 (TP=1 时 = 全局)

    flat = topk_ids.reshape(-1)                       # [R=M*topk] 全局 expert id
    weights = topk_weights.reshape(-1)                # [R]

    local_map = None
    if expert_map is not None:
        local_map = expert_map.cpu().numpy()

    flat_np = flat.cpu().numpy()
    order = np.argsort(flat_np, kind="stable")
    counts = np.bincount(flat_np, minlength=E)
    offsets = np.concatenate([[0], np.cumsum(counts)])

    out = torch.zeros(num_tokens, hidden_dim, dtype=hidden_states.dtype, device=dev)
    for e in range(E):
        n = int(counts[e])
        if n == 0:
            continue
        local_e = e if local_map is None else int(local_map[e])
        if local_e < 0:
            continue
        rows = order[offsets[e]:offsets[e + 1]]
        tok = torch.from_numpy(rows // topk).to(dev)
        kk = torch.from_numpy(rows % topk).to(dev)
        xe = hidden_states[tok]                       # [n, H]
        w1e = w1[local_e]                             # [2F, H]
        g = xe @ w1e[:ffn_hd].T                       # [n, F]  gate
        u = xe @ w1e[ffn_hd:].T                       # [n, F]  up
        inter = torch.nn.functional.silu(g) * u       # SwiGLU
        ye = inter @ w2[local_e].T                    # w2[local_e]: [H, F] -> [n, H]
        w = weights[torch.from_numpy(rows).to(dev)].unsqueeze(-1)
        out.index_add_(0, tok, ye * w)
    return out
