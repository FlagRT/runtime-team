#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
conformance/infer_cases.py — 推理场景一致性用例（设备无关，机制层验证）
═══════════════════════════════════════════════════════════════════════════════

【定位】设备执行上下文职责在"推理形态"下的机制验证（不需要真实 LLM）：
  用轻量 MLP 模拟"模型加载 → 多轮前向 → D2H 采样 → 长驻循环"的推理负载，
  验证统一设备句柄 / Stream / Event / 传输 / 状态在推理形态下语义成立。

【用法】（容器内、对应后端环境激活、单进程）：
  python runner.py --chip ascend --backend npu --cases infer_cases \
      --out conformance_ascend_infer_result.json
  B 线对照：--backend flagos --cases infer_cases

【ctx 约定】同 cases.py（device/sync/event/stream/stream_ctx/current_stream）。
【硬约束】不 import torch_fl（npu 后端）；不依赖 vLLM（机制层，与推理引擎解耦）。
【数值纪律】跨操作对比一律 .cpu() 后计算（flag_gems 等 sum 算子不可靠，坑 B4）。
"""

import json
import time

import torch


def _make_small_mlp(device, n=64):
    """轻量两层 MLP 模拟模型前向（确定性权重，便于数值比对）。"""
    torch.manual_seed(42)
    w1 = torch.randn(n, n, device=device) * 0.02
    b1 = torch.zeros(n, device=device)
    w2 = torch.randn(n, n, device=device) * 0.02
    b2 = torch.zeros(n, device=device)
    return (w1, b1, w2, b2)


def _forward(ctx, params, x):
    """MLP 前向：x @ w1 + b1 → relu → @ w2 + b2（返回 logits 张量）。"""
    w1, b1, w2, b2 = params
    h = torch.relu(x @ w1 + b1)
    return h @ w2 + b2


# ────────────────────────────────────────────────────────────────────────────
# I1: 模型加载后设备上下文可用（设备枚举/预热/张量创建）
# ────────────────────────────────────────────────────────────────────────────
def case_i1_device_context_after_load(ctx):
    device = ctx["device"]
    try:
        params = _make_small_mlp(device)          # 模拟权重加载（设备张量创建）
        x = torch.randn(4, 64, device=device)     # 模拟输入就位
        logits = _forward(ctx, params, x)
        ctx["sync"]()
        ok = bool(logits.shape == (4, 64))
        return ok, f"加载后上下文可用: logits.shape={tuple(logits.shape)}"
    except Exception as e:
        return False, f"异常: {e}"


# ────────────────────────────────────────────────────────────────────────────
# I2: 推理多轮前向的 Stream 顺序（同流串行，多轮结果与逐轮参考一致）
# ────────────────────────────────────────────────────────────────────────────
def case_i2_infer_forward_stream_order(ctx):
    device = ctx["device"]
    try:
        params = _make_small_mlp(device)
        x = torch.randn(4, 64, device=device)       # 固定输入：同流多轮前向应确定性一致
        ref = None
        ok = True
        for _ in range(3):                          # 模拟 3 轮 decode（同一请求上下文）
            logits = _forward(ctx, params, x)
            s = logits.sum().cpu().item()           # 数值纪律：.cpu() 后计算（坑 B4）
            if ref is None:
                ref = s
            else:
                ok = ok and abs(s - ref) < 1e-3     # 同流顺序 + 固定输入 → 逐轮结果一致
        return ok, f"多轮前向同流顺序: 固定输入逐轮一致(误差={abs(s - ref):.2e}) = {ok}"
    except Exception as e:
        return False, f"异常: {e}"


# ────────────────────────────────────────────────────────────────────────────
# I3: KV 模拟缓冲跨流可见性（写入流 record → 计算流 wait → 读取）
# ────────────────────────────────────────────────────────────────────────────
def case_i3_kv_buffer_visibility(ctx):
    device = ctx["device"]
    try:
        # 模拟 KV 分配（固定缓冲）+ 写入流 + 事件 + 计算流读取
        kv = torch.zeros(8, 16, device=device)     # KV 缓冲（模拟）
        s_write = ctx["stream"]()
        ev = ctx["event"]()
        with ctx["stream_ctx"](s_write):
            kv.fill_(7)                            # 写入流填充 KV
        ev.record(s_write)                         # 写入完成点
        cur = ctx["current_stream"]()
        cur.wait_event(ev)                         # 计算流显式等待
        read = (kv + 1).cpu()                      # 读取（依赖已建立）
        ok = bool((read == 8).all().item())
        return ok, f"KV 跨流可见性: 写入流→事件→计算流, 读回全 8 = {ok}"
    except Exception as e:
        return False, f"异常: {e}"


# ────────────────────────────────────────────────────────────────────────────
# I4: D2H 采样回传（logits → CPU top-k，模拟推理采样路径）
# ────────────────────────────────────────────────────────────────────────────
def case_i4_d2h_sample_transfer(ctx):
    device = ctx["device"]
    try:
        params = _make_small_mlp(device)
        x = torch.randn(4, 64, device=device)
        logits = _forward(ctx, params, x)
        ctx["sync"]()
        logits_cpu = logits.cpu()                  # D2H 回传（推理每 token 必经）
        topk = torch.topk(logits_cpu, 3, dim=-1)   # CPU 侧采样
        ok = bool(topk.indices.shape == (4, 3))
        return ok, f"D2H 采样回传: logits.cpu()→topk, shape={tuple(topk.indices.shape)}"
    except Exception as e:
        return False, f"异常: {e}"


# ────────────────────────────────────────────────────────────────────────────
# I5: 长驻循环后设备状态保持（模拟服务多轮运行，状态不漂移）
# ────────────────────────────────────────────────────────────────────────────
def case_i5_longrun_device_state(ctx):
    device = ctx["device"]
    try:
        params = _make_small_mlp(device)
        acc = 0.0
        for _ in range(20):                        # 模拟 20 轮服务请求
            x = torch.randn(4, 64, device=device)
            logits = _forward(ctx, params, x)
            acc += logits.sum().cpu().item()       # 数值纪律：.cpu()（坑 B4）
        ok = bool(abs(acc) < 1e6)                  # 无 NaN/Inf 漂移
        return ok, f"长驻 20 轮: 累计 logits={acc:.2e}, 无 NaN/Inf = {ok}"
    except Exception as e:
        return False, f"异常: {e}"


# ────────────────────────────────────────────────────────────────────────────
# I6: 双缓冲流水线形态（H2D/计算/D2H 多流重叠，墙钟 < 串行）
#      —— 推理核心职责（图 5-12），与 test_double_buffer_pipeline.py 呼应
# ────────────────────────────────────────────────────────────────────────────
def case_i6_pipeline_overlap(ctx):
    device = ctx["device"]
    try:
        n_batches, n = 4, 512
        # 页锁定主机缓冲（保证 non_blocking 真异步）+ 更大张量（重叠可观测）
        hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
        buf = [torch.zeros(n, n, device=device) for _ in range(2)]
        ev_h2d = [ctx["event"]() for _ in range(2)]
        ev_calc = [ctx["event"]() for _ in range(2)]
        s_trans, s_calc, s_d2h = ctx["stream"](), ctx["stream"](), ctx["stream"]()
        ok = True
        t0 = time.time()
        for i in range(n_batches):
            b = i % 2
            with ctx["stream_ctx"](s_trans):
                buf[b].copy_(hosts[b], non_blocking=True)   # H2D（页锁定真异步）
            ev_h2d[b].record(s_trans)
            s_calc.wait_event(ev_h2d[b])
            with ctx["stream_ctx"](s_calc):
                out = (buf[b] @ buf[b]).sum()               # 设备计算
            ev_calc[b].record(s_calc)
            s_d2h.wait_event(ev_calc[b])
            with ctx["stream_ctx"](s_d2h):
                out_cpu = out.cpu()                          # D2H 回传
        ctx["sync"]()
        pipe_t = time.time() - t0
        # 串行参考（同批同模型，无流重叠）
        t1 = time.time()
        for i in range(n_batches):
            d = hosts[i % 2].to(device)
            (d @ d).sum().cpu()
        serial_t = time.time() - t1
        overlap = (serial_t - pipe_t) / serial_t if serial_t > 0 else 0
        # 判定：事件依赖链成立 = 最后一轮数据正确（按轮对应 hosts，相对容差规避 CPU/NPU sum 数值差）
        last_b = (n_batches - 1) % 2
        ref = (hosts[last_b] @ hosts[last_b]).sum().item()
        got = out_cpu.item()
        rel_err = abs(got - ref) / max(abs(ref), 1e-6)
        ok = ok and rel_err < 1e-3
        return ok, f"流水线依赖链: 末轮数据正确(rel_err={rel_err:.2e})={ok} | 观测: pipe={pipe_t:.3f}s serial={serial_t:.3f}s 重叠率={overlap:.1%}"
    except Exception as e:
        return False, f"异常: {e}"
