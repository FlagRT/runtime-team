#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
probe_graph_capture_stream.py — D5 graph capture 流语义验证（P1-④，A 线）
═══════════════════════════════════════════════════════════════════════════════

【背景】映射文档 D5 显式列了"graph capture 流语义"但从未覆盖。本探针验证：
  G1 capture 正确性：捕获 matmul → replay 结果与 eager 一致
  G2 replay 确定性：同输入两次 replay 结果逐位一致
  G3 输入更新：更新输入后 replay 使用新值（图内固定地址，值可变）
  G4 流语义：capture 内使用非默认流（torch.npu.stream 上下文）不破坏捕获；
             replay 后结果仍正确 —— 验证"capture 期间流切换"语义
  G5 capture 用流 vs eager 用流一致性

【用法】容器内 A 线环境：python3 probe_graph_capture_stream.py
【输出】stdout + graph_capture_stream_result.json，判定 GRAPH_CAPTURE_PASS/FAIL
【硬约束】A 线 torch_npu（不走 torch_fl）。
"""
import json
import torch
import torch_npu  # noqa: F401

DEV = "npu"


def main():
    print("=== probe_graph_capture_stream.py: D5 graph capture 流语义（P1-④）===")
    print(f"[env] torch_npu={getattr(torch_npu, '__version__', 'unknown')} devices={torch.npu.device_count()}")
    torch.zeros(1, device=DEV)  # 设备预热

    result = {"verdict": "FAIL", "checks": {}, "note": ""}
    n = 1024

    # 固定输入（避免随机影响确定性判定）
    x = torch.randn(n, n, device=DEV)
    w = torch.randn(n, n, device=DEV)
    eager_ref = (x @ w).sum().item()

    # ══════════ G1+G2+G3：标准 capture/replay 语义 ══════════
    g1_g2_g3_ok = True
    try:
        g = torch.npu.CUDAGraph() if hasattr(torch.npu, "CUDAGraph") else torch.npu.NPUGraph()
        # 预热 + 稳定内存池（CUDA graph 惯例：先跑一次 eager 定型）
        y = (x @ w).sum()
        torch.npu.synchronize()

        with torch.npu.graph(g):
            y = (x @ w).sum()
        torch.npu.synchronize()
        # CUDA graph 语义：capture 仅记录不执行，必须先 replay 才产生结果
        g.replay()
        torch.npu.synchronize()
        val1 = y.item()
        # G2：同输入再 replay 一次（确定性）
        g.replay()
        torch.npu.synchronize()
        val2 = y.item()
        # G3：换输入（写回同一地址，图内指针固定）后 replay
        x.copy_(torch.randn(n, n, device=DEV))
        g.replay()
        torch.npu.synchronize()
        val3 = y.item()

        rel1 = abs(val1 - eager_ref) / max(abs(eager_ref), 1.0)
        rel2 = abs(val1 - val2) / max(abs(val1), 1.0)
        # G3 校验：val3 应等于「新 x @ 原 w」的 eager 值
        eager3 = (x @ w).sum().item()
        rel3 = abs(val3 - eager3) / max(abs(eager3), 1.0)

        g1_g2_g3_ok = rel1 < 1e-3 and rel2 == 0.0 and rel3 < 1e-3
        print(f"[G1] capture vs eager: val={val1:.6f} ref={eager_ref:.6f} rel_err={rel1:.2e} "
              f"{'✅' if rel1 < 1e-3 else '❌'}")
        print(f"[G2] replay 确定性: val1={val1:.6f} val2={val2:.6f} 差={rel2:.2e} "
              f"{'✅' if rel2 == 0.0 else '❌'}")
        print(f"[G3] 输入更新后 replay: val3={val3:.6f} eager3={eager3:.6f} rel_err={rel3:.2e} "
              f"{'✅' if rel3 < 1e-3 else '❌'}")
        result["checks"]["G1_capture_correct"] = {"ok": rel1 < 1e-3, "detail": f"rel_err={rel1:.2e}"}
        result["checks"]["G2_replay_deterministic"] = {"ok": rel2 == 0.0, "detail": f"两次 replay 差={rel2:.2e}"}
        result["checks"]["G3_input_update_replay"] = {"ok": rel3 < 1e-3, "detail": f"rel_err={rel3:.2e}"}
    except Exception as e:
        g1_g2_g3_ok = False
        print(f"[G1-G3] 异常: {type(e).__name__}: {str(e)[:200]}")
        result["checks"]["G1_capture_correct"] = {"ok": False, "detail": str(e)[:150]}

    # ══════════ G4：capture 内流切换语义 ══════════
    g4_ok = True
    try:
        g4 = torch.npu.NPUGraph()
        s_cap = torch.npu.Stream()
        x4 = torch.randn(n, n, device=DEV)
        w4 = torch.randn(n, n, device=DEV)
        y4_ref = (x4 @ w4).sum().item()
        (x4 @ w4).sum()  # 预热
        torch.npu.synchronize()
        with torch.npu.graph(g4):
            # capture 期间切换到命名流执行（模拟 capture 内多流）
            with torch.npu.stream(s_cap):
                y4 = (x4 @ w4).sum()
        torch.npu.synchronize()
        g4.replay()
        torch.npu.synchronize()
        val4 = y4.item()
        rel4 = abs(val4 - y4_ref) / max(abs(y4_ref), 1.0)
        g4_ok = rel4 < 1e-3
        print(f"[G4] capture 内 stream 切换: val={val4:.6f} ref={y4_ref:.6f} rel_err={rel4:.2e} "
              f"{'✅' if g4_ok else '❌'}")
        result["checks"]["G4_capture_stream_semantics"] = {"ok": g4_ok, "detail": f"rel_err={rel4:.2e}"}
    except Exception as e:
        g4_ok = False
        print(f"[G4] 异常: {type(e).__name__}: {str(e)[:200]}")
        result["checks"]["G4_capture_stream_semantics"] = {"ok": False, "detail": str(e)[:150]}

    # ══════════ G5：capture 指定 stream 参数 ══════════
    g5_ok = True
    try:
        g5 = torch.npu.NPUGraph()
        s5 = torch.npu.Stream()
        x5 = torch.randn(n, n, device=DEV)
        w5 = torch.randn(n, n, device=DEV)
        y5_ref = (x5 @ w5).sum().item()
        (x5 @ w5).sum()
        torch.npu.synchronize()
        # 显式传 stream 参数（与默认流区分）
        with torch.npu.graph(g5, stream=s5):
            y5 = (x5 @ w5).sum()
        torch.npu.synchronize()
        g5.replay()
        torch.npu.synchronize()
        rel5 = abs(y5.item() - y5_ref) / max(abs(y5_ref), 1.0)
        g5_ok = rel5 < 1e-3
        print(f"[G5] capture 显式 stream 参数: rel_err={rel5:.2e} {'✅' if g5_ok else '❌'}")
        result["checks"]["G5_capture_explicit_stream"] = {"ok": g5_ok, "detail": f"rel_err={rel5:.2e}"}
    except Exception as e:
        g5_ok = False
        print(f"[G5] 异常: {type(e).__name__}: {str(e)[:200]}")
        result["checks"]["G5_capture_explicit_stream"] = {"ok": False, "detail": str(e)[:150]}

    all_ok = g1_g2_g3_ok and g4_ok and g5_ok
    result["verdict"] = "GRAPH_CAPTURE_PASS" if all_ok else "GRAPH_CAPTURE_FAIL"
    result["note"] = ("D5 graph capture 流语义验证：capture/replay/输入更新/流切换/显式流 5 项"
                      f"{'全部通过' if all_ok else '存在失败项'}")
    print(f"\n{result['verdict']}: {result['note']}")

    with open("graph_capture_stream_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已写入 graph_capture_stream_result.json")


if __name__ == "__main__":
    main()
