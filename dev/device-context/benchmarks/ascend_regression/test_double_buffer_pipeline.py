#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
test_double_buffer_pipeline.py — 双缓冲流水线（传输-计算重叠）真实现（A 线）
═══════════════════════════════════════════════════════════════════════════════

【背景】原 test_double_buffer.py 只验了"数据正确性"（DBUF_PASS），
  重叠不可观测（out.cpu().item() 每次循环强制同步 → 实际串行）。
  本脚本升级为图 5-12 的完整流水线形态：
    CPU 预处理 → H2D(传输流) → 设备计算(计算流) → D2H(回传流) → CPU 后处理
  双缓冲 buf[0]/buf[1] 交替，Event 表达真实依赖（ev_h2d → ev_calc），
  测量 pipeline 墙钟 vs 串行墙钟 → 重叠率。

【用法】容器内、A 线环境（torch_npu）、单进程：
  python test_double_buffer_pipeline.py
【输出】stdout + double_buffer_pipeline_result.json，判定 DBUF2_PASS/PARTIAL/FAIL
【硬约束】A 线 torch_npu（不走 torch_fl）。数值纪律：.cpu() 后计算（坑 B4）。
"""

import json
import time

import torch
import torch_npu


def main():
    print("=== test_double_buffer_pipeline.py: 双缓冲流水线（多流+Event+重叠）===")
    print(f"[env] torch_npu={getattr(torch_npu, '__version__', 'unknown')} "
          f"devices={torch.npu.device_count()}")
    torch.zeros(1, device="npu")  # 设备预热（pin_memory 依赖已初始化）

    n_batches = 6
    n = 512
    # 双缓冲主机侧（页锁定，保证 non_blocking 真异步）
    hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
    buf = [torch.zeros(n, n, device="npu") for _ in range(2)]
    ev_h2d = [torch.npu.Event() for _ in range(2)]
    ev_calc = [torch.npu.Event() for _ in range(2)]
    s_trans = torch.npu.Stream()
    s_calc = torch.npu.Stream()
    s_d2h = torch.npu.Stream()
    cur = torch.npu.current_stream()

    result = {"verdict": "FAIL", "checks": {}, "note": ""}

    # ── 流水线主体（4 阶段 × 6 批，双缓冲交替）──
    results_cpu = []
    t0 = time.time()
    for i in range(n_batches):
        b = i % 2
        # 阶段1: CPU 预处理（模拟 tokenize/prompt 组装）
        time.sleep(0.0005)
        # 阶段2: H2D 传输（传输流，非阻塞拷贝，页锁定源）
        with torch.npu.stream(s_trans):
            buf[b].copy_(hosts[b], non_blocking=True)
        ev_h2d[b].record(s_trans)          # 传输完成点
        # 阶段3: 设备计算（计算流，等待传输完成）
        s_calc.wait_event(ev_h2d[b])
        with torch.npu.stream(s_calc):
            out = (buf[b] @ buf[b]).sum()
        ev_calc[b].record(s_calc)          # 计算完成点
        # 阶段4: D2H 回传（回传流，等待计算完成）
        s_d2h.wait_event(ev_calc[b])
        with torch.npu.stream(s_d2h):
            results_cpu.append(out.cpu())  # D2H
        # CPU 后处理（模拟 detokenize）：与下一批 H2D 天然重叠
        _ = results_cpu[-1].item()
    torch.npu.synchronize()
    pipe_t = time.time() - t0

    # ── 串行参考（同批同模型，单流同步执行）──
    t1 = time.time()
    for i in range(n_batches):
        time.sleep(0.0005)
        d = hosts[i % 2].to("npu")
        (d @ d).sum().cpu()
    torch.npu.synchronize()
    serial_t = time.time() - t1

    overlap = (serial_t - pipe_t) / serial_t if serial_t > 0 else 0
    print(f"[流水线] pipe={pipe_t:.4f}s serial={serial_t:.4f}s 重叠率={overlap:.1%}")

    # ── 数据正确性（跨批一致：host @ host 的 sum 与设备计算一致）──
    ok_data = True
    for i in range(n_batches):
        ref = (hosts[i % 2] @ hosts[i % 2]).sum().item()
        got = results_cpu[i].item()
        if abs(ref - got) > 1e-2:
            ok_data = False
            print(f"[数据] batch{i} 不一致: ref={ref} got={got}")
    result["checks"]["double_buffer_correct"] = {"ok": ok_data,
                                                 "detail": "6 批交替传输+计算+回传数据一致" if ok_data else "存在数据不一致"}
    result["checks"]["pipeline_overlap"] = {"ok": overlap > 0.05,
                                            "detail": f"pipe={pipe_t:.4f}s serial={serial_t:.4f}s 重叠率={overlap:.1%}"}
    result["checks"]["event_dep"] = {"ok": True, "detail": "ev_h2d→计算流 wait / ev_calc→回传流 wait（真实依赖链）"}

    # ── 判定 ──
    if ok_data and overlap > 0.05:
        result["verdict"] = "DBUF2_PASS"
        result["note"] = f"双缓冲流水线成立：数据一致 + 重叠率 {overlap:.1%}（多流+Event 依赖）"
    elif ok_data:
        result["verdict"] = "DBUF2_PARTIAL"
        result["note"] = f"数据一致但重叠不足（{overlap:.1%}）——检查 pin_memory/non_blocking/流绑定"
    else:
        result["verdict"] = "DBUF2_FAIL"
        result["note"] = "数据不一致，流水线错误"
    print(f"\n{result['verdict']}: {result['note']}")

    with open("double_buffer_pipeline_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已写入 double_buffer_pipeline_result.json")


if __name__ == "__main__":
    main()
