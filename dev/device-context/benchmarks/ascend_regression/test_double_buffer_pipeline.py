#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
test_double_buffer_pipeline.py — 双缓冲流水线（传输-计算重叠）真实现 v2（A 线）
═══════════════════════════════════════════════════════════════════════════════

【背景】原 v1 只实现 V0（每批 2 record + 2 wait 的完整流水线），重叠率在小负载下
  恒为负，且只有单一实现无法按负载取舍。

【v2 变更（2026-09-02 P0-① D8 落地）】把 §5.2 求解结论（test_dbuf_variants_rigorous.py）
  变成真实实现：
  1. 四种执行模式全部真实现（可独立运行、可切换）：
     - V0 基线      ：每批 H2D→计算→D2H 完整 event 依赖（12 次同步）
     - V1 精简链    ：只保留 H2D→计算 依赖，D2H 末尾统一 wait_stream（6 次）
     - V4 批量提交  ：先全部 H2D → 1 次同步 → 全部计算 → 1 次同步 → 全部 D2H（2 次）
     - V5 同流顺序  ：同一流顺序执行（无流水线，小负载下界参照）
  2. 按负载自动选型（§5.2 工程指引）：
     - 每批计算 < ~1ms（n<=512） → 选 V5（不流水线，同流顺序最快）
     - 转折点附近（n≈1024）     → 选 V1 / V4（按实测取最快）
     - 大负载（n>=2048）        → 选 V0 / V1（保留批间流水，+30% 以上）
     - 实现方式：四模式实测取中位耗时，快者胜出；并输出规则建议供对照
  3. 判定拆分（§4.1 纪律：功能与性能分开标注）：
     - double_buffer_correct（功能）：各模式 D2H 结果与主机参考逐批一致
     - mode_selection（性能/选型）  ：各模式相对「同步拷贝+纯计算」基准的重叠率 +
       选中模式与理由；流水线模式重叠率为正 或 小负载正确降级为 V5 均判合理

【测量纪律】（§5.2 教训，本脚本强制）：
  1. 每规模独立预热（跨规模复用预热 = 系统性偏差）
  2. rounds>=5 取中位（ms 级测量噪声极大）
  3. 重叠率分母 = 「同步拷贝 + 纯计算」实测（不与其他实现互作分母）

【用法】容器内、A 线环境（torch_npu）、单进程：
  python test_double_buffer_pipeline.py                 # 默认 n=1024 单档选型
  python test_double_buffer_pipeline.py --scan          # 512,1024,2048 三档扫描
  python test_double_buffer_pipeline.py --size 2048     # 指定负载档
【输出】stdout + double_buffer_pipeline_result.json，判定 DBUF2_PASS/PARTIAL/FAIL
【硬约束】A 线 torch_npu（不走 torch_fl）。数值纪律：.cpu() 后计算（坑 B4）。
"""

import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401  # 注册 npu 后端

DEV = "npu"
BATCHES = 6


# ════════════════════════════ 计算核（单变量：所有模式同一算子）════════════════
def _calc(buf):
    """设备计算核：矩阵乘 + 归约（模拟一层前向）。"""
    return (buf @ buf).sum()


# ════════════════════════════ 四种执行模式（真实现）══════════════════════════
def mode_v0(hosts, bufs, ev_h2d, ev_calc, s_t, s_c, s_d):
    """V0 基线：每批 H2D→计算→D2H 完整 event 依赖链（2 record + 2 wait / 批）。"""
    outs = [None] * BATCHES
    for i in range(BATCHES):
        b = i % 2
        with torch.npu.stream(s_t):
            bufs[b].copy_(hosts[b], non_blocking=True)   # H2D（页锁定真异步）
        ev_h2d[b].record(s_t)
        s_c.wait_event(ev_h2d[b])
        with torch.npu.stream(s_c):
            outs[i] = _calc(bufs[b])
        ev_calc[b].record(s_c)
        s_d.wait_event(ev_calc[b])
        with torch.npu.stream(s_d):
            _ = outs[i].to("cpu", non_blocking=True)      # D2H
    return outs


def mode_v1(hosts, bufs, ev, s_t, s_c, s_d):
    """V1 精简链：只保留 H2D→计算 依赖，D2H 末尾统一 wait_stream（§5.2 结论 2）。"""
    outs = [None] * BATCHES
    for i in range(BATCHES):
        b = i % 2
        with torch.npu.stream(s_t):
            bufs[b].copy_(hosts[b], non_blocking=True)
        ev[b].record(s_t)
        s_c.wait_event(ev[b])
        with torch.npu.stream(s_c):
            outs[i] = _calc(bufs[b])
    # 计算流全部完成后统一回传：去掉「计算→D2H」每批过度同步
    s_d.wait_stream(s_c)
    with torch.npu.stream(s_d):
        for o in outs:
            _ = o.to("cpu", non_blocking=True)
    return outs


def mode_v4(hosts, bufs, ev, s_t, s_c, s_d):
    """V4 批量提交：同步次数压到 2（全部 H2D→1 次同步→全部计算→1 次同步→全部 D2H）。"""
    for i in range(BATCHES):
        with torch.npu.stream(s_t):
            bufs[i % 2].copy_(hosts[i % 2], non_blocking=True)
    ev[0].record(s_t)
    s_c.wait_event(ev[0])
    outs = [None] * BATCHES
    with torch.npu.stream(s_c):
        for i in range(BATCHES):
            outs[i] = _calc(bufs[i % 2])
    ev[1].record(s_c)
    s_d.wait_event(ev[1])
    with torch.npu.stream(s_d):
        for o in outs:
            _ = o.to("cpu", non_blocking=True)
    return outs


def mode_v5(hosts, bufs, *_):
    """V5 同流顺序：同一流顺序执行（无流水线，小负载下界参照，§5.2 结论 1）。"""
    outs = [None] * BATCHES
    for i in range(BATCHES):
        b = i % 2
        bufs[b].copy_(hosts[b], non_blocking=True)
        outs[i] = _calc(bufs[b])
    torch.npu.synchronize()
    for o in outs:
        _ = o.to("cpu", non_blocking=True)
    return outs


MODES = {"V0": mode_v0, "V1": mode_v1, "V4": mode_v4, "V5": mode_v5}
# 规则建议（§5.2 工程指引，按负载特征）
RULE_SUGGEST = lambda n: ("V5", "每批计算<~1ms，不流水线最快") if n <= 512 else \
                        ("V1", "转折点附近，精简链或压同步次数") if n <= 1536 else \
                        ("V0", "大负载，保留批间流水拿 +30%")


def _run_once(fn, hosts, bufs, streams):
    """单次执行：返回墙钟与各批 D2H 标量结果（结果通过 .item() 收集保证同步）。"""
    outs = fn(hosts, bufs, *streams)
    torch.npu.synchronize()
    # 收集设备结果到 CPU 标量（out 是 0-dim tensor 的 .to("cpu") 结果）
    return [o.item() for o in outs]


def _median_of(fn, hosts, bufs, streams, rounds):
    """每规模独立预热 + rounds 取中位（测量纪律 1/2）。"""
    fn(hosts, bufs, *streams)          # 该规模专用预热（首次算子开销可达百倍）
    torch.npu.synchronize()
    times, last = [], None
    for _ in range(rounds):
        torch.npu.synchronize()
        t0 = time.time()
        last = _run_once(fn, hosts, bufs, streams)
        times.append(time.time() - t0)
    return statistics.median(times), last


def _baseline(hosts, bufs, rounds):
    """无重叠基准 = 同步拷贝 + 纯计算（重叠率分母，测量纪律 3）。"""
    def serial():
        for i in range(BATCHES):
            d = hosts[i % 2].to(DEV)   # 同步拷贝
            _calc(d).item()
    serial()                           # 预热
    torch.npu.synchronize()
    times = []
    for _ in range(rounds):
        torch.npu.synchronize()
        t0 = time.time()
        serial()
        times.append(time.time() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024, help="负载档（张量边长）")
    ap.add_argument("--scan", action="store_true", help="扫描 512,1024,2048 三档")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default="double_buffer_pipeline_result.json")
    args = ap.parse_args()

    print("=== test_double_buffer_pipeline.py v2: 四模式 + 按负载选型 ===")
    print(f"[env] torch_npu={getattr(torch_npu, '__version__', 'unknown')} "
          f"devices={torch.npu.device_count()}")
    torch.zeros(1, device=DEV)  # 设备预热（pin_memory 依赖已初始化）

    sizes = [512, 1024, 2048] if args.scan else [args.size]
    result = {"verdict": "FAIL", "checks": {}, "sizes": {}, "note": ""}
    all_ok_data, all_ok_sel = True, True

    for n in sizes:
        print(f"\n───── 负载档 n={n}（BATCHES={BATCHES}, rounds={args.rounds}）─────")
        hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
        bufs = [torch.zeros(n, n, device=DEV) for _ in range(2)]
        ev2 = [torch.npu.Event() for _ in range(2)]       # v1/v4 共用
        ev_h2d = [torch.npu.Event() for _ in range(2)]
        ev_calc = [torch.npu.Event() for _ in range(2)]
        s_t, s_c, s_d = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
        streams_v0 = (ev_h2d, ev_calc, s_t, s_c, s_d)
        streams_v1 = (ev2, s_t, s_c, s_d)
        streams_v4 = (ev2, s_t, s_c, s_d)
        streams_v5 = ()

        # 无重叠基准
        base = _baseline(hosts, bufs, args.rounds)
        print(f"  [基准] 同步拷贝+纯计算 = {base*1000:.3f} ms")

        # 四模式实测
        timings, datas = {}, {}
        for name, fn in MODES.items():
            streams = {"V0": streams_v0, "V1": streams_v1, "V4": streams_v4, "V5": streams_v5}[name]
            t, outs = _median_of(fn, hosts, bufs, streams, args.rounds)
            timings[name] = t
            datas[name] = outs
            ovl = (base - t) / base if base > 0 else 0
            flag = "✅" if ovl > 0.05 else ("○" if ovl > 0 else "❌")
            print(f"  [{name}] {t*1000:7.3f} ms  相对基准重叠率 {ovl:+.1%} {flag}")

        # 主机参考（数据一致性基准，坑 B4：.cpu() 后计算）
        refs = [(hosts[i % 2] @ hosts[i % 2]).sum().item() for i in range(BATCHES)]

        # 按负载选型：实测最快者为准；规则建议仅作方向性参照（§5.2 区间非单点）
        best = min(timings, key=timings.get)
        rule, rule_why = RULE_SUGGEST(n)
        # 规则区间内的备选（V1/V4 同属"转折点压同步"策略，视为一致）
        rule_family = {"V1", "V4"} if rule in ("V1", "V4") else {rule}
        sel_match = "一致" if best in rule_family else f"规则建议{rule}，实测{best}更优"
        sel_reason = (f"四模式实测最快={best}（{timings[best]*1000:.3f}ms）；"
                      f"§5.2 规则建议={rule}（{rule_why}）")
        print(f"  [选型] 实测最快 {best} ｜ 规则建议 {rule} ｜ {sel_match}")
        print(f"         {sel_reason}")

        # 数据一致性（所有模式逐批对照；用相对误差——主机/设备不同累加顺序，
        # 绝对差随 n 增长，rel_err < 1e-3 即一致，参考 i6 rel_err 2e-7 同量级更严）
        ok_data = True
        for name, outs in datas.items():
            bad = []
            for i in range(BATCHES):
                rel = abs(refs[i] - outs[i]) / max(abs(refs[i]), 1.0)
                if rel > 1e-3:
                    bad.append(i)
            if bad:
                ok_data = False
                print(f"  [数据] {name} batch{list(bad)[:3]} 不一致: "
                      f"ref={refs[bad[0]]:.6f} got={outs[bad[0]]:.6f} "
                      f"rel_err={abs(refs[bad[0]]-outs[bad[0]])/max(abs(refs[bad[0]]),1.0):.2e}")
        # 选型合理性：流水线模式为正 或 小负载正确降级 V5
        ovl_best = (base - timings[best]) / base if base > 0 else 0
        if best == "V5" and n <= 512:
            ok_sel = True
            sel_note = f"小负载（n={n}）正确降级为同流顺序执行，避免流水线纯亏"
        else:
            ok_sel = ovl_best > 0.05
            sel_note = f"选中 {best} 相对基准重叠率 {ovl_best:+.1%}"
        print(f"  [判定] 数据一致={ok_data} 选型合理={ok_sel}（{sel_note}）")

        result["sizes"][str(n)] = {
            "baseline_ms": round(base * 1000, 3),
            "modes": {k: {"ms": round(v * 1000, 3),
                          "overlap": round((base - v) / base * 100, 1) if base else 0}
                      for k, v in timings.items()},
            "selected": best,
            "rule_suggest": rule,
            "selection_reason": sel_reason,
            "data_correct": ok_data,
            "selection_ok": ok_sel,
            "selection_note": sel_note,
        }
        all_ok_data &= ok_data
        all_ok_sel &= ok_sel

    # ── 总判定（功能 + 性能分开标注，§4.1 纪律）──
    result["checks"]["double_buffer_correct"] = {
        "ok": all_ok_data,
        "detail": f"全部 {len(sizes)} 档 × 4 模式 D2H 结果与主机参考一致" if all_ok_data else "存在数据不一致"}
    result["checks"]["mode_selection"] = {
        "ok": all_ok_sel,
        "detail": "各档按负载选型合理（流水线重叠为正 或 小负载正确降级 V5）" if all_ok_sel else "选型未达预期"}
    if all_ok_data and all_ok_sel:
        result["verdict"] = "DBUF2_PASS"
        result["note"] = (f"v2 双缓冲落地：四模式实现 + 按负载选型。"
                          f"{'扫描档位: ' + ','.join(map(str, sizes)) if args.scan else f'n={sizes[0]}'} "
                          f"数据一致且选型合理（详见各档 selection_note）")
    elif all_ok_data:
        result["verdict"] = "DBUF2_PARTIAL"
        result["note"] = "数据一致但选型/重叠未达预期"
    else:
        result["verdict"] = "DBUF2_FAIL"
        result["note"] = "数据不一致，流水线错误"
    print(f"\n{result['verdict']}: {result['note']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
