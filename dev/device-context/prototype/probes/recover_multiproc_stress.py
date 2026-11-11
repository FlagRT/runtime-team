#!/usr/bin/env python3
"""A2 · 多卡多进程「设备级恢复（`mode="real"`）」压测（**芯片无关**，按 `--backend` 复用）。

═══════════════════════════════════════════════════════════════════════════════
【它回答什么问题】
  现网默认走 `mode="probe"`（进程内安全）。真重建（`destroyContext → aclrtResetDevice →
  setDevice → 重建`）在**多 rank 并发**下会不会互扰 —— 这一条**此前没有生产级证据**
  （`DEVICE_CONTEXT_INFERENCE_MAPPING_20260831.md` 明写"real/hybrid 在生产默认启用前
  需压力测试验证多卡并发恢复"）。本脚本就是那个压测。

【判定（沿用 09-02 的 S1/S2/S3 口径）】
  S1 恢复者：`recovered=True` + 契约**五键** + **context 三键**取值正确
             （`context_recreated` 必须**为 True** —— 真重建必须被如实声明）
  S2 隔离性：**其余 rank 完全不受影响** —— 逐轮
             · 逐位相同的确定性 digest 不变（同输入、整数可精确表示的 fp32 运算）
             · `device_state()` 仍 `available`
             · `context_query()` 快照逐键不变
             · `probe_device()` 仍 True、本轮无异常
  S3 重建后可继续：重置后重新 `set_device` + 再建流 + 重算 digest 正确
  ≥ `--rounds`（默认 30）轮，零失败

【⚠️ 前置条件与边界（如实声明，不外推）】
  ① **ISOLATED 是"构造"出来的**：经**公开入口** `runtime.set_device_state(dev, "isolated")`
     驱动本层四态账本（2026-09-29 补的**只增**入口 —— 在此之前公开面无法置隔离，
     `mode="real"` 在健康设备上**永远走"无需重建"分支**，本脚本过去只能用内部模块构造）。
     ⇒ 这**不是**真实 L4 硬件故障；"真实故障触发链"不在本脚本结论范围内。
  ② 只验证**同机多卡多进程**（同厂商），不涉及多机。
  ③ `digest` 用整数可精确表示的 fp32 运算 ⇒ 比较是**逐位相同**而非容差近似。
  ④ 结论只在本次档位/环境（镜像、驱动、卡数）成立。

【用法】
  DC_ROOT=<prototype> python3 probes/recover_multiproc_stress.py \
      --backend ascend --devices 0,1,2 --rounds 30 --out /path/to/out_dir
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))            # prototype/probes
ROOT = os.environ.get("DC_ROOT") or os.path.dirname(_HERE)     # prototype

N_SIDE = 64


def _expected_digest() -> int:
    """纯 Python 复算期望值（与设备侧**同一算式**），用于"算得对"而非仅"没变化"。

    `A = tril(ones)` ⇒ `A[i][j] = 1 当且仅当 i >= j` ⇒ `y[j] = Σ_{i>=j} x[i] = Σ_{i=j}^{N-1}(i+1)`。
    （等价闭式：`Σ_{k=1..N} k² = N(N+1)(2N+1)/6`，N=64 ⇒ **89440**。2026-09-29 首次跑时这里写成了
    上三角约定（`i<=j`），得 45760 —— 与设备实测 89440 不符，**说明"算得对"这条判据是有牙齿的**；
    设备侧三 rank 一致给出 89440 ⇒ 错的是复算公式，已修。）
    """
    total = 0
    for j in range(N_SIDE):
        total += sum(i + 1 for i in range(j, N_SIDE))
    assert total == N_SIDE * (N_SIDE + 1) * (2 * N_SIDE + 1) // 6, "复算式与闭式不一致"
    return total


def _child_worker(rank: int, dev: int, ranks: int, rounds: int, backend: str,
                  root: str, out: dict, barrier) -> None:      # noqa: C901
    """一个 rank = 一个进程 = 一张卡。"""
    sys.path.insert(0, root)
    res = {"rank": rank, "device": dev, "rounds": [], "errors": []}
    try:
        import torch
        import runtime
        rt = runtime.use(backend)
        rt.set_device(dev)

        def _tok(st):
            """设备状态 → 对外 token（契约：`DeviceState` 的 `.value` 即 token）。"""
            return getattr(st, "value", None) or str(st).split(".")[-1].lower()

        DEV = rt.device_type

        def make_tensors():
            """下三角 mask × 1..64：所有中间值都是**整数**且远小于 2^24
            ⇒ fp32 可精确表示，**与归约顺序无关** ⇒ 比较是"逐位相同"而非容差近似。"""
            a = torch.tril(torch.ones(N_SIDE, N_SIDE, dtype=torch.float32, device=DEV))
            x = torch.arange(1, N_SIDE + 1, dtype=torch.float32, device=DEV)
            return x, a

        def compute_digest(x, a):
            rt.synchronize(dev)
            return int((x @ a).sum().item())

        # ⭐ 非恢复者**整轮复用同一批张量**（不重建）：若本进程的上下文被他人重建波及，
        #   这批张量就会失效/失真 ⇒ digest 逐轮不变 = 比"重新算一遍"更强的"未受影响"证据。
        xs, A = make_tensors()

        def snapshot():
            st = _tok(rt.device_state(dev))
            try:
                cq = dict(rt.context_query())
            except Exception as e:                                # noqa: BLE001
                cq = {"error": f"{type(e).__name__}: {e}"[:120]}
            return {"state": st, "context_query": cq,
                    "probe": bool(rt.probe_device(dev))}

        expected = _expected_digest()
        base_digest = compute_digest(xs, A)
        base_snap = snapshot()
        res["expected_digest"] = expected
        res["base_digest"] = base_digest
        res["base_snapshot"] = base_snap
        res["declared_recovery_real"] = bool(rt.supports("recovery_real"))

        for r in range(rounds):
            barrier.wait()
            victim = r % ranks
            row = {"round": r, "victim_rank": victim, "i_am_victim": (rank == victim)}
            try:
                if rank == victim:
                    pre = snapshot()
                    # ⭐ 经**公开入口**构造隔离（2026-09-29 补的只增入口）
                    rt.set_device_state(dev, "isolated",
                                        f"a2-stress r{r}: 构造隔离（公开入口）")
                    rec = rt.recover_device(dev, mode="real",
                                            reason=f"a2-stress r{r}: 多卡并发 real 重建")
                    row["recover"] = rec
                    row["pre_snapshot"] = pre
                    # real 会重置本进程默认上下文 ⇒ 必须重新 set_device 才能继续用
                    rt.set_device(dev)
                    _s = rt.create_stream()                        # 重建后可再建流
                    row["stream_after"] = type(_s).__name__
                    # ⚠️ reset 已释放默认上下文 ⇒ 旧张量作废，必须**重新分配**后才能继续算。
                    #   （此处**不**去碰旧张量：无法预判是抛错还是读出垃圾/挂死，属未做项，如实标注。）
                    xs, A = make_tensors()
                    row["digest"] = compute_digest(xs, A)
                    row["post_state"] = _tok(rt.device_state(dev))
                else:
                    row["digest"] = compute_digest(xs, A)
                    row["snapshot"] = snapshot()
            except Exception as e:                                # noqa: BLE001
                row["exception"] = f"{type(e).__name__}: {e}"[:200]
                row["traceback"] = traceback.format_exc()[-800:]
            barrier.wait()
            res["rounds"].append(row)
        # 注：状态机的 `last_transition` 事件流**未公开**，此处不采集（逐轮 post_state 已足够）；
        #     如需事件订阅，属接口面扩张，另行裁定。
        res["uses_public_api"] = True
        res["ok"] = True
    except Exception as e:                                        # noqa: BLE001
        res["errors"].append(f"{type(e).__name__}: {e}"[:300])
        res["traceback"] = traceback.format_exc()[-1500:]
        res["ok"] = False
    finally:
        out[f"rank{rank}"] = res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("DC_BACKEND", "ascend"))
    ap.add_argument("--devices", default="0,1,2", help="逗号分隔的**容器内可见设备序号**")
    ap.add_argument("--ranks", type=int, default=0, help="0 = 取设备数")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(os.path.abspath("."), "out_a2"))
    ap.add_argument("--max-seconds", type=int, default=1800, help="看门狗：超时强制 dump 退出")
    args = ap.parse_args()

    devs = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    ranks = args.ranks or len(devs)
    assert ranks <= len(devs), "ranks 不得超过给定设备数"

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== A2 多卡多进程 real 恢复压测：backend={args.backend} "
          f"devices={devs[:ranks]} ranks={ranks} rounds={args.rounds} ===")
    print("[前置声明] ISOLATED 由**公开入口** `runtime.set_device_state(dev, \"isolated\")` 构造"
          "（本层四态账本）；非真实硬件 L4 故障。")

    sys.path.insert(0, ROOT)
    import runtime as _rt
    bk = _rt.use(args.backend)
    if not bk.supports("recovery_real"):
        print(f"[如实跳过] backend={args.backend} **未声明 `recovery_real`** "
              f"⇒ A2（real 重建压测）对本后端**不适用**。"
              f"这不计为失败（能力缺失已如实声明）。")
        print("A2_BOUNDARY: SKIP_UNSUPPORTED —— 未做任何设备操作。")
        return 3

    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    out = mgr.dict()
    barrier = ctx.Barrier(ranks)
    procs = []
    faulthandler.dump_traceback_later(args.max_seconds, exit=True)
    t0 = time.time()
    for r in range(ranks):
        p = ctx.Process(target=_child_worker,
                        args=(r, devs[r], ranks, args.rounds, args.backend, ROOT, out, barrier),
                        name=f"rank{r}")
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=args.max_seconds)
    faulthandler.cancel_dump_traceback_later()
    elapsed = round(time.time() - t0, 1)

    results = {k: out[k] for k in sorted(out.keys())}
    res = {"backend": args.backend, "devices": devs[:ranks], "ranks": ranks,
           "rounds": args.rounds, "elapsed_s": elapsed, "per_rank": results}

    # ── 判定 ──
    expected = _expected_digest()
    FIVE = ("ordinal", "mode", "recovered", "state", "detail")
    CTX3 = ("context_supported", "context_count", "context_recreated")
    STATES = ("available", "degraded", "isolated", "destroyed")

    s1_bad, s2_bad, s3_bad, other_bad = [], [], [], []
    victim_rounds = 0
    for rname, rres in results.items():
        if not rres.get("ok"):
            other_bad.append(f"{rname} 进程级失败：{rres.get('errors')}")
            continue
        if rres.get("base_digest") != expected:
            other_bad.append(f"{rname} 基线 digest={rres.get('base_digest')} 期望 {expected}")
        for row in rres["rounds"]:
            r = row["round"]
            if row.get("exception"):
                other_bad.append(f"{rname} r{r} 异常：{row['exception']}")
                continue
            if row["i_am_victim"]:
                victim_rounds += 1
                rec = row.get("recover") or {}
                miss = [k for k in FIVE + CTX3 if k not in rec]
                if miss:
                    s1_bad.append(f"{rname} r{r} 缺键 {miss}")
                else:
                    if rec.get("recovered") is not True:
                        s1_bad.append(f"{rname} r{r} recovered={rec.get('recovered')!r}")
                    if rec.get("mode") != "real":
                        s1_bad.append(f"{rname} r{r} mode={rec.get('mode')!r}")
                    if rec.get("ordinal") != rres["device"]:
                        s1_bad.append(f"{rname} r{r} ordinal={rec.get('ordinal')!r}")
                    if rec.get("state") not in STATES:
                        s1_bad.append(f"{rname} r{r} state={rec.get('state')!r}")
                    if rec.get("context_recreated") is not True:
                        s1_bad.append(f"{rname} r{r} context_recreated={rec.get('context_recreated')!r}"
                                      "（real 真重建必须如实声明）")
                    if rec.get("context_supported") is not True:
                        s1_bad.append(f"{rname} r{r} context_supported={rec.get('context_supported')!r}")
                    if not isinstance(rec.get("context_count"), int):
                        s1_bad.append(f"{rname} r{r} context_count={rec.get('context_count')!r}")
                if row.get("digest") != expected:
                    s3_bad.append(f"{rname} r{r} 重建后 digest={row.get('digest')} 期望 {expected}")
                if not row.get("stream_after"):
                    s3_bad.append(f"{rname} r{r} 重建后未能再建流")
                if row.get("post_state") != "available":
                    s3_bad.append(f"{rname} r{r} 重建后 device_state={row.get('post_state')!r}"
                                  "（R4 要求回到 available）")
            else:
                if row.get("digest") != expected:
                    s2_bad.append(f"{rname} r{r} digest={row.get('digest')} 期望 {expected}")
                snap = row.get("snapshot") or {}
                if snap.get("state") != "available":
                    s2_bad.append(f"{rname} r{r} device_state={snap.get('state')!r}（受他人重建影响）")
                if snap.get("probe") is not True:
                    s2_bad.append(f"{rname} r{r} probe_device=False")
                if snap.get("context_query") != rres.get("base_snapshot", {}).get("context_query"):
                    s2_bad.append(f"{rname} r{r} context_query 快照变化：{snap.get('context_query')}")

    res["victim_rounds"] = victim_rounds
    res["checks"] = {
        "S1_recoverer_real_rebuild": {"ok": not s1_bad, "detail": s1_bad[:6],
                                       "count": f"{victim_rounds} 次恢复者轮次"},
        "S2_peers_unaffected": {"ok": not s2_bad, "detail": s2_bad[:6]},
        "S3_usable_after_rebuild": {"ok": not s3_bad, "detail": s3_bad[:6]},
        "S0_no_exception": {"ok": not other_bad, "detail": other_bad[:6]},
    }
    all_ok = all(v["ok"] for v in res["checks"].values()) and victim_rounds >= args.rounds
    res["verdict"] = "MULTIPROC_REAL_RECOVER_PASS" if all_ok else "MULTIPROC_REAL_RECOVER_FAIL"
    res["expected_digest"] = expected

    print(f"\n[结果] 恢复者轮次 {victim_rounds}（每 rank 各 ~{args.rounds // max(ranks,1)} 次）")
    for k, v in res["checks"].items():
        print(f"  {'PASS' if v['ok'] else 'FAIL'}  {k}" +
              ("" if v["ok"] else f"  → {v['detail']}"))
    for rname, rres in results.items():
        bad = [row["round"] for row in rres.get("rounds", []) if row.get("exception")]
        print(f"  {rname}: ok={rres.get('ok')} victim轮次="
              f"{sum(1 for x in rres.get('rounds', []) if x['i_am_victim'])} "
              f"异常轮={bad[:5]}")
    print(f"\n{res['verdict']}（{elapsed}s）")

    outp = os.path.join(out_dir, "a2_recover_multiproc_result.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"结果 -> {outp}")
    print("A2_BOUNDARY: 仅同机同厂商多卡多进程；ISOLATED 为构造而非真实硬件故障；"
          "结论只在本档位/环境成立，不外推到其他实例或多机。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
