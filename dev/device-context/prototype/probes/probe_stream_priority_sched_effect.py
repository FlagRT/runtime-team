#!/usr/bin/env python3
"""
probe_stream_priority_sched_effect.py — 流优先级「调度效果」对照实验（台账 D2）
═══════════════════════════════════════════════════════════════════════════════

【为什么单独一个探针】
  契约 §1.9 只承诺 **接口能力**（能设、能回读），**不承诺调度效果** —— 「范围可读 ≠ 效果可用」。
  台账 **D2 =「`priority` 的调度效果对照实验」**：**没有它就不该拿优先级做调度决策**。
  本探针就是那一条性能侧证据。

【与两个旧探针的关系（二者都保留作历史对照）】
  · `probe_stream_priority_effect.py`（2026-09-30 起自标「已被取代」）：方向标反 + 测法与机制不匹配。
  · `probe_stream_priority_queued.py`：改「两流同时排队再放行」，方向按官方语义取 `min`。
  ⚠️ 本探针相对上述两者的关键改动 —— 都是 **判据可信度** 问题，不是换姿势重做：

    ① **必须经本层统一面建流**（`runtime.create_stream(priority=…)`）。
       MLU590 实测厂商 torch 侧**会静默丢弃**该参数（设备 C API 回读 `4`，而 `.priority` 属性回显 `7`）
       ⇒ 用 `torch.<dt>.Stream(priority=…)` 做实验，等于测了一条 **参数并没进设备** 的路径，
       由此得到的「无效果」是**假结论**。

    ② **消除两类系统性偏向**：
       · *入队顺序偏向* —— 同优先级时「谁先入队谁先跑」，旧探针把顺序写死。
         本探针**每轮随机化入队顺序** ⇒ 顺序效应在统计上被抵消，剩下的一侧才会被读成「优先级效果」。
       · *观测顺序偏向* —— 旧探针固定「先查 hi 再查 lo」，同一轮内两者都完成时**必然记 hi 先完成**。
         本探针**每轮随机化扫描顺序**，并额外检测「同一趟扫描里两条都完成」⇒ 该轮记为
         `indeterminate`（平局），**不计入胜负**、单独上报。
         顺带：`D1_measurement_resolution` 把「扫描一趟要多久」变成判据（相对工作量级），
         分辨力不够就 FAIL —— 否则「测不出」可能只是**看不清**。

    ③ **gate 必须是真 gate**。旧探针把「两流 wait gate」**先入队**、`gate.record()` **后调用**；
       而对**未记录**的事件做等待在 CUDA 语义下等价于 **空操作** ⇒ 两条流其实在入队时就放行了，
       「gated 组」退化成 naive 组。本探针改为：**先把一段慢活排在同一条流上，再记录 gate**，
       使两条流的等待都**在 gate 触发前**入队；并自带有效性判据 **V1**（见下）。

    ④ **配正对照（非空转验证）**：让「高优先级」那条流的工作量**减半**、而优先级**相同** ⇒
       测量**必须**判出「hi 先完成」。没有这一步，「未观测到效果」与「测量根本分辨不出」
       无法区分 —— 后者会让一条**仪器缺陷**伪装成一条**设备结论**。

【关键比较对象（别读错）】
  G1 的读数是「高优先级先完成几轮」。它**只**应与 **G3（同优先级、同入队顺序分布、等量工作）** 比 ——
  那才是**噪声基线**。不要拿 G1 与「理论上的 50%」比，也不要用「ge 50%」当有无效果的判据。

【判据（先写死，不许事后解释）】
  E1 `range_and_applicability`      范围可解析 ⇒ `hi=min` / `lo=max`（官方语义：数值越小越高）；
                                    `hi == lo`（单点区间）或未声明 control ⇒ **applicable=False**
                                    ⇒ 如实走 NOT_APPLICABLE，并仍跑仪器有效性那两组。
  E2 `setup_readback_equals_request` 两条流经统一面建成后 `stream_priority_readback` **必须等于请求**
                                    —— 否则整场实验测的不是我们以为的东西 ⇒ FAIL（不放行）。
  E3 `work_quantum_calibrated`      量出「每流工作量级」与「一趟扫描耗时」⇒ 供判断分辨力。
  D1 `measurement_resolution`       一趟扫描耗时必须 ≪ 工作量级（默认 ≤ 5%）⇒ 否则 FAIL。
  V1 `gate_actually_gated`          两条流的完成时刻都必须 ≥ 慢活时长的 80%
                                    —— 若某条流在 gate 触发前就完成，说明它的等待是**空操作** ⇒ FAIL。
  G1 `gate_high_priority_wins`      **主判据**：两流同一 gate 同时就绪、入队顺序随机化。
                                    判定 = **胜率 ∧ 幅度**（默认 ≥75% ∧ ≥25% 工作量级）——
                                    只看胜率会把「差一个测量步长」读成效果（本轮实测教训）。
                                    高优先级先完成 `hi_wins / ROUNDS`。结论写进 `effect_observed`。
  G2 `positive_control_unequal_work` **非空转**：同优先级、hi 工作量减半 ⇒ 必须 ≥ 阈值，
                                    否则 `instrument_valid=False` ⇒ FAIL（结论不可信，先修仪器）。
  G3 `equal_priority_baseline`      同优先级、等量工作 ⇒ **噪声基线**（不作 pass/fail）。
  G4 `naive_submit_order_control`   旧测法（**固定**低优先级先提交、已进入运行态）⇒ 决定性地回答
                                    「后提交的高优先级流能否反超」。
  G7 `ungated_equal_priority_control` G4 的**必要对照**：同样「lo 先提交」但两流取值相同 ⇒
                                    排除「恒为后提交者获胜」这一混淆，G4 的效应才可归因于优先级。
  G8 `ungated_reversed_order`       与 G4 顺序相反的同一对照（未 gated）⇒ 与 G4/G7 合起来构成 2×2。
  G9 `gated_large_vs_mid_crosscheck` gated 场景换一对非端点取值，刻画那个 ~0.23 ms 偏斜的方向。
  G10 `ungated_effect_summary`       把未 gated 场景（G4/G8 两向 + G7 同取值对照）合成一个布尔结论
                                    —— **gated 与未 gated 必须分开报**，否则会漏掉一半事实。

【判定】
  STREAM_PRIORITY_SCHED_EFFECT_PASS           实验有效（`effect_observed` 给出结论，正负皆可）
  STREAM_PRIORITY_SCHED_EFFECT_NOT_APPLICABLE 本机不具备对照条件（单点区间 / 未声明 control）
  STREAM_PRIORITY_SCHED_EFFECT_FAIL           实验本身不可信（回读不符 / 分辨力不足 / gate 无效 /
                                              正对照失败）

【⚠️ 结论边界（必须随结论一起引用）】
  · 本实验是**单卡、同进程两条流、人为构造的争用**；**不等于**生产负载下的收益。
  · 「未观测到效果」**不等于**「优先级无效」：可能取决于负载形态、设备占用、驱动调度策略。
    本探针只回答「**在本场景下**，两流同时就绪时完成顺序是否与优先级一致」。
  · 本探针**不引用**任何厂商调度文档作为机制依据：MLU590 侧**未取得**对应文档
    （CANN 侧文档有「不抢占已运行任务、不动态重评估队列」的表述，但**不得**外推到本家）。

【环境变量 / 用法】
  DC_BACKEND / DC_OUT_DIR / DC_TAG / DC_PRIO_ROUNDS(8) / DC_PRIO_PASS_N(6)
  DC_PRIO_WORK(2048 方阵边长) / DC_PRIO_ITERS(60 每流链式 matmul 次数) / DC_PRIO_CALIB(3)
  用法：DC_ROOT=<原型根> DC_BACKEND=<vendor> python3 probes/probe_stream_priority_sched_effect.py --dev 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import torch  # noqa: E402
import runtime  # noqa: E402

ROUNDS = int(os.environ.get("DC_PRIO_ROUNDS", "8"))
PASS_N = int(os.environ.get("DC_PRIO_PASS_N", "6"))
WORK = int(os.environ.get("DC_PRIO_WORK", "2048"))
ITERS = int(os.environ.get("DC_PRIO_ITERS", "60"))
CALIB = int(os.environ.get("DC_PRIO_CALIB", "3"))
V1_ROUNDS = int(os.environ.get("DC_PRIO_V1_ROUNDS", "3"))
MAX_RES_FRAC = float(os.environ.get("DC_PRIO_MAX_RES_FRAC", "0.05"))
#: 「观测到调度效果」= **胜率**分量 ∧ **幅度**分量（两者都要）。
#: 为什么不能只看胜率：若两条流其实**并行**跑完、只差一个测量步长，胜率也能靠「非 0 值 vs 0 值」
#: 的微小偏向凑到接近 100%（本轮 MLU590 实测正是如此：8/8 稳定领先，但幅度只有 0.23 ms）。
#: 那种差异对"拿优先级做调度决策"毫无价值 ⇒ 必须同时要求幅度达到工作量级的一个可感比例。
EFFECT_MIN_WIN_FRAC = float(os.environ.get("DC_PRIO_EFFECT_MIN_WIN_FRAC", "0.75"))
EFFECT_MIN_GAP_FRAC = float(os.environ.get("DC_PRIO_EFFECT_MIN_GAP_FRAC", "0.25"))
HOST_TIMEOUT_MS = int(os.environ.get("DC_PRIO_TIMEOUT_MS", "600000"))
#: gate 慢活的**下限时长**（ms）。为什么要有下限：gate 的作用是「把两条流的等待都挤在
#: gate 触发之前入队」。宿主机入队两条 60 次链的成本是**毫秒量级**，所以慢活只有几毫秒时
#: gate 就可能来不及生效（退化成空操作）。此值由**实测**驱动（见 E3 的自适应放大）。
MIN_SLOW_MS = float(os.environ.get("DC_PRIO_MIN_SLOW_MS", "30"))


def main() -> int:                                          # noqa: C901  (探针，允许长)
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("DC_BACKEND", "ascend"))
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    out_dir = Path(a.out).parent if a.out else Path(os.environ.get("DC_OUT_DIR", str(_HERE)))
    tag = os.environ.get("DC_TAG", "")
    checks: dict = {}
    notes: list = []
    rng = random.Random(int(os.environ.get("DC_PRIO_SEED", "20261008")))

    def judge(name, ok, detail, **extra):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if extra:
            checks[name].update(extra)

    def skip(name, detail, **extra):
        checks[name] = {"ok": True, "skipped": True, "detail": detail}
        if extra:
            checks[name].update(extra)

    bk = runtime.use(a.backend)
    n_dev = bk.device_count()
    dt = bk.device_type
    ns = getattr(torch, dt)
    runtime.set_device(a.dev)

    def sync():
        """设备同步（不同栈签名略异，按可用性退让）。"""
        try:
            ns.synchronize()
        except BaseException:                                 # noqa: BLE001
            try:
                ns.synchronize(a.dev)
            except BaseException:                             # noqa: BLE001
                pass

    print("=== probe_stream_priority_sched_effect.py: 流优先级「调度效果」对照实验（台账 D2）===",
          flush=True)
    try:
        dev_name = ns.get_device_name(a.dev)
    except BaseException:                                     # noqa: BLE001
        dev_name = f"{dt}:{a.dev}"
    print(f"[env] backend={a.backend} device_type={dt} devices={n_dev} torch={torch.__version__} "
          f"dev={a.dev}（{dev_name}）", flush=True)
    print(f"[env] rounds={ROUNDS} pass_n={PASS_N} work={WORK} iters={ITERS} calib={CALIB} "
          f"v1_rounds={V1_ROUNDS}", flush=True)

    # ══════════════════ E1 范围与适用性 ══════════════════
    raw = bk.stream_priority_range()
    hi = lo = None
    if isinstance(raw, (tuple, list)) and len(raw) in (2, 3):
        try:
            vals = [int(raw[0]), int(raw[1])]
            rc = int(raw[2]) if len(raw) == 3 else 0
            if rc == 0:
                hi, lo = min(vals), max(vals)     # 官方语义：greatest priority 数值最小
        except (TypeError, ValueError):
            pass
    sup_ctrl = bool(bk.supports("stream_priority_control"))
    sup_rb = bool(bk.supports("stream_priority_readback"))
    applicable = bool(hi is not None and lo is not None and hi != lo and sup_ctrl)
    why_not = []
    if raw is None:
        why_not.append("范围不可读（stream_priority_range() = None）")
    elif hi is None:
        why_not.append(f"范围形状不可解析（raw={raw!r}）")
    elif hi == lo:
        why_not.append(f"优先级空间**退化为单点**（raw={raw!r} ⇒ hi=lo={hi}）⇒ 无从构造对照")
    if not sup_ctrl:
        why_not.append("未声明 `stream_priority_control`（如实不具备「能设置」）")
    print(f"[E1] range_raw={raw!r} ⇒ hi(最高)={hi} lo(最低)={lo} "
          f"supports_ctrl={sup_ctrl} supports_rb={sup_rb} applicable={applicable}", flush=True)
    if not applicable:
        print(f"     ⇒ 不适用原因：{'；'.join(why_not)}", flush=True)
    judge("E1_range_and_applicability", True,
          f"raw={raw!r}; hi={hi}; lo={lo}; supports[control]={sup_ctrl}; "
          f"supports[readback]={sup_rb}; applicable={applicable}"
          + ("" if applicable else f"；不适用原因：{'；'.join(why_not)}"),
          range_raw=list(raw) if isinstance(raw, (tuple, list)) else None,
          hi_priority=hi, lo_priority=lo, supports_control=sup_ctrl,
          supports_readback=sup_rb, applicable=applicable, not_applicable_reason=why_not)

    dev = f"{dt}:{a.dev}"
    m_hi = torch.randn(WORK, WORK, device=dev)
    m_lo = torch.randn(WORK, WORK, device=dev)
    m_slow = torch.randn(WORK, WORK, device=dev)
    sync()
    keep: dict = {}                       # 每「角色」只保活最后一个结果张量（换手前已 sync）
    wait_path = {"v": None}

    def mk(prio):
        return runtime.create_stream(priority=prio) if prio is not None else runtime.create_stream()

    def ev_new():
        return runtime.create_event()

    def chain(stream, mat, iters, role):
        """在当前流上下文里跑 iters 次链式 matmul（数据依赖 ⇒ 串行，不可被拆散）。"""
        with stream.context():
            acc = mat.clone()
            for _ in range(iters):
                acc = acc @ mat
        keep[role] = acc
        return acc

    def gate_wait(gate, s):
        """让流 s 等待 gate。优先走「Event.wait(Stream)」（底层 = 厂商 event.wait(stream)）。"""
        try:
            gate.wait(s)
            wait_path["v"] = wait_path["v"] or "Event.wait(Stream)"
            return
        except BaseException as e1:                           # noqa: BLE001
            notes.append(f"Event.wait(Stream) 不可用：{type(e1).__name__}: {str(e1)[:80]}")
        s.wait_event(gate)                                    # 退化路径
        wait_path["v"] = wait_path["v"] or "Stream.wait_event(Event)"

    # ── 单轮：返回该轮记录 ──
    def one_round(s_hi, s_lo, s_gate, it_hi, it_lo, use_gate, hi_first_enq, slow_iters):
        e_hi, e_lo = ev_new(), ev_new()
        t0 = time.monotonic()
        if use_gate:
            # ⭐ 关键顺序：**先**把慢活排到 gate 流上、**再**记录 gate。
            #    这样下面两次 wait 都在 gate 触发**之前**入队 ⇒ 两条流在 gate 触发时
            #    **同时**变为就绪（真 gate）。反过来写（先入队 wait、后 record gate）对
            #    未记录事件的等待等价于空操作 ⇒ gate 失效（旧探针的缺陷）。
            chain(s_gate, m_slow, slow_iters, "slow")
            g = ev_new()
            g.record(s_gate)
            gate_wait(g, s_lo)
            gate_wait(g, s_hi)
        todo = [(s_lo, e_lo, m_lo, it_lo, "lo"), (s_hi, e_hi, m_hi, it_hi, "hi")]
        if hi_first_enq:
            todo.reverse()
        for s, e, m, it, role in todo:
            chain(s, m, it, role)
            e.record(s)

        # ── 观测：busy-wait 扫两条完成事件，**每轮随机化扫描顺序** ──
        poll = ["hi", "lo"] if rng.random() < 0.5 else ["lo", "hi"]
        got: dict = {}
        indeterminate = False
        sweep_n = 0
        sweep_times = []
        deadline = t0 + HOST_TIMEOUT_MS / 1000.0
        while len(got) < 2:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{HOST_TIMEOUT_MS} ms 内有流未完成（已观测到 {sorted(got)}）")
            ts = time.monotonic()
            new = []
            for w in poll:
                if w in got:
                    continue
                ev = e_hi if w == "hi" else e_lo
                if ev.query():
                    got[w] = (time.monotonic() - t0) * 1000.0
                    new.append(w)
            sweep_times.append((time.monotonic() - ts) * 1e6)     # µs
            sweep_n += 1
            if len(new) == 2:
                # 同一趟扫描里两条都完成 ⇒ 完成时刻差 < 一趟扫描时长 ⇒ 记平局，不计胜负
                indeterminate = True
        return {"hi_ms": round(got["hi"], 4), "lo_ms": round(got["lo"], 4),
                "hi_first": bool(got["hi"] < got["lo"]),
                "indeterminate": indeterminate,
                "enqueue_first": "hi" if hi_first_enq else "lo",
                "poll_order": poll, "sweeps": sweep_n,
                "sweep_us_median": round(statistics.median(sweep_times), 2)}

    def run_group(name, prio_hi, prio_lo, it_hi, it_lo, use_gate, rounds, slow_iters,
                  enq_mode="random"):
        """enq_mode: random（每轮随机化入队顺序）| lo_first | hi_first（固定）。
        ⚠️ 固定顺序只用于**故意保留偏向**的对照组（G4 旧测法），不得用于主判据。"""
        s_hi = mk(prio_hi)
        s_lo = mk(prio_lo)
        s_gate = runtime.create_stream() if use_gate else None
        try:
            recs = []
            for _ in range(rounds):
                if enq_mode == "random":
                    hf = rng.random() < 0.5
                else:
                    hf = (enq_mode == "hi_first")
                recs.append(one_round(s_hi, s_lo, s_gate, it_hi, it_lo,
                                      use_gate, hf, slow_iters))
                sync()                                   # 轮间同步：保证换手张量时设备已用完
            decided = [r for r in recs if not r["indeterminate"]]
            wins = sum(1 for r in decided if r["hi_first"])
            return {"group": name, "rounds": rounds, "decided": len(decided),
                    "indeterminate": rounds - len(decided), "hi_wins": wins,
                    "hi_win_rate_decided": (round(wins / len(decided), 3) if decided else None),
                    "median_hi_ms": round(statistics.median([r["hi_ms"] for r in recs]), 4),
                    "median_lo_ms": round(statistics.median([r["lo_ms"] for r in recs]), 4),
                    "median_gap_ms": round(statistics.median(
                        [r["lo_ms"] - r["hi_ms"] for r in recs]), 4),
                    "records": recs}
        finally:
            sync()
            for s in (s_hi, s_lo, s_gate):
                if s is None:
                    continue
                try:
                    runtime.release_stream(s)   # 本层拥有的流必须释放（契约 §1.10 规则 3）
                except BaseException:                 # noqa: BLE001
                    pass

    # ══════════════════ E2 设置后回读必须一致 ══════════════════
    rb_hi = rb_lo = None
    if applicable:
        try:
            s_h, s_l = mk(hi), mk(lo)
            rb_hi = runtime.stream_priority_readback(s_h)
            rb_lo = runtime.stream_priority_readback(s_l)
            ok = (rb_hi == hi and rb_lo == lo)
            judge("E2_setup_readback_equals_request", ok,
                  f"请求 hi={hi} ⇒ 回读 {rb_hi!r}；请求 lo={lo} ⇒ 回读 {rb_lo!r}"
                  + ("" if ok else "  ⚠️ 回读 ≠ 请求 ⇒ 本场实验测的不是我们以为的东西，结论作废"))
            sync()
            runtime.release_stream(s_h)
            runtime.release_stream(s_l)
        except BaseException as e:                            # noqa: BLE001
            judge("E2_setup_readback_equals_request", False,
                  f"{type(e).__name__}: {str(e)[:160]}")
    else:
        skip("E2_setup_readback_equals_request",
             f"applicable=False ⇒ 不经优先级路径建流（{'；'.join(why_not)}）")

    # ══════════════════ E3 量级标定 + D1 分辨力 ══════════════════
    per_iter = None
    slow_iters = max(4, 2 * ITERS)
    slow_ms = None
    quantum_ms = None
    try:
        s = mk(hi) if applicable else runtime.create_stream()
        # ⚠️ 先热身：首次调用含**惰性初始化**（实测首次 5.9 ms / 后续 ~0.02 ms per op）
        #    ⇒ 不热身会把 per_iter 估大一个量级，进而把 gate 慢活估得**过短**。
        chain(s, m_hi, max(2, min(ITERS, 8)), "warmup")
        sync()
        qs = []
        for _ in range(CALIB):
            e0 = ev_new()
            t0 = time.monotonic()
            e0.record(None)
            chain(s, m_hi, ITERS, "calib")
            e1 = ev_new()
            e1.record(s)
            e1.wait_host(600000)
            qs.append((time.monotonic() - t0) * 1000.0)
        quantum_ms = statistics.median(qs)
        per_iter = quantum_ms / ITERS if ITERS else None
        sync()
        runtime.release_stream(s)
        # 慢活时长：由 per_iter 推 slow_iters，再实测一次作为 V1 的基准
        if per_iter:
            target = max(MIN_SLOW_MS, 2.0 * quantum_ms)   # ≥ 下限 且 ≥2×工作量
            slow_iters = max(4, int(math.ceil(target / per_iter)))
        # 实测 + **自适应放大**：per_iter 是估的，实测若不够长就按比例放大重测（最多 4 轮）
        slow_tries = []
        for _try in range(4):
            s2 = runtime.create_stream()
            e0 = ev_new()
            t0 = time.monotonic()
            e0.record(None)
            chain(s2, m_slow, slow_iters, "slowcalib")
            e1 = ev_new()
            e1.record(s2)
            e1.wait_host(HOST_TIMEOUT_MS)
            slow_ms = (time.monotonic() - t0) * 1000.0
            sync()
            runtime.release_stream(s2)
            slow_tries.append({"iters": slow_iters, "slow_ms": round(slow_ms, 3)})
            if slow_ms >= MIN_SLOW_MS:
                break
            slow_iters = max(slow_iters + 4, int(slow_iters * max(2.0, MIN_SLOW_MS * 1.5 / max(slow_ms, 0.01))))
        judge("E3_work_quantum_calibrated", True,
              f"每流 {ITERS} 次链式 {WORK}² matmul 墙钟中位数 = {round(quantum_ms, 3)} ms"
              f"（单次 ≈ {round(per_iter, 4)} ms）；gate 慢活 {slow_iters} 次实测 "
              f"{round(slow_ms, 3)} ms（下限 {MIN_SLOW_MS} ms）",
              quantum_ms=round(quantum_ms, 3), per_iter_ms=round(per_iter, 5),
              slow_iters=slow_iters, slow_ms=round(slow_ms, 3),
              slow_tries=slow_tries, min_slow_ms=MIN_SLOW_MS,
              calib_samples=[round(v, 3) for v in qs])
    except BaseException as e:                                # noqa: BLE001
        judge("E3_work_quantum_calibrated", False, f"{type(e).__name__}: {str(e)[:160]}")

    def resolution_check(sweep_us_val, q_ms):
        if not sweep_us_val or not q_ms:
            judge("D1_measurement_resolution", False,
                  f"扫描耗时或工作量级不可得（sweep={sweep_us_val!r} quantum={q_ms!r}）")
            return
        frac = (sweep_us_val / 1000.0) / q_ms
        ok = frac <= MAX_RES_FRAC
        judge("D1_measurement_resolution", ok,
              f"一趟扫描 {sweep_us_val} µs = 工作量级（{round(q_ms, 3)} ms）的 {frac * 100:.3f}%"
              f"（阈值 ≤ {MAX_RES_FRAC * 100:.1f}%）⇒ "
              + ("分辨力足够" if ok else
                 "⚠️ 分辨力不足 ⇒ 真有效果也可能被读成「同时完成」⇒ 结论不可信"),
              sweep_us=sweep_us_val, resolution_frac=round(frac, 6))

    groups: dict = {}
    dir_find: dict = {}
    effect_gated = None
    effect_ungated = None
    instrument_valid = None
    effect_observed = None

    # ══════════════════ G2 正对照（非空转）══════════════════
    try:
        # ⭐ 正对照**走 gate**：这样它验证的是「gate + 测量」整条链，而不只是测量本身
        g2 = run_group("G2_positive_control", None, None, ITERS, 2 * ITERS, True, ROUNDS,
                       slow_iters)
        groups["G2"] = g2
        sweeps = [r["sweep_us_median"] for r in g2["records"]]
        resolution_check(statistics.median(sweeps) if sweeps else None, quantum_ms)
        iv = g2["decided"] > 0 and g2["hi_wins"] >= min(PASS_N, g2["decided"])
        instrument_valid = iv
        judge("G2_positive_control_unequal_work", iv,
              f"同优先级、同一 gate、hi 工作量 = lo 的一半 ⇒ hi 先完成 "
              f"{g2['hi_wins']}/{g2['decided']} 有效轮"
              f"（平局 {g2['indeterminate']}；阈值 {PASS_N}）；中位差 {g2['median_gap_ms']} ms ⇒ "
              + ("测量**能**判出差异（仪器有效）" if iv else
                 "⚠️ 测量**判不出**已知存在的差异 ⇒ 仪器无效，任何「未观测到效果」的结论作废"),
              instrument_valid=iv, **{k: v for k, v in g2.items() if k != "records"})
    except BaseException as e:                                # noqa: BLE001
        judge("G2_positive_control_unequal_work", False, f"{type(e).__name__}: {str(e)[:160]}")
        instrument_valid = False

    # ══════════════════ G3 同优先级噪声基线（两种情形都要跑）══════════════════
    try:
        g3 = run_group("G3_equal_priority_baseline",
                       hi if applicable else None, hi if applicable else None,
                       ITERS, ITERS, True, ROUNDS, slow_iters)
        groups["G3"] = g3
        judge("G3_equal_priority_baseline", True,
              f"同优先级、等量工作、同一 gate、入队顺序随机化：hi 先完成 "
              f"{g3['hi_wins']}/{g3['decided']} 有效轮（平局 {g3['indeterminate']}）"
              f"；中位差 {g3['median_gap_ms']} ms ⇒ 这是**噪声基线**（G1 的读数只应与它比）",
              **{k: v for k, v in g3.items() if k != "records"})
    except BaseException as e:                                # noqa: BLE001
        judge("G3_equal_priority_baseline", False, f"{type(e).__name__}: {str(e)[:160]}")

    if applicable:
        # ══════════════════ V1 gate 有效性 ══════════════════
        try:
            v1 = run_group("V1_gate_validity", hi, lo, ITERS, ITERS, True, V1_ROUNDS, slow_iters)
            groups["V1"] = v1
            thr = 0.8 * (slow_ms or 0.0)
            bad = [r for r in v1["records"] if min(r["hi_ms"], r["lo_ms"]) < thr]
            ok = (not bad) and bool(slow_ms)
            judge("V1_gate_actually_gated", ok,
                  f"慢活基准 {round(slow_ms or 0.0, 3)} ms；{V1_ROUNDS} 轮的完成时刻 "
                  f"hi={[r['hi_ms'] for r in v1['records']]} lo={[r['lo_ms'] for r in v1['records']]}"
                  f"（阈值 ≥ {round(thr, 3)} ms）⇒ "
                  + ("两条流都在 gate 触发**之后**才完成 ⇒ 等待非空操作" if ok else
                     f"⚠️ {len(bad)} 轮有流早于 gate 触发完成 ⇒ 其等待是**空操作**，gated 组数据无效"),
                  gate_slow_ms=round(slow_ms or 0.0, 3), threshold_ms=round(thr, 3))
        except BaseException as e:                            # noqa: BLE001
            judge("V1_gate_actually_gated", False, f"{type(e).__name__}: {str(e)[:160]}")

        # ══════════════════ G1 主判据 ══════════════════
        try:
            g1 = run_group("G1_gate_hi_vs_lo", hi, lo, ITERS, ITERS, True, ROUNDS, slow_iters)
            groups["G1"] = g1
            need = max(1, int(math.ceil(EFFECT_MIN_WIN_FRAC * max(g1["decided"], 1))))
            gap_need = EFFECT_MIN_GAP_FRAC * (quantum_ms or 0.0)
            win_ok = g1["decided"] > 0 and g1["hi_wins"] >= need
            gap_ok = g1["median_gap_ms"] >= gap_need
            eff = bool(win_ok and gap_ok)
            effect_observed = eff
            judge("G1_gate_high_priority_wins", True,
                  f"同一 gate 同时就绪、入队顺序随机化：高优先级取值（{hi}）那条流先完成 "
                  f"{g1['hi_wins']}/{g1['decided']} 有效轮（平局 {g1['indeterminate']}；"
                  f"需 ≥ {need} = {EFFECT_MIN_WIN_FRAC:.0%}）"
                  f"；中位 hi={g1['median_hi_ms']} ms / lo={g1['median_lo_ms']} ms、"
                  f"中位差 {g1['median_gap_ms']} ms（需 ≥ {round(gap_need, 4)} = "
                  f"{EFFECT_MIN_GAP_FRAC:.0%} 工作量级 {round(quantum_ms or 0, 3)} ms）⇒ "
                  + ("**观测到优先级调度效果**" if eff else
                     "**未观测到实质性优先级调度效果（如实记录，非探针失败）**"
                     + ("（胜率够但幅度不够）" if (win_ok and not gap_ok) else ""))
                  + "｜⚠️ 本组只回答『两流**同时就绪**』这一场景；"
                    "「一条流已在运行、另一条才提交」的场景见 G4/G8（那里测到的是另一个量级）"
                  + (f"；噪声基线（G3）为 {groups['G3']['hi_wins']}/{groups['G3']['decided']}"
                     if "G3" in groups else ""),
                  effect_observed=eff, win_criterion_met=win_ok, gap_criterion_met=gap_ok,
                  win_need=need, gap_need_ms=round(gap_need, 4),
                  **{k: v for k, v in g1.items() if k != "records"})
        except BaseException as e:                            # noqa: BLE001
            judge("G1_gate_high_priority_wins", False, f"{type(e).__name__}: {str(e)[:160]}")

        # ══════════════════ G4 旧测法对照 ══════════════════
        try:
            # ⚠️ **入队顺序固定为 lo 先**（低优先级先提交、已进入运行态）= 旧测法。
            #    本组是**决定性**的一组：若先提交的低优先级流仍然获胜 ⇒ 提交顺序主导、
            #    优先级无实质作用（与「不抢占已运行任务」一致）；若后提交的高优先级流
            #    反超 ⇒ 说明优先级能「插队」（那是**实质**效果）。
            g4 = run_group("G4_naive_submit_order", hi, lo, ITERS, ITERS, False, ROUNDS,
                           slow_iters, enq_mode="lo_first")
            groups["G4"] = g4
            judge("G4_naive_submit_order_control", True,
                  f"旧测法（**入队顺序固定**：取值 {lo} 先提交、已进入运行态，随后才提交 "
                  f"取值 {hi}）：取值 {hi}（**后提交**）那条流先完成 "
                  f"{g4['hi_wins']}/{g4['decided']} 有效轮（平局 {g4['indeterminate']}）"
                  f"；中位差 {g4['median_gap_ms']} ms"
                  f"（**正 = 后提交的取值 {hi} 反超获胜；负 = 先提交者仍获胜**）⇒ "
                  + (f"后提交者反超 {g4['median_gap_ms']} ms ⇒ 疑似**优先级插队**"
                     if g4["median_gap_ms"] > 0 else "先提交者仍获胜 ⇒ **提交顺序主导**"),
                  **{k: v for k, v in g4.items() if k != "records"})

        except BaseException as e:                            # noqa: BLE001
            judge("G4_naive_submit_order_control", False, f"{type(e).__name__}: {str(e)[:160]}")

        # ⚠️ **G4 的必要对照（否则 G4 的结论不成立）**：G4 同时变了**两个变量** ——
        #    先提交者的取值（{lo}）与提交顺序 ⇒ 单看 G4 **分不清**
        #    「优先级插队」与「恒为后提交者获胜」。G7 把两流取值取成**同一个**、
        #    提交顺序保持 lo 先 ⇒ 唯一差别只剩「谁先提交」，干净排除该混淆。
        try:
            g7 = run_group("G7_ungated_equal_prio", hi, hi, ITERS, ITERS, False, ROUNDS,
                           slow_iters, enq_mode="lo_first")
            groups["G7"] = g7
            judge("G7_ungated_equal_priority_control", True,
                  f"同 G4 但两流**取值相同**（均 {hi}）、提交顺序仍为角色 lo 先："
                  f"后提交的角色 hi 先完成 {g7['hi_wins']}/{g7['decided']} 有效轮"
                  f"（平局 {g7['indeterminate']}）；中位差 {g7['median_gap_ms']} ms ⇒ "
                  + ("后提交者也获胜 ⇒ 「恒为后者获胜」**不能排除**，"
                     "G4 的效应因此**不能**归因于优先级"
                     if g7["median_gap_ms"] > 0.5 else
                     "基本同速/先提交者胜 ⇒ 「恒为后者获胜」被排除，"
                     "G4 的效应**可**归因于优先级差值"),
                  **{k: v for k, v in g7.items() if k != "records"})
        except BaseException as e:                            # noqa: BLE001
            judge("G7_ungated_equal_priority_control", False,
                  f"{type(e).__name__}: {str(e)[:160]}")

        # ⚠️ **G8：把 G4 的提交顺序反过来**（取值 {hi} 先提交、取值 {lo} 后提交）⇒ 与 G4/G7 一起
        #    构成未 gated 场景的 2×2：若两向都是「取值 {hi} 胜且幅度 ~7 ms」，
        #    则「{hi} 优先级更高 + 争用时份额更大」就是**可复现的实质效果**。
        try:
            g8 = run_group("G8_ungated_reversed", hi, lo, ITERS, ITERS, False, ROUNDS,
                           slow_iters, enq_mode="hi_first")
            groups["G8"] = g8
            judge("G8_ungated_reversed_order", True,
                  f"与 G4 顺序相反（取值 {hi} **先**提交、取值 {lo} **后**提交）："
                  f"取值 {hi} 那条流先完成 {g8['hi_wins']}/{g8['decided']} 有效轮"
                  f"（平局 {g8['indeterminate']}）；中位差 {g8['median_gap_ms']} ms ⇒ "
                  + (f"先提交的高优先级流大胜 {'%.2f' % g8['median_gap_ms']} ms ⇒ "
                     "与 G4/G7 合并支持「取值 %d 更高、争用时份额更大」" % hi
                     if g8["median_gap_ms"] > 3.0 else
                     "未出现大胜 ⇒ 与 G4 不一致，需按 2×2 重新判读"),
                  **{k: v for k, v in g8.items() if k != "records"})
        except BaseException as e:                            # noqa: BLE001
            judge("G8_ungated_reversed_order", False, f"{type(e).__name__}: {str(e)[:160]}")

        # ══════════════════ G5/G6 方向交叉核验（记录性，不作 pass/fail）══════════════════
        # ⭐ 为什么必须有：G1 只用「(最高档, 最低档)」这一对值。若 G1 出现**系统性偏向**
        #    （而非掷硬币），单看它**分不清**下面三种解释：
        #      ① 取值方向与本层约定相反（本栈「数值大 = 优先级高」）；
        #      ② 只是「非 0 值 vs 0 值」的队列差异（与方向无关）；
        #      ③ 与优先级无关的角色/位置假象。
        #    G5（把两个取值在角色间**对调**）能分开 ①/③：若获胜方仍跟着**取值**走 ⇒ ①；
        #    若获胜方跟着**角色**走 ⇒ ③。G6（1 vs 0）能进一步分开 ①/②。
        try:
            g5 = run_group("G5_direction_swapped", lo, hi, ITERS, ITERS, True, ROUNDS, slow_iters)
            groups["G5"] = g5
            judge("G5_direction_crosscheck_swapped", True,
                  f"**把两个取值在角色间对调**（角色 hi 用取值 {lo}，角色 lo 用取值 {hi}）："
                  f"取值 {lo} 那条流先完成 {g5['hi_wins']}/{g5['decided']} 有效轮"
                  f"（平局 {g5['indeterminate']}）；中位差 {g5['median_gap_ms']} ms ⇒ "
                  f"与 G1（取值 {hi} 先完成 {groups['G1']['hi_wins']}/{groups['G1']['decided']}）"
                  f"对照，判断获胜方跟『取值』还是跟『角色』",
                  **{k: v for k, v in g5.items() if k != "records"})
            dir_find["g5_value_wins"] = g5["hi_wins"]
        except BaseException as e:                            # noqa: BLE001
            judge("G5_direction_crosscheck_swapped", False, f"{type(e).__name__}: {str(e)[:160]}")

        try:
            if hi < 1 <= lo:
                g6 = run_group("G6_nonzero_vs_zero", 1, hi, ITERS, ITERS, True, ROUNDS, slow_iters)
                groups["G6"] = g6
                judge("G6_nonzero_vs_zero_crosscheck", True,
                      f"取值 1 vs 取值 {hi}（其余相同）：取值 1 那条流先完成 "
                      f"{g6['hi_wins']}/{g6['decided']} 有效轮（平局 {g6['indeterminate']}）"
                      f"；中位差 {g6['median_gap_ms']} ms ⇒ 用来区分「方向相反」与"
                      f"「非 0 值 vs 0 值的队列差异」",
                      **{k: v for k, v in g6.items() if k != "records"})
                dir_find["g6_value1_wins"] = g6["hi_wins"]
            else:
                skip("G6_nonzero_vs_zero_crosscheck", f"取值 1 不在区间 [{hi}, {lo}] 内 ⇒ 如实跳过")

        except BaseException as e:                            # noqa: BLE001
            judge("G6_nonzero_vs_zero_crosscheck", False, f"{type(e).__name__}: {str(e)[:160]}")

        # ⚠️ **G9：gated 场景下换一对「都不是端点」的取值**（{lo} vs 1）⇒ 刻画那个
        #    0.228 ms 偏斜到底跟「数值大小」还是跟「是否非 0」有关。
        #    （G1/G5/G6 只能说明「较大取值领先一个步长」；G9 把 7 与 1 直接对上。）
        try:
            if lo > 1:
                g9 = run_group("G9_gated_large_vs_mid", lo, 1, ITERS, ITERS, True, ROUNDS,
                               slow_iters)
                groups["G9"] = g9
                judge("G9_gated_large_vs_mid_crosscheck", True,
                      f"gated 场景 取值 {lo} vs 取值 1（均为非 0）：取值 {lo} 那条流先完成 "
                      f"{g9['hi_wins']}/{g9['decided']} 有效轮（平局 {g9['indeterminate']}）"
                      f"；中位差 {g9['median_gap_ms']} ms ⇒ "
                      + ("较大取值稳定领先 ⇒ 偏斜随**数值大小**（与『非 0』无关）"
                         if g9["median_gap_ms"] > 0.15 else
                         "没有稳定偏向 ⇒ 该偏斜只在『0 vs 非 0』之间出现"),
                      **{k: v for k, v in g9.items() if k != "records"})
                dir_find["g9_large_value_wins"] = g9["hi_wins"]
            else:
                skip("G9_gated_large_vs_mid_crosscheck", f"取值 1 不在区间 [{hi}, {lo}] 内 ⇒ 如实跳过")
        except BaseException as e:                            # noqa: BLE001
            judge("G9_gated_large_vs_mid_crosscheck", False, f"{type(e).__name__}: {str(e)[:160]}")

        # ⚠️ **必须把两种场景分开报** —— 它们测到的是**不同现象**，合成一个布尔值会漏掉一侧：
        #    · gated（两流**同时就绪**）：本轮实测两条流并行跑完 ⇒ 只差一个量测步长；
        #    · 未 gated（一条已在运行、另一条才提交）：本轮实测优先级带来 **~6.5–7.3 ms** 的
        #      份额差异（对照 G7 排除「恒为后者获胜」）⇒ 这是**实质**效果。
        #    若只报 gated 的布尔值，读者会误以为「优先级完全没用」。
        effect_gated = effect_observed
        effect_ungated = None
        if "G4" in groups and "G8" in groups and "G7" in groups:
            g4, g8, g7 = groups["G4"], groups["G8"], groups["G7"]
            gap_need_u = EFFECT_MIN_GAP_FRAC * (quantum_ms or 0.0)
            n4 = max(1, int(math.ceil(EFFECT_MIN_WIN_FRAC * max(g4["decided"], 1))))
            n8 = max(1, int(math.ceil(EFFECT_MIN_WIN_FRAC * max(g8["decided"], 1))))
            effect_ungated = bool(g4["median_gap_ms"] >= gap_need_u
                                  and g8["median_gap_ms"] >= gap_need_u
                                  and g4["hi_wins"] >= n4 and g8["hi_wins"] >= n8
                                  and (g4["median_gap_ms"] - g7["median_gap_ms"]) >= gap_need_u)
            judge("G10_ungated_effect_summary", True,
                  f"未 gated 场景汇总：G4（取值 {lo} 先提交）= {g4['median_gap_ms']} ms / "
                  f"{g4['hi_wins']}/{g4['decided']}；G8（取值 {hi} 先提交）= "
                  f"{g8['median_gap_ms']} ms / {g8['hi_wins']}/{g8['decided']}；同序同取值对照 "
                  f"G7 = {g7['median_gap_ms']} ms ⇒ "
                  + (f"**观测到实质优先级效果**（两向都让取值 {hi} 领先 ≥ "
                     f"{round(gap_need_u, 3)} ms，且与对照的差值也达阈值）"
                     if effect_ungated else
                     "未达到「两向一致 ∧ 幅度阈值 ∧ 与对照可区分」⇒ 如实标 False"),
                  effect_observed_ungated=effect_ungated,
                  g4_gap_ms=g4["median_gap_ms"], g8_gap_ms=g8["median_gap_ms"],
                  g7_control_gap_ms=g7["median_gap_ms"], gap_need_ms=round(gap_need_u, 4))
        effect_observed = bool(effect_gated) or bool(effect_ungated)
    else:
        for k in ("V1_gate_actually_gated", "G1_gate_high_priority_wins",
                  "G4_naive_submit_order_control", "G5_direction_crosscheck_swapped",
                  "G6_nonzero_vs_zero_crosscheck", "G7_ungated_equal_priority_control",
                  "G8_ungated_reversed_order", "G9_gated_large_vs_mid_crosscheck"):
            skip(k, f"applicable=False ⇒ 如实跳过（{'；'.join(why_not)}）")

    # ══════════════════ 汇总 ══════════════════
    failed = [k for k, v in checks.items() if not v.get("ok")]
    if not applicable:
        verdict = "STREAM_PRIORITY_SCHED_EFFECT_NOT_APPLICABLE"
    elif failed:
        verdict = "STREAM_PRIORITY_SCHED_EFFECT_FAIL"
    else:
        verdict = "STREAM_PRIORITY_SCHED_EFFECT_PASS"
    print(f"\n{verdict}（applicable={applicable} · instrument_valid={instrument_valid} · "
          f"effect_observed={effect_observed}）", flush=True)
    for k, v in checks.items():
        tag_ = "SKIP" if v.get("skipped") else ("PASS" if v["ok"] else "FAIL")
        print(f"  [{tag_}] {k}: {str(v.get('detail'))[:170]}", flush=True)
    if notes:
        print("[notes] " + " ｜ ".join(notes), flush=True)

    out = {"verdict": verdict, "backend": a.backend, "device_type": dt, "device_name": dev_name,
           "torch": torch.__version__, "dev": a.dev,
           "range_raw": list(raw) if isinstance(raw, (tuple, list)) else None,
           "hi_priority": hi, "lo_priority": lo,
           "supports_control": sup_ctrl, "supports_readback": sup_rb,
           "applicable": applicable, "not_applicable_reason": why_not,
           "instrument_valid": instrument_valid, "effect_observed": effect_observed,
           "effect_observed_gated": effect_gated,
           "effect_observed_ungated": effect_ungated,
           "wait_path": wait_path["v"],
           "rounds": ROUNDS, "pass_n": PASS_N, "work": WORK, "iters": ITERS,
           "slow_iters": slow_iters, "quantum_ms": (None if quantum_ms is None else round(quantum_ms, 3)),
           "slow_ms": (None if slow_ms is None else round(slow_ms, 3)),
           "backread_hi": rb_hi, "backread_lo": rb_lo,
           "direction_finding": dir_find,
           "queue_jump_finding": {
               "g4_gap_ms": (groups.get("G4") or {}).get("median_gap_ms"),
               "g7_equal_prio_gap_ms": (groups.get("G7") or {}).get("median_gap_ms"),
               "g8_reversed_gap_ms": (groups.get("G8") or {}).get("median_gap_ms"),
               "interpretation": ("G4（取值 lo 先提交）与 G8（取值 hi 先提交）两向都让取值 hi 大胜 "
                                  "⇒ 未 gated 场景下优先级有**实质**效果；G7 用同顺序+同取值作对照，"
                                  "排除「恒为后提交者获胜」"),
           },
           "gated_tilt_finding": {
               "g1_gap_ms": (groups.get("G1") or {}).get("median_gap_ms"),
               "g5_gap_ms": (groups.get("G5") or {}).get("median_gap_ms"),
               "g6_gap_ms": (groups.get("G6") or {}).get("median_gap_ms"),
               "g9_gap_ms": (groups.get("G9") or {}).get("median_gap_ms"),
               "interpretation": ("gated（同时就绪）场景下只有**一个测量步长**量级的偏斜；"
                                  "其方向与未 gated 场景的大效应相反 ⇒ 两个场景分开报，"
                                  "**不就『方向』下统一结论**"),
           },
           "effect_criteria": {"min_win_frac": EFFECT_MIN_WIN_FRAC,
                               "min_gap_frac": EFFECT_MIN_GAP_FRAC,
                               "min_res_frac": MAX_RES_FRAC,
                               "min_slow_ms": MIN_SLOW_MS},
           "groups": groups, "checks": checks, "notes": notes,
           "boundary": ("单卡 · 同进程两条流 · 人为构造争用；不等于生产负载收益。"
                        "「未观测到效果」≠「优先级无效」。本探针不引用厂商调度文档作机制依据。")}
    p = Path(a.out) if a.out else (out_dir / f"stream_priority_sched_effect_result{tag}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果 -> {p}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
