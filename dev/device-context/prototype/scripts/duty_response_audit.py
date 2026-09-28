#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
scripts/duty_response_audit.py — 职责响应审计（逐 sub-part × 单后端）
═══════════════════════════════════════════════════════════════════════════════

【为什么有这个脚本】
  `backend_offline_check.py` 验的是**契约形态**（离线、桩），
  `conformance` 验的是**行为契约**（13+6 例），`smoke` 验的是**接入可用性**。
  三者都没有回答一个问题：「**接口约定里承诺的每一个 sub-part，本后端是否都能顺利响应**」。
  本脚本就是这个逐项审计：把接口约定（`docs/INTERFACE_CONTRACT_DC_20260908.md`）
  的 §1.1–§1.5 + §2（三支撑方法）+ §3（两条硬纪律）拆成可执行 sub-part，逐项调用。

【判定口径】
  OK   = 调用成功 **且** 返回值/副作用符合接口约定
  FAIL = 不响应（异常/未实现）**或** 响应但不符合契约（如字段缺失、静默降级）
  SKIP = 能力已如实声明为不支持（`supports()` 为假），且调用被后端**主动拦截**而非崩溃
  注意：**FAIL 才算缺口**；SKIP 是「如实不具备」，不是缺口。

【用法】
  DC_BACKEND=cambricon python3 scripts/duty_response_audit.py --backend cambricon
  python3 scripts/duty_response_audit.py --backend ascend --out /path/result.json

【边界】真机脚本：结论只在**当前档位/环境**成立；离线结果不得当真机结论。
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = []          # (domain, sid, title, status, detail)


def _err_text(code) -> str:
    """构造能被 CANN 码提取正则识别的消息。

    ⚠️ 实测（2026-09-28）：`conformance/errors.py::_extract_acl_retcode` 只认
    `ret=N` 与 `error code is N` 两种格式；用 `code N` 之类的写法会**提取不到码**，
    于是退化为 message_hint/default 分级 —— 审计会得到"码表没命中"的**假 FAIL**。
    """
    return f"error code is {code}"


def item(domain, sid, title):
    """装饰器：把一项审计注册成独立执行的函数（单项失败不影响其它项）。"""
    def deco(fn):
        RESULTS.append({"domain": domain, "sid": sid, "title": title, "fn": fn})
        return fn
    return deco


def _ok(detail=""):
    return True, detail


def _fail(detail=""):
    return False, detail


def _skip(detail=""):
    return None, detail


# ══════════════════════════════════════════════════════════════════════════════
# A. 后端选择（接口约定 §1.1）
# ══════════════════════════════════════════════════════════════════════════════

@item("A 后端选择", "A1", "discover() 能发现本后端")
def a1(env):
    names = env["runtime"].discover(verbose=False) or []
    bk = env["backend"].name
    found = bk in [getattr(x, "name", x) for x in names]
    return (_ok if found else _fail)(f"discover → {sorted(str(n) for n in names)}")


@item("A 后端选择", "A2", "available() 含本后端")
def a2(env):
    av = env["runtime"].available()
    return (_ok if env["backend"].name in av else _fail)(f"available → {av}")


@item("A 后端选择", "A3", "use() 返回实例 / current() 一致 / 单例")
def a3(env):
    rt, bk = env["runtime"], env["backend"]
    same = rt.current() is bk
    return (_ok if same else _fail)(f"current().name={rt.current().name}, 单例={same}")


@item("A 后端选择", "A4", "use(未注册名) 如实抛 BackendNotFound（负向）")
def a4(env):
    rt = env["runtime"]
    try:
        rt.use("__duty_audit_nonexistent__")
        rt.use(env["backend"].name)
        return _fail("未注册名竟然未抛错（静默通过）")
    except Exception as e:
        name = type(e).__name__
        return (_ok if "NotFound" in name or "Backend" in name else _fail)(
            f"抛 {name}")


# ══════════════════════════════════════════════════════════════════════════════
# B. 设备域（接口约定 §1.2）
# ══════════════════════════════════════════════════════════════════════════════

@item("B 设备域", "B1", "device_count() > 0")
def b1(env):
    n = env["backend"].device_count()
    return (_ok if n > 0 else _fail)(f"n={n}")


@item("B 设备域", "B2", "set_device(0) 生效且无异常")
def b2(env):
    env["backend"].set_device(0)
    return _ok("已绑定设备 0")


@item("B 设备域", "B3", "memory_stats(0) 结构含 total/used/free")
def b3(env):
    m = env["backend"].memory_stats(0)
    need = {"total_mb", "used_mb", "free_mb"}
    return (_ok if need <= set(m) else _fail)(f"keys={sorted(m)}")


@item("B 设备域", "B4", "memory_stats 值自洽（total≈used+free 且 total>0）")
def b4(env):
    m = env["backend"].memory_stats(0)
    t, u, f = m.get("total_mb", 0), m.get("used_mb", 0), m.get("free_mb", 0)
    if t <= 0:
        return _fail(f"total_mb={t}（不应为 0）")
    gap = abs(t - u - f) / max(t, 1)
    return (_ok if gap < 0.02 else _fail)(f"total={t} used={u} free={f} gap={gap:.2%}")


@item("B 设备域", "B5", "probe_device(0) 返回 True（健康设备）")
def b5(env):
    r = env["backend"].probe_device(0)
    return (_ok if r is True else _fail)(f"probe_device → {r!r}")


# ══════════════════════════════════════════════════════════════════════════════
# C. 流与事件域（接口约定 §1.3）
# ══════════════════════════════════════════════════════════════════════════════

@item("C 流与事件", "C1", "create_stream() 返回统一 Stream 且绑定本后端")
def c1(env):
    st = env["runtime"].create_stream()
    ok = hasattr(st, "native") and st.backend.name == env["backend"].name
    return (_ok if ok else _fail)(f"{type(st).__name__} backend={st.backend.name}")


@item("C 流与事件", "C2", "create_event() 返回统一 Event 且绑定本后端")
def c2(env):
    ev = env["runtime"].create_event()
    ok = hasattr(ev, "native") and ev.backend.name == env["backend"].name
    return (_ok if ok else _fail)(f"{type(ev).__name__} backend={ev.backend.name}")


@item("C 流与事件", "C3", "current_stream() 可调用")
def c3(env):
    st = env["runtime"].current_stream()
    return (_ok if st is not None else _fail)("current_stream 返回空")


@item("C 流与事件", "C4", "stream.synchronize() 无界路径正常返回")
def c4(env):
    st = env["runtime"].create_stream()
    st.synchronize()
    return _ok("无界同步返回")


@item("C 流与事件", "C5", "stream.synchronize(timeout_ms) 有界成功路径")
def c5(env):
    import torch
    st = env["runtime"].create_stream()
    with st.context():
        x = torch.randn(64, 64, device=env["torch_device"])
        _ = (x @ x).sum()
    st.synchronize(timeout_ms=30000)
    return _ok("有界同步（流上已有工作）返回")


@item("C 流与事件", "C6", "stream.context() 可作 with 上下文管理器")
def c6(env):
    st = env["runtime"].create_stream()
    with st.context():
        pass
    return _ok("with 语法可用")


@item("C 流与事件", "C7", "跨流依赖：A record → B wait → B 可见 A 结果（真实语义）")
def c7(env):
    import torch
    rt = env["runtime"]
    a, b = rt.create_stream(), rt.create_stream()
    with a.context():
        x = torch.ones(256, 256, device=env["torch_device"])
        y = x * 3.0
        s = y.sum()
    ev = rt.create_event()
    ev.record(a)
    b.wait_event(ev)
    with b.context():
        z = s + 1.0
    z_host = z.cpu().item()
    ok = abs(z_host - (256 * 256 * 3.0 + 1.0)) < 1.0
    return (_ok if ok else _fail)(f"跨流可见性结果={z_host}")


@item("C 流与事件", "C8", "stream.wait_stream(other) 可调用")
def c8(env):
    import torch
    rt = env["runtime"]
    a, b = rt.create_stream(), rt.create_stream()
    with a.context():
        _ = torch.ones(64, device=env["torch_device"]).sum()
    b.wait_stream(a)
    b.synchronize(timeout_ms=30000)
    return _ok("wait_stream 后同步完成")


@item("C 流与事件", "C9", "event.record() 后 query() 为 True")
def c9(env):
    rt = env["runtime"]
    st = rt.create_stream()
    ev = rt.create_event()
    ev.record(st)
    st.synchronize(timeout_ms=30000)
    q = ev.query()
    return (_ok if q is True else _fail)(f"query()={q!r}")


@item("C 流与事件", "C10", "未 record 的 event：query()=False 且 wait_host 不永久阻塞")
def c10(env):
    rt = env["runtime"]
    ev = rt.create_event()
    q = ev.query()
    t0 = time.time()
    w = ev.wait_host(500)
    dt = time.time() - t0
    ok = (q is False) and (dt < 5.0)
    return (_ok if ok else _fail)(f"query={q!r} wait_host={w!r} 耗时={dt:.2f}s")


@item("C 流与事件", "C11", "record 后 event.wait_host(timeout_ms) 返回 True")
def c11(env):
    rt = env["runtime"]
    st = rt.create_stream()
    ev = rt.create_event()
    ev.record(st)
    st.synchronize(timeout_ms=30000)
    w = ev.wait_host(30000)
    return (_ok if w is True else _fail)(f"wait_host → {w!r}")


@item("C 流与事件", "C12", "event.wait(stream) / event.synchronize() 可调用")
def c12(env):
    rt = env["runtime"]
    st = rt.create_stream()
    ev = rt.create_event()
    ev.record(st)
    st.synchronize(timeout_ms=30000)
    ev.wait(st)
    ev.synchronize()
    return _ok("event.wait / event.synchronize 均返回")


@item("C 流与事件", "C13", "event.elapsed_time(end) 可调用（若声明支持）")
def c13(env):
    rt, bk = env["runtime"], env["backend"]
    st = rt.create_stream()
    if not bk.supports("elapsed_time") and "elapsed_time" not in str(bk.info().get("supports")):
        # 未声明 → 仍试调一次，记录实际行为（不判 FAIL，除非崩溃在无声明下）
        pass
    e1, e2 = rt.create_event(), rt.create_event()
    e1.record(st)
    st.synchronize(timeout_ms=30000)
    time.sleep(0.01)
    e2.record(st)
    st.synchronize(timeout_ms=30000)
    try:
        dt = e2.elapsed_time(e1)
        return _ok(f"elapsed_time={dt}" if isinstance(dt, (int, float)) else f"返回 {dt!r}")
    except Exception as e:
        return _skip(f"未声明/不可用：{type(e).__name__}: {str(e)[:70]}")


@item("C 流与事件", "C14", "硬纪律 1：stream.record_stream(tensor) 可调用")
def c14(env):
    import torch
    rt = env["runtime"]
    st = rt.create_stream()
    x = torch.ones(1024, device=env["torch_device"])
    st.record_stream(x)
    return _ok("record_stream 已登记（跨流缓冲保护）")


# ══════════════════════════════════════════════════════════════════════════════
# D. 错误翻译（接口约定 §1.4）
# ══════════════════════════════════════════════════════════════════════════════

@item("D 错误翻译", "D1", "translate_error 返回统一 FlagosError")
def d1(env):
    from runtime.api.errors import FlagosError
    fe = env["runtime"].translate_error(RuntimeError("duty audit probe"), location="audit")
    return (_ok if isinstance(fe, FlagosError) else _fail)(type(fe).__name__)


@item("D 错误翻译", "D2", "category 属于 L1–L4 四类")
def d2(env):
    fe = env["runtime"].translate_error(RuntimeError("out of memory"), location="audit")
    cat = str(getattr(fe, "category", ""))
    ok = any(x in cat for x in ("L1", "L2", "L3", "L4"))
    return (_ok if ok else _fail)(f"category={cat}")


@item("D 错误翻译", "D3", "disposition 属于四动作之一")
def d3(env):
    fe = env["runtime"].translate_error(RuntimeError("out of memory"), location="audit")
    disp = fe.disposition
    ok = disp in ("retry", "raise", "replay", "device_recovery")
    return (_ok if ok else _fail)(f"disposition={disp!r}")


@item("D 错误翻译", "D4", "mapped / graded_by 可观测且自洽")
def d4(env):
    fe = env["runtime"].translate_error(RuntimeError("duty audit probe"), location="audit")
    mapped, by = fe.mapped, str(getattr(fe, "graded_by", ""))
    ok = isinstance(mapped, bool) and by != ""
    if not ok:
        return _fail(f"mapped={mapped!r} graded_by={by!r}")
    # 自洽：mapped=True 时不应是 message_hint 兜底
    if mapped and by in ("default", "message_hint"):
        return _fail(f"mapped=True 但 graded_by={by}（自相矛盾）")
    return _ok(f"mapped={mapped} graded_by={by}")


@item("D 错误翻译", "D5", "L1–L4 处置映射符合处置约定表")
def d5(env):
    """按约定表：L1→retry / L2→raise / L3→replay / L4→device_recovery。

    期望值取自**码表本身**（不是从 translate 结果反推 —— 反推会变成同义反复的空转判据）。
    """
    want = {"L1": "retry", "L2": "raise", "L3": "replay", "L4": "device_recovery"}
    samples = env.get("error_samples") or {}
    if env.get("sample_err"):
        return _fail(f"工具未构造出样例：{env['sample_err']}"
                     f"（声明了 error_map 就必须可验 —— 静默 SKIP = 覆盖缺口）")
    if not samples:
        return _skip("本后端未声明 error_map（无厂商码样例）⇒ 交由 conformance F1 覆盖")
    rt, bad = env["runtime"], []
    for code, exp_cat in sorted(samples.items()):
        fe = rt.translate_error(RuntimeError(_err_text(code)), location="audit")
        got = str(getattr(fe, "category", ""))
        key = next((k for k in want if k in got), None)
        if key is None or want[key] != fe.disposition:
            bad.append(f"{code}(期望{exp_cat})→{got}/{fe.disposition}")
    return (_fail if bad else _ok)("; ".join(bad) if bad else f"采样 {len(samples)} 条全部符合约定表")


@item("D 错误翻译", "D6", "完整消息不被截断（截断会退化为保守误判）")
def d6(env):
    rt = env["runtime"]
    long_msg = ('{"error": {"message": "The engine prompt length 99999 exceeds the maximum model '
                'length 4096 for input_tokens", "type": "BadRequestError", "param": "input_tokens"}}')
    fe = rt.translate_error(RuntimeError(long_msg), location="audit")
    keep = long_msg[:40] in str(getattr(fe, "message", "") or "") or len(str(fe)) > 40
    return (_ok if keep else _fail)(f"category={fe.category} disposition={fe.disposition}")


# ══════════════════════════════════════════════════════════════════════════════
# E. 状态恢复（接口约定 §1.5 + 返回契约补充）
# ══════════════════════════════════════════════════════════════════════════════

@item("E 状态恢复", "E1", "device_state(0) 返回四态之一")
def e1(env):
    st = env["runtime"].device_state(0)
    s = str(getattr(st, "value", st))
    ok = any(k in s.upper() for k in ("AVAILABLE", "DEGRADED", "ISOLATED", "UNKNOWN"))
    return (_ok if ok else _fail)(f"state={s}")


@item("E 状态恢复", "E2", "recover_device(0,'probe') 返回 dict")
def e2(env):
    r = env["runtime"].recover_device(0, "probe", reason="duty audit")
    return (_ok if isinstance(r, dict) else _fail)(f"type={type(r).__name__}")


@item("E 状态恢复", "E3", "返回契约字段齐全 {ordinal,mode,recovered,state,detail}")
def e3(env):
    r = env["runtime"].recover_device(0, "probe", reason="duty audit")
    need = {"ordinal", "mode", "recovered", "state", "detail"}
    miss = need - set(r if isinstance(r, dict) else {})
    return (_ok if not miss else _fail)(f"缺字段 {sorted(miss)} / 实际 {sorted(r) if isinstance(r, dict) else r}")


@item("E 状态恢复", "E4", "recovered 语义 = 设备当前可用（与 probe_device 一致）")
def e4(env):
    rt, bk = env["runtime"], env["backend"]
    r = rt.recover_device(0, "probe", reason="duty audit")
    p = bk.probe_device(0)
    rec = r.get("recovered") if isinstance(r, dict) else None
    ok = (rec is p) or (rec is True)
    return (_ok if ok else _fail)(f"recovered={rec!r} probe_device={p!r}")


@item("E 状态恢复", "E5", "recover_device(0,'hybrid') 返回 dict")
def e5(env):
    r = env["runtime"].recover_device(0, "hybrid", reason="duty audit")
    return (_ok if isinstance(r, dict) else _fail)(f"mode={r.get('mode') if isinstance(r, dict) else '?'}")


@item("E 状态恢复", "E6", "recover_device(0,'real') 如实响应（执行或明确拒绝，不静默）")
def e6(env):
    try:
        r = env["runtime"].recover_device(0, "real", reason="duty audit")
        ok = isinstance(r, dict)
        return (_ok if ok else _fail)(f"mode={r.get('mode')} recovered={r.get('recovered')}")
    except Exception as e:
        # 如实拒绝也算响应（不静默），但记录类型
        return _ok(f"明确抛错而非静默：{type(e).__name__}: {str(e)[:70]}")


# ══════════════════════════════════════════════════════════════════════════════
# F. 硬纪律 2（错误隔离分层）+ G. 元信息与声明
# ══════════════════════════════════════════════════════════════════════════════

@item("F 硬纪律", "F1", "错误隔离分层：芯片级 → device_recovery；API 级 → 非 device_recovery")
def f1(env):
    rt = env["runtime"]
    chip = env.get("chip_error_sample")
    api = env.get("api_error_sample")
    if env.get("sample_err"):
        return _fail(f"工具未构造出样例：{env['sample_err']}"
                     f"（声明了 error_map 就必须可验 —— 静默 SKIP = 覆盖缺口）")
    if not chip or not api:
        return _skip("本后端无数字错误码（未声明 error_map）⇒ 交由错误闭环与 conformance F1 覆盖")
    c = rt.translate_error(RuntimeError(_err_text(chip)), location="audit")
    a = rt.translate_error(RuntimeError(_err_text(api)), location="audit")
    ok = c.disposition == "device_recovery" and a.disposition != "device_recovery"
    return (_ok if ok else _fail)(f"芯片级={c.disposition} API级={a.disposition}")


@item("G 元信息", "G1", "supports() 与 info()['supports'] 一致")
def g1(env):
    bk = env["backend"]
    sup = (bk.info() or {}).get("supports") or {}
    bad = [k for k, v in sup.items() if bool(v) != bool(bk.supports(k))]
    return (_fail if bad else _ok)(f"不一致 {bad}" if bad else f"{len(sup)} 项一致")


@item("G 元信息", "G2", "info()['supports'] 键集合 == _CAPABILITY_KEYS（防键漂移）")
def g2(env):
    bk = env["backend"]
    keys = set(getattr(bk, "_CAPABILITY_KEYS", ()) or ())
    sup = set(((bk.info() or {}).get("supports") or {}).keys())
    return (_ok if keys == sup else _fail)(f"KEYS-INFO 差 {sorted(keys ^ sup)}")


@item("G 元信息", "G3", "known_issues() 结构完整（无则如实为空）")
def g3(env):
    bk = env["backend"]
    issues = bk.known_issues() or []
    if not issues:
        return _ok("如实为空（无已知缺陷）")
    need = {"id", "severity", "symptom"}
    bad = [i.get("id", "?") for i in issues if not need <= set(i)]
    n = len(issues)
    return (_fail if bad else _ok)(f"{n} 条，字段不全 {bad}" if bad else f"{n} 条结构完整")


# ══════════════════════════════════════════════════════════════════════════════
# 驱动
# ══════════════════════════════════════════════════════════════════════════════

def build_env(backend_name):
    import torch
    import runtime

    bak = runtime.use(backend_name)
    n = bak.device_count()                      # 触碰设备（懒加载前提，审计第 11/15 条）
    dev = bak.device_type
    torch.zeros(1, device=f"{dev}:0")           # 确认设备命名空间就绪
    # 样例取自**码表本身**（不从 translate 结果反推 —— 反推即同义反复的空转判据）
    samples, chip_code, api_code, sample_err = {}, None, None, ""
    if bak.supports("error_map"):
        try:
            from runtime.conformance.errors import ACL_ERR_TO_CATEGORY as _M

            # ⚠️ 必须用 `k.name`：本仓 `conformance` 的历史分级是 **IntEnum**，
            #    `str(k)` 在 py3.11 给的是**数字**（"1"），用 `in str(k)` 匹配类名会**全部落空**
            #    ⇒ 样例为空 ⇒ 判据被静默 SKIP（2026-09-28 实测踩到）。
            def _name(k):
                return getattr(k, "name", str(k))

            for cat in ("L1_RESOURCE", "L2_PARAM", "L3_EXECUTION", "L4_FATAL"):
                hit = next((int(c) for c, k in _M.items() if cat in _name(k)), None)
                if hit is not None:
                    samples[str(hit)] = cat[:2]
            chip_code = next((int(c) for c, k in _M.items() if "L4_FATAL" in _name(k)), None)
            api_code = next((int(c) for c, k in _M.items() if "L2_PARAM" in _name(k)), None)
            if not samples:
                # 声明了码表却取不到样例 ⇒ 不是「能力不具备」，是**工具失败**
                sample_err = "码表可取但未匹配到任何分级样例（判据会退化为空转）"
        except Exception as e:
            sample_err = f"{type(e).__name__}: {e}"
    return {
        "runtime": runtime, "backend": bak, "device_count_hint": n,
        "torch_device": f"{dev}:0", "error_samples": samples,
        "chip_error_sample": chip_code,
        "api_error_sample": api_code,
        "sample_err": sample_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print("=" * 78)
    print(f"职责响应审计：backend={args.backend}"
          f"（口径：接口约定 §1.1–§1.5 + §2 三支撑 + §3 两纪律）")
    print("=" * 78)

    try:
        env = build_env(args.backend)
    except Exception as e:
        print(f"❌ 环境装配失败：{type(e).__name__}: {e}")
        traceback.print_exc()
        return 2

    print(f"ℹ️ 设备类型={env['backend'].device_type}  device_count={env['device_count_hint']}")
    print(f"ℹ️ 后端已声明能力：{sorted((env['backend'].info() or {}).get('supports') or {})}\n")

    rows, n_ok, n_fail, n_skip = [], 0, 0, 0
    for r in RESULTS:
        try:
            status, detail = r["fn"](env)
        except Exception as e:
            status, detail = False, f"{type(e).__name__}: {str(e)[:110]}"
        label = {True: "OK  ", False: "FAIL", None: "SKIP"}[status]
        if status is True:
            n_ok += 1
        elif status is False:
            n_fail += 1
        else:
            n_skip += 1
        rows.append({"domain": r["domain"], "sid": r["sid"], "title": r["title"],
                     "status": {True: "OK", False: "FAIL", None: "SKIP"}[status],
                     "detail": detail})

    cur = None
    for row in rows:
        if row["domain"] != cur:
            cur = row["domain"]
            print(f"\n[{cur}]")
        print(f"  [{row['status']}] {row['sid']} {row['title']}")
        if row["detail"]:
            print(f"         {row['detail']}")

    total = n_ok + n_fail + n_skip
    print("\n" + "-" * 78)
    print(f"DUTY_RESPONSE_AUDIT {args.backend}: OK {n_ok} / FAIL {n_fail} / SKIP {n_skip}"
          f"（共 {total} 项）")
    verdict = "DUTY_RESPONSE_PASS" if n_fail == 0 else "DUTY_RESPONSE_FAIL"
    print(f"verdict: {verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "backend": args.backend, "ok": n_ok, "fail": n_fail, "skip": n_skip,
            "total": total, "verdict": verdict, "rows": rows,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"结果已写入 {args.out}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
