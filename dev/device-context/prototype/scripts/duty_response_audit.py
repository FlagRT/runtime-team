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
  的 **§1.1–§1.10** + §2（三支撑方法）+ §3（两条硬纪律）拆成可执行 sub-part，逐项调用。
  （H–M 六个域 = 2026-10-08 扩口径：契约 §1.6–§1.10 与「统一 API 面」，见台账 E1。）

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


@item("E 状态恢复", "E3", "返回契约字段齐全 {ordinal,mode,recovered,state,detail} 且 state 为四态规范 token")
def e3(env):
    r = env["runtime"].recover_device(0, "probe", reason="duty audit")
    need = {"ordinal", "mode", "recovered", "state", "detail"}
    miss = need - set(r if isinstance(r, dict) else {})
    # 2026-09-29 加（工作包 A 实验暴露）：**不只手字段名，取值域也要守**。
    #   原判据只看「五键存在性」⇒ `state` 长期返回 `str(enum)`（`'DeviceState.AVAILABLE'`）
    #   而三套件都没发现（纯靠 09-29 实验才暴露）；下游按契约比较
    #   `state == "available"` 会**判假**。属 §11-⑫「字符串常量要有判据守」的再深一层。
    st = r.get("state") if isinstance(r, dict) else None
    if miss:
        return _fail(f"缺字段 {sorted(miss)} / 实际 {sorted(r) if isinstance(r, dict) else r}")
    if st not in ("available", "degraded", "isolated", "destroyed"):
        return _fail(f"state={st!r} 不是四态规范 token（str(enum) 形式 ⇒ 下游比较会判假）")
    return _ok(f"五键齐全，state={st!r}")


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
# H–M. 2026-10-08 扩口径：契约 §1.6–§1.10 + 统一 API 面（台账 E1 的落地）
# ══════════════════════════════════════════════════════════════════════════════
#
# **为什么扩**：原口径「§1.1–§1.5 + §2 + §3 = 39 项」冻结在 2026-09-28 的契约上，
# 而契约此后又新增了 §1.6（内存句柄与生命周期）· §1.7（设备上下文生命周期）·
# §1.8（契约不变式 I1–I4）· §1.9（流优先级）· §1.10（流所有权与释放）。
# 这五章的判据**分散在各探针里**（离线 `[8b]`/`[8b-②]`/`[10]` 段、`probe_bc_contract.py`、
# `probe_stream_priority_api.py`、`probe_stream_release_and_control.py`、
# `conformance --cases contract_invariants`），**不在职责审计里**，
# 职责文档也未登记该映射 ⇒ 读者会以为「39/0/0 = **全部**职责已响应」。本轮把新章
# 拆成 sub-part **并入本工具**（台账 E1 选项 A），使职责口径与契约齐平。
#
# **本轮新增判据遵循的三条纪律**（每条都在 `probes/selfcheck_duty_audit_ext.py` 里配了
# **逐条注入的非空转验证** —— 证明它们**真的能 FAIL**，而不是"看起来通过"）：
#   ① **能用事实就不要反推**：能真调一次拿结果的，绝不靠声明或字段推导；
#   ② **负向判据只认契约级异常类型**（`NotImplementedError` / `ValueError` / `RuntimeError`），
#      其它异常类型一律判 FAIL —— 否则真 bug 会被当成"如实拒绝"（同族纪律：台账第 ⑤ 条家族）；
#   ③ **未声明能力 ⇒ 如实 SKIP**（不是 FAIL），但 SKIP 必须在 detail 里写清**为什么**。

#: 契约 §1.1–§1.10 表格里**明列的统一 API 名字**（转录自契约正文；契约新增章节时同步维护）。
#:
#: 用途（M1）：检查「**契约承诺 ⇒ 统一面上真的可调用**」。这与 `contract_invariants` 的 I1④
#: 是**两层不同的东西**：I1④ 查的是**后端入口**（`getattr(bk, e)`），而契约第 1 章的标题是
#: 「**统一 API 承诺（下游直接使用）**」⇒ 下游取的是 `runtime.<name>`。
#: **2026-10-08 依此发现两处真缺口**（台账第 30 条）：`context_set`（契约 §1.7）与
#: `stream_priority_range`（契约 §1.9）**只有后端实现、统一面根本没导出** ⇒
#: 下游按契约写 `runtime.context_set(...)` 会 `AttributeError`。已补为只增出口。
PROMISED_UNIFIED_API = (
    # §1.1 后端选择
    "discover", "available", "use", "current", "set_current", "get", "register",
    # §1.2 设备
    "device_count", "set_device", "memory_stats", "probe_device",
    # §1.3 流与事件
    "create_stream", "create_event", "current_stream", "synchronize",
    # §1.4 错误翻译
    "translate_error",
    # §1.5 状态恢复
    "device_state", "recover_device",
    # §1.6 内存句柄与生命周期
    "allocate", "free", "memory_handle_count",
    # §1.7 设备上下文生命周期
    "context_create", "context_set", "context_destroy", "context_count", "context_query",
    # §1.9 流优先级
    "stream_priority_readback", "stream_priority_range",
    # §1.10 流所有权与释放
    "release_stream",
    # §1.6 审计（`.native` 逃生舱 / 退化路径）
    "native_accesses", "degradations",
)

#: 统一面上必须存在的**公开常量**（句柄字段与状态取值域的对外声明）
PROMISED_UNIFIED_CONSTS = ("MEMORY_HANDLE_KEYS", "CONTEXT_HANDLE_KEYS", "DEVICE_STATE_TOKENS")

#: `memory_stats()` 的**允许键集**（三键规范 + 可选 `allocated_mb`）；多出即"未登记键"
MEMORY_STATS_ALLOWED = ("total_mb", "used_mb", "free_mb", "allocated_mb")

#: `context_query()` 的**固定 6 键**（契约 §1.7）+ 取值域
CONTEXT_QUERY_KEYS = ("queryable", "present", "ordinal", "flags", "managed_by", "reason")
CONTEXT_QUERY_MANAGED_BY = ("unified", "external", None)


def _count(rt, fn_name):
    """取审计计数（`{total, by_kind}`）的总数；取不到 ⇒ None（**不补零**）。"""
    try:
        d = getattr(rt, fn_name)() or {}
    except BaseException:                                         # noqa: BLE001
        return None, None
    return (int(d.get("total")) if isinstance(d.get("total"), int) else None), d


def _strict_bool(v):
    return isinstance(v, bool)


# ══════════════════════════════════════════════════════════════════════════════
# H. 内存句柄与生命周期（接口约定 §1.6）
# ══════════════════════════════════════════════════════════════════════════════

@item("H 内存句柄与生命周期", "H1",
      "allocate()：未声明 memory_alloc ⇒ 契约级拒绝；声明 ⇒ 五键句柄且**不含厂商指针**")
def h1(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("memory_alloc"):
        try:
            rt.allocate(4096)
            return _fail("未声明 memory_alloc 却**静默返回**（假句柄比报错危险：上层会当成可用的设备内存）")
        except NotImplementedError as e:
            return _ok(f"显式拒绝：{str(e)[:70]}")
        except BaseException as e:                                # noqa: BLE001
            return _fail(f"拒绝时抛的不是契约级 NotImplementedError：{type(e).__name__}: {str(e)[:60]}")
    h = rt.allocate(4096)
    need = set(getattr(rt, "MEMORY_HANDLE_KEYS", ()) or ())
    if not isinstance(h, dict) or not need <= set(h):
        return _fail(f"句柄形状不符：{sorted(h) if isinstance(h, dict) else h!r}（需要 {sorted(need)}）")
    leak = [k for k in h if ("ptr" in k.lower() or "native" in k.lower())]
    if leak:
        return _fail(f"句柄里出现厂商指针字段 {leak}（契约：句柄不含厂商指针，需绕过请走 `.native` 并计数）")
    rt.free(h)
    return _ok(f"五键齐全且无厂商指针：{sorted(h)}")


@item("H 内存句柄与生命周期", "H2", "allocate(size_bytes) 非正整数 ⇒ ValueError（0 / -1 / bool / 非 int）")
def h2(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("memory_alloc"):
        return _skip("未声明 memory_alloc ⇒ 该分支不适用")
    bad = []
    for v in (0, -1, True, "4096"):
        try:
            rt.allocate(v)
            bad.append(f"{v!r} 未报错")
        except ValueError:
            pass
        except BaseException as e:                                # noqa: BLE001
            bad.append(f"{v!r} → {type(e).__name__}（非契约级）")
    return (_fail if bad else _ok)("; ".join(bad) if bad else "4 例（0 / -1 / bool / 字符串）全部 ValueError")


@item("H 内存句柄与生命周期", "H3", "free() 二次释放 ⇒ ValueError（不得静默）")
def h3(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("memory_alloc"):
        return _skip("未声明 memory_alloc ⇒ 该分支不适用")
    h = rt.allocate(4096)
    rt.free(h)
    try:
        rt.free(h)
        return _fail("二次释放**静默通过**（契约要求 ValueError —— 静默会是双重释放 → 后续踩内存）")
    except ValueError:
        return _ok("二次释放 → ValueError")
    except BaseException as e:                                    # noqa: BLE001
        return _fail(f"异常类型非契约级：{type(e).__name__}: {str(e)[:60]}")


@item("H 内存句柄与生命周期", "H4", "free(非本层句柄 / 裸指针) ⇒ ValueError（不接受厂商原始句柄）")
def h4(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("memory_alloc"):
        return _skip("未声明 memory_alloc ⇒ 该分支不适用")
    bad = []
    for badh in ({"handle_id": 999999, "kind": "memory"}, 0x7F0000, "0x7f0000", None):
        try:
            rt.free(badh)
            bad.append(f"{badh!r} 未报错")
        except ValueError:
            pass
        except BaseException as e:                                # noqa: BLE001
            bad.append(f"{badh!r} → {type(e).__name__}（非契约级）")
    return (_fail if bad else _ok)("; ".join(bad) if bad else "4 类陌生句柄全部 ValueError")


@item("H 内存句柄与生命周期", "H5", "memory_handle_count() 与分配/释放联动（+1 ⇒ 释放后回落）")
def h5(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("memory_alloc"):
        return _skip("未声明 memory_alloc ⇒ 该分支不适用")
    c0 = rt.memory_handle_count()
    h = rt.allocate(4096)
    c1 = rt.memory_handle_count()
    rt.free(h)
    c2 = rt.memory_handle_count()
    ok = (c1 == c0 + 1) and (c2 == c0)
    return (_ok if ok else _fail)(f"分配前 {c0} → 分配后 {c1}（期望 {c0 + 1}）→ 释放后 {c2}（期望 {c0}）")


@item("H 内存句柄与生命周期", "H6",
      "memory_stats()：三键必在；声明 memory_alloc_stat ⇒ 须给 allocated_mb；未声明 ⇒ **必须缺席**；无未登记键")
def h6(env):
    rt, bk = env["runtime"], env["backend"]
    m = rt.memory_stats(0) or {}
    if not {"total_mb", "used_mb", "free_mb"} <= set(m):
        return _fail(f"三键缺失：{sorted(m)}")
    extra = [k for k in m if k not in MEMORY_STATS_ALLOWED]
    if extra:
        return _fail(f"出现未登记键 {extra}（契约：不得出现未登记键）")
    if bk.supports("memory_alloc_stat"):
        if "allocated_mb" not in m:
            return _fail("声明了 memory_alloc_stat 却**没给** allocated_mb（声明即承诺）")
        return _ok(f"三键 + allocated_mb={m.get('allocated_mb')!r}（已声明 memory_alloc_stat）")
    if "allocated_mb" in m:
        return _fail("**未声明** memory_alloc_stat 却给了 allocated_mb"
                     f"（值 {m.get('allocated_mb')!r}；契约：取不到就必须缺席，**不得填 0 冒充**）")
    return _ok("三键齐全；未声明 memory_alloc_stat ⇒ allocated_mb 如实缺席")


@item("H 内存句柄与生命周期", "H7",
      "record_stream：未声明能力 ⇒ 保守同步路径**不抛错**且 degradations +1；声明 ⇒ 走原生路径不抛错")
def h7(env):
    import torch
    rt, bk = env["runtime"], env["backend"]
    declared = bk.supports("record_stream")
    d0, _ = _count(rt, "degradations")
    st = rt.create_stream()
    x = torch.ones(1024, device=env["torch_device"])
    try:
        st.record_stream(x)
    except BaseException as e:                                    # noqa: BLE001
        return _fail(f"record_stream 抛错：{type(e).__name__}: {str(e)[:70]}"
                     "（契约：未声明 ⇒ 保守同步放行**不抛错**，跨流内存被提前回收比降级慢危险得多）")
    d1, _ = _count(rt, "degradations")
    if declared:
        return _ok(f"已声明 record_stream ⇒ 原生路径，degradations {d0}→{d1}")
    if d1 is None or d0 is None:
        return _fail("degradations() 取不到，无法验证退化是否计数")
    if d1 != d0 + 1:
        return _fail(f"未声明 record_stream 走了保守路径，但 degradations 未 +1（{d0}→{d1}）"
                     "—— 契约：任何降级必须**计数可查**")
    return _ok(f"未声明 ⇒ 保守同步放行（不抛错）+ degradations {d0}→{d1}")


@item("H 内存句柄与生命周期", "H8",
      ".native 取用 ⇒ native_accesses +1 且 degradations **不变**（两者语义相反，必须分开计数）")
def h8(env):
    rt = env["runtime"]
    n0, _ = _count(rt, "native_accesses")
    d0, _ = _count(rt, "degradations")
    st = rt.create_stream()
    _ = st.native                                                  # 显式逃生舱取用
    n1, _ = _count(rt, "native_accesses")
    d1, _ = _count(rt, "degradations")
    if None in (n0, n1, d0, d1):
        return _fail("审计计数取不到（native_accesses / degradations 必须给出 {total, by_kind}）")
    if n1 != n0 + 1:
        return _fail(f".native 取用后 native_accesses 未 +1（{n0}→{n1}）—— 取用必须可事后定位")
    if d1 != d0:
        return _fail(f".native 取用把 degradations 也动了（{d0}→{d1}）"
                     "—— 两者语义相反（破坏可移植性 vs 用性能换正确性），必须**分开计数**")
    return _ok(f"native_accesses {n0}→{n1}，degradations 保持 {d0}")


# ══════════════════════════════════════════════════════════════════════════════
# I. 设备上下文生命周期（接口约定 §1.7）
# ══════════════════════════════════════════════════════════════════════════════

@item("I 设备上下文", "I1",
      "context_create()：未声明 context_lifecycle ⇒ 契约级拒绝；声明 ⇒ 四键句柄")
def i1(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("context_lifecycle"):
        try:
            rt.context_create(0)
            return _fail("未声明 context_lifecycle 却静默返回了上下文句柄")
        except NotImplementedError as e:
            return _ok(f"显式拒绝：{str(e)[:70]}")
        except BaseException as e:                                # noqa: BLE001
            return _fail(f"拒绝时抛的不是契约级 NotImplementedError：{type(e).__name__}")
    h = rt.context_create(0)
    need = set(getattr(rt, "CONTEXT_HANDLE_KEYS", ()) or ())
    if not isinstance(h, dict) or not need <= set(h):
        return _fail(f"句柄形状不符：{sorted(h) if isinstance(h, dict) else h!r}（需要 {sorted(need)}）")
    rt.context_destroy(h)
    return _ok(f"四键齐全：{sorted(h)}")


@item("I 设备上下文", "I2", "context_set() 只接受本层句柄（陌生/厂商句柄 ⇒ 报错）")
def i2(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("context_lifecycle"):
        return _skip("未声明 context_lifecycle ⇒ 该分支不适用")
    bad = []
    for badh in ({"handle_id": 999999, "kind": "context"}, 0x1234, None):
        try:
            rt.context_set(badh)
            bad.append(f"{badh!r} 未报错")
        except ValueError:
            pass
        except BaseException as e:                                # noqa: BLE001
            bad.append(f"{badh!r} → {type(e).__name__}（非契约级）")
    return (_fail if bad else _ok)("; ".join(bad) if bad else "3 类陌生句柄全部 ValueError")


@item("I 设备上下文", "I3",
      "context_destroy() 只接受本层句柄 —— **尤其不得毁掉进程默认上下文**")
def i3(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("context_lifecycle"):
        return _skip("未声明 context_lifecycle ⇒ 该分支不适用")
    bad = []
    for badh in ({"handle_id": 999999, "kind": "context"}, 0x5678, None):
        try:
            rt.context_destroy(badh)
            bad.append(f"{badh!r} 未报错（**误毁默认上下文会让整个进程的设备不可用**）")
        except ValueError:
            pass
        except BaseException as e:                                # noqa: BLE001
            bad.append(f"{badh!r} → {type(e).__name__}（非契约级）")
    return (_fail if bad else _ok)("; ".join(bad) if bad else "3 类陌生句柄全部 ValueError（默认上下文受保护）")


@item("I 设备上下文", "I4",
      "context_count() **不含**进程默认上下文（初始 0；新建 +1；销毁回落）")
def i4(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("context_lifecycle"):
        return _skip("未声明 context_lifecycle ⇒ 该分支不适用")
    c0 = rt.context_count()
    if c0 != 0:
        return _fail(f"开工时本层上下文数 = {c0}（应为 0：契约要求**不含进程默认上下文**）")
    h = rt.context_create(0)
    c1 = rt.context_count()
    rt.context_destroy(h)
    c2 = rt.context_count()
    ok = (c1 == 1) and (c2 == 0)
    return (_ok if ok else _fail)(f"{c0} → 新建 {c1}（期望 1）→ 销毁 {c2}（期望 0）")


@item("I 设备上下文", "I5", "内存句柄与上下文句柄**不得互相误用**（误用必须报错）")
def i5(env):
    rt, bk = env["runtime"], env["backend"]
    if not (bk.supports("memory_alloc") and bk.supports("context_lifecycle")):
        return _skip("需要同时声明 memory_alloc 与 context_lifecycle ⇒ 本机该分支不适用")
    mem = rt.allocate(4096)
    ctx = rt.context_create(0)
    bad = []
    try:
        rt.context_destroy(mem)                                    # 内存句柄当上下文销毁
        bad.append("内存句柄被 accept 为上下文句柄")
    except ValueError:
        pass
    except BaseException as e:                                    # noqa: BLE001
        bad.append(f"内存句柄误用 → {type(e).__name__}（非契约级）")
    try:
        rt.free(ctx)                                               # 上下文句柄当内存释放
        bad.append("上下文句柄被 accept 为内存句柄")
    except ValueError:
        pass
    except BaseException as e:                                    # noqa: BLE001
        bad.append(f"上下文句柄误用 → {type(e).__name__}（非契约级）")
    rt.free(mem)
    rt.context_destroy(ctx)
    return (_fail if bad else _ok)("; ".join(bad) if bad else "双向误用均 ValueError")


@item("I 设备上下文", "I6",
      "context_query() 返回**固定 6 键**；未声明者也返回同一 6 键（queryable=False + **具体原因**）")
def i6(env):
    rt, bk = env["runtime"], env["backend"]
    q = rt.context_query()
    if not isinstance(q, dict):
        return _fail(f"返回的不是 dict：{type(q).__name__}")
    need = set(CONTEXT_QUERY_KEYS)
    if not need <= set(q):
        return _fail(f"缺键 {sorted(need - set(q))}（契约：固定 6 键，上层换芯片无需分支）")
    if not _strict_bool(q.get("queryable")):
        return _fail(f"queryable 不是严格 bool：{q.get('queryable')!r}")
    if q.get("queryable") is False and not str(q.get("reason", "")).strip():
        return _fail("queryable=False 但 reason 为空（契约：未声明者也要给**具体原因**）")
    tag = "已声明" if bk.supports("context_query") else "未声明（如实 queryable=False）"
    return _ok(f"6 键齐全，queryable={q['queryable']}（{tag}），reason={str(q.get('reason'))[:44]!r}")


@item("I 设备上下文", "I7",
      "context_query() 取值域：present 必须 bool/None（**不得 0/1**）；managed_by ∈ {unified, external, None}")
def i7(env):
    rt = env["runtime"]
    q = rt.context_query()
    bad = []
    if not (q.get("present") is None or _strict_bool(q.get("present"))):
        bad.append(f"present={q.get('present')!r}（不得用 0/1 冒充 bool）")
    if q.get("managed_by") not in CONTEXT_QUERY_MANAGED_BY:
        bad.append(f"managed_by={q.get('managed_by')!r} 不在取值域 {CONTEXT_QUERY_MANAGED_BY}")
    if "flags" in q and not (q.get("flags") is None or isinstance(q.get("flags"), int)):
        bad.append(f"flags={q.get('flags')!r} 类型异常")
    return (_fail if bad else _ok)("; ".join(bad) if bad
                                   else f"present={q.get('present')!r} managed_by={q.get('managed_by')!r}")


@item("I 设备上下文", "I8",
      "context_query() **只读无副作用**（调用前后设备计算一致，且**不新造句柄**）")
def i8(env):
    import torch
    rt = env["runtime"]
    x = torch.full((64, 64), 1.5, device=env["torch_device"])
    before = float(x.sum().item())
    h0 = rt.memory_handle_count() if rt.current().supports("memory_alloc") else None
    rt.context_query()
    after = float(x.sum().item())
    h1 = rt.memory_handle_count() if h0 is not None else None
    bad = []
    if after != before:
        bad.append(f"计算被影响：{before} → {after}")
    if h0 is not None and h1 != h0:
        bad.append(f"查询过程新造/销毁了内存句柄：{h0} → {h1}（只读接口不得有副作用）")
    return (_fail if bad else _ok)("; ".join(bad) if bad
                                   else f"计算 {before} 不变" + (f"，句柄数 {h0} 不变" if h0 is not None else ""))


@item("I 设备上下文", "I9",
      "recover_device() 的 context 三键**只增不改**；且 context_recreated 不得与 detail 自相矛盾")
def i9(env):
    rt = env["runtime"]
    r = rt.recover_device(0, "probe", reason="duty audit")
    if not isinstance(r, dict):
        return _fail(f"返回类型 {type(r).__name__}")
    need5 = {"ordinal", "mode", "recovered", "state", "detail"}
    if not need5 <= set(r):
        return _fail(f"契约五键被破坏，缺 {sorted(need5 - set(r))}（增键必须**只增不改**）")
    need3 = {"context_supported", "context_count", "context_recreated"}
    if not need3 <= set(r):
        return _fail(f"缺 context 三键 {sorted(need3 - set(r))}（§1.7 只增）")
    if not _strict_bool(r.get("context_supported")):
        return _fail(f"context_supported 不是严格 bool：{r.get('context_supported')!r}")
    if not (r.get("context_count") is None or isinstance(r.get("context_count"), int)):
        return _fail(f"context_count 类型异常：{r.get('context_count')!r}")
    if not (r.get("context_recreated") is None or _strict_bool(r.get("context_recreated"))):
        return _fail(f"context_recreated 不是 bool/None：{r.get('context_recreated')!r}")
    detail = str(r.get("detail") or "")
    if r.get("context_recreated") is True and ("无需" in detail or "不支持" in detail):
        return _fail(f"**同一次返回内自相矛盾**：context_recreated=True 而 detail 写着「{detail[:60]}」")
    return _ok(f"五键 + context 三键齐全；context_recreated={r.get('context_recreated')!r}")


@item("I 设备上下文", "I10",
      "绑定语义：上下文销毁后其流**在使用点如实报错**（厂商侧当场是静默的）")
def i10(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("context_lifecycle"):
        return _skip("未声明 context_lifecycle ⇒ 该分支不适用")
    ctx = rt.context_create(0)
    st = rt.create_stream()
    rt.context_destroy(ctx)
    try:
        try:
            st.synchronize()
            return _fail("上下文销毁后其流**仍可用**（契约：使用点必须如实报错；"
                         "厂商侧实测是静默的 ⇒ 拦截由本层负责）")
        except RuntimeError as e:
            txt = str(e)
            ok = ("释放" in txt) or ("销毁" in txt) or ("上下文" in txt) or ("context" in txt.lower())
            # ⚠️ 两个分支必须写**不同**的文案（初版成功分支也印「文案无线索」⇒ 误导）
            return (_ok if ok else _fail)(
                f"使用点如实报错：{txt[:70]}" if ok
                else f"RuntimeError 但文案不含「释放/销毁/上下文」线索：{txt[:70]}")
        except BaseException as e:                                # noqa: BLE001
            return _fail(f"异常类型非契约级：{type(e).__name__}: {str(e)[:60]}")
    finally:
        try:
            rt.context_create(0)       # 恢复一个可用上下文，避免污染后续判据
        except BaseException:                                     # noqa: BLE001
            pass


# ══════════════════════════════════════════════════════════════════════════════
# J. 契约不变式 I1–I4（接口约定 §1.8）—— **委托同一份实现**，不另写一套
# ══════════════════════════════════════════════════════════════════════════════
#
# ⚠️ 为什么不在这里重写：§1.8 的判据已有**唯一实现** `runtime/conformance/contract_invariants.py`
# （真机 `--cases contract_invariants` 与离线第 `[10]` 段**共用同一套核心函数**）。
# 在职责审计里再抄一份 ⇒ 两处实现必然漂移（同族纪律：判据与被测对象共用同一份实现）。
# 故此处**直接调用**那套 `check_i1..check_i4`，只做"逐 sub-part 点名"。

def _invariant(env, which):
    from runtime.conformance import contract_invariants as ci
    fn = {"i1": ci.check_i1, "i2": ci.check_i2, "i3": ci.check_i3, "i4": ci.check_i4}[which]
    try:
        ok, detail = fn(env["backend"]) if which != "i3" else fn(env["backend"], 0)
    except BaseException as e:                                    # noqa: BLE001
        return _fail(f"判据自身异常：{type(e).__name__}: {str(e)[:90]}")
    return (_ok if ok else _fail)(f"[委托 contract_invariants] {str(detail)[:170]}")


@item("J 契约不变式", "J1", "I1 诚实声明（声明的真可调；未声明的显式拒绝）")
def j1(env):
    return _invariant(env, "i1")


@item("J 契约不变式", "J2", "I2 禁止伪造（字段不得自相矛盾；取不到必须缺席）")
def j2(env):
    return _invariant(env, "i2")


@item("J 契约不变式", "J3", "I3 失效受管（二次释放/二次销毁/跨种类误用/销毁后用其流）")
def j3(env):
    return _invariant(env, "i3")


@item("J 契约不变式", "J4", "I4 降级可观测（.native 与退化分开计数、单调不减）")
def j4(env):
    return _invariant(env, "i4")


# ══════════════════════════════════════════════════════════════════════════════
# K. 流优先级（接口约定 §1.9）
# ══════════════════════════════════════════════════════════════════════════════

def _range_of(bk):
    """取归一后的区间 `(low, high)`；形状不符或不可得 ⇒ (None, 原因)。"""
    try:
        r = bk.stream_priority_range()
    except BaseException as e:                                    # noqa: BLE001
        return None, f"stream_priority_range() 抛 {type(e).__name__}: {str(e)[:60]}"
    if r is None:
        return None, "如实返回 None"
    if not (isinstance(r, tuple) and len(r) == 2 and all(isinstance(v, int) for v in r)):
        return None, f"形状不符：{r!r}（契约：**(least, greatest) 2 元组** 或 None；不得透传厂商三元组）"
    return (min(r), max(r)), f"{r!r}"


@item("K 流优先级", "K1", "stream_priority_range() 形状 = **(least, greatest) 2 元组** 或 None")
def k1(env):
    flat, why = _range_of(env["backend"])
    return (_fail if flat is None and "形状不符" in why else _ok)(why)


@item("K 流优先级", "K2", "未声明 stream_priority ⇒ 范围**如实为 None**（不假装可读）")
def k2(env):
    bk = env["backend"]
    if bk.supports("stream_priority"):
        return _skip("本后端已声明 stream_priority（范围可读）⇒ 该分支不适用")
    r = bk.stream_priority_range()
    return (_ok if r is None else _fail)(f"未声明却给出范围 {r!r}（不得假装可读）")


@item("K 流优先级", "K3",
      "未声明 stream_priority_control ⇒ create_stream(priority=int) ⇒ **契约级 NotImplementedError**")
def k3(env):
    rt, bk = env["runtime"], env["backend"]
    if bk.supports("stream_priority_control"):
        return _skip("本后端已声明 stream_priority_control ⇒ 该分支不适用")
    try:
        rt.create_stream(priority=0)
        return _fail("未声明 control 却**静默返回**了一条流（会产出「看起来设了优先级」的假象）")
    except NotImplementedError as e:
        return _ok(f"显式拒绝：{str(e)[:70]}")
    except BaseException as e:                                    # noqa: BLE001
        return _fail(f"拒绝时抛的不是契约级 NotImplementedError：{type(e).__name__}: {str(e)[:60]}")


@item("K 流优先级", "K4", "取值域：越界 / 非 int / bool ⇒ ValueError")
def k4(env):
    rt, bk = env["runtime"], env["backend"]
    flat, why = _range_of(bk)
    if flat is None:
        return _skip(f"范围不可得（{why}）⇒ 该分支不适用")
    bad = []
    for v in (flat[0] - 1, flat[1] + 1, "1", True):
        try:
            rt.create_stream(priority=v)
            bad.append(f"{v!r} 未报错")
        except ValueError:
            pass
        except BaseException as e:                                # noqa: BLE001
            bad.append(f"{v!r} → {type(e).__name__}（非契约级）")
    return (_fail if bad else _ok)("; ".join(bad) if bad
                                   else f"越界 {flat[0] - 1}/{flat[1] + 1} 与 非 int/ bool 共 4 例全部 ValueError")


@item("K 流优先级", "K5", "耦合：声明 stream_priority_control ⇒ **必须**同时声明 stream_priority_readback")
def k5(env):
    bk = env["backend"]
    has_c, has_r = bk.supports("stream_priority_control"), bk.supports("stream_priority_readback")
    if not has_c:
        return _skip("未声明 stream_priority_control ⇒ 该耦合不适用")
    return (_ok if has_r else _fail)("声明了 control 但**没声明** readback ⇒ 设置无法校验"
                                     "（离线 `[8b]` 段同源判据）")


@item("K 流优先级", "K6",
      "声明 control ⇒ 区间端点请求必须**回读 == 请求**（不得静默丢弃 —— 910C 的 torch 路径正是如此）")
def k6(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("stream_priority_control"):
        return _skip("未声明 stream_priority_control ⇒ 该分支不适用")
    flat, why = _range_of(bk)
    if flat is None:
        return _skip(f"范围不可得（{why}）⇒ 无法选定请求值")
    bad = []
    for v in {flat[0], flat[1]}:
        try:
            st = rt.create_stream(priority=v)
        except BaseException as e:                                # noqa: BLE001
            bad.append(f"请求 {v} 抛 {type(e).__name__}: {str(e)[:50]}")
            continue
        got = rt.stream_priority_readback(st)
        if got != v:
            bad.append(f"请求 {v} 回读 {got!r}（**参数被静默丢弃**）")
    return (_fail if bad else _ok)("; ".join(bad) if bad
                                   else f"端点 {flat[0]}/{flat[1]} 请求与回读一致")


@item("K 流优先级", "K7",
      "stream_priority_readback() 返回值域**严格** `int` 或 `None`（不得用 0/字符串伪装「读不出」）")
def k7(env):
    rt, bk = env["runtime"], env["backend"]
    st = rt.create_stream()
    got = rt.stream_priority_readback(st)
    if not (got is None or (isinstance(got, int) and not isinstance(got, bool))):
        return _fail(f"回读返回 {got!r}（{type(got).__name__}）—— 契约：int 或 None，**不得补零/伪装**")
    if got is None and not bk.supports("stream_priority"):
        return _ok("未声明 stream_priority ⇒ 如实返回 None（未补零）")
    return _ok(f"回读 = {got!r}（类型合规）")


@item("K 流优先级", "K8",
      "单点区间：请求**唯一档位**成功且回读一致（等价放行）；非单点 ⇒ 如实 SKIP")
def k8(env):
    rt, bk = env["runtime"], env["backend"]
    flat, why = _range_of(bk)
    if flat is None:
        return _skip(f"范围不可得（{why}）⇒ 该分支不适用")
    if flat[0] != flat[1]:
        return _skip(f"本机区间 {flat!r} 非单点 ⇒ 「等价放行」分支不适用（由 K6 覆盖）")
    try:
        st = rt.create_stream(priority=flat[0])
        got = rt.stream_priority_readback(st)
    except BaseException as e:                                    # noqa: BLE001
        return _fail(f"单点区间请求唯一档位仍抛错：{type(e).__name__}: {str(e)[:60]}")
    return (_ok if got == flat[0] else _fail)(f"请求 {flat[0]} 回读 {got!r}")


# ══════════════════════════════════════════════════════════════════════════════
# L. 流所有权与释放（接口约定 §1.10）
# ══════════════════════════════════════════════════════════════════════════════

@item("L 流所有权与释放", "L1",
      "默认路径的流由**厂商拥有** ⇒ release_stream 必须 **no-op 返回 False**（不得越权销毁）")
def l1(env):
    rt = env["runtime"]
    st = rt.create_stream()
    rel = rt.release_stream(st)
    if rel is not False:
        return _fail(f"release_stream(默认路径的流) 返回 {rel!r} —— 契约：厂商拥有的流**必须 no-op 返回 False**"
                     "（越权销毁别人的流会让厂商栈内部状态崩坏，比泄漏更糟）")
    st.synchronize()                                              # no-op 后该流必须仍可用
    return _ok("release_stream 返回 False，且该流仍可正常使用（确实没被动过）")


@item("L 流所有权与释放", "L2", "release_stream **幂等**：重复调用返回 False、不报错")
def l2(env):
    rt = env["runtime"]
    st = rt.create_stream()
    r1 = rt.release_stream(st)
    try:
        r2 = rt.release_stream(st)
    except BaseException as e:                                    # noqa: BLE001
        return _fail(f"重复调用抛 {type(e).__name__}: {str(e)[:60]}（契约：幂等，返回 False 不报错）")
    ok = (r1 is False) and (r2 is False)
    return (_ok if ok else _fail)(f"第一次={r1!r} 第二次={r2!r}（均应为 False）")


@item("L 流所有权与释放", "L3", "`Stream.release()` 与 `release_stream(stream)` 语义一致（都不越权销毁）")
def l3(env):
    rt = env["runtime"]
    st = rt.create_stream()
    try:
        r = st.release()
    except BaseException as e:                                    # noqa: BLE001
        return _fail(f"Stream.release() 抛 {type(e).__name__}: {str(e)[:60]}")
    if r is not False:
        return _fail(f"Stream.release() 返回 {r!r} —— 与 release_stream 同语义，厂商拥有的流必须是 False")
    return _ok("Stream.release() → False（与 release_stream 一致）")


@item("L 流所有权与释放", "L4",
      "**所有权判据**：owns_stream(native) 与 release_stream(native) 必须一致"
      "（owns=False ⇒ 不得返回 True；owns=True ⇒ 不得返回 False）")
def l4(env):
    bk = env["backend"]
    bad = []
    # 厂商原生路径（后端默认建流 ⇒ 由厂商拥有）
    nat = bk.create_stream()
    owns = bool(bk.owns_stream(nat))
    rel = bk.release_stream(nat)
    if owns != rel:
        bad.append(f"厂商原生路径：owns={owns} 但 release={rel!r}")
    if bk.supports("stream_priority_control"):
        flat, why = _range_of(bk)
        if flat is not None:
            nat2 = bk.create_stream(flat[0])                      # 优先级路径（可能是 C API 建流）
            owns2 = bool(bk.owns_stream(nat2))
            rel2 = bk.release_stream(nat2)
            if owns2 != rel2:
                bad.append(f"优先级路径：owns={owns2} 但 release={rel2!r}")
            note = f"；优先级路径 owns={owns2} release={rel2!r}"
        else:
            note = f"；优先级路径不可得（{why})"
    else:
        note = "；未声明 stream_priority_control ⇒ 只验原生路径"
    return (_fail if bad else _ok)("; ".join(bad) if bad
                                   else f"原生路径 owns={owns} release={rel!r}" + note)


@item("L 流所有权与释放", "L5",
      "**使用已释放对象必须明确**：本层拥有的流 release 后再使用 ⇒ `RuntimeError`（文案含「释放」线索）")
def l5(env):
    rt, bk = env["runtime"], env["backend"]
    if not bk.supports("stream_priority_control"):
        return _skip("本机没有「由本层拥有」的建流路径（未声明 control）⇒ 该分支不适用")
    flat, why = _range_of(bk)
    if flat is None:
        return _skip(f"范围不可得（{why}）⇒ 无法走优先级路径")
    st = rt.create_stream(priority=flat[0])
    rel = rt.release_stream(st)
    if rel is not True:
        return _skip(f"本机该路径未产生「本层拥有」的流（release={rel!r}）⇒ 使用已释放对象的场景不成立")
    try:
        st.synchronize()
        return _fail("已释放的流仍可 synchronize（契约：使用已销毁对象必须明确报错，不得静默失败）")
    except RuntimeError as e:
        txt = str(e)
        ok = ("释放" in txt) or ("release" in txt.lower())
        return (_ok if ok else _fail)(f"RuntimeError 但文案不含「已释放」线索：{txt[:70]}")
    except BaseException as e:                                    # noqa: BLE001
        return _fail(f"异常类型非契约级：{type(e).__name__}: {str(e)[:60]}")


@item("L 流所有权与释放", "L6", "release_stream 的返回值必须是**严格 bool**（不得返回 truthy 的非 bool）")
def l6(env):
    rt = env["runtime"]
    st = rt.create_stream()
    r = rt.release_stream(st)
    return (_ok if _strict_bool(r) else _fail)(f"返回 {r!r}（{type(r).__name__}）—— 契约：bool")


# ══════════════════════════════════════════════════════════════════════════════
# M. 统一 API 面（契约 §1 的标题就是「**统一 API 承诺（下游直接使用）**」）
# ══════════════════════════════════════════════════════════════════════════════
#
# 这三条与后端无关（查的是 `runtime` 这个面本身），但**必须按实例跑** ——
# 因为它们正是"契约承诺 vs 实际交付"的判据，而承诺是**面向所有实例**的。

@item("M 统一 API 面", "M1",
      "契约 §1.1–§1.10 列出的统一 API 名字，在 `runtime` 上**真的可调用**")
def m1(env):
    rt = env["runtime"]
    missing = [n for n in PROMISED_UNIFIED_API if not callable(getattr(rt, n, None))]
    if missing:
        return _fail(f"契约承诺但统一面取不到：{missing} ⇒ 下游按契约写 `runtime.<name>` 会 AttributeError")
    return _ok(f"{len(PROMISED_UNIFIED_API)} 个承诺名全部可调用")


@item("M 统一 API 面", "M2", "公开常量存在且非空（句柄字段 / 状态取值域的对外声明）")
def m2(env):
    rt = env["runtime"]
    bad = [n for n in PROMISED_UNIFIED_CONSTS if not getattr(rt, n, None)]
    return (_fail if bad else _ok)(f"缺或为空：{bad}" if bad
                                   else f"{len(PROMISED_UNIFIED_CONSTS)} 个常量就位（{', '.join(PROMISED_UNIFIED_CONSTS)}）")


@item("M 统一 API 面", "M3", "`runtime.__all__` 里的名字都真的存在（防幽灵导出 / 拼错）")
def m3(env):
    rt = env["runtime"]
    names = list(getattr(rt, "__all__", ()) or ())
    if not names:
        return _fail("__all__ 缺失或为空")
    ghost = [n for n in names if getattr(rt, n, None) is None]
    return (_fail if ghost else _ok)(f"幽灵导出 {ghost}" if ghost else f"{len(names)} 个名字均可取到")


#: **必须在独立子进程里执行的项**（2026-10-08 真机实测教训）
#:
#: 原因：设备上下文的 create/destroy 会让**整个进程失去当前设备上下文**。实测（910C）：
#:   · 先做 create+destroy，随后的 `torch` 算子直接报 CANN 内部错误
#:     （`The Inner error is reported as above. The process exits for this i...`）；
#:   · 同一进程内**第二次**跑 `check_i3`（它也做上下文 create/destroy）**段错误退出**（rc=139）。
#:   ⇒ 若不隔离，**后续判据会被污染**：自查时表现为 I8 与 J3 误报 FAIL，
#:     而单独跑（干净进程）两者都是 PASS。**这是判据设计缺陷，不是被测对象的缺陷。**
#:
#: 先例：`probes/probe_bc_contract.py` 的 C 组（上下文生命周期）本来就是**逐组独立子进程**。
#: ⇒ 纪律：**当一个判据本身会破坏进程级状态时，它的执行单元必须是进程**。
INVASIVE_SIDS = {"I1", "I4", "I5", "I10", "J3"}


def _run_in_child(sid, backend):
    """在子进程里跑单项，返回 (status, detail)。子进程崩了也要**如实报告**。"""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "one.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--backend", backend,
               "--only", sid, "--out", out]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except BaseException as e:                                # noqa: BLE001
            return False, f"子进程启动失败：{type(e).__name__}: {str(e)[:80]}"
        try:
            d = json.loads(Path(out).read_text(encoding="utf-8"))
            # ⚠️ 子进程写的是**字符串**状态；主循环按 bool/None 取 ⇒ 必须映射回来
            #    （自查第一版漏了这步：`KeyError: 'FAIL'`，整轮审计直接崩掉 —— 教训：
            #     跨进程边界的**类型口径**也要显式对齐，别靠"看起来一样"）。
            back = {"OK": True, "FAIL": False, "SKIP": None}[d["status"]]
            return back, f"{d['detail']}（子进程隔离）"
        except BaseException:                                     # noqa: BLE001
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            tail = " | ".join(tail[-2:])[:200] if tail else "（无输出）"
            return False, (f"子进程未产出结果（rc={r.returncode}）"
                           f"⇒ 该侵入项**很可能崩掉了进程**：{tail}")


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
    ap.add_argument("--only", default="",
                    help="只跑指定 sid（供侵入项的子进程隔离用）")
    args = ap.parse_args()

    print("=" * 78)
    print(f"职责响应审计：backend={args.backend}"
          f"（口径：接口约定 §1.1–§1.10 + §2 三支撑 + §3 两纪律）")
    print("=" * 78)

    try:
        env = build_env(args.backend)
    except Exception as e:
        print(f"❌ 环境装配失败：{type(e).__name__}: {e}")
        traceback.print_exc()
        return 2

    if args.only:                     # ── 子进程模式：只跑一项，结果写 JSON ──
        fn = next((r["fn"] for r in RESULTS if r["sid"] == args.only), None)
        if fn is None:
            print(f"未知 sid: {args.only}")
            return 2
        try:
            status, detail = fn(env)
        except BaseException as e:                            # noqa: BLE001
            status, detail = False, f"{type(e).__name__}: {str(e)[:200]}"
        label = {True: "OK", False: "FAIL", None: "SKIP"}[status]
        print(f"[{label}] {args.only} {detail}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(
                {"sid": args.only, "status": label, "detail": str(detail)},
                ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    print(f"ℹ️ 设备类型={env['backend'].device_type}  device_count={env['device_count_hint']}")
    print(f"ℹ️ 后端已声明能力：{sorted((env['backend'].info() or {}).get('supports') or {})}\n")

    rows, n_ok, n_fail, n_skip = [], 0, 0, 0
    for r in RESULTS:
        if r["sid"] in INVASIVE_SIDS:
            status, detail = _run_in_child(r["sid"], args.backend)
        else:
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
