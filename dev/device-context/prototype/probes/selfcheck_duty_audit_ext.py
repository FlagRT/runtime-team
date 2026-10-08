#!/usr/bin/env python3
"""非空转验证：对 `duty_response_audit.py` 的 **H–M 六个域逐条注入缺陷**，
证明这些判据**真的能 FAIL** —— 而不是「看起来通过」。

【为什么必须做】
  空转判据比没有判据**更糟**：它给的是**虚假的安全感**。本项目为此已付出过代价
  （工作包 B/C 曾出现「能力矩阵只对 stub 跑 ⇒ 真实后端长期无人守」、
  「判据与被测对象不在同一个世界 ⇒ 静默空转」等）。故**新增判据一律配注入验证**。

【做法】
  每个被测项跑两遍：
    ① **原状**（期望 `OK`，或本机如实 `SKIP`）
    ② **注入后**（期望 **`FAIL`**）
  注入通过**只读代理**实现（`_Ov`）：`runtime` 是模块、`backend` 是实例，
  两者都能被代理覆盖单个属性，**不改原型代码、不污染真实状态**（每次注入用全新代理）。

【判定】
  · 非 SKIP 的项：`原状 == OK` 且 `注入后 == FAIL` ⇒ **抓到了**；
  · 原状即 `SKIP` 的项 ⇒ 如实标「不适用（本机该分支不成立）」并**计入覆盖缺口**，
    不算通过（换芯片/换实例时这些分支会在别的机器上被真正行使）。
    ⚠️ **2026-10-08 更正**：本节原写「L5 只在声明 `stream_priority_control` 且区间非单点的
    实例上成立，**目前只有 MLU590 满足**」—— 该预测**已被实机证伪**：MLU590 区间确为
    `(0, -3)` 非单点，但它的建流走 torch 侧（`torch.mlu.Stream(priority=…)`，本家**没有**
    独立的句柄式 C API）⇒ `release_stream` 恒 `False`（不属于「本层拥有」）⇒ L5 **同样 SKIP**。
    ⇒ **「区间非单点」只是 L5 的必要条件，不是充分条件**；充分条件还要求该后端真的存在
    「厂商 C API 建流 + 本层包装」的路径（目前只有 P800 的 C API 路径，而它区间退化单点
    ⇒ 真机走的是等价放行分支）。**结论：三家现役实例目前都行使不了 L5。**

【§0 委派翻译层自检（无需设备）】
  J 域把 `contract_invariants.check_i1..i4` **委托**过来（同一份实现，避免两处漂移）。
  被委托方的返回是三值的（`True` / `False` / `None` 或 `True` + `NOT_APPLICABLE` 哨兵），
  本审计的状态也是三值的（`OK` / `FAIL` / `SKIP`）⇒ **必须逐值对齐**。
  这一节用**受控替身**直接喂四种返回形状并断言映射 —— 因为「如实跳过 / 如实不适用」
  这两条分支在**声明了两项能力的实例上永远走不到**（910C 两项都声明、P800 声明
  `memory_alloc`），单靠真机跑不动它。⭐ 纪律：**新分支必须证明它能真的报**。

【用法】
  DC_BACKEND=ascend python3 probes/selfcheck_duty_audit_ext.py --backend ascend --out <json>
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = os.environ.get("DC_ROOT") or str(HERE.parent)
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "duty_response_audit", str(HERE.parent / "scripts" / "duty_response_audit.py"))
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)          # 导入即注册全部 item


class _Ov:
    """只读代理：覆盖若干属性，其余透传（用于注入缺陷）。"""

    def __init__(self, obj, **ov):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_ov", ov)

    def __getattr__(self, k):
        ov = object.__getattribute__(self, "_ov")
        if k in ov:
            return ov[k]
        return getattr(object.__getattribute__(self, "_obj"), k)


class _FakeStream:
    """假流：用于注入「record_stream 抛错」「release 假成功」「synchronize 不报错」。"""

    def __init__(self, exc=None, release_ret=True):
        self._exc, self._release_ret = exc, release_ret

    def synchronize(self, timeout_ms=None):
        return None

    def record_stream(self, tensor):
        if self._exc is not None:
            raise self._exc
        return None

    def release(self):
        return self._release_ret

    @property
    def native(self):
        return None


class _MappedTrue:
    """把真实错误对象包一层，只把 `mapped` 改成 True（注入 I2② 自相矛盾）。"""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, k):
        return getattr(self._real, k)

    @property
    def mapped(self):
        return True


def _env_with(env, rt_ov=None, bk_ov=None):
    e = dict(env)
    if rt_ov:
        e["runtime"] = _Ov(env["runtime"], **rt_ov)
    if bk_ov:
        e["backend"] = _Ov(env["backend"], **bk_ov)
    return e


def _rt(env, name):
    return getattr(env["runtime"], name)


def build_injections():
    """返回 `[(sid, 注入函数(env)->env', 说明)]`（顺序与域顺序一致）。"""
    inj = []

    def add(sid, fn, note):
        inj.append((sid, fn, note))

    # ── H 内存句柄与生命周期 ──
    add("H1", lambda e: _env_with(
        e,
        rt_ov={"allocate": lambda *a, **k: {"handle_id": 1}},
        bk_ov={"supports": lambda k: False if k == "memory_alloc" else e["backend"].supports(k)}),
        "未声明能力却静默返回假句柄")
    add("H2", lambda e: _env_with(
        e, rt_ov={"allocate": lambda *a, **k: {"handle_id": 1, "kind": "memory",
                                               "backend": "x", "ordinal": 0, "size_bytes": 0}}),
        "allocate 对非法 size_bytes 不报错")
    add("H3", lambda e: _env_with(e, rt_ov={"free": lambda h: None}), "free 永不报错（二次释放静默）")
    add("H4", lambda e: _env_with(e, rt_ov={"free": lambda h: None}), "free 接受任意陌生句柄")
    add("H5", lambda e: _env_with(e, rt_ov={"memory_handle_count": lambda: 0}),
        "句柄计数恒为 0（与分配/释放不联动）")
    add("H6", lambda e: _env_with(
        e, rt_ov={"memory_stats": lambda o=0: {"total_mb": 1, "used_mb": 0, "free_mb": 1,
                                               "allocated_mb": 0, "not_a_registered_key": 1}}),
        "memory_stats 出现未登记键")
    add("H7", lambda e: _env_with(
        e, rt_ov={"create_stream": lambda priority=None: _FakeStream(exc=TypeError("no record_stream"))}),
        "record_stream 直接抛错（契约要求保守同步放行）")
    add("H8", lambda e: _env_with(e, rt_ov={"native_accesses": lambda: {"total": 424242, "by_kind": {}}}),
        ".native 取用不计数（恒值）")

    # ── I 设备上下文 ──
    add("I1", lambda e: _env_with(
        e,
        rt_ov={"context_create": lambda *a, **k: {"handle_id": 1}},
        bk_ov={"supports": lambda k: False if k == "context_lifecycle" else e["backend"].supports(k)}),
        "未声明 context_lifecycle 却静默返回句柄")
    add("I2", lambda e: _env_with(e, rt_ov={"context_set": lambda h: None}), "context_set 接受任意句柄")
    add("I3", lambda e: _env_with(e, rt_ov={"context_destroy": lambda h: None}),
        "context_destroy 接受任意句柄（含默认上下文）")
    add("I4", lambda e: _env_with(e, rt_ov={"context_count": lambda: 1}),
        "context_count 恒为 1（把进程默认上下文也数进去）")
    add("I5", lambda e: _env_with(e, rt_ov={"context_destroy": lambda h: None}),
        "两种句柄可互相误用（不报错）")
    add("I6", lambda e: _env_with(e, rt_ov={"context_query": lambda: {"queryable": True}}),
        "context_query 缺键")
    add("I7", lambda e: _env_with(
        e, rt_ov={"context_query": lambda: {**e["runtime"].context_query(), "present": 1}}),
        "present 用 0/1 冒充 bool")
    add("I8", lambda e: _env_with(
        e, rt_ov={"context_query": lambda: (e["runtime"].allocate(4096), e["runtime"].context_query())[1]}),
        "只读接口产生副作用（新造句柄）")
    add("I9", lambda e: _env_with(
        e, rt_ov={"recover_device": lambda *a, **k: {"ordinal": 0, "mode": "probe", "recovered": True,
                                                     "state": "available", "detail": "x"}}),
        "recover_device 缺 context 三键")
    # ⚠️ 注入换成**不崩进程**的形式：原版把 `context_destroy` 变空操作 ⇒ 退出时段错误（rc=139）
    #    ⇒ 子进程产出不了结果，**证明不了判据能 FAIL**（那种"抓到"是假的）。
    add("I10", lambda e: _env_with(
        e, rt_ov={"create_stream": lambda priority=None: _FakeStream()}),
        "销毁上下文后其流仍可用（使用点未拦截）")

    # ── J 契约不变式（委托同一份实现；注入打在它实际读取的面上）──
    # ⚠️ 注入必须打在被测判据**实际探测的那个键**上（自查第一版打的是自定义键 ⇒ 没翻红）：
    #    `check_i1` ② 的探测名单是 ["", "no_such_capability_xyzzy", "__class__", "supports"]。
    add("J1", lambda e: _env_with(
        e, bk_ov={"supports": lambda k: True if k == "no_such_capability_xyzzy"
                  else e["backend"].supports(k)}),
        "能力全集之外的键被判 True（I1② 违反）")
    # ⚠️ 注入打在被测判据实际读取的面上（自查第一版打在 `runtime.` ⇒ 没翻红）：
    #    `check_i2` 调的是 `bk.translate_error`。
    add("J2", lambda e: _env_with(
        e, bk_ov={"translate_error": lambda *a, **k: _MappedTrue(
            e["backend"].translate_error(*a, **k))}),
        "陌生消息却声称 mapped=True（I2② 自相矛盾）")
    add("J3", lambda e: _env_with(e, bk_ov={"free": lambda h: None}), "二次释放静默（I3 违反）")
    add("J4", lambda e: _env_with(e, bk_ov={"native_accesses": lambda: {"total": 0, "by_kind": {}}}),
        ".native 取用不计数（I4② 违反）")

    # ── K 流优先级 ──
    add("K1", lambda e: _env_with(e, bk_ov={"stream_priority_range": lambda: (7, 0, 0)}),
        "范围透传厂商三元组（形状违约）")
    add("K2", lambda e: _env_with(
        e, bk_ov={"stream_priority_range": lambda: (0, 7),
                  "supports": lambda k: False if k == "stream_priority" else e["backend"].supports(k)}),
        "未声明 stream_priority 却给出范围")
    add("K3", lambda e: _env_with(e, rt_ov={"create_stream": lambda priority=None: object()}),
        "未声明 control 却静默返回流（不拒绝）")
    add("K4", lambda e: _env_with(e, rt_ov={"create_stream": lambda priority=None: object()}),
        "非法 priority 不报错")
    add("K5", lambda e: _env_with(
        e, bk_ov={"supports": lambda k: False if k == "stream_priority_readback" else e["backend"].supports(k)}),
        "声明 control 但不声明 readback（耦合破裂）")
    # ⚠️ 注入值必须**必然不等于端点**：初版注入「回读恒 0」，而单点区间（kunlun `(0,0)`）
    #    的请求值本来就是 0 ⇒ 注入不可区分、没翻红（自查抓到，已改）。
    add("K6", lambda e: _env_with(e, rt_ov={"stream_priority_readback": lambda s: 1}),
        "回读与请求不一致（参数被静默丢弃）")
    add("K7", lambda e: _env_with(e, rt_ov={"stream_priority_readback": lambda s: "0"}),
        "回读返回字符串（用伪装值表示「读不出」）")
    add("K8", lambda e: _env_with(e, rt_ov={"create_stream": lambda priority=None: object()}),
        "单点区间请求唯一档位也失败/不校验")

    # ── L 流所有权与释放 ──
    add("L1", lambda e: _env_with(e, rt_ov={"release_stream": lambda s: True}),
        "对厂商拥有的流返回 True（越权销毁）")
    add("L2", lambda e: _env_with(e, rt_ov={"release_stream": lambda s: True}),
        "release_stream 不幂等（恒 True）")
    add("L3", lambda e: _env_with(e, rt_ov={"create_stream": lambda priority=None: _FakeStream(release_ret=True)}),
        "Stream.release() 越权返回 True")
    add("L4", lambda e: _env_with(e, bk_ov={"release_stream": lambda s: True}),
        "owns=False 但 release 返回 True（不一致）")
    add("L5", lambda e: _env_with(e, rt_ov={"release_stream": lambda s: False}),
        "本层拥有的流未被真销毁（release 恒 False）⇒ 已释放语义不成立")
    add("L6", lambda e: _env_with(e, rt_ov={"release_stream": lambda s: 1}),
        "返回值不是严格 bool")

    # ── M 统一 API 面 ──
    add("M1", lambda e: _env_with(e, rt_ov={"context_set": None}),
        "契约承诺的统一 API 名不可调用")
    add("M2", lambda e: _env_with(e, rt_ov={"MEMORY_HANDLE_KEYS": ()}), "公开常量为空")
    add("M3", lambda e: _env_with(e, rt_ov={"__all__": ["definitely_not_a_real_name_xyz"]}),
        "__all__ 里出现幽灵导出（拼错/未定义）")

    return inj


#: 侵入项**必须与审计脚本保持同一份定义**（单一来源，避免两处漂移）
INVASIVE = set(getattr(audit, "INVASIVE_SIDS", ()) or ())


def _run_single(sid, env):
    items = {r["sid"]: r["fn"] for r in audit.RESULTS}
    try:
        st, det = items[sid](env)
    except BaseException as exc:                                  # noqa: BLE001
        st, det = False, f"{type(exc).__name__}: {str(exc)[:90]}"
    return {True: "OK", False: "FAIL", None: "SKIP"}[st], det


def _child(sid, backend, inject):
    """在子进程里跑一项（侵入项用）。原状与注入都走子进程，避免污染本进程。"""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "one.json")
        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "--backend", backend, "--only", sid, "--out", out]
        if inject:
            cmd.append("--inject")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        try:
            d = json.loads(Path(out).read_text(encoding="utf-8"))
            return d["status"], f"{d['detail']}（子进程隔离）"
        except BaseException:                                     # noqa: BLE001
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            # ⚠️ 返回 **`CRASH`** 而不是 `FAIL`/`False`：
            #   「子进程崩掉」≠「判据翻红」。把它算成"抓到"就是**假通过**。
            return "CRASH", "子进程未产出结果（rc=%s）：%s" % (
                r.returncode, " | ".join(tail[-2:])[:160] if tail else "（无输出）")


def check_delegation_mapping():
    """§0 委派翻译层自检（**无需设备**）：委派返回形状 → 审计状态 的三值映射。

    断言四条（依据 = 审计工具自身的三值约定 `_ok=True / _fail=False / _skip=None`）：
        真实通过   (True,  "…")                  -> OK   (True)
        如实跳过   (None,  "…")                  -> SKIP (None)
        如实不适用 (True,  NOT_APPLICABLE + "…") -> SKIP (None)   ← 2026-10-08 修的正是这条
        真实失败   (False, "…")                  -> FAIL (False)

    为什么不能靠真机：后两条分支要求**后端不声明**相应能力才会触发；910C 两项都声明、
    P800 声明 memory_alloc ⇒ 它们的 J3 永远走真实分支。MLU590 是首个两项都不声明的实例，
    但它只能证明「如实不适用」这一条；「如实跳过」（`None`）目前**没有任何被委托方会返回**
    ⇒ 只能在这里用受控替身证明它真的被识别，而不是一条永不触发的 if。
    """
    from runtime.conformance import contract_invariants as ci

    cases = [
        ("真实通过", (True, "判据真正执行且通过"), True, "OK"),
        ("如实跳过", (None, "委派方按三值约定跳过"), None, "SKIP"),
        ("如实不适用", (True, ci.NOT_APPLICABLE + "未声明 `memory_alloc` 等"),
         None, "SKIP"),
        ("真实失败", (False, "二次释放静默（I3 违反）"), False, "FAIL"),
    ]
    rows, bad = [], []
    saved = ci.check_i3
    try:
        for label, ret, want, want_name in cases:
            # 受控替身：只换被委托的那一个函数，其余一律不动
            ci.check_i3 = lambda bk, ordinal=0, _r=ret: _r
            got = audit.j3({"backend": object()})[0]
            good = (got is want)
            rows.append({"case": label, "delegated": f"{ret[0]!r} / {str(ret[1])[:26]}",
                         "want": want_name,
                         "mapped": {True: "OK", False: "FAIL", None: "SKIP"}.get(got, repr(got)),
                         "ok": good})
            if not good:
                bad.append(f"{label}：期望 {want_name}，实得 {got!r}")
    finally:
        ci.check_i3 = saved                                     # ⚠️ 必须还原，否则污染后续真机判据
    return (not bad), ("；".join(bad) if bad else "四种返回形状的三值映射全部符合约定"), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("DC_BACKEND", "ascend"))
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="", help="只跑一项（供侵入项的子进程隔离用）")
    ap.add_argument("--inject", action="store_true", help="在 --only 模式里应用该 sid 的注入")
    a = ap.parse_args()

    map_rows = []
    if not a.only:
        # ── §0 委派翻译层自检（**无需设备**；先跑，不过就别浪费后面的设备时间）──
        map_ok, map_detail, map_rows = check_delegation_mapping()
        print("=" * 90)
        print("§0 委派翻译层自检（无需设备）：委派返回形状 → 审计状态 的三值映射")
        print("=" * 90)
        for r in map_rows:
            print(f"  [{'OK  ' if r['ok'] else 'FAIL'}] {r['case']:<10} "
                  f"委派={r['delegated']:<36} → 期望 {r['want']:<5} 实得 {r['mapped']}")
        print(f"  {map_detail}")
        if not map_ok:
            print("-" * 90)
            print("SELFCHECK_DUTY_EXT_FAIL（§0 未过 ⇒ 后续注入验证的结论不可信）")
            if a.out:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps({
                    "backend": a.backend, "verdict": "SELFCHECK_DUTY_EXT_FAIL",
                    "stage": "delegation_mapping", "rows": map_rows,
                }, ensure_ascii=False, indent=1), encoding="utf-8")
            return 1
        print()

    items = {r["sid"]: r["fn"] for r in audit.RESULTS}
    env = audit.build_env(a.backend)

    if a.only:                        # ── 子进程模式 ──
        status, detail = _run_single(a.only, env)
        if a.inject:
            inj = {s: f for s, f, _ in build_injections()}
            if a.only not in inj:
                print("该 sid 没有注入定义")
                return 2
            status, detail = _run_single(a.only, inj[a.only](env))
            detail = f"[注入后] {detail}"
        print(f"[{status}] {a.only} {detail}")
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(
                {"sid": a.only, "status": status, "detail": str(detail)},
                ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    print("=" * 90)
    print(f"非空转验证：duty_response_audit H–M 域逐条注入 ｜ backend={a.backend}")
    print("=" * 90)

    caught, na, broken = [], [], []
    rows = []
    for sid, fn, note in build_injections():
        if sid in INVASIVE:            # 侵入项：原状与注入都在子进程里跑
            base, bdet = _child(sid, a.backend, inject=False)
            if base == "SKIP":         # ⚠️ 本机不适用 ⇒ 计「不适用」，**不得**计「未抓到」
                na.append(sid)         #    （初版漏了这一步：3 个 SKIP 的侵入项被误报未抓到）
                rows.append({"sid": sid, "base": base, "injected": "-", "caught": False,
                             "note": note, "detail": f"本机如实 SKIP：{bdet[:110]}"})
                continue
            injected, idet = _child(sid, a.backend, inject=True)
            ok = (base == "OK" and injected == "FAIL")
            (caught if ok else broken).append(sid)
            rows.append({"sid": sid, "base": base, "injected": injected, "caught": ok,
                         "note": note, "detail": f"注入后：{idet[:130]}"})
            continue
        base, bdet = _run_single(sid, env)
        if base == "SKIP":
            na.append(sid)
            rows.append({"sid": sid, "base": base, "injected": "-", "caught": False,
                         "note": note, "detail": f"本机如实 SKIP：{bdet[:110]}"})
            continue
        injected, idet = _run_single(sid, fn(env))
        ok = (base == "OK" and injected == "FAIL")
        (caught if ok else broken).append(sid)
        rows.append({"sid": sid, "base": base, "injected": injected, "caught": ok,
                     "note": note, "detail": f"注入后：{idet[:130]}"})

    for r in rows:
        flag = ("抓到" if r["caught"]
                else "不适用" if r["base"] == "SKIP"
                else "★崩溃未证★" if r["injected"] == "CRASH"
                else "★未抓到★")
        print(f"  [{flag}] {r['sid']:<4} 原状={r['base']:<5} 注入后={r['injected']:<5} {r['note']}")
        if r["detail"]:
            print(f"           {r['detail']}")

    n = len(rows)
    verdict = "SELFCHECK_DUTY_EXT_PASS" if not broken else "SELFCHECK_DUTY_EXT_FAIL"
    print("-" * 90)
    print(f"共 {n} 项：抓到 {len(caught)} · 本机不适用 {len(na)} · **未抓到 {len(broken)}**")
    if broken:
        print(f"★ 未抓到（判据可能是空转，必须修）：{broken}")
    if na:
        print(f"（本机不适用 {na} —— 换实例会真正行使，见各自的 SKIP 原因）")
    print(f"verdict = {verdict}")
    print("⚠️ 边界：本脚本只证「判据能 FAIL」，不证「判据的期望值是对的」——"
          "期望值依据是契约 §1.6–§1.10 的条款原文。")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "backend": a.backend, "verdict": verdict, "total": n,
            "delegation_mapping": map_rows,
            "caught": caught, "not_applicable": na, "not_caught": broken, "rows": rows,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"结果已写入 {a.out}")
    return 0 if not broken else 1


if __name__ == "__main__":
    sys.exit(main())
