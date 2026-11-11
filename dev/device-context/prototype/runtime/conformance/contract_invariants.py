#!/usr/bin/env python3
"""契约不变式 I1–I4 的判据实现（真机 conformance 用例 + 离线桩**共用同一套核心函数**）。

【它解决什么问题】
  `VERIFICATION_MANIFEST` §1 第 9 条与 §3 G8 早已登记
  「契约不变式（I1–I4）**未实现**，目前靠人工检查」—— 而四条**一直只有名字、没有定义**。
  2026-09-29 在《接口约定》新增 **§1.8** 补齐定义；**本模块是它的判据实现**。

  四条不变式约束的**不是新 API，而是既有 API 的可观测性** —— 即「本层说的话是否可信」：
    I1 诚实声明 · I2 禁止伪造 · I3 失效受管 · I4 降级可观测

【两处使用（单一事实来源，不会漂移）】
  · **真机**：`runner.py --backend $B --cases contract_invariants` ⇒ `CONTRACT_INVARIANTS_PASS 4/4`
  · **离线**：`scripts/backend_offline_check.py --backend $B` 第 `[10]` 段（用 stub 后端调 `CHECKS`）
    ⇒ **无设备即可拦住回归**（这正是本项被定为"不需卡"的原因）

【边界（勿外推）】
  · 四条都是**可观测性**约束，**不评判性能**。
  · 能力**未声明**时如实标「不适用」（判为通过但**显式注明**）——
    **不得**因不适用而把未声明能力当作已验，也不得据此声称该芯片不具备。
    实现方式：`return True, NOT_APPLICABLE + 原因`（哨兵前缀见 `NOT_APPLICABLE` 常量）。
    ⚠️ **委派消费方必须据此转成"跳过"而不是"通过"**（2026-10-08 教训）：
      本模块的「不适用」在 conformance 口径下**算通过**（四条不变式是一条整体判据），
      但在**逐条点名的职责审计**口径下必须记 `SKIP` —— 否则一个**从未运行**的分支
      会被记成「已响应」，而同一份审计里同样原因的其他项（H2/I3/K2…）记的是 SKIP
      ⇒ 同一原因两种标签。
  · 本模块只碰**本层 API 与文本**；唯一的设备副作用是 I3 的内存/上下文生命周期，且**用完即清理**。
  · 离线桩结论**不得**当真机结论（厂商原语真实形态仍须上机实测）。
"""

from __future__ import annotations

import warnings

#: 「**本机不适用**」的哨兵前缀：所需能力未声明时，检查函数以
#: `(True, NOT_APPLICABLE + 原因)` 返回（见模块 docstring「边界」第 2 条）。
#: ⚠️ 该字符串是**跨模块契约**：`scripts/duty_response_audit.py` 的 J 域据此把
#: 委派结果如实转成 `SKIP`（而不是 `OK`）。两侧**共用本常量**，避免文案漂移
#: 导致审计侧静默退化成「通过」。
NOT_APPLICABLE = "不适用："

# ───────────────────────── 常量：能力键 → 公共入口（I1④ 用）─────────────────────────
#: 只列**本层有明确公共入口**的能力；未列入的键不做入口存在性检查（在 detail 里如实说明覆盖数）。
CAPABILITY_ENTRYPOINTS = {
    "device": ("device_count", "set_device"),
    "memory": ("memory_stats",),
    "memory_alloc": ("allocate", "free", "memory_handle_count"),
    "memory_alloc_stat": ("memory_stats",),
    "stream": ("create_stream", "current_stream", "stream_context"),
    "event": ("create_event",),
    "bounded_sync": ("synchronize", "synchronize_stream", "wait_event_host"),
    "error_map": ("translate_error",),
    "recovery_probe": ("probe_device", "recover_device"),
    "recovery_real": ("recover_device",),
    "device_state": ("device_state",),
    # 2026-09-29（A2 收尾新增能力键）：四态的**驱动**入口（原只有查询）。
    # 与 `device_state` 分开声明的原因同 `context_query` / `context_lifecycle`：**能查 ≠ 能改**。
    "device_state_control": ("set_device_state",),
    "context_lifecycle": ("context_create", "context_destroy", "context_set", "context_count"),
    "context_query": ("context_query",),
    # 2026-09-30（(A) 方案落地）：流优先级**拆三把钥匙**（能读范围 / 能设置 / 能回读）。
    # 拆开的原因同 `context_query` / `context_lifecycle`：**能读 ≠ 能改**。
    # 三家的取值不同（910C：范围✅ 回读✅ 设置❌ / P800：范围✅ 回读✅ 设置❌ / MLU590：全✅），
    # 正是"如实拆分"要表达的信息。
    "stream_priority": ("stream_priority_range",),
    "stream_priority_control": ("create_stream",),
    "stream_priority_readback": ("stream_priority_readback",),
}

#: 能力**未声明**时必须「显式拒绝」的探测（I1④ 反向）。
#: 只选**安全可调**且当前三家都有明确拒绝行为的三项；不做全量（避免误伤合法保守声明）。
REFUSE_PROBES = {
    "memory_alloc": ("allocate(4096)", lambda bk: bk.allocate(4096)),
    "context_lifecycle": ("context_create(0)", lambda bk: bk.context_create(0)),
    "context_query": ("context_query()", lambda bk: bk.context_query()),
    # 2026-09-30：未声明 `stream_priority_control` ⇒ `create_stream(priority=…)` 必须**显式拒绝**。
    # 用 0 作参数：能力门禁在取值域校验**之前**，故该值本身合不合法不影响本探测。
    "stream_priority_control": ("create_stream(0)", lambda bk: bk.create_stream(0)),
}

#: `graded_by` 的**已登记取值域**（见 conformance/errors.py 的字段说明）
KNOWN_GRADED_BY = {"code_map", "message_hint", "default"}
#: 「码表 / 规则命中」类 —— `mapped=True` 必须落在这里（I2②）
CODE_MAP_GRADED_BY = {"code_map"}
#: `disposition` 取值域（api/errors.py 的 `DISPOSITION`）
KNOWN_DISPOSITIONS = {"retry", "raise", "replay", "device_recovery"}
#: `category` 取值域（四级分级）
KNOWN_CATEGORIES = {"L1_RESOURCE", "L2_PARAM", "L3_EXECUTION", "L4_FATAL"}

#: 一条**不含任何已知关键词**的消息（I2① 用）。
#: 刻意避开 `timeout` / `AICORE` / `out of memory` 等规则词，也不含 `ret=` / `error code is` 形态
#: ⇒ 任何按码表或消息规则分级的实现都不应命中它。
STRANGER_MSG = "zzq frobnicate plugh xyzzy quux 0000 (no known keyword)"


# ───────────────────────── 小工具 ─────────────────────────
def _capability_keys(bk):
    """后端声明的**能力全集**（`_CAPABILITY_KEYS`）；缺失即视为违反 I1①。"""
    keys = getattr(bk, "_CAPABILITY_KEYS", None)
    if not keys:
        return None
    return list(keys)


def _is_strict_bool(v):
    return isinstance(v, bool)


# ───────────────────────── I1 诚实声明 ─────────────────────────
def check_i1(bk):
    """I1 诚实声明：`supports()` 与 `info()["capabilities"]` 必须忠实，且未声明的必须**显式拒绝**。"""
    problems = []
    notes = []
    keys = _capability_keys(bk)
    if keys is None:
        return False, "后端未定义 `_CAPABILITY_KEYS`（能力全集缺失 ⇒ 无法判定声明是否诚实）"

    # ① 全集内：严格 bool
    bad_types = [(k, type(bk.supports(k)).__name__) for k in keys
                 if not _is_strict_bool(bk.supports(k))]
    if bad_types:
        problems.append(f"① supports() 非严格 bool: {bad_types[:3]}")

    # ② 全集外：必须 False（不得 True）
    outside = ["", "no_such_capability_xyzzy", "__class__", "supports"]
    leaked = [k for k in outside if bk.supports(k) is True]
    if leaked:
        problems.append(f"② 能力全集外的键被判 True: {leaked}")

    # ③ info()["capabilities"] 自洽。
    # ⚠️ **别名感知**（初版判据在这里写错了，值得留记录）：
    #   初版写成「capabilities == 所有 `supports(k) is True` 的键」—— 这在**别名键**上必然误报：
    #   `sync_timeout` 是 `bounded_sync` 的**弃用别名**（见基类 `_CAPABILITY_ALIASES`），
    #   `supports("sync_timeout")` 按设计为 True，但它**不在** `capabilities` 里
    #   ⇒ 判据在 kunlun / cambricon 上误报。
    #   而这个误报**顺带暴露了一处真缺陷**：ascend 的 `_capabilities` 里**确实**列了该别名
    #   ⇒ 三家 `info()["capabilities"]` 的在册集合不一致（台账第 18 条，已修）。
    #   ⇒ 教训：**判据本身要跟着"契约的扩展/弃用机制"走**（与"精确键集在只增不改契约下必然误报"同族）。
    aliases = getattr(bk, "_CAPABILITY_ALIASES", {}) or {}
    declared = set(getattr(bk, "_capabilities", set()) or ())
    caps = bk.info().get("capabilities")
    expect = sorted(declared)
    if not isinstance(caps, list):
        problems.append(f"③ info()['capabilities'] 非 list: {type(caps).__name__}")
    else:
        if caps != expect:
            problems.append(f"③ capabilities != 声明集: 得到 {caps} / 期望 {expect}")
        if len(caps) != len(set(caps)):
            problems.append("③ capabilities 含重复项")
        if not set(caps) <= set(keys):
            problems.append(f"③ capabilities 含未登记键: {sorted(set(caps) - set(keys))}")
        # 规范键：supports() 与 capabilities 必须互相印证
        canon_bad = [k for k in keys if k not in aliases
                     and (bk.supports(k) is True) != (k in declared)]
        if canon_bad:
            problems.append(f"③ 规范键在 supports() 与 capabilities 之间不一致: {canon_bad}")
        # 别名键：取值随规范键，且**不得**出现在 capabilities（弃用项不是在册能力）
        alias_bad = []
        for a, c in aliases.items():
            if a in declared:
                alias_bad.append(f"别名 {a!r} 出现在 capabilities（应在能力全集 + 别名映射里，但不在声明集）")
            if a in keys and bk.supports(a) != bk.supports(c):
                alias_bad.append(f"别名 {a!r} 取值与规范键 {c!r} 不一致")
        if alias_bad:
            problems.append("③ " + "；".join(alias_bad))

    # ④ 声明为 True ⇒ 公共入口存在
    covered, missing_entry = 0, []
    for cap, entries in CAPABILITY_ENTRYPOINTS.items():
        if cap not in keys or not bk.supports(cap):
            continue
        covered += 1
        for e in entries:
            if not callable(getattr(bk, e, None)):
                missing_entry.append(f"{cap}→{e}")
    if missing_entry:
        problems.append(f"④ 声明为支持但公共入口不可调用: {missing_entry}")
    notes.append(f"④ 入口存在性覆盖 {covered}/{len(CAPABILITY_ENTRYPOINTS)} 项能力")

    # ⑤ 声明为 False ⇒ 必须**显式拒绝**（抛错，或返回带 queryable=False + 原因的显式结构）。
    #
    # ⚠️ **收紧点（2026-09-29，非空转验证暴露）**：初版把"**抛任何异常**"都当成显式拒绝
    #   ⇒ 注入"去掉声明守卫"后**未被抓到**：更底层仍抛 `NotImplementedError`（因为该能力确实没实现），
    #   判据便误判为"已拒绝" —— 这是**假通过**。
    #   收紧为：**只有契约级 `NotImplementedError` 才算如实拒绝**；
    #   · 其它异常类型（`AttributeError`/`TypeError`/…）⇒ **判 FAIL**（很可能是真 bug 被当成了"拒绝"）；
    #   · 返回句柄/值（静默成功）⇒ 判 FAIL。
    #   （同族纪律：负向判据必须要求**契约级**异常类型，见工作包 B/C 的"6 条负向判据收紧为 ValueError"。）
    silent, wrong_exc = [], []
    for cap, (desc, probe) in REFUSE_PROBES.items():
        if cap not in keys or bk.supports(cap):
            continue
        try:
            ret = probe(bk)
        except NotImplementedError:
            continue                                             # 契约级如实拒绝 ✓
        except BaseException as e:                                # noqa: BLE001
            wrong_exc.append(f"{cap}({desc}) → {type(e).__name__}: {str(e)[:60]}")
            continue
        # 未抛错：只接受"显式拒绝结构"（如 context_query 的 queryable=False + reason）
        if (isinstance(ret, dict) and ret.get("queryable") is False
                and str(ret.get("reason", "")).strip()):
            continue
        silent.append(f"{cap}({desc}) → 未抛错且非显式拒绝结构: {str(ret)[:60]!r}")
    if silent:
        problems.append(f"⑤ 未声明能力被静默接受: {silent}")
    if wrong_exc:
        problems.append(f"⑤ 拒绝时抛的不是契约级 NotImplementedError（疑似真 bug 被当成拒绝）: {wrong_exc}")

    detail = ("；".join(notes) if notes else "") + ("｜问题: " + "；".join(problems) if problems else "")
    return (not problems), detail


# ───────────────────────── I2 禁止伪造 ─────────────────────────
def check_i2(bk):
    """I2 禁止伪造：可观测字段之间不得自相矛盾；取不到的字段必须**缺席**而非填 0 冒充。"""
    problems = []

    # ①② 分级自洽：用**陌生异常**（不含任何已知关键词）
    try:
        fe = bk.translate_error(RuntimeError(STRANGER_MSG), location="contract_invariants")
    except BaseException as e:                                    # noqa: BLE001
        return False, f"translate_error 自身抛错: {type(e).__name__}: {e}"

    mapped = getattr(fe, "mapped", None)
    graded_by = getattr(fe, "graded_by", None)
    if mapped is not False:
        problems.append(f"① 陌生消息却 mapped={mapped!r}（不得声称有依据）")
    if getattr(fe, "error_code", None) is not None:
        problems.append(f"① 陌生消息却给出 error_code={fe.error_code!r}")
    if graded_by not in KNOWN_GRADED_BY:
        problems.append(f"② graded_by 取值未登记: {graded_by!r}")
    elif mapped is True and graded_by not in CODE_MAP_GRADED_BY:
        problems.append(f"② 自相矛盾: mapped=True 但 graded_by={graded_by!r}")
    elif mapped is False and graded_by in CODE_MAP_GRADED_BY:
        problems.append(f"② 自相矛盾: mapped=False 但 graded_by={graded_by!r}")

    # ③ 取值域：category / disposition
    cat_name = getattr(getattr(fe, "category", None), "name", None)
    if cat_name not in KNOWN_CATEGORIES:
        problems.append(f"③ category 取值越界: {cat_name!r}")
    disp = getattr(fe, "disposition", None)
    if disp not in KNOWN_DISPOSITIONS:
        problems.append(f"③ disposition 取值越界: {disp!r}")

    # ③ present 必须是 bool 或 None（不得用 0/1 冒充）——仅在声明了 context_query 时检查
    if getattr(bk, "_CAPABILITY_KEYS", None) and bk.supports("context_query"):
        present = bk.context_query().get("present")
        if not (present is None or isinstance(present, bool)):
            problems.append(f"③ context_query()['present'] 非 bool/None: {present!r}")

    # ④ 取不到就必须缺席：memory_stats 的 allocated_mb 只在声明 memory_alloc_stat 时出现
    try:
        stats = bk.memory_stats(0)
    except BaseException as e:                                    # noqa: BLE001
        return False, f"memory_stats() 抛错: {type(e).__name__}: {e}（已有判据）"
    has_key = "allocated_mb" in stats
    declared = bool(bk.supports("memory_alloc_stat"))
    if declared and not has_key:
        problems.append("④ 声明了 memory_alloc_stat 却缺 allocated_mb")
    if (not declared) and has_key:
        problems.append(f"④ 未声明 memory_alloc_stat 却给出 allocated_mb={stats['allocated_mb']!r}"
                        "（**填值冒充**）")
    if has_key and (not isinstance(stats["allocated_mb"], (int, float)) or isinstance(stats["allocated_mb"], bool)):
        problems.append(f"④ allocated_mb 非数值: {stats['allocated_mb']!r}")

    return (not problems), ("；".join(problems) if problems else
                            f"陌生消息 → {cat_name}/{disp}（mapped={mapped}, graded_by={graded_by!r}）")


# ───────────────────────── I3 失效受管 ─────────────────────────
def check_i3(bk, ordinal=0):
    """I3 失效受管：二次释放 / 二次销毁 / 跨种类误用 / 销毁后用其流 —— 四类都必须**如实报错**。"""
    problems, notes = [], []
    mins = getattr(bk, "_CAPABILITY_KEYS", None)
    if not mins:
        return False, "后端未定义 `_CAPABILITY_KEYS`（无法判定）"

    ok_mem = bool(bk.supports("memory_alloc"))
    ok_ctx = bool(bk.supports("context_lifecycle"))
    if not (ok_mem or ok_ctx):
        return True, (NOT_APPLICABLE + "本后端未声明 `memory_alloc` 与 `context_lifecycle`"
                      "（如实不具备；**未验证 ≠ 已确认不具备**）")

    mem_handle = ctx_handle = None
    try:
        # ── A 内存句柄（声明了才验） ──
        if ok_mem:
            mem_handle = bk.allocate(4096)
            bk.free(mem_handle)
            try:
                bk.free(mem_handle)
                problems.append("A 二次释放未报错（静默）")
            except ValueError:
                notes.append("A 二次释放 → ValueError ✓")
            except BaseException as e:                            # noqa: BLE001
                problems.append(f"A 二次释放抛的是 {type(e).__name__}（要求契约级 ValueError）")
            # 跨种类误用：内存句柄当上下文销毁（仅当上下文能力可用，否则无可比对象）
            if ok_ctx:
                try:
                    bk.context_destroy(mem_handle)
                    problems.append("A 跨种类误用（内存句柄当上下文）未报错")
                except ValueError:
                    notes.append("A 跨种类误用 → ValueError ✓")
                except BaseException as e:                        # noqa: BLE001
                    problems.append(f"A 跨种类误用抛 {type(e).__name__}（要求 ValueError）")

        # ── B 上下文与绑定语义（声明了才验） ──
        if ok_ctx:
            ctx_handle = bk.context_create(ordinal)
            native = bk.create_stream()                           # 在该上下文之下创建流
            bk.note_stream_created(native)
            bk.context_destroy(ctx_handle)
            try:
                bk.context_destroy(ctx_handle)
                problems.append("B 二次销毁未报错")
            except ValueError:
                notes.append("B 二次销毁 → ValueError ✓")
            except BaseException as e:                            # noqa: BLE001
                problems.append(f"B 二次销毁抛 {type(e).__name__}（要求 ValueError）")
            # ★ 绑定语义：销毁上下文后使用其流必须**如实报错**（厂商侧实测为静默）
            try:
                bk.check_stream_usable(native)
                problems.append("B 销毁上下文后使用其流未报错（厂商静默被透传）")
            except RuntimeError:
                notes.append("B 销毁后使用其流 → RuntimeError ✓")
            except BaseException as e:                            # noqa: BLE001
                problems.append(f"B 销毁后使用其流抛 {type(e).__name__}（要求 RuntimeError）")
            if ok_mem and mem_handle is not None:
                # 反向跨种类：内存句柄已释放，这里只验"上下文句柄当内存句柄"
                h2 = bk.allocate(4096)
                try:
                    bk.free({"handle_id": 10 ** 9, "kind": "memory"})
                    problems.append("B 未登记句柄（伪 id）未报错")
                except ValueError:
                    notes.append("B 未登记句柄 → ValueError ✓")
                except BaseException as e:                        # noqa: BLE001
                    problems.append(f"B 未登记句柄抛 {type(e).__name__}（要求 ValueError）")
                bk.free(h2)
    finally:
        # 清理：不把设备/登记表留在半途状态（I3 的唯一设备副作用必须收干净）
        for h, fn in ((mem_handle, "free"), (ctx_handle, "context_destroy")):
            if h is None:
                continue
            try:
                getattr(bk, fn)(h)
            except BaseException:                                 # noqa: BLE001
                pass
        try:
            bk.set_device(ordinal)                                # 恢复当前设备（销毁上下文后必须重设）
        except BaseException:                                     # noqa: BLE001
            pass
        if bk.memory_handle_count() != 0:
            problems.append(f"清理后仍有在世内存句柄: {bk.memory_handle_count()}（本判据泄漏）")

    return (not problems), "；".join(notes + problems)


# ───────────────────────── I4 降级可观测 ─────────────────────────
def check_i4(bk):
    """I4 降级可观测：降级必须计数可查；`.native` 取用与退化**分开计数**（语义相反）。"""
    problems, notes = [], []

    # ① 结构 + 单调不减
    for name, fn in (("degradations", bk.degradations), ("native_accesses", bk.native_accesses)):
        a = fn()
        b = fn()
        if not isinstance(a, dict) or not {"total", "by_kind"} <= set(a):
            problems.append(f"① {name}() 结构应为 {{total, by_kind}}: {a!r}")
            continue
        if not isinstance(a["total"], int) or a["total"] < 0 or not isinstance(a["by_kind"], dict):
            problems.append(f"① {name}() 取值不合法: {a!r}")
        if not (isinstance(a["total"], int) and isinstance(b["total"], int) and b["total"] >= a["total"]):
            problems.append(f"① {name}() 不单调: {a['total']} → {b['total']}")

    # ② `.native` 取用：native +1 而 **退化不变**（两者语义相反，混一起即不可观测）
    try:
        from runtime.api.stream import Stream as _Stream
        d0 = bk.degradations()["total"]
        n0 = bk.native_accesses()["total"]
        st = _Stream(bk, bk.create_stream())
        _ = st.native
        n1 = bk.native_accesses()["total"]
        d1 = bk.degradations()["total"]
        if n1 != n0 + 1:
            problems.append(f"② `.native` 取用未计数: {n0} → {n1}（应 +1）")
        if d1 != d0:
            problems.append(f"② `.native` 取用污染了退化计数: {d0} → {d1}（应不变）")
        else:
            notes.append(f"② .native 计数 {n0}→{n1}、退化保持 {d0} ✓")
    except BaseException as e:                                    # noqa: BLE001
        problems.append(f"② `.native` 审计路径不可用: {type(e).__name__}: {e}")

    # ③ `info()` 必须同时暴露二者（降级可观测的对外形态）
    info = bk.info()
    for k in ("native_accesses", "degradations"):
        if k not in info:
            problems.append(f"③ info() 缺 {k}（降级不可观测）")

    # ④ 跨流保护**不得静默通过**：tensor 侧无 record_stream 时，要么原生调用、要么退化 +1
    try:
        from runtime.api.stream import Stream as _Stream2
        st2 = _Stream2(bk, bk.create_stream())

        class _WithRec:
            def __init__(self):
                self.calls = 0

            def record_stream(self, s):
                self.calls += 1

        class _NoRec:
            pass

        d0 = bk.degradations()["total"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bk.supports("record_stream"):
                t = _WithRec()
                st2.record_stream(t)
                if t.calls != 1:
                    problems.append("④ 声明 record_stream 却未走原生路径")
                elif bk.degradations()["total"] != d0:
                    problems.append("④ 原生路径不应计退化")
                else:
                    notes.append("④ 声明 record_stream ⇒ 原生路径、零退化 ✓")
            t2 = _NoRec()                                          # tensor 无 record_stream
            st2.record_stream(t2)                                  # 必须不抛错
            d2 = bk.degradations()["total"]
            if d2 != d0 + 1:
                problems.append(f"④ tensor 无 record_stream 时退化未计数: {d0} → {d2}（应 +1）")
            else:
                notes.append("④ 无原生 record_stream ⇒ 保守路径 + 退化 +1 ✓")
    except BaseException as e:                                    # noqa: BLE001
        problems.append(f"④ 跨流保护路径异常: {type(e).__name__}: {e}")

    return (not problems), "；".join(notes + problems)


#: 判据串标签（runner 用它拼出 `CONTRACT_INVARIANTS_PASS/FAIL`，使该腿的结论**自证是哪一项**）
VERDICT_TAG = "CONTRACT_INVARIANTS"

#: 核心检查表（真机与离线**共用**；新增不变式只改这里）
CHECKS = (
    ("I1_honest_declaration", "诚实声明", check_i1),
    ("I2_no_fabrication", "禁止伪造", check_i2),
    ("I3_invalidated_governed", "失效受管", check_i3),
    ("I4_degradation_observable", "降级可观测", check_i4),
)


# ───────────────────────── conformance 用例入口 ─────────────────────────
def _backend_of(ctx):
    import runtime
    return runtime.current()


def case_i1_honest_declaration(ctx):
    """I1 诚实声明（定义见《接口约定》§1.8）。"""
    return check_i1(_backend_of(ctx))


def case_i2_no_fabrication(ctx):
    """I2 禁止伪造（定义见《接口约定》§1.8）。"""
    return check_i2(_backend_of(ctx))


def case_i3_invalidated_governed(ctx):
    """I3 失效受管（定义见《接口约定》§1.8）。"""
    return check_i3(_backend_of(ctx))


def case_i4_degradation_observable(ctx):
    """I4 降级可观测（定义见《接口约定》§1.8）。"""
    return check_i4(_backend_of(ctx))
