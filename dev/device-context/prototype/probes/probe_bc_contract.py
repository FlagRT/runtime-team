#!/usr/bin/env python3
"""工作包 B/C 契约的真机验证探针（后端无关）。

设计纪律（沿用 `exp_divergence_cost.py`）：
  · **每组独立子进程**：上下文/句柄类操作一旦失败会污染进程后续状态；
  · 只通过**统一 API** 操作（不直接 import 厂商扩展），厂商原生的对照行为单独标注；
  · 结论边界：只在当前档位/环境成立。

用法：
    python3 probes/probe_bc_contract.py --backend ascend --out out.json       # 全部组
    python3 probes/probe_bc_contract.py --backend ascend --group c1           # 单组
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROTO = HERE.parent
sys.path.insert(0, str(PROTO))

GROUPS = ("m1", "m2", "c1", "c2", "c3", "c4")


def _emit(name, **kw):
    print("RESULT_JSON " + json.dumps({"group": name, **kw}, ensure_ascii=False))


# ─────────────────────────── 各组实现 ───────────────────────────

def group_m1(runtime):
    """B-1：内存句柄 —— 申请有真实占用、成对释放、负向必须报错。"""
    bk = runtime.current()
    out = {"capability": bk.supports("memory_alloc")}
    if not bk.supports("memory_alloc"):
        _emit("m1", skipped="未声明 memory_alloc（如实不具备）", **out)
        return
    torch = bk.torch
    ns = getattr(torch, bk.device_type if bk.device_type != "cuda" else "cuda")

    def snap():
        d = {}
        try:
            free, total = ns.mem_get_info()
            d["free_mb"] = round(free / 1048576, 1)
        except Exception as e:                                # noqa: BLE001
            d["free_mb"] = f"err:{type(e).__name__}"
        try:
            d["stats"] = bk.memory_stats(0)
        except Exception as e:                                # noqa: BLE001
            d["stats"] = f"err:{type(e).__name__}"
        return d

    before = snap()
    h = bk.allocate(8 * 1024 * 1024)
    after_alloc = snap()
    out.update({"handle_keys": sorted(h), "handle_has_ptr": "ptr" in h,
                "before": before, "after_alloc": after_alloc,
                "handle_count_after_alloc": bk.memory_handle_count()})
    bk.free(h)
    out["after_free"] = snap()
    out["handle_count_after_free"] = bk.memory_handle_count()
    # 负向
    try:
        bk.free(h)
        out["double_free"] = "未报错（静默）"
    except Exception as e:                                    # noqa: BLE001
        out["double_free"] = f"{type(e).__name__}: {str(e)[:90]}"
    try:
        bk.allocate(0)
        out["alloc_zero"] = "未报错"
    except Exception as e:                                    # noqa: BLE001
        out["alloc_zero"] = f"{type(e).__name__}"
    _emit("m1", **out)


def group_m2(runtime):
    """B-3/B-4：record_stream 能力位与真机路径 + `.native` 审计。"""
    bk = runtime.current()
    out = {"record_stream_capability": bk.supports("record_stream")}
    st = runtime.create_stream()
    t = bk.torch.ones(8, device=f"{bk.device_type}:0")
    d0 = bk.degradations()["total"]
    try:
        st.record_stream(t)
        out["record_stream_real"] = ("原生路径（无退化）" if bk.degradations()["total"] == d0
                                     else "走了保守同步（退化计数 +1）")
    except Exception as e:                                    # noqa: BLE001
        out["record_stream_real"] = f"{type(e).__name__}: {str(e)[:90]}"
    bk.reset_native_accesses()
    _ = st.native
    out["native_after_public_access"] = bk.native_accesses()["total"]
    bk.reset_native_accesses()
    ev = runtime.create_event()
    st.wait_event(ev)                                          # 层内部路径
    out["native_after_internal_path"] = bk.native_accesses()["total"]
    st.synchronize()
    _emit("m2", **out)


def group_c1(runtime):
    """C-1：上下文创建/切换/销毁/计数 + 重复销毁必须报错。"""
    bk = runtime.current()
    out = {"capability": bk.supports("context_lifecycle")}
    if not bk.supports("context_lifecycle"):
        _emit("c1", skipped="未声明 context_lifecycle（如实不具备）", **out)
        return
    out["count_before"] = bk.context_count()
    ctx = bk.context_create(0)
    out["handle_keys"] = sorted(ctx)
    out["count_after_create"] = bk.context_count()
    bk.context_set(ctx)
    out["set"] = "ok"
    out["compute_in_ctx"] = float(bk.torch.ones(4).sum())
    bk.context_destroy(ctx)
    out["count_after_destroy"] = bk.context_count()
    try:
        bk.context_destroy(ctx)
        out["double_destroy"] = "未报错（静默）"
    except Exception as e:                                    # noqa: BLE001
        out["double_destroy"] = f"{type(e).__name__}: {str(e)[:90]}"
    out["compute_after_destroy"] = float(bk.torch.ones(4).sum())
    _emit("c1", **out)


def group_c2(runtime):
    """C-2：绑定语义 —— 销毁上下文后使用其流。

    ① 本层使用点拦截（统一 API）：**必须如实报错**；
    ② 顺带取证：**厂商原生流**销毁后直接使用是什么行为（不经过本层）。
    """
    bk = runtime.current()
    out = {}
    if not bk.supports("context_lifecycle"):
        _emit("c2", skipped="未声明 context_lifecycle（如实不具备）")
        return
    ctx = bk.context_create(0)
    st = runtime.create_stream()                               # 在该上下文下创建
    bk.context_destroy(ctx)
    try:
        st.context()
        out["unified_use_after_destroy"] = "未报错（静默成功）"
    except Exception as e:                                    # noqa: BLE001
        out["unified_use_after_destroy"] = f"{type(e).__name__}: {str(e)[:110]}"
    # 厂商原生对照（**只取证，不判 FAIL**）
    try:
        ns = getattr(bk.torch, bk.device_type if bk.device_type != "cuda" else "cuda")
        nstream = ns.Stream()
        with ns.stream(nstream):
            _ = bk.torch.ones(4)
        ns.synchronize()
        out["vendor_native_use_after_destroy"] = "未报错（静默成功；退出清理阶段可能才暴露）"
    except Exception as e:                                    # noqa: BLE001
        out["vendor_native_use_after_destroy"] = f"{type(e).__name__}: {str(e)[:110]}"
    _emit("c2", **out)


def group_c3(runtime):
    """C-3：多上下文隔离 —— 可共存、各自可销毁、互不串。"""
    bk = runtime.current()
    if not bk.supports("context_lifecycle"):
        _emit("c3", skipped="未声明 context_lifecycle（如实不具备）")
        return
    out = {}
    a = bk.context_create(0)
    b = bk.context_create(0)
    out["count_with_two"] = bk.context_count()
    bk.context_set(a)
    v1 = float(bk.torch.ones(4).sum())
    bk.context_set(b)
    v2 = float(bk.torch.ones(4).sum())
    out.update({"value_in_a": v1, "value_in_b": v2, "pairwise_isolated": v1 == v2 == 4.0})
    bk.context_destroy(b)
    out["count_after_destroy_b"] = bk.context_count()
    bk.context_set(a)
    out["value_back_in_a"] = float(bk.torch.ones(4).sum())
    bk.context_destroy(a)
    out["count_after_destroy_a"] = bk.context_count()
    out["compute_after_all"] = float(bk.torch.ones(4).sum())
    _emit("c3", **out)


def group_c4(runtime):
    """C-4：上下文**只读观测**（工作包 C·P800 专项）—— 真机取证。

    取证目标（每条对应一个可能出错的地方）：
      · 真机上能否**读到**框架自建的上下文（present / ordinal / flags）；
      · `managed_by` 是否**如实** —— 本层一个上下文都没造，就不许自称 `unified`；
      · **只读安全**：查询前后做**同一次计算**，结果必须一致（证明查询无副作用）。

    P800 上的期望值：`managed_by == "external"`（上下文由 XPytorch/XRE 自建）。
    """
    bk = runtime.current()
    if not bk.supports("context_query"):
        _emit("c4", skipped="未声明 context_query（如实不具备）", capability=False)
        return
    out = {"capability": True}
    try:
        import torch
        out["device_count"] = bk.device_count()
        dev = "cuda" if getattr(bk, "device_type", "") == "cuda" else None
        a = torch.ones(8, 8, device=dev)
        out["compute_before"] = float((a @ a).sum())        # 触碰设备（让上下文建起来）
        q = bk.context_query()
        out["query"] = q
        out["present"] = q.get("present")
        out["ordinal"] = q.get("ordinal")
        out["flags"] = q.get("flags")
        out["managed_by"] = q.get("managed_by")
        out["reason"] = q.get("reason")
        out["compute_after"] = float((a @ a).sum())         # 只读 ⇒ 必须与 before 相同
        out["readonly_safe"] = (out["compute_before"] == out["compute_after"])
        out["context_query_works"] = bool(q.get("queryable") and q.get("present"))
    except Exception as e:                                    # noqa: BLE001
        out["err"] = f"{type(e).__name__}: {str(e)[:160]}"
    _emit("c4", **out)


_FUNCS = {"m1": group_m1, "m2": group_m2, "c1": group_c1, "c2": group_c2,
          "c3": group_c3, "c4": group_c4}


def run_one(backend, group):
    import runtime
    runtime.use(backend)
    _FUNCS[group](runtime)


def run_all(backend, out_path, ordinal=0):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    res = {"backend": backend, "groups": {}, "vendor": None}
    for g in GROUPS:
        cmd = [sys.executable, "-u", str(pathlib.Path(__file__).resolve()), "--backend", backend, "--group", g]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
            res["groups"][g] = (json.loads(line[len("RESULT_JSON "):]) if line
                                else {"fatal": p.stderr.strip()[-400:] or "no RESULT_JSON"})
            res["groups"][g]["returncode"] = p.returncode
        except subprocess.TimeoutExpired:
            res["groups"][g] = {"fatal": "timeout(300s)"}
    try:
        import runtime
        runtime.use(backend)
        bk = runtime.current()
        res["vendor"] = {"backend": bk.name, "device_type": bk.device_type,
                         "torch": getattr(bk.torch, "__version__", None),
                         "device_count": bk.device_count(),
                         "capabilities": sorted(bk._capabilities),
                         "info_native_accesses": bk.info().get("native_accesses"),
                         "info_degradations": bk.info().get("degradations")}
    except Exception as e:                                    # noqa: BLE001
        res["vendor"] = {"error": f"{type(e).__name__}: {e}"}

    # 汇总判据（本探针自己给结论）
    g = res["groups"]
    verdict = {}
    if bk_ok := (res["vendor"] or {}).get("capabilities") is not None:
        caps = set((res["vendor"] or {}).get("capabilities") or [])
        if "memory_alloc" in caps:
            m1 = g.get("m1", {})
            verdict["B1_allocate_free_pairs"] = bool(
                m1.get("handle_count_after_free") == 0
                and isinstance(m1.get("double_free"), str) and "ValueError" in m1.get("double_free", ""))
        if "record_stream" in caps:
            verdict["B3_record_stream_native_path"] = (
                g.get("m2", {}).get("record_stream_real") == "原生路径（无退化）")
        verdict["B4_native_audit_isolated"] = bool(
            g.get("m2", {}).get("native_after_public_access") == 1
            and g.get("m2", {}).get("native_after_internal_path") == 0)
        if "context_lifecycle" in caps:
            c1 = g.get("c1", {})
            verdict["C1_context_lifecycle"] = bool(
                c1.get("count_after_create") == 1 and c1.get("count_after_destroy") == 0
                and "ValueError" in str(c1.get("double_destroy")))
            verdict["C2_binding_intercepted"] = "RuntimeError" in str(
                g.get("c2", {}).get("unified_use_after_destroy"))
            verdict["C3_multi_context_isolated"] = bool(
                g.get("c3", {}).get("count_with_two") == 2
                and g.get("c3", {}).get("pairwise_isolated") is True)
        if "context_query" in caps:
            c4 = g.get("c4", {})
            # 只读观测：读得到 + **查询无副作用** + managed_by 为合法取值
            # （P800 期望 `external` —— 上下文由框架自建；该值记在 JSON 里供报告引用）
            verdict["C4_context_query_readonly"] = bool(
                c4.get("context_query_works") is True
                and c4.get("readonly_safe") is True
                and c4.get("managed_by") in ("external", "unified"))
    res["verdict"] = verdict
    res["PASS"] = all(verdict.values()) if verdict else False
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if out_path:
        pathlib.Path(out_path).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--group", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.group:
        run_one(a.backend, a.group)
    else:
        run_all(a.backend, a.out)
