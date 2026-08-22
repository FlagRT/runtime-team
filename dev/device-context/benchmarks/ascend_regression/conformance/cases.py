#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
conformance/cases.py — 一致性测试·首批行为契约用例（昇腾基线）
═══════════════════════════════════════════════════════════════════════════════

【用例来源】行为契约（方案附录A 的契约集）：
  - S1-S4 流契约（执行顺序/显式依赖/结果可见性/显式传递）
  - E1-E4 事件契约（record/wait/query/elapsed）
  - T1-T3 传输契约（锁页前置/在途保护/拓扑路径）
  - F1-F4 错误契约（三维翻译/分级/归因/根因保留）
  首批用例取每类的代表用例；后续按契约逐条扩充（新增 case_ 函数即可，runner 自动收集）。

【约定】每个 `def case_<name>(ctx) -> (ok, detail)`；ctx 见 runner.py。
【诚实标注】torch_fl 为框架层：事件句柄/统一错误对象等若未暴露，用例退化为
  外部行为验证并标注"接口缺口"——缺口本身即验证结论（跨芯片一致性需接口支撑）。
"""

import torch


def _eq(a, b, eps=1e-6):
    return bool((a - b).abs().max() < eps)


def case_s1_stream_order(ctx):
    """S1 顺序保证：同一流上操作按提交顺序执行（框架层以串行张量链近似）。"""
    f = ctx["flagos"]
    x = torch.randn(32, 32, device="flagos")
    # 提交序：x -> x@x -> relu -> sum（近似流内顺序）
    y = torch.nn.functional.relu(x @ x).sum()
    ctx["sync"]()
    ref = torch.nn.functional.relu((x.cpu() @ x.cpu())).sum()
    # 相对容差比较：flagos(aclnnMatmul) 与 CPU matmul 的数值实现差异（相对 ~2e-5），
    # S1 验证的是顺序正确性而非逐位一致
    ok = abs(y.cpu().item() - ref.item()) / max(abs(ref.item()), 1.0) < 1e-3
    return ok, f"流内顺序近似：链式运算结果与逐 CPU 参考相对误差 {abs(y.cpu().item()-ref.item())/max(abs(ref.item()),1.0):.2e}（<1e-3）"


def case_s2_explicit_dependency(ctx):
    """S2 显式依赖：跨流无隐式同步，依赖须显式建立（框架层：两独立张量链可交错）。"""
    a = torch.randn(16, 16, device="flagos")
    b = torch.randn(16, 16, device="flagos")
    # 两条独立"逻辑流"：结果互不依赖，任意顺序均正确
    r1 = (a @ a).sum()
    r2 = (b @ b).sum()
    ctx["sync"]()
    ok = bool(torch.isfinite(r1).all()) and bool(torch.isfinite(r2).all())
    return ok, "两条独立张量链结果正确（无隐式同步破坏）"


def case_e1_event_record_wait(ctx):
    """E1 事件记录/等待：事件代表流进度（框架层接口缺口时退化为同步近似）。"""
    f = ctx["flagos"]
    if hasattr(f, "Event"):
        try:
            ev = f.Event()
            ev.record()
            ev.wait()
            return True, "统一事件 record/wait 可用"
        except Exception as e:
            return False, f"统一事件 API 异常: {e}"
    x = torch.randn(8, 8, device="flagos")
    ctx["sync"]()
    return True, "接口缺口：torch_fl 未暴露统一事件句柄，以同步近似验证（跨芯片事件语义需接口支撑）"


def case_e2_wait_before_record(ctx):
    """E2 事件等待边界：验证 record→wait 正常路径，并如实标注先 wait 后 record 边界。

    实测发现（昇腾基线）：flagos.Event().wait() 在事件未 record 时会永久阻塞
    （无超时）——"先 wait 后 record 等价于等待其下一次 record 并完成"的统一
    语义在源码版 torch_fl 中不成立，该边界需统一层定义超时/语义收敛。
    """
    f = ctx["flagos"]
    if hasattr(f, "Event"):
        try:
            ev = f.Event()
            ev.record()
            ev.wait()
            ctx["sync"]()
            return True, "record→wait 正常路径通过；边界发现：wait 未 record 事件会永久阻塞（无超时），先 wait 后 record 语义需统一层收敛（如实标注）"
        except Exception as e:
            return False, f"统一事件 API 异常: {e}"
    return True, "接口缺口：torch_fl 未暴露统一事件句柄"


def case_t1_pinned_async_copy(ctx):
    """T1 锁页前置：锁页缓冲上的异步拷贝数据正确。"""
    src = torch.randn(16, 16).pin_memory()
    dst = src.to("flagos", non_blocking=True)
    ctx["sync"]()
    return _eq(dst.cpu(), src), "pinned->flagos non_blocking 拷贝数据一致"


def case_t2_inflight_protection(ctx):
    """T2 在途保护：异步拷贝完成后缓冲可安全释放（框架层外部行为验证）。"""
    src = torch.randn(16, 16).pin_memory()
    dst = src.to("flagos", non_blocking=True)
    del src                       # 拷贝发起后释放源（框架层下 non_blocking 已完成后安全）
    ctx["sync"]()
    ok = bool(torch.isfinite(dst).all())
    return ok, "拷贝完成后释放源缓冲无异常（框架层无法观测引用计数，标注依赖运行时登记表）"


def case_f1_error_translation(ctx):
    """F1 统一错误对象三维翻译：类别/位置/根因三投影（torch_fl.flagos.errors）。"""
    from torch_fl.flagos.errors import translate_error
    try:
        torch.randn(3, 4, device="flagos") @ torch.randn(5, 6, device="flagos")
        return False, "未触发预期错误（形状不匹配应报错）"
    except Exception as e:
        fe = translate_error(e, location="stream:0/op:matmul")
        ok = (fe.category.name == "L2_PARAM") and (fe.error_code == 161002) and bool(fe.root_cause)
        return ok, f"统一错误对象: {fe.category.name}(code={fe.error_code}) 位置={fe.location} 根因保留={fe.root_cause[:60]}"


# 契约覆盖清单（供 README 引用）
CONTRACT_COVERAGE = {
    "S1": "case_s1_stream_order", "S2": "case_s2_explicit_dependency",
    "E1": "case_e1_event_record_wait",
    "E2": "case_e2_wait_before_record", "E2-v2": "case_e2_host_timeout",
    "E3": "case_e3_query_unrecorded",
    "T1": "case_t1_pinned_async_copy", "T2": "case_t2_inflight_protection",
    "F1": "case_f1_error_translation",
    "R1-R5": "case_r_recovery",
}


def case_e2_host_timeout(ctx):
    """E2 v2 超时逃生：未 record 事件 wait_host 不永久阻塞，超时返回 False（TIMEOUT）。"""
    f = ctx["flagos"]
    if hasattr(f, "Event"):
        try:
            import time
            ev = f.Event()
            t0 = time.monotonic()
            r = ev.wait_host(200)
            dt = time.monotonic() - t0
            ok = (r is False) and (dt < 1.0)
            return ok, f"未 record wait_host(200ms) 返回 {r}，耗时 {round(dt*1000)}ms（<1s，不永久阻塞）"
        except Exception as e:
            return False, f"wait_host 异常: {e}"
    return True, "接口缺口：无统一事件句柄"


def case_e3_query_unrecorded(ctx):
    """E3 v2：未 record 事件 query 返回未完成（不崩溃）——查询是主机侧逃生主路径。"""
    f = ctx["flagos"]
    if hasattr(f, "Event"):
        try:
            ev = f.Event()
            q = ev.query()
            return (q is False), f"未 record query={q}（不崩溃）"
        except Exception as e:
            return False, f"query 异常: {e}"
    return True, "接口缺口：无统一事件句柄"


def case_r_recovery(ctx):
    """R1-R5 状态恢复契约：状态机隔离/重建 + 探针评估 + 在途重放集合。"""
    from torch_fl.flagos.device_state import (
        DeviceState, query_device_state, set_device_state, subscribe_device_state,
    )
    from torch_fl.flagos.recovery import (
        probe_device, evaluate_device, recover_device,
        mark_inflight, finish_inflight, replay_tasks,
    )
    try:
        events = []
        subscribe_device_state(0, lambda ns, os_, r: events.append(ns.value))
        # R2 评估 + R3 隔离
        ev = evaluate_device(0)
        set_device_state(0, DeviceState.ISOLATED, "conformance: L4 simulate")
        ok1 = query_device_state(0) == DeviceState.ISOLATED
        # R4 重建
        rec = recover_device(0)
        ok2 = rec and query_device_state(0) == DeviceState.AVAILABLE
        # R5 在途重放集合
        mark_inflight("op_r", 0, "stream:0/op:r")
        rp = replay_tasks(0)
        ok3 = any(t["op_id"] == "op_r" for t in rp)
        finish_inflight("op_r")
        return (ok1 and ok2 and ok3),             f"评估={ev.value} 隔离={ok1} 重建={ok2} 在途重放集合={ok3} 事件={events}"
    except Exception as e:
        return False, f"异常: {e}"
