"""昆仑芯 P800 设备上下文五域探针（只读探测，不改环境）"""
import json, sys, traceback
R = {}
def t(name, fn):
    try:
        R[name] = fn()
    except Exception as e:
        R[name] = f"EXC {type(e).__name__}: {str(e)[:160]}"

import torch
t("torch", lambda: torch.__version__)
t("device_count", lambda: torch.cuda.device_count())
t("device_name_0", lambda: torch.cuda.get_device_name(0))

# 域1 设备抽象
t("1_current_device", lambda: torch.cuda.current_device())
def _mem():
    free, total = torch.cuda.mem_get_info(0)
    return {"free_MiB": round(free/2**20), "total_MiB": round(total/2**20)}
t("1_mem_get_info", _mem)
t("1_mem_allocated", lambda: torch.cuda.memory_allocated(0))

# 域2 多流 Stream / Event
t("2_create_stream", lambda: type(torch.cuda.Stream()).__name__)
def _prio():
    lo, hi = torch.cuda.Stream.priority_range() if hasattr(torch.cuda.Stream, "priority_range") else (None, None)
    return {"priority_range": [lo, hi]}
t("2_stream_priority", _prio)
def _evt():
    e = torch.cuda.Event(enable_timing=True)
    e.record(); torch.cuda.synchronize(); e.record()
    torch.cuda.synchronize()
    return {"event_elapsed_ms": e.elapsed_time(e)}
t("2_event_timing", _evt)
def _ctx():
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x = torch.ones(2048, 2048, device="cuda"); y = x + 1
    s.synchronize()
    return {"stream_context_ok": True, "y_sum": float(y.sum())}
t("2_stream_context", _ctx)
t("2_current_stream", lambda: type(torch.cuda.current_stream()).__name__)
def _cross():
    s1 = torch.cuda.Stream(); s2 = torch.cuda.Stream()
    a = torch.zeros(4096, device="cuda")
    with torch.cuda.stream(s1):
        a += 1; e = torch.cuda.Event(); e.record(s1)
    s2.wait_event(e)
    with torch.cuda.stream(s2):
        a += 10
    torch.cuda.synchronize()
    return {"cross_stream_visible": float(a.sum()), "expect": 4096*11.0}
t("2_cross_stream", _cross)
def _record_stream():
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x = torch.empty(1024, 1024, device="cuda")
    x.record_stream(s)
    return "record_stream OK"
t("2_record_stream", _record_stream)

# 域4 错误翻译（拿真实错误码）
def _err_oob():
    try:
        torch.cuda.set_device(99)
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:200]}"
t("4_err_bad_device", _err_oob)
def _err_oom():
    try:
        torch.empty(int(1e12), dtype=torch.float32, device="cuda")
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:200]}"
t("4_err_oom", _err_oom)

# 域5 状态恢复 / 分布式
t("5_dist_available", lambda: torch.distributed.is_available())
t("5_dist_backends", lambda: sorted(torch.distributed.Backend.backend_list) if hasattr(torch.distributed, "Backend") else "n/a")
t("5_bkcl_registered", lambda: "bkcl" in str(torch.distributed.Backend.backend_list).lower())
t("5_cuda_reset_fn", lambda: hasattr(torch.cuda, "empty_cache"))

print(json.dumps(R, ensure_ascii=False, default=str, indent=1))
