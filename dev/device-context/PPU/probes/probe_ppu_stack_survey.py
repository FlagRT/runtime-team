#!/usr/bin/env python3
"""PPU（平头哥 真武 ZW810E）设备上下文栈 **能力探测**（只读，不改任何状态）。

用途：按《新芯片接入手册》的接入纪律 —— **先探测、后声明**。本探针的输出
      **直接决定** `runtime/backends/<ppu>/backend.py` 里 `_capabilities` 能写哪些键。
      （铁律：没实测到的一律不声明；「未验证」与「已确认不具备」不得混写。）

纪律：
  · 每一项独立 try/except ⇒ 单项失败**不得**影响其余项（也不得让整轮崩掉、连汇总都打不出）；
  · 输出机器可读 `KEY = VALUE`；取不到就写 `UNAVAILABLE(<原因>)`，**不猜、不补零**；
  · 本探针**不创建/销毁上下文**、不改设备状态（只做只读查询 + 一次张量分配）。

运行（容器内，已挂 PPU 设备）：
    python3 probe_ppu_stack_survey.py
"""
from __future__ import annotations

import ctypes
import os
import sys
import traceback

RESULT = []


def rec(key, val):
    """记录一条探测结果（同时打印，便于落日志）。"""
    RESULT.append((key, val))
    print(f"{key} = {val}", flush=True)


def probe(key, fn):
    """执行单项探测；异常一律转成 UNAVAILABLE 文本，不中断。"""
    try:
        rec(key, fn())
    except BaseException as e:                                  # noqa: BLE001
        rec(key, f"UNAVAILABLE({type(e).__name__}: {str(e)[:160]})")
    return None


def main():
    import torch
    import torch.distributed as dist

    print("=" * 78)
    print("PPU 设备上下文栈能力探测（只读）")
    print("=" * 78)

    # ── A. 基本形态（决定走哪条路 A/B/C/D）──────────────────────────────
    probe("A1.torch_version", lambda: torch.__version__)
    probe("A2.torch_cuda_version", lambda: getattr(torch.version, "cuda", None))
    probe("A3.cuda_is_available", lambda: torch.cuda.is_available())
    probe("A4.device_count", lambda: torch.cuda.device_count())
    probe("A5.device_name_0", lambda: torch.cuda.get_device_name(0))
    probe("A6.current_device", lambda: torch.cuda.current_device())
    for ns in ("npu", "mlu", "xpu", "cuda", "hggc", "ppu"):
        probe(f"A7.namespace_torch_{ns}", lambda ns=ns: f"has={hasattr(torch, ns)}")

    # ── B. 设备属性（真实能力，不是命名空间名字）────────────────────────
    probe("B1.device_properties_0",
          lambda: {k: getattr(torch.cuda.get_device_properties(0), k, None)
                   for k in ("name", "major", "minor", "total_memory", "multi_processor_count")})
    probe("B2.tensor_matmul_on_device0", lambda: float(
        (lambda a: (a @ a).sum().item())(torch.randn(8, 8, device="cuda:0"))))

    # ── C. 内存（决定 memory / memory_alloc / memory_alloc_stat）─────────
    probe("C1.memory_stats_keys", lambda: sorted(torch.cuda.memory_stats(0).keys())[:8])
    probe("C2.mem_get_info_0_MB", lambda: tuple(
        int(x / 1024 / 1024) for x in torch.cuda.mem_get_info(0)))
    probe("C3.memory_allocated_MB", lambda: int(torch.cuda.memory_allocated(0) / 1024 / 1024))
    probe("C4.has_caching_allocator_alloc",
          lambda: hasattr(torch.cuda, "caching_allocator_alloc"))
    probe("C5.has_caching_allocator_delete",
          lambda: hasattr(torch.cuda, "caching_allocator_delete"))
    probe("C6.has_caching_allocator_free", lambda: hasattr(torch.cuda, "caching_allocator_free"))
    probe("C7.has_memory_reserved", lambda: hasattr(torch.cuda, "memory_reserved"))

    # ── D. 流（决定 stream / stream_priority / stream_priority_control）──
    probe("D1.create_stream", lambda: type(torch.cuda.Stream()).__name__)
    probe("D2.has_ExternalStream", lambda: hasattr(torch.cuda, "ExternalStream"))
    probe("D3.Stream_priority_range_attr",
          lambda: hasattr(torch.cuda.Stream, "priority_range"))
    probe("D4.Stream_priority_range_call",
          lambda: torch.cuda.Stream.priority_range() if hasattr(
              torch.cuda.Stream, "priority_range") else "NO_ATTR")
    probe("D5.current_stream", lambda: type(torch.cuda.current_stream()).__name__)
    probe("D6.stream_query", lambda: bool(torch.cuda.current_stream().query()))
    probe("D7.stream_synchronize_signature",
          lambda: str(__import__("inspect").signature(torch.cuda.Stream.synchronize)))
    probe("D8.has_stream_ctx_mgr", lambda: hasattr(torch.cuda, "stream"))
    probe("D9.stream_priority_ctor_accepts", lambda: type(
        torch.cuda.Stream(priority=0)).__name__)

    # ── E. 事件（决定 event / bounded_sync；重点验 E3：未 record 的 query 不得报完成）──
    probe("E1.create_event", lambda: type(torch.cuda.Event()).__name__)
    probe("E2.event_unrecorded_query", lambda: bool(torch.cuda.Event().query()))
    probe("E3.event_record_then_query", lambda: (
        lambda ev, st: (ev.record(st), bool(ev.query()))[1])(
            torch.cuda.Event(), torch.cuda.current_stream()))
    probe("E4.event_wait_timeout_signature",
          lambda: str(__import__("inspect").signature(torch.cuda.Event.wait)))
    probe("E5.has_event_synchronize", lambda: hasattr(torch.cuda.Event, "synchronize"))
    probe("E6.has_event_elapsed_time", lambda: hasattr(torch.cuda.Event, "elapsed_time"))

    # ── F. 跨流内存保护 / 图捕获 / 多设备 ────────────────────────────────
    probe("F1.has_tensor_record_stream",
          lambda: hasattr(torch.Tensor, "record_stream"))
    probe("F2.has_cuda_graph", lambda: hasattr(torch.cuda, "graph"))
    probe("F3.has_CUDAGraph", lambda: hasattr(torch.cuda, "CUDAGraph"))
    probe("F4.multidevice_all_names",
          lambda: [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())][:4])

    # ── G. 驱动库符号（决定能否走 C API 真原语：优先级/上下文只读）──────
    def _driver():
        """按 soname 逐个尝试（**不拿任意副本路径** —— 会引入第二份副本导致版本错配）。"""
        found = {}
        for soname in ("libcuda.so.1", "libcuda.so", "libpccl.so.2", "libalixpu.so"):
            try:
                found[soname] = ctypes.CDLL(soname)
            except OSError as e:
                found[soname] = f"({type(e).__name__})"
        return found

    libs = {}
    probe("G1.driver_libs_loaded",
          lambda: {k: ("OK" if not isinstance(v, str) else v)
                   for k, v in (libs.update(_driver()) or libs).items()})

    def _sym(libname, sym):
        lib = libs.get(libname)
        if not isinstance(lib, ctypes.CDLL) and lib is not None and not isinstance(lib, str):
            pass
        if lib is None or isinstance(lib, str):
            return f"LIB_UNAVAILABLE({lib})"
        return f"has={hasattr(lib, sym)}"

    for soname in ("libcuda.so.1", "libpccl.so.2"):
        for sym in ("cuCtxGetCurrent", "cuCtxGetDevice", "cuCtxGetFlags",
                    "cuCtxGetStreamPriorityRange", "cuStreamCreateWithPriority",
                    "cuStreamGetPriority", "cuStreamDestroy_v2", "cuCtxCreate_v2",
                    "cuCtxDestroy_v2", "cuCtxResetPersistingL2Cache", "cuDevicePrimaryCtxGetState"):
            probe(f"G2.{soname}.{sym}", lambda s=soname, y=sym: _sym(s, y))

    # ── H. 上下文只读观测（决定 context_query）──────────────────────────
    def _ctx_probe():
        lib = libs.get("libcuda.so.1")
        if lib is None or isinstance(lib, str):
            return "LIB_UNAVAILABLE"
        cur = ctypes.c_void_p(0)
        rc = lib.cuCtxGetCurrent(ctypes.byref(cur))
        dev = ctypes.c_int(-1)
        rc_d = lib.cuCtxGetDevice(ctypes.byref(dev))
        return {"rc": int(rc), "ctx_nonnull": bool(cur.value),
                "rc_get_device": int(rc_d), "ordinal": int(dev.value)}

    probe("H1.cuCtxGetCurrent_after_torch_init", _ctx_probe)

    # ── I. 集合通信（决定训练腿 DC_DIST_BT）────────────────────────────
    probe("I1.default_device_backend_map",
          lambda: dict(dist.Backend.default_device_backend_map))
    probe("I2.nccl_version", lambda: ".".join(map(str, torch.cuda.nccl.version()))
          if hasattr(torch.cuda, "nccl") else "NO torch.cuda.nccl")
    probe("I3.dist_backend_members",
          lambda: [m for m in dir(dist.Backend) if m.isupper() and not m.startswith("_")])
    probe("I4.nccl_env",
          lambda: {k: os.environ.get(k) for k in
                   ("NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "NCCL_DEBUG", "NCCL_IB_HCA")})

    # ── J. 平台与选卡变量（推理腿服务化 + 选卡纪律）─────────────────────
    probe("J1.vllm_current_platform", lambda: str(
        __import__("vllm.platforms", fromlist=["current_platform"]).current_platform))
    probe("J2.env_CUDA_VISIBLE_DEVICES", lambda: os.environ.get("CUDA_VISIBLE_DEVICES"))
    probe("J3.env_PPU_SDK", lambda: os.environ.get("PPU_SDK"))
    probe("J4.env_PPU_VERSION", lambda: os.environ.get("PPU_VERSION"))

    # ── 汇总 ────────────────────────────────────────────────────────────
    print("=" * 78)
    unavail = [k for k, v in RESULT if isinstance(v, str) and v.startswith("UNAVAILABLE")]
    print(f"探测完成：{len(RESULT)} 项，其中取不到 {len(unavail)} 项")
    if unavail:
        print("取不到的项：")
        for k in unavail:
            print(f"  · {k}")
    print("⚠️ 边界：本探针只读探测**接口形态**；是否真能用于业务，须由 conformance / 两条腿判定。")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except BaseException:                                       # noqa: BLE001
        print("!! 顶层异常（仍照常打印已有结果）", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
