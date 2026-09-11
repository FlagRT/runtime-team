#!/usr/bin/env python3
"""probe_allocator_profile.py — torch_fl caching allocator 现状画像探针（只读，不改造）。

重建自 docs/allocator-画像报告-20260817.md 的测试矩阵（原探针未入库，本脚本按报告复刻）：
  1) 环境与开关：FLAGOS_USE_CACHING_ALLOCATOR 默认开启、device_count、暴露接口
  2) 缓存复用：10 次 1GiB alloc/free 循环 → 期望仅 1 次 device malloc
  3) 交错释放：alloc A/B/C(2GiB) → del B → alloc D(2GiB) → 期望 0 次新 malloc
  4) 尺寸抖动：20 次随机 64MiB~1GiB 半保留半释放 → 报告碎片冗余占比
  5) 大块切分：alloc 8GiB → free → 8×512MiB → 期望 0 次新 malloc
  6) OOM 重试：num_alloc_retries

自适应：先用 aclrtGetMemInfo 探测 HBM 空闲量，不足时按比例缩放块大小
（报告中的绝对值依赖当时空卡环境，语义判定不依赖块大小）。
用法：/root/vllm-venv312/bin/python probe_allocator_profile.py [device]
"""
import ctypes
import os
import random
import sys

import torch
import torch_fl  # noqa: F401  (必须先于 torch 使用? 见安装文档: 先 import torch_fl)

# ---------------------------------------------------------------------------
# HBM free 探测（ctypes 调 aclrtGetMemInfo，ACL_HBM_MEM=1）
# ---------------------------------------------------------------------------


def hbm_free_mb(device: int = 0) -> int:
    try:
        lib = ctypes.CDLL("libascendcl.so")
        lib.aclInit.restype = ctypes.c_int32
        lib.aclrtSetDevice.restype = ctypes.c_int32
        lib.aclrtGetMemInfo.restype = ctypes.c_int32
        lib.aclrtResetDevice.restype = ctypes.c_int32
        lib.aclFinalize.restype = ctypes.c_int32
        lib.aclInit(ctypes.c_void_p(0))
        rc = lib.aclrtSetDevice(ctypes.c_int32(device))
        if rc != 0:
            print(f"[warn] aclrtSetDevice rc={rc}")
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        rc = lib.aclrtGetMemInfo(ctypes.c_int32(1), ctypes.byref(free), ctypes.byref(total))
        lib.aclrtResetDevice(ctypes.c_int32(device))
        lib.aclFinalize()
        if rc != 0:
            print(f"[warn] aclrtGetMemInfo rc={rc}, 回退估计")
            return 8 * 1024  # 保守估计: sglang 占用下每卡约剩 8GiB
        return free.value // (1024 * 1024)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] aclrtGetMemInfo 不可用 ({e})，按 8GiB 估计")
        return 8 * 1024


def dev(s: str) -> str:
    return f"flagos:{DEV}"


def stats() -> dict:
    return torch_fl.flagos.memory_stats(DEV)


def dev_malloc_count() -> int:
    return stats()["num_device_malloc"]


def make(mib: int) -> torch.Tensor:
    """分配 mib MiB 的 flagos 张量（int32 元素）。"""
    return torch.empty(mib * 1024 * 1024 // 4, dtype=torch.int32, device=dev("flagos:0"))


def main() -> None:
    global DEV
    DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print("=" * 72)
    print("torch_fl caching allocator 画像探针（重建自 allocator-画像报告-20260817）")
    print(f"torch {torch.__version__} | device flagos:{DEV} | HBM free ≈ {hbm_free_mb(DEV)} MiB")
    env = os.environ.get("FLAGOS_USE_CACHING_ALLOCATOR")
    print(f"FLAGOS_USE_CACHING_ALLOCATOR = {env!r} → {'开启' if env != '0' else '关闭（passthrough）'}")
    print(f"device_count = {torch_fl.flagos.device_count()}")
    for name in ("empty_cache", "memory_stats", "memory_allocated", "memory_reserved",
                 "reset_peak_memory_stats"):
        print(f"  interface {name}: {'OK' if hasattr(torch_fl.flagos, name) else 'MISSING'}")
    print("=" * 72)

    # 自适应缩放：块大小基准 1GiB，若空闲 < 16GiB 则按比例缩小（>=1GiB 保底到 256MiB）
    free_mb = hbm_free_mb(DEV)
    scale = max(0.25, min(1.0, (free_mb - 1024) / (16 * 1024)))
    G = int(1024 * scale)  # 1GiB 基准的缩放后 MiB
    print(f"[scale] 块基准 = {G} MiB（HBM 空闲 {free_mb} MiB）\n")

    torch_fl.flagos.empty_cache()
    torch_fl.flagos.reset_peak_memory_stats(DEV)

    def snap() -> dict:
        return {"malloc": dev_malloc_count(), "stats": stats()}

    # ---- 1) 缓存复用：10 次同尺寸 alloc/free ----
    print("─ 1) 缓存复用：10 次同尺寸 alloc/free 循环")
    before = snap()
    for i in range(10):
        t = make(G)
        assert t.device.type == "flagos"
        del t
    s = stats()
    d = s["num_device_malloc"] - before["malloc"]
    print(f"    device_malloc 增量 = {d}  (期望 1: 首次后 9 次全命中池)")
    print(f"    reserved = {s['reserved_bytes'] / 2**30:.3f} GiB, allocated = {s['allocated_bytes'] / 2**30:.3f} GiB")
    ok1 = d <= 1
    print(f"    → {'✅ 复用生效' if ok1 else '❌ 未命中池'}")
    torch_fl.flagos.empty_cache()

    # ---- 2) 交错释放：A/B/C(2G) → del B → D(2G) ----
    # 报告口径: A/B/C 的 3 次 malloc 是预期内的; 判定 D 是否命中同尺寸池(0 次新 malloc)
    print("─ 2) 交错释放：alloc A/B/C(2G) → del B → alloc D(2G)")
    a, b, c = make(2 * G), make(2 * G), make(2 * G)
    before = snap()
    del b
    d = make(2 * G)
    del a, c, d
    d2 = dev_malloc_count() - before["malloc"]
    ok2 = d2 == 0
    print(f"    D 的 device_malloc 增量 = {d2}  (期望 0: 同尺寸池命中; A/B/C 3 次为预期)")
    print(f"    → {'✅ 同尺寸池命中' if ok2 else '❌ 未命中'}")
    torch_fl.flagos.empty_cache()

    # ---- 3) 尺寸抖动：20 次随机 64MiB~1GiB，半保留半释放 ----
    print("─ 3) 尺寸抖动：20 次随机 64MiB~1GiB，半保留半释放")
    before = snap()
    random.seed(20260817)
    kept: list[torch.Tensor] = []
    for i in range(20):
        mb = random.randint(64, 1024) * scale
        t = make(int(mb))
        if i % 2 == 0:
            kept.append(t)
        else:
            del t
    s = stats()
    frag = s["reserved_bytes"] - s["allocated_bytes"]
    ratio = frag / s["reserved_bytes"] * 100 if s["reserved_bytes"] else 0
    print(f"    reserved = {s['reserved_bytes'] / 2**30:.3f} GiB, allocated = {s['allocated_bytes'] / 2**30:.3f} GiB")
    print(f"    碎片冗余 = {frag / 2**30:.3f} GiB ({ratio:.1f}%)  (报告原值 0.54GiB / 4.9%)")
    ok3 = ratio < 15
    print(f"    → {'✅ 碎片控制良好(<15%)' if ok3 else '⚠️ 碎片偏高'}")
    del kept
    torch_fl.flagos.empty_cache()

    # ---- 4) 大块切分：alloc 8G → free → 8×512MiB ----
    # 报告口径: 8G 那次 malloc 是预期内的; 判定 8 个 512MiB 块是否切分自该块(0 次新 malloc)
    print("─ 4) 大块切分：alloc 8G → free → 8×512MiB")
    big = make(8 * G)
    del big
    before = snap()
    chunks = [make(G // 2) for _ in range(8)]
    del chunks
    d4 = dev_malloc_count() - before["malloc"]
    ok4 = d4 == 0
    print(f"    8×512MiB 的 device_malloc 增量 = {d4}  (期望 0: 切分自已释放的 8G 块)")
    print(f"    → {'✅ 切分复用生效' if ok4 else '❌ 未切分复用'}")
    torch_fl.flagos.empty_cache()

    # ---- 5) OOM 重试统计 ----
    s = stats()
    print(f"─ 5) OOM 重试：num_alloc_retries = {s['num_alloc_retries']}  (期望 0)")
    ok5 = s["num_alloc_retries"] == 0
    print(f"    → {'✅' if ok5 else '⚠️ 有重试'}")

    torch_fl.flagos.empty_cache()
    print("\n" + "=" * 72)
    all_ok = ok1 and ok2 and ok3 and ok4 and ok5
    print(f"总体判定: {'✅ 全部通过 —— 与 2026-08-17 画像一致（语义层可复现）' if all_ok else '❌ 存在不一致，见上'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
