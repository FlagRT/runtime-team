#!/usr/bin/env python3
"""
统一运行时 API 演示（demos/demo_unified.py）

展示核心主张：**同一份用户代码，切换芯片只改 use() 一行**。

    python3 demo_unified.py --backend ascend
    python3 demo_unified.py --backend kunlun     # 9 月为 stub，会报告未实现能力

覆盖能力（对应本方向职责子层）：
  1. 设备上下文：设备枚举 / 绑定 / 显存查询
  2. 多流 Stream：流创建 / 跨流依赖 / 流上下文切换 / 有界同步
  3. 错误码翻译：厂商错误码 → L1-L4 统一分级 + 处置策略
  4. 状态恢复：设备探活 + 重建接口（real 模式仅在真实需要时调用）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime
from runtime.api.errors import ErrorCategory


def sep(title):
    print(f"\n{'─' * 56}\n{title}\n{'─' * 56}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ascend", help="后端名（ascend / kunlun）")
    ap.add_argument("--ordinal", type=int, default=0)
    args = ap.parse_args()

    print("=" * 56)
    print("统一运行时 API 演示 · 设备上下文与多流 Stream")
    print("=" * 56)

    # ── 0. 选择后端（唯一与芯片相关的代码）──
    sep(f"[0] 选择后端：runtime.use(\"{args.backend}\")")
    try:
        backend = runtime.use(args.backend)
    except runtime.BackendNotFound as e:
        print(f"  ✗ {e}")
        print("  提示：kunlun 为 9 月 stub，可能尚未注册")
        return 1
    print(f"  ✓ 已加载：{backend.info()}")

    # ── 1. 设备上下文 ──
    sep("[1] 设备上下文（设备句柄 + 内存）")
    n = runtime.device_count()
    print(f"  设备数量：{n}")
    if n == 0:
        print("  ✗ 无可用设备")
        return 1
    runtime.set_device(args.ordinal)
    print(f"  已绑定设备：{args.ordinal}")
    mem = runtime.memory_stats(args.ordinal)
    print(f"  显存：total={mem['total_mb']}MB  used={mem['used_mb']}MB  free={mem['free_mb']}MB")

    # ── 2. 多流 Stream ──
    sep("[2] 多流 Stream（创建 / 跨流依赖 / 上下文切换）")
    s1 = runtime.create_stream()
    s2 = runtime.create_stream()
    ev = runtime.create_event()
    print(f"  {s1!r}")
    print(f"  {s2!r}")
    print(f"  {ev!r}")

    torch = getattr(backend, "torch", None)
    if torch is not None:
        dev = f"{backend.device_type}:{args.ordinal}"
        # 流 s1 上计算并记录事件
        with s1.context():
            x = torch.ones(64, 64, device=dev)
            y = (x * 3) + 2            # 期望 5
        ev.record(s1)
        # 流 s2 等待事件后读取（跨流可见性）
        s2.wait_event(ev)
        with s2.context():
            z = y.mean()
        s2.synchronize()
        print(f"  跨流可见性：流 s1 计算 → 流 s2 读取 = {z.cpu().item():.4f}（期望 5.0）")

        # 有界同步（超时路径）
        try:
            s1.synchronize(timeout_ms=5000)
            print("  有界同步（5s）：✓ 完成")
        except TimeoutError as e:
            print(f"  有界同步超时：{e}")

        # 主机侧事件有界等待
        print(f"  主机侧事件等待：{ev.wait_host(timeout_ms=3000)}")
    else:
        print("  （无 torch，跳过张量级演示）")

    # ── 3. 错误码翻译 ──
    sep("[3] 错误码翻译（厂商错误码 → 统一分级 + 处置）")
    samples = [
        ("ACL stream sync timeout, error code is 507046", "流同步超时"),
        ("device reset failed, error code is 507015", "设备级致命"),
        ("VLLMValidationError: prompt too long", "框架参数校验"),
    ]
    for msg, desc in samples:
        fe = runtime.translate_error(RuntimeError(msg), location="demo:op")
        print(f"  {desc:12s} → {fe.category.value:14s} "
              f"处置={fe.disposition:14s} mapped={fe.mapped} by={fe.graded_by}")

    # ── 4. 状态恢复 ──
    sep("[4] 状态恢复（探活 + 重建接口）")
    ok = runtime.probe_device(args.ordinal)
    print(f"  设备探活：{ok}")
    if hasattr(backend, "device_state"):
        try:
            st = backend.device_state(args.ordinal)
            print(f"  设备状态：{st}")
        except Exception as e:
            print(f"  设备状态查询：{type(e).__name__}: {e}")
    print("  重建接口：recover_device(mode='probe'/'real'/'hybrid')")
    print("  注意：real 模式会重置当前进程默认上下文，演示中不实际调用")

    sep("演示结束")
    print("  同一份代码切换芯片只需改：runtime.use(\"<backend>\")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
