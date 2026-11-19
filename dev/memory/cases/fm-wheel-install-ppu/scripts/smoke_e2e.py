#!/usr/bin/env python3
"""fm-wheel-install-ppu · C3 功能面：pip 安装形态下的异步搬运最小闭环（消费方视角）。

前提：flagmemory 已由 wheel pip 安装（本样例 run.sh 第 3 步），无源码仓、无 PYTHONPATH 挂载。
判据（对应 FM 仓 tests/tiering/test_ppu_backend_realcard.py 的真卡用例，最小裁剪）：
  ① PpuTierBackend 懒加载真 DmaCopyBackend；② D2H 异步换出+事件确认+数值逐字一致；
  ③ H2D 异步换入同上；④ 锁页能力真环境探测到一或二级。
输出：SMOKE PASS / SMOKE FAIL 判定行 + 退出码。
"""
import sys

import torch

from flagmemory.tiering.backends.ppu_backend import (
    PINNED_LEVEL_HOST_REGISTER,
    PINNED_LEVEL_PIN_MEMORY,
    PpuTierBackend,
)


def main() -> int:
    checks = []

    adapter = PpuTierBackend(backend_id="ppu")
    checks.append(("懒加载:bind 前不持有 connector", adapter._copy_backend is None))

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    n_blocks, block_bytes = 4, 4096
    gpu = torch.arange(n_blocks * block_bytes, dtype=torch.int8, device=dev).view(n_blocks, block_bytes)
    cpu = torch.zeros(n_blocks, block_bytes, dtype=torch.int8, pin_memory=True)
    adapter.bind(
        {"layer.0": gpu}, {"layer.0": cpu}, dev,
        torch.cuda.Stream(), torch.cuda.Stream(), block_bytes=block_bytes,
    )
    from flagmemory.tiering.offload.ppu.ppu_copy_backend import DmaCopyBackend
    checks.append(("bind 后为真 DmaCopyBackend", isinstance(adapter._copy_backend, DmaCopyBackend)))

    def wait(ticket):
        for _ in range(1000):
            if adapter.confirm(ticket):
                return True
            torch.cuda.synchronize(dev)
        return adapter.confirm(ticket)

    # ② D2H：异步换出 → 事件确认 → 主机侧逐字核对
    ticket = adapter.copy_out("e1", 2, block_bytes, stream_id=0)
    adapter.register_entry("e1", 2)
    ok = wait(ticket)
    checks.append(("D2H 事件确认", ok is True))
    checks.append(("D2H 数值逐字一致", bool(torch.equal(cpu[2], gpu[2].cpu()))))

    # ③ H2D：数据写在条目记录源行（cpu[2]，e1→行2）→ 异步换行换入 gpu[3] → 取回核对。
    # （2026-11-19 修正：原写法把数据写进 cpu[3]，按契约实际拷的是源行 cpu[2]——
    #   与 FM 仓 test_ppu_backend_realcard 同源错误，修复分支 08af9f3 同批定论）
    cpu[2] = 42
    ticket = adapter.copy_in("e1", 3, block_bytes, stream_id=0)
    ok = wait(ticket)
    checks.append(("H2D 事件确认", ok is True))
    checks.append(("H2D 数值一致", bool(torch.equal(gpu[3].cpu(), cpu[2]))))

    adapter.drop_entry("e1")
    adapter.shutdown()

    probed = adapter.probe_capabilities()
    checks.append(("锁页能力探测为真锁页",
                   bool(probed.supports_pinned) and adapter.probed_pinned_level in (
                       PINNED_LEVEL_HOST_REGISTER, PINNED_LEVEL_PIN_MEMORY)))

    n_pass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"SMOKE {'PASS' if n_pass == len(checks) else 'FAIL'} ({n_pass}/{len(checks)})")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
