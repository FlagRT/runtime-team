#!/usr/bin/env python3
"""P9: net.cc send/recv 完成判定 streamQuery -> eventSynchronize.

修复 1/10 偶发数据错（集合级稳定性循环）：
- 根因：rank1(910C, CANN) send 侧 `aclrtStreamQuery` 返回 COMPLETE 早于 D2H
  数据对 CPU 可见（aarch64 DMA/cache 窗口），isend 偶发发出旧 buffer（实锤：
  失败轮 P4-SEND-DATA 显示 isend 前 buffer 为 AG 旧数据残留 11，PASS 轮为 2.0）。
- 修复：send/recv 两侧 `streamQuery(cpStream)` -> `eventSynchronize(cpEvents[step])`，
  阻塞等到 D2H/H2D 事件真正完成（= 数据落位），不依赖 stream 状态查询语义。
- 验证：两轮 10/10 + 一轮 10/10（recv 侧加入后）全过，0 死锁；50 步训练无回归。

用法：python3 patch_p9_eventsync_completion.py <net.cc 路径>
（910C: /workspace/FlagCX/flagcx/core/net.cc（RAID 挂载）
 4090-1: /home/data/hongbinliu/FlagCX/flagcx/core/net.cc）
"""
import sys

SEND_OLD = """        // Kistich(fix-stale-data): stream-based completion. Per-chunk
        // eventQuery races with event state reuse across collectives and
        // could fire isend on a buffer whose D2H copy had not executed.
        if (deviceAdaptor->streamQuery(resources->cpStream) ==
            flagcxSuccess) {
          args->copied++;
          done = 1;
        }"""
SEND_NEW = """        // Kistich(fix-stale-data): per-chunk eventSynchronize. streamQuery on
        // CANN can report COMPLETE before the D2H data is CPU-visible
        // (aarch64 DMA/cache window), so isend would transmit a stale buffer.
        // Blocking on the chunk event guarantees the copy has executed.
        if (deviceAdaptor->eventSynchronize(resources->cpEvents[step]) ==
            flagcxSuccess) {
          args->copied++;
          done = 1;
        }"""

RECV_OLD = """        if (deviceAdaptor->streamQuery(resources->cpStream) ==
            flagcxSuccess) {
          args->copied++;
        }"""
RECV_NEW = """        if (deviceAdaptor->eventSynchronize(resources->cpEvents[step]) ==
            flagcxSuccess) {
          args->copied++;
        }"""


def main():
    if len(sys.argv) != 2:
        print("usage: python3 patch_p9_eventsync_completion.py <net.cc>")
        sys.exit(1)
    path = sys.argv[1]
    s = open(path).read()
    n_send = s.count(SEND_OLD)
    n_recv = s.count(RECV_OLD)
    # recv 侧可能已被 send 版 patch 误判，用不含注释的代码锚点
    recv_anchor = """        if (deviceAdaptor->streamQuery(resources->cpStream) ==
            flagcxSuccess) {
          args->copied++;
        }"""
    n_recv = s.count(recv_anchor)
    if n_send == 1:
        s = s.replace(SEND_OLD, SEND_NEW, 1)
        print("[ok] send 侧已替换 (streamQuery -> eventSynchronize)")
    elif "eventSynchronize(resources->cpEvents[step])" in s and n_send == 0:
        print("[skip] send 侧已是 eventSynchronize")
    else:
        print(f"[warn] send 侧模式匹配异常: n_send={n_send}")
    if n_recv == 1:
        s = s.replace(recv_anchor, RECV_NEW, 1)
        print("[ok] recv 侧已替换 (streamQuery -> eventSynchronize)")
    elif "eventSynchronize(resources->cpEvents[step])" in s and n_recv == 0:
        print("[skip] recv 侧已是 eventSynchronize")
    else:
        print(f"[warn] recv 侧模式匹配异常: n_recv={n_recv}")
    open(path, "w").write(s)
    left = s.count("streamQuery(resources->cpStream)")
    print(f"[done] 剩余 streamQuery(resources->cpStream) 次数: {left} (应为 0)")


if __name__ == "__main__":
    main()
