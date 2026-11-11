"""T11 §10 第 3 条缺口的判别实验（纯 torch，进程内**完全不 import runtime**）。

目的：把「弱引用悬垂」与「大量对象被回收」这两个候选触发因素分开。
判据变量只有一个：**给谁挂弱引用 / 丢不丢强引用**。

变体（各独立进程）：
  no_wr_drop      N 条流，**不挂弱引用**，丢弃
  wr_hold         N 条流，挂弱引用，**保留强引用**
  wr_tensor       N 个张量，挂弱引用，丢弃
  wr_stream       N 条流，挂弱引用，丢弃            （已知崩）
剂量组：5 / 10（ow_stream）

⚠️ 全部**不得**在退出前 gc.collect()。
"""
import sys
import weakref

MODE = sys.argv[1]
N = int(sys.argv[2])

print(f"[disc] mode={MODE} N={N} runtime_imported={'runtime' in sys.modules}", flush=True)

import torch  # noqa: E402

keep_refs = []
self_refs = []

if MODE == "no_wr_drop":
    for _ in range(N):
        s = torch.cuda.Stream()
        del s
elif MODE == "wr_hold":
    for _ in range(N):
        s = torch.cuda.Stream()
        self_refs.append(weakref.ref(s))
        keep_refs.append(s)          # 强引用保留
elif MODE == "wr_tensor":
    for _ in range(N):
        t = torch.zeros(4, device="cuda")
        self_refs.append(weakref.ref(t))
        del t
elif MODE == "wr_stream":
    for _ in range(N):
        s = torch.cuda.Stream()
        self_refs.append(weakref.ref(s))
        del s
else:
    raise SystemExit(f"未知 mode {MODE}")

print(f"[disc] done; weakrefs={len(self_refs)} kept_strong={len(keep_refs)} "
      f"runtime_imported={'runtime' in sys.modules}", flush=True)
