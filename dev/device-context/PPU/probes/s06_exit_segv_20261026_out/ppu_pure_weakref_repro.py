"""T11 §10 自审缺口的补跑：**进程内完全不含本层**（不 import runtime、不 use()）的纯净复现。

T11 原文的保留意见：当时那个"最小复现"脚本**也 import runtime 并 use()**，
严格说只是「不调用本层的流 API」。这里给两个纯净变体：

  pure_many   : 30 条原生 torch.cuda.Stream()，各挂一个 weakref.ref，然后丢弃
  pure_one    : 同上但 N=1（对照：T11 记录 N=1 不崩）

每个变体独立进程，各跑 2 轮；打印首行以证明**没有 import runtime**。
"""
import gc
import sys
import weakref

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

print(f"[pure] runtime_imported={'runtime' in sys.modules} N={N}", flush=True)

import torch  # noqa: E402

print(f"[pure] after torch: runtime_imported={'runtime' in sys.modules}", flush=True)

refs = []
for _ in range(N):
    s = torch.cuda.Stream()
    refs.append(weakref.ref(s))
    del s

print(f"[pure] created {len(refs)} streams + weakrefs, {sum(1 for r in refs if r() is None)} dead",
      flush=True)
print(f"[pure] runtime_imported={'runtime' in sys.modules}", flush=True)
gc.collect()
print("[pure] gc done, 正常退出中（退出的最终化阶段才是观察点）", flush=True)
