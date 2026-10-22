#!/usr/bin/env python3
"""
t11_attrib_verify_20261014.py — T11「平台侧弱引用缺陷」**归属复核**（完全不含本层）
════════════════════════════════════════════════════════════════════════════════

【为什么要做这一次复核】（对既有结论的两处质疑，必须补实测）
  既有报告 `PPU/docs/PPU_T11_EXIT_SEGV_ROOTCAUSE_20261014.md` §5.3 把最小复现写成
  「**不含本层**」——但那个变体（`t11_exit_segv_matrix.py::vendor_weakref_many`）
  实际位于**统一引导段之后**：文件顶部有
      `import runtime` + `_bk = runtime.use(os.environ.get("DC_BACKEND", "ppu"))`
  ⇒ 「不含本层」严格说只是「**不调用本层的流 API**」，不是「进程里没有本层代码」。
  本脚本用 **纯 torch**（不 import 本层任何模块）重做同一形态，并**自证**这一点。

  另有一处**逻辑缺项**：既有矩阵只测了
      · N=1 + 弱引用 + 持有（`native_weakref_alive`）→ 干净
      · N=30 + 弱引用 + 丢弃（`vendor_weakref_many`）→ 崩
  但**没有**「N=30 + 无弱引用 + 丢弃」这个对照 ⇒ 无法区分
      「触发因素是**弱引用悬垂**」还是「触发因素是**大量流被丢弃**（与弱引用无关）」。
  本脚本把它补齐，并加两组判别：
      · N=30 + 弱引用 + **持有到退出**（对象不死 ⇒ 弱引用不可能悬垂）→ 若不崩，则坐实「必须先死」
      · N=30 + 弱引用 + 丢弃，但对象**不是 torch 流**（普通 `object()`）→ 若不崩，则坐实「与厂商流对象相关」

【判据（先写死）】
  读数 = **进程退出码**（`0` 干净 / `139` 解释器退出期 SIGSEGV）。每个变体独立进程；轮次交替。
  · `pure_baseline` 必须干净 —— 否则本窗口无效。
  · `pure_wref_30_dropped` 若为 139 ⇒ 既有「平台侧」归属**成立**（进程内无本层）。
  · `pure_nowref_30_dropped` 若**也**为 139 ⇒ 既有「弱引用是触发因素」的表述**不成立**，须更正为
    「大量流对象被回收」这一更宽的因素（但仍属平台侧）。

【自证】
  末尾打印 `runtime_loaded=[...]` —— 必须是空列表，否则本次读数不可用于「归属」判断。
  另有 `BRANCH=` / `EXIT_REACHED=` 两个标记：两者都在 ⇒ 崩溃确实发生在**判据之后、退出期**。

【用法】
  CUDA_VISIBLE_DEVICES=<卡> python3 -u t11_attrib_verify_20261014.py <变体名>
  变体：pure_baseline | pure_stream_30_held | pure_stream_30_dropped
        | pure_wref_30_held | pure_wref_30_dropped | pure_wref_10_dropped
        | pure_wref_5_dropped | pure_wref_30_plainobj_dropped
"""
from __future__ import annotations

import sys
import weakref

V = sys.argv[1] if len(sys.argv) > 1 else "help"
N = 30

print(f"BRANCH={V}", flush=True)

import torch  # noqa: E402

# ⚠️ 所有变体先做**同一段**最小真设备运算（非空转 + 统一"触碰设备"这一步）：
#    对照组之间只允许差「建流 / 挂弱引用 / 是否丢弃」这三件事。
_t = torch.zeros(1, device="cuda")
print(f"torch={torch.__version__} zero={float(_t.sum()):.1f}", flush=True)


def _work() -> float:
    a = torch.randn(64, 64, device="cuda")
    return float((a @ a).sum().item())


if V == "pure_baseline":
    # 只有一次真运算，不建流、不挂弱引用
    print(f"sum={_work():.1f}", flush=True)

elif V == "pure_stream_30_held":
    # 建 30 条原生流并**持有到退出**，不挂弱引用
    keep = [torch.cuda.Stream() for _ in range(N)]
    print(f"sum={_work():.1f} held={len(keep)}", flush=True)

elif V == "pure_stream_30_dropped":
    # ⭐ 缺项对照：建 30 条流、**每条建完即丢**、**不挂任何弱引用**
    for _ in range(N):
        s = torch.cuda.Stream()
        del s
    print(f"sum={_work():.1f} dropped={N}", flush=True)

elif V == "pure_wref_30_held":
    # ⭐ 判别：弱引用挂了 30 个，但对象**活到退出** ⇒ 弱引用不可能悬垂
    keep = [torch.cuda.Stream() for _ in range(N)]
    ws = [weakref.ref(s) for s in keep]
    print(f"sum={_work():.1f} alive={sum(1 for w in ws if w() is not None)}", flush=True)

elif V == "pure_wref_30_dropped":
    # ⭐ 与既有 `vendor_weakref_many` 同形，但**进程内没有本层**
    ws = []
    for _ in range(N):
        s = torch.cuda.Stream()
        ws.append(weakref.ref(s))
        del s
    print(f"sum={_work():.1f} alive={sum(1 for w in ws if w() is not None)}", flush=True)

elif V == "pure_wref_10_dropped":
    ws = []
    for _ in range(10):
        s = torch.cuda.Stream()
        ws.append(weakref.ref(s))
        del s
    print(f"sum={_work():.1f} alive={sum(1 for w in ws if w() is not None)}", flush=True)

elif V == "pure_wref_5_dropped":
    ws = []
    for _ in range(5):
        s = torch.cuda.Stream()
        ws.append(weakref.ref(s))
        del s
    print(f"sum={_work():.1f} alive={sum(1 for w in ws if w() is not None)}", flush=True)

elif V == "pure_wref_30_plainobj_dropped":
    # ⭐ 判别：弱引用 + 丢弃，但对象**不是**厂商流对象
    ws = []
    for _ in range(N):
        o = object()
        ws.append(weakref.ref(o))
        del o
    print(f"sum={_work():.1f} alive={sum(1 for w in ws if w() is not None)}", flush=True)

elif V == "pure_wref_30_tensor_dropped":
    # ⭐ 判别：弱引用 + 丢弃，对象是**设备张量**（厂商对象的另一类）
    ws = []
    for _ in range(N):
        o = torch.zeros(8, device="cuda")
        ws.append(weakref.ref(o))
        del o
    print(f"sum={_work():.1f} alive={sum(1 for w in ws if w() is not None)}", flush=True)

else:
    print("UNKNOWN_VARIANT", flush=True)
    sys.exit(2)

# ⭐ 自证：本进程**从未加载本层**（否则本次读数不能用于「归属」判断）
_rt = sorted(m for m in sys.modules if m == "runtime" or m.startswith("runtime."))
print(f"runtime_loaded={_rt}", flush=True)
print("EXIT_REACHED", flush=True)
