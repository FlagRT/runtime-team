#!/usr/bin/env python3
"""
t11_exit_segv_matrix.py — 平头哥 PPU「解释器退出期段错误」最小对照矩阵（执行单 T11）
═══════════════════════════════════════════════════════════════════════════════

【要回答的问题】
  现象（10-10 首次发现，10-14 复现面扩大）：**判据通过、JSON 已落盘之后**，进程在
  解释器退出期 SIGSEGV（`rc=139`）。已排除的对象见
  `PPU/docs/PPU_MODEL_AND_TWO_LEGS_20261010.md` §4.2（E1–E16 共 16 组）。
  本轮把面收窄到**「经本层统一面建流」这一件事**，用最小变体逐项二分。

【为什么要重做（本轮的新事实，不是重复劳动）】
  逐文件核对「建流方式」后发现一个**天然对照**：
    · `probes/probe_stream_semantics_full.py`（T5）**只用原生** `torch.cuda.Stream()`（`dev.Stream()`）
      ⇒ `rc=0`（干净退出），而它建的流**数量更多、还有两流/双流并发**；
    · `probes/probe_stream_priority_sched_effect.py`（T6）经 `runtime.create_stream()` 建流
      ⇒ `rc=139`；
    · `runtime/proto/proto_train_leg.py` / `proto_infer_leg.py` 同样经统一面建流 ⇒ `rc=139`。
  更关键：**训练腿那条流只被 `repr` 过一次（`f"...统一流 {st!r}"`），全程没有任何流操作**
  ⇒ 触发条件与「怎么用流」无关，只与「**建了这条统一流**」有关。
  ⇒ 于是可以把它做成**几行的最小复现**，用退出码当判据逐项二分。

【判据（先写死，不许事后解释）】
  本探针的读数 = **进程退出码**（`0` 干净 / `139` 解释器退出期 SIGSEGV）。
  ① `native_plain`（原生 `torch.cuda.Stream()`）**必须干净** —— 否则本矩阵的前提不成立，
     整场读数作废（说明本窗口存在与「本层」无关的第三因素）。
  ② `ours_wrapper`（经 `runtime.create_stream()`）**必须复现 139** —— 这是同期阳性对照；
     若它不崩，则本窗口**测不出该现象**，结论只能是「本轮未复现」而**不得**写成「已修复」。
  ③ 其余变体各去掉/加上一个因素 ⇒ 与 ①② 比对，指向**触发因素**。
  ⚠️ 非空转：每个变体都打印 `BRANCH=` 与 `EXIT_REACHED=`，二者都在 ⇒ 段错误确实发生在
     **判据之后、退出期**，不是运行期崩溃被误读。
  ⚠️ 每个变体**一个独立进程**（进程级状态会被段错误破坏 ⇒ 不得同进程连跑）。
  ⚠️ 轮次必须**交替**（A/B/A/B…，由驱动脚本保证）：分组跑分不清「变量效应」与「时间漂移」。

【环境变量 / 用法】
  DC_ROOT=<原型根> DC_BACKEND=ppu CUDA_VISIBLE_DEVICES=<卡>
  python3 t11_exit_segv_matrix.py <变体名>
"""
from __future__ import annotations

import gc
import os
import sys
import weakref

V = sys.argv[1] if len(sys.argv) > 1 else "help"

ROOT = os.environ.get("DC_ROOT", "")
if ROOT and ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"BRANCH={V}", flush=True)

# ⚠️ **所有变体统一先做这一段引导**（含 `native_*`）——
#    对照组要只差「建流那一步」，而不是差「有没有 import 本层」。
#    `runtime.current()` **不读环境变量**；不显式 `use()` 时，`discover()` 按
#    `_KNOWN_BACKENDS` 顺序注册，**第一个注册的会占住 current 槽**（PPU 机器上是 ascend，
#    其惰性加载要到首次用 npu 才炸）⇒ 实测会 `ModuleNotFoundError: torch_npu`。
#    这就是本矩阵首跑 rc=1 的原因，已按各探针的既有写法对齐（`bk = runtime.use(a.backend)`）。
import torch  # noqa: E402
import runtime  # noqa: E402

_bk = runtime.use(os.environ.get("DC_BACKEND", "ppu"))
print(f"backend={_bk.name}", flush=True)


def _work(torch):
    """一段**真设备运算**（非空转：必须真的把张量算出来）。"""
    a = torch.randn(64, 64, device="cuda")
    return float((a @ a).sum().item())


# ───────────────────────── 前提组：原生路径（应干净）─────────────────────────
if V == "native_plain":
    st = torch.cuda.Stream()
    print(f"sum={_work(torch):.1f} repr={st!r}", flush=True)

elif V == "native_weakref_alive":
    # 去掉「本层」、只保留「对原生流持一个 weakref 且活到退出」
    st = torch.cuda.Stream()
    print(f"sum={_work(torch):.1f}", flush=True)
    _wr = weakref.ref(st)
    print(f"weakref_alive={_wr() is st}", flush=True)

elif V == "native_sync":
    st = torch.cuda.Stream()
    torch.cuda.synchronize()
    print(f"sum={_work(torch):.1f} synced=1", flush=True)

# ───────────── 阳性对照组：本层统一面（预期 rc=139）─────────────
elif V == "ours_wrapper":
    st = runtime.create_stream()
    print(f"sum={_work(torch):.1f} repr={st!r}", flush=True)

elif V == "ours_wrapper_del":
    # 退出前主动丢弃包装对象（10-10 E15 的形态：`del` + `gc.collect()`）
    st = runtime.create_stream()
    print(f"sum={_work(torch):.1f}", flush=True)
    del st
    print(f"gc_collected={gc.collect()}", flush=True)

elif V == "ours_wrapper_os_exit":
    # 跳过解释器退出期 ⇒ 用于**归属**判断：若 139 消失，则崩溃在退出期而非运行期
    st = runtime.create_stream()
    print(f"sum={_work(torch):.1f}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

# ───────────── 二分：把「统一面」拆成它的几个可分离动作 ─────────────
elif V == "ours_note_off":
    # 只关掉 `note_stream_created`（`_stream_ctx` 里那次**弱引用**登记）
    from runtime.backends.ppu.backend import PpuBackend
    PpuBackend.note_stream_created = lambda self, s: None
    st = runtime.create_stream()
    print(f"sum={_work(torch):.1f} patched=1", flush=True)

elif V == "ours_registry_only":
    # 只登记、不持有统一包装对象（`runtime.create_stream()` 的等价动作之一）
    b = runtime.current()
    n = b.create_stream(None)
    b.note_stream_created(n)
    print(f"sum={_work(torch):.1f}", flush=True)

elif V == "ours_wrapper_only":
    # 只持有统一包装对象、不做登记（另一等价动作）
    from runtime.api.stream import Stream
    b = runtime.current()
    st = Stream(b, b.create_stream(None))
    print(f"sum={_work(torch):.1f} repr={st!r}", flush=True)

elif V == "ours_driver_loaded":
    # 诊断「驱动库句柄（ctypes.CDLL→PyCapsule→dlclose）在退出期被卸载」这一假说：
    # **原生**建流 + 本层加载驱动库 ⇒ 与 `native_plain` 只差「驱动库被本层持有」
    lib = runtime.current()._driver_handle()
    st = torch.cuda.Stream()
    print(f"lib={type(lib).__name__} sum={_work(torch):.1f}", flush=True)

elif V == "driver_only":
    # 只加载驱动库、**不建任何流**（`_driver_handle()` 的副作用单独看）
    lib = runtime.current()._driver_handle()
    print(f"lib={type(lib).__name__} no_stream=1", flush=True)

elif V == "ours_prio_once":
    # 优先级路径的最小形态：本层 C API 建流（ExternalStream）+ 归属登记 + **显式释放**
    st = runtime.create_stream(priority=-1)
    print(f"sum={_work(torch):.1f} prio={runtime.stream_priority_readback(st)}", flush=True)
    print(f"released={runtime.release_stream(st)}", flush=True)

elif V == "ours_prio_once_norelease":
    # 同上但**不释放**（区分「释放动作」与「建流动作」）
    st = runtime.create_stream(priority=-1)
    print(f"sum={_work(torch):.1f} prio={runtime.stream_priority_readback(st)}", flush=True)

elif V == "ours_default_many":
    # 多条默认流（放大「持有本层包装对象」这一面，仍不走驱动库）
    sts = [runtime.create_stream() for _ in range(8)]
    print(f"sum={_work(torch):.1f} n={len(sts)}", flush=True)

elif V == "ours_prio_many":
    # 多条优先级流：建 + 释放各 8 次（T6 探针的节奏特征）
    for i in range(8):
        s = runtime.create_stream(priority=-1 if i % 2 == 0 else 0)
        runtime.release_stream(s)
    print(f"sum={_work(torch):.1f} rounds=8", flush=True)

# ───────── 依据原生现场（Py_FinalizeEx → _PyWeakref_ClearRef）设计的判定组 ─────────
elif V == "vendor_weakref_min":
    # ⭐ **不含本层**的最小复现：原生流 + 一个弱引用 + 丢掉流
    #    原生现场（10-14，t4_native*/t6_native）显示崩溃在
    #    `Py_Exit → Py_FinalizeEx → … → _PyWeakref_ClearRef`、`si_addr=0xab`
    #    ⇒ 若本变体也崩，则**触发条件里不需要本层**（本层只是把它暴露出来）。
    st = torch.cuda.Stream()
    print(f"sum={_work(torch):.1f}", flush=True)
    _wr = weakref.ref(st)
    del st
    print(f"dropped gc={gc.collect()} weakref_alive={_wr() is not None}", flush=True)

elif V == "vendor_weakref_gc_del":
    # 同上，但连弱引用对象一起丢掉（对照：弱引用没了还会不会崩）
    st = torch.cuda.Stream()
    print(f"sum={_work(torch):.1f}", flush=True)
    _wr = weakref.ref(st)
    del st
    gc.collect()
    del _wr
    print(f"dropped both gc={gc.collect()}", flush=True)

elif V == "vendor_stream_dropped":
    # 只丢流、不建弱引用 ⇒ 与上一变体的唯一差别就是"有没有弱引用"
    st = torch.cuda.Stream()
    print(f"sum={_work(torch):.1f}", flush=True)
    del st
    print(f"dropped gc={gc.collect()}", flush=True)

elif V == "ours_default_dropped":
    # 本层默认路径：建 30 条、**每条建完即丢**（对象被回收 ⇒ 登记表里的弱引用悬垂）
    for _ in range(30):
        runtime.create_stream()
    print(f"sum={_work(torch):.1f} n=30", flush=True)

elif V == "ours_prio_released_many":
    # 本层优先级路径：建 + 释放 10 轮（`_owned_streams`/`_released_streams` 都会挂弱引用）
    for _ in range(10):
        s = runtime.create_stream(priority=-1)
        runtime.release_stream(s)
    print(f"sum={_work(torch):.1f} rounds=10", flush=True)

elif V == "vendor_weakref_many":
    # ⭐ **不含本层**的"同形"复现：30 条原生流，每条建完就挂一个弱引用，然后丢掉。
    #    与 `ours_default_dropped` 的差别**只有"建流是否经本层"** ⇒ 用来判「归谁」。
    _ws = []
    for _ in range(30):
        s = torch.cuda.Stream()
        _ws.append(weakref.ref(s))
        del s
    print(f"sum={_work(torch):.1f} n=30 weakrefs_alive={sum(1 for w in _ws if w() is not None)}", flush=True)

elif V == "vendor_weakref_many_gc":
    # 同上，但每轮都 `gc.collect()`（排除"需要 GC 干预才复现"）
    _ws = []
    for _ in range(30):
        s = torch.cuda.Stream()
        _ws.append(weakref.ref(s))
        del s
        gc.collect()
    print(f"sum={_work(torch):.1f} n=30", flush=True)

else:
    print("UNKNOWN_VARIANT", flush=True)
    sys.exit(2)

print("EXIT_REACHED", flush=True)
