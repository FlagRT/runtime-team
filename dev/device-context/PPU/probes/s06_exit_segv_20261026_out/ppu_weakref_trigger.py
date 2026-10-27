"""把「退出期段错误」的触发条件收窄到**本层经厂商 C API 建/销上下文**这一步。

三变体（各独立进程），唯一变量是"上下文是否经本层 C API 建/销"：

  A torch_only   : 原生 torch 流 + weakref，丢弃              （已知 rc=0）
  B ctx_then_del : 本层 context_create → 在其下建流 → 挂 weakref → context_destroy → 丢弃流
  C ctx_keep     : 同 B，但只销毁上下文、不丢弃流的引用（对照）

所有变体都在"正常退出"下观察（退出期才是崩溃点）。
"""
import gc
import os
import sys
import weakref

sys.path.insert(0, os.getcwd())

MODE = sys.argv[1]
import runtime  # noqa: E402

runtime.use("ppu")
bk = runtime.current()

print(f"[probe] mode={MODE}", flush=True)

if MODE == "torch_only":
    import torch
    s = torch.cuda.Stream()
    wr = weakref.ref(s)
    del s
elif MODE in ("ctx_then_del", "ctx_keep"):
    ctx = bk.context_create(0)
    print(f"[probe] ctx={ctx}", flush=True)
    native = bk.create_stream()
    wr = weakref.ref(native)
    bk.context_destroy(ctx)
    print("[probe] ctx destroyed", flush=True)
    try:
        bk.set_device(0)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] set_device 失败：{e}", flush=True)
    if MODE == "ctx_then_del":
        del native

print(f"[probe] weakref alive={wr() is not None}", flush=True)
gc.collect()
print(f"[probe] gc done; runtime_imported={'runtime' in sys.modules}", flush=True)
