"""自核验复现：`release_stream(默认路径的流)` 为何返回 True（契约要求 False）。

分两组，隔离「基类/后端交互」与「id 复用」两种可能：
  A 单独跑：create_stream() → release_stream()   （audit L1 的原始形态）
  B 先做一轮优先级创建+释放，再重复 A（制造「本层拥有的流」登记/回收历史）
"""
import os
import sys

sys.path.insert(0, "/workspace/runtime-team/dev/device-context/prototype")

import runtime  # noqa: E402

runtime.use("ppu")
b = runtime.current()
print(f"[env] backend={b.name} dev={b.device_count()}", flush=True)


def show(tag: str) -> None:
    st = runtime.create_stream()
    native = st._native_obj
    owns = b.owns_stream(native)
    rel = runtime.release_stream(st)
    print(f"[{tag}] native={type(native).__name__} owns_stream={owns} release_stream={rel!r} "
          f"| owned_reg={list(b._reg('_owned_streams').keys())} released_reg={list(b._reg('_released_streams').keys())}",
          flush=True)
    try:
        st.synchronize()
        print(f"[{tag}] 之后 synchronize OK", flush=True)
    except BaseException as e:                                   # noqa: BLE001
        print(f"[{tag}] 之后 synchronize 抛 {type(e).__name__}: {str(e)[:80]}", flush=True)


print("========== A 组：单独跑 ==========", flush=True)
show("A")

print("========== B 组：先做一轮优先级创建+释放 ==========", flush=True)
p = runtime.create_stream(priority=-1)
print(f"[B-pre] 优先级流 owns={b.owns_stream(p._native_obj)}", flush=True)
print(f"[B-pre] release={runtime.release_stream(p)!r}", flush=True)
print(f"[B-pre] owned_reg={list(b._reg('_owned_streams').keys())} "
      f"released_reg={list(b._reg('_released_streams').keys())}", flush=True)
show("B")

print("========== C 组：不释放优先级流（制造「未释放的在册条目」）后再跑 ==========", flush=True)
_keep = runtime.create_stream(priority=0)
print(f"[C-pre] 故意不释放，owned_reg={list(b._reg('_owned_streams').keys())}", flush=True)
show("C")
