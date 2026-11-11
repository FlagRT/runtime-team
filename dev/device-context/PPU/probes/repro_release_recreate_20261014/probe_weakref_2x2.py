#!/usr/bin/env python3
"""PPU：weakref 语义 2×2 判据实验（有无 gc × 是否跟随地址复用）。

已知（已复现）：
  · 循环建/释放后，`_released_streams` 里为**旧对象**创建的 weakref 会解析出**新对象**
    ⇒ 本层把"刚建好的流"误判为"已释放"（T6 D2 全线 FAIL 的直接原因）。
  · 单发（release → del → **gc.collect()** → 再建）时，同一 weakref 却**不**跟随。

本实验把「是否插入 gc.collect()」作为唯一变量，各测若干轮，用显式弱引用复核：
    w_old() is None ？（已置死）   w_old() is new_obj ？（跟随地址）
用法：DC_BACKEND=ppu CUDA_VISIBLE_DEVICES=1 python3 -u <此脚本>
"""
import gc
import os
import sys
import weakref

_ROOT = os.environ.get("DC_ROOT", "/workspace/runtime-team/dev/device-context/prototype")
sys.path.insert(0, _ROOT)

import runtime  # noqa: E402

runtime.use(os.environ.get("DC_BACKEND", "ppu"))
runtime.set_device(0)
PRIO = -3


def one_round(do_gc):
    s1 = runtime.create_stream(priority=PRIO)
    n1 = s1._native_obj
    w1 = weakref.ref(n1)
    id1 = id(n1)
    runtime.release_stream(s1)
    del s1, n1
    if do_gc:
        gc.collect()
    # 立刻再建一条
    s2 = runtime.create_stream(priority=PRIO)
    n2 = s2._native_obj
    id2 = id(n2)
    got = w1()
    res = {
        "id1": id1, "id2": id2, "same_id": id1 == id2, "same_handle": n2.cuda_stream == got.cuda_stream if got is not None else None,
        "dead": got is None, "followed": (got is n2),
    }
    try:
        s2.synchronize()
        res["use"] = "OK"
    except BaseException as e:                                # noqa: BLE001
        res["use"] = f"{type(e).__name__}"
    runtime.release_stream(s2)
    del s2, n2
    return res


for tag, do_gc in (("不 gc", False), ("gc.collect()", True)):
    print(f"=== {tag} ===")
    stats = {"dead": 0, "followed": 0, "use_fail": 0, "same_id": 0}
    for i in range(8):
        r = one_round(do_gc)
        stats["dead"] += r["dead"]
        stats["followed"] += r["followed"]
        stats["same_id"] += r["same_id"]
        stats["use_fail"] += (r["use"] != "OK")
        if i < 3:
            print(f"  round {i+1}: same_id={r['same_id']} w() is None={r['dead']} "
                  f"w() is new={r['followed']} 使用点={r['use']}")
    print(f"  ⇒ 8 轮统计：同 id={stats['same_id']} · 已置死={stats['dead']} · "
          f"跟随新对象={stats['followed']} · 使用失败={stats['use_fail']}")
    gc.collect()

print("判读：若「不 gc」时 followed>0 而「gc」时为 0 ⇒ 平台弱引用失效**与回收路径相关**")
