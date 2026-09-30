#!/usr/bin/env python3
"""910C：流优先级与配额上限的「真正的入口在哪」核查（只读 + 少量建流）。

针对上一轮的两条 B2 结论，分别找**本栈上正确的 API 路径**：
  Q1 优先级：`torch.npu.Stream(priority=…)` 到底有没有把 priority 传给 ACL？
     pyACL 侧的正确入口是不是 `create_stream_with_config`（argb 里有 priority 字段）？
  Q2 配额：pyACL 有 `get_stream_available_num()`（设备级"可用流数"）与
     `get_stream_res_limit()/set_stream_res_limit()` ⇒ 直接用它们**问设备**上限是多少，
     而不是靠"建到失败"。
"""
import inspect

import acl  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402

ns = torch.npu


def sec(t):
    print("\n" + "=" * 6, t, "=" * 6)


sec("Q1-a 完整 docstring（看 priority 是否被文档化）")
doc = ns.Stream.__doc__ or ""
print(doc[:1500])
print("...")
print("含 'priority' 字样:", "priority" in doc.lower())

sec("Q1-b pyACL 的流创建入口与属性常量")
print("create_stream 签名           :", end=" ")
try:
    print(inspect.signature(acl.rt.create_stream))
except Exception as e:  # noqa: BLE001
    print("取不到", type(e).__name__)
for fn in ("create_stream_with_config", "get_stream_attribute", "set_stream_attribute",
           "get_stream_available_num", "get_stream_res_limit", "set_stream_res_limit",
           "reset_stream_res_limit"):
    f = getattr(acl.rt, fn, None)
    d = (getattr(f, "__doc__", "") or "").strip().replace("\n", " ")[:220]
    print(f"  {fn:28s} doc: {d}")
consts = sorted(n for n in dir(acl) if "STREAM" in n.upper() and ("PRIOR" in n.upper() or "ATTR" in n.upper()))
print("  acl 里 STREAM/PRIOR/ATTR 常量:", consts[:30])

sec("Q2-a 直接问设备：可用流数 / 流资源上限")
for fn in ("get_stream_available_num", "get_stream_res_limit"):
    f = getattr(acl.rt, fn, None)
    if f is None:
        print(f"  {fn}: 本版 pyACL 无此入口")
        continue
    try:
        print(f"  {fn}() ->", repr(f()))
    except Exception as e:  # noqa: BLE001
        print(f"  {fn}() 失败:", type(e).__name__, str(e)[:160])

sec("Q2-b 建流前后 available_num 的变化（看它是否真的在计数）")
try:
    before = acl.rt.get_stream_available_num()
    S = [ns.Stream() for _ in range(64)]
    after64 = acl.rt.get_stream_available_num()
    S += [ns.Stream() for _ in range(320)]
    after384 = acl.rt.get_stream_available_num()
    print(f"  建流前={before}  建 64 后={after64}  建 384 后={after384}")
    S.clear()
    torch.npu.synchronize()
    print("  释放后=", acl.rt.get_stream_available_num())
except Exception as e:  # noqa: BLE001
    print("  失败:", type(e).__name__, str(e)[:200])

sec("Q1-c torch.npu.Stream(priority=…) 是否真的把 priority 传下去")
print("  先看 C 层是否接受该 kwargs（不报错 ≠ 生效）：")
for prio in (0, 7, -1):
    try:
        s = ns.Stream(priority=prio)
        print(f"    Stream(priority={prio:>2}) OK  stream_id={s.stream_id} npu_stream={getattr(s,'npu_stream',None)}")
    except Exception as e:  # noqa: BLE001
        print(f"    Stream(priority={prio:>2}) 报错: {type(e).__name__}: {str(e)[:120]}")
print("  ⚠️ 关键判断：若三次的 stream_id/句柄都不同 ⇒ 只是各建了一条流，**没有证据**说明 priority 进了设备；")
print("     需对比 device 侧调度行为（B2 已做：双向对照 ⇒ 提交顺序主导 ⇒ 未观测到效果）。")

sec("Q1-d 用 pyACL 原生路径建流（带 priority config）看能否成立")
try:
    print("  acl.rt.create_stream 返回:", repr(acl.rt.create_stream()))
except Exception as e:  # noqa: BLE001
    print("  create_stream 失败:", type(e).__name__, str(e)[:160])
try:
    sig = inspect.signature(acl.rt.create_stream_with_config)
    print("  create_stream_with_config 签名:", sig)
except Exception as e:  # noqa: BLE001
    print("  取签名失败:", type(e).__name__)
try:
    st = acl.rt.create_stream_with_config({"priority": 7})
    print("  create_stream_with_config({priority:7}) ->", repr(st))
except Exception as e:  # noqa: BLE001
    print("  create_stream_with_config(dict) 失败:", type(e).__name__, str(e)[:200])

print("\nBOUNDARY: 只读 + 稀疏建流；结论只在本栈（torch_npu 2.11.0 / CANN 9.0.0）成立。")
