# 最小独立复验：flag_gems 在 910C 上对 silu / triu / lift_fresh(空张量) 的实际行为
# 目的：把「重放失败」与「kernel 本身失败」区分开 —— 这里不经过任何重放框架
import torch, torch_npu, json, traceback
torch.npu.set_device(0)
d = "npu:0"
import flag_gems
res = {}

def try_op(tag, fn):
    try:
        out = fn()
        torch.npu.synchronize()
        res[tag] = {"ok": True, "sample": float(out.flatten()[0]) if out.numel() else None}
        print("[OK]   %-34s %s" % (tag, res[tag]), flush=True)
    except Exception as exc:
        res[tag] = {"ok": False, "err": "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:120])}
        print("[FAIL] %-34s %s" % (tag, res[tag]["err"]), flush=True)

x = torch.randn(4, 8, device=d)

# 臂 1：vendor 原生
print("--- vendor 原生 ---", flush=True)
try_op("vendor.silu", lambda: torch.nn.functional.silu(x))
try_op("vendor.triu", lambda: torch.triu(torch.randn(8, 8, device=d)))
try_op("vendor.lift_fresh.empty", lambda: torch.randn(0, device=d).clone())

# 臂 2：启用 flag_gems
flag_gems.enable()
print("--- flag_gems 启用后（同一输入）---", flush=True)
try_op("gems.silu", lambda: torch.nn.functional.silu(x))
try_op("gems.triu", lambda: torch.triu(torch.randn(8, 8, device=d)))
try_op("gems.lift_fresh.empty", lambda: torch.randn(0, device=d).clone())
try_op("gems.matmul", lambda: torch.randn(64, 64, device=d) @ torch.randn(64, 64, device=d))
try_op("gems.add", lambda: x + x)

print(json.dumps(res, ensure_ascii=False, indent=1))
