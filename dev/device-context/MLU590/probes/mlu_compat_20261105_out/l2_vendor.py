import sys, json, torch, torch_mlu
print("=== L2 VENDOR (device-bearing, flag_gems NOT imported) ===", flush=True)
print("torch", torch.__version__, "torch_mlu", torch_mlu.__version__, flush=True)
print("device_count", torch.mlu.device_count(), flush=True)
print("flag_gems_in_modules", "flag_gems" in sys.modules, flush=True)
OPS = ["aten::mm", "aten::add.Tensor", "aten::silu", "aten::matmul"]
def priv(op):
    try:
        t = torch._C._dispatch_dump_table(op)
    except Exception as e:
        return ["dump_failed:%s" % type(e).__name__]
    return [l.strip() for l in t.splitlines() if "PrivateUse1" in l][:1]
for op in OPS:
    print("  REG %-18s %s" % (op, priv(op)), flush=True)

torch.mlu.set_device(0)
torch.manual_seed(20261105)
a = torch.randn(64, 64)
b = torch.randn(64, 64)
am, bm = a.to("mlu:0"), b.to("mlu:0")
x = torch.randn(32, 64).to("mlu:0")
res = {}
def snap(tag, t):
    c = t.detach().cpu()
    res[tag] = {
        "head": [round(float(v), 6) for v in c.flatten()[:8].tolist()],
        "sum": round(float(c.sum()), 6),
        "nan": int(torch.isnan(c).sum()),
        "inf": int(torch.isinf(c).sum()),
    }
snap("gemm64", am @ bm)
snap("matmul", x @ am)
snap("add", x + x)
snap("mul", x * x)
snap("silu", torch.nn.functional.silu(x))
snap("relu", torch.relu(x))
snap("softmax", torch.softmax(x, dim=-1))
snap("layernorm", torch.nn.functional.layer_norm(x, (64,)))
snap("triu", torch.triu(x))
torch.mlu.synchronize()
print("NUMJSON:" + json.dumps(res, sort_keys=True), flush=True)
print("VENDOR_RC_OK", flush=True)
