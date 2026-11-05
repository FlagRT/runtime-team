import sys, json, torch, torch_mlu
print("=== L2 GEMS (device-bearing, import + enable) ===", flush=True)
torch.mlu.set_device(0)
print("device_count", torch.mlu.device_count(), flush=True)
OPS = ["aten::mm", "aten::add.Tensor", "aten::silu", "aten::matmul"]
def priv(op):
    try:
        t = torch._C._dispatch_dump_table(op)
    except Exception as e:
        return ["dump_failed:%s" % type(e).__name__]
    return [l.strip() for l in t.splitlines() if "PrivateUse1" in l][:1]
before = {op: priv(op) for op in OPS}
try:
    import triton
    print("triton", triton.__version__, flush=True)
    import flag_gems
    print("flag_gems_imported", getattr(flag_gems, "__version__", "no __version__"), flush=True)
except Exception as e:
    print("GEMS_IMPORT_FAIL %s: %s" % (type(e).__name__, str(e).splitlines()[0][:200]), flush=True)
    print("GEMS_RC_FAIL", flush=True)
    sys.exit(0)
try:
    flag_gems.enable()
    print("ENABLE_OK", flush=True)
except Exception as e:
    print("ENABLE_FAIL %s: %s" % (type(e).__name__, str(e).splitlines()[0][:200]), flush=True)
after = {op: priv(op) for op in OPS}
for op in OPS:
    print("  REG_BEFORE %-18s %s" % (op, before[op]), flush=True)
    print("  REG_AFTER  %-18s %s" % (op, after[op]), flush=True)
    print("  REG_CHANGED %-18s %s" % (op, before[op] != after[op]), flush=True)

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
try:
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
    print("GEMS_RC_OK", flush=True)
except Exception as e:
    print("GEMS_OPS_FAIL %s: %s" % (type(e).__name__, str(e).splitlines()[0][:200]), flush=True)
    import traceback
    print("\n".join(traceback.format_exc().splitlines()[-8:]), flush=True)
    print("GEMS_RC_FAIL", flush=True)
