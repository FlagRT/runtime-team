import sys, torch, torch_mlu
OPS = ["aten::mm", "aten::matmul", "aten::add.Tensor", "aten::mul.Tensor", "aten::silu", "aten::triu"]
def priv(op):
    try:
        t = torch._C._dispatch_dump_table(op)
    except Exception as e:
        return "dump_failed:%s:%s" % (type(e).__name__, e)
    return [l.strip() for l in t.splitlines() if "PrivateUse1" in l][:2]
print("=== GEMS PATH (import + enable) ===", flush=True)
try:
    import triton
    print("triton:", triton.__version__, triton.__file__, flush=True)
except Exception as e:
    print("triton IMPORT_FAIL:", type(e).__name__, e, flush=True)
try:
    import flag_gems
    print("flag_gems:", getattr(flag_gems, "__version__", "no __version__"), flag_gems.__file__, flush=True)
except Exception as e:
    print("flag_gems IMPORT_FAIL:", type(e).__name__, e, flush=True)
    sys.exit(0)
before = {op: priv(op) for op in OPS}
try:
    flag_gems.enable()
    print("ENABLE_OK", flush=True)
except Exception as e:
    print("ENABLE_FAIL:", type(e).__name__, str(e).splitlines()[0][:200], flush=True)
after = {op: priv(op) for op in OPS}
for op in OPS:
    print("  %-18s before=%s" % (op, before[op]), flush=True)
    print("  %-18s after =%s" % (op, after[op]), flush=True)
    print("  %-18s changed=%s" % (op, before[op] != after[op]), flush=True)
