import sys, torch, torch_mlu
OPS = ["aten::mm", "aten::matmul", "aten::add.Tensor", "aten::mul.Tensor", "aten::silu", "aten::triu"]
def priv(op):
    try:
        t = torch._C._dispatch_dump_table(op)
    except Exception as e:
        return "dump_failed:%s:%s" % (type(e).__name__, e)
    return [l.strip() for l in t.splitlines() if "PrivateUse1" in l][:2]
print("=== VENDOR PATH (no flag_gems import) ===", flush=True)
print("torch:", torch.__version__, "torch_mlu:", torch_mlu.__version__, flush=True)
print("flag_gems_in_sys_modules:", "flag_gems" in sys.modules, flush=True)
print("triton_in_sys_modules:", "triton" in sys.modules, flush=True)
for op in OPS:
    print("  %-18s %s" % (op, priv(op)), flush=True)
