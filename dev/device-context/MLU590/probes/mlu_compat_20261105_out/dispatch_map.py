import sys, torch, torch_mlu
torch.mlu.set_device(0)
OPS = {
    "gemm64 -> aten::mm":        "aten::mm",
    "matmul -> aten::matmul":    "aten::matmul",
    "add -> aten::add.Tensor":   "aten::add.Tensor",
    "mul -> aten::mul.Tensor":   "aten::mul.Tensor",
    "silu -> aten::silu":        "aten::silu",
    "relu -> aten::relu":        "aten::relu",
    "softmax -> aten::_softmax": "aten::_softmax",
    "layernorm -> native_layer_norm": "aten::native_layer_norm",
    "triu -> aten::triu":        "aten::triu",
}
def priv(op):
    try:
        t = torch._C._dispatch_dump_table(op)
    except Exception as e:
        return "DUMP_FAIL:%s" % type(e).__name__
    ls = [l.strip() for l in t.splitlines() if "PrivateUse1" in l][:1]
    return ls[0] if ls else "(no PrivateUse1 row)"
before = {k: priv(v) for k, v in OPS.items()}
print("flag_gems_in_modules_before:", "flag_gems" in sys.modules, flush=True)
import triton, flag_gems
flag_gems.enable()
print("ENABLE_OK", flush=True)
after = {k: priv(v) for k, v in OPS.items()}
for k in OPS:
    b, a = before[k], after[k]
    src_a = "FlagGems(/dev/null)" if "/dev/null" in a else ("torch_mlu(RegisterMLU)" if "RegisterMLU" in a else "other/none")
    print("DISPATCH_ROW: %-34s changed=%-5s after=%-22s | %s" % (k, b != a, src_a, a[:110]), flush=True)
print("DISPATCH_MAP_DONE", flush=True)
