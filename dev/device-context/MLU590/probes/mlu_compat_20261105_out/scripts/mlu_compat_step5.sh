#!/usr/bin/env bash
# MLU590 · Step5 dispatch 归属逐算子对照（enable 前后，同一进程内；带卡但不改任何配置）
# 用法：ssh Mlu-1 'bash -s' < /tmp/mlu_compat_step5.sh
set -u
C=dc-mlu590-hliu553
DEV=2
WD=/tmp/mlu_compat_20261105
mkdir -p "$WD"
TS=$(date +%H%M%S)

echo "=== SELF_CHECK mlu_compat_step5 2026-11-05 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="

cat > "$WD/dispatch_map.py" <<'PY'
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
PY

echo '--- dispatch map (gems process) ---'
timeout 180 docker exec -i -e MLU_VISIBLE_DEVICES=$DEV -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/opt/triton "$C" /flagos/bin/python3 - < "$WD/dispatch_map.py" > "$WD/dispatch_map_$TS.out" 2>&1
echo "dispatch_rc=$?"
grep -aE '^(ENABLE_OK|flag_gems_in_modules_before|DISPATCH_ROW:|DISPATCH_MAP_DONE)' "$WD/dispatch_map_$TS.out" | head -20
echo '--- cnmon AFTER ---'; cnmon 2>&1 | grep -cE '0 MiB/ 98304 MiB'
echo "=== END step5 (TS=$TS) ==="
