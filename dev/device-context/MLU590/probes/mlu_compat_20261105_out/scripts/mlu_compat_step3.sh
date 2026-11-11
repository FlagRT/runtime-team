#!/usr/bin/env bash
# MLU590 兼容性核对 · Step3 L2 带卡判据（厂商原生 vs FlagGems 启用）
# 纪律：只启动/使用本方向自己的容器 dc-mlu590-hliu553；单卡 MLU_VISIBLE_DEVICES=2；
#       两个路径用**独立进程**避免 enable() 全局注册污染；证据按目录只追加、不覆盖。
# 用法：ssh Mlu-1 'bash -s' < /tmp/mlu_compat_step3.sh
set -u
C=dc-mlu590-hliu553
DEV=2
WD=/tmp/mlu_compat_20261105
mkdir -p "$WD"
TS=$(date +%H%M%S)

echo "=== SELF_CHECK mlu_compat_step3 2026-11-05 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo '--- cnmon BEFORE (busy check) ---'; cnmon 2>&1 | grep -E 'MiB/|No running processes' | head -12
echo '--- start our own container ---'
docker start "$C" 2>&1 | head -3
sleep 3
docker ps --filter "name=$C" --format '{{.Names}}|{{.Status}}' 2>&1

cat > "$WD/l2_vendor.py" <<'PY'
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
PY

cat > "$WD/l2_gems.py" <<'PY'
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
PY

echo "--- L2 vendor process (MLU_VISIBLE_DEVICES=$DEV) ---"
docker exec -i -e MLU_VISIBLE_DEVICES=$DEV -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/opt/triton "$C" /flagos/bin/python3 - < "$WD/l2_vendor.py" > "$WD/l2_vendor_$TS.out" 2>&1
echo "vendor_rc=$?"
grep -aE '^(===|torch |device_count|flag_gems_in_modules|  REG|NUMJSON|VENDOR_RC|RuntimeError|Traceback)' "$WD/l2_vendor_$TS.out" | head -20

echo "--- L2 gems process (independent process, same device) ---"
docker exec -i -e MLU_VISIBLE_DEVICES=$DEV -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/opt/triton "$C" /flagos/bin/python3 - < "$WD/l2_gems.py" > "$WD/l2_gems_$TS.out" 2>&1
echo "gems_rc=$?"
grep -aE '^(===|device_count|triton |flag_gems_imported|GEMS_|ENABLE_|  REG_|NUMJSON|RuntimeError|Traceback)' "$WD/l2_gems_$TS.out" | head -35

echo '--- numeric parity (host python3 compares the two NUMJSON lines) ---'
python3 - "$WD/l2_vendor_$TS.out" "$WD/l2_gems_$TS.out" <<'PY'
import sys, json
def load(p):
    for l in open(p, errors="replace"):
        if l.startswith("NUMJSON:"):
            return json.loads(l[8:])
    return None
v, g = load(sys.argv[1]), load(sys.argv[2])
if not v or not g:
    print("PARITY_INCOMPLETE vendor=%s gems=%s" % (bool(v), bool(g)))
else:
    print("op          max_abs_head_diff  sum_diff  nan/inf(v/g)  verdict")
    for k in sorted(set(v) & set(g)):
        a, b = v[k], g[k]
        d = max(abs(x-y) for x, y in zip(a["head"], b["head"]))
        ds = abs(a["sum"]-b["sum"])
        bad = (a["nan"]+a["inf"]+b["nan"]+b["inf"]) > 0
        ok = (d < 1e-3) and (ds < 1e-2) and not bad
        print("%-11s %-18.6f %-9.6f %-13s %s" % (k, d, ds, "%d/%d" % (a["nan"]+a["inf"], b["nan"]+b["inf"]), "PASS" if ok else "FAIL"))
PY

echo '--- cnmon AFTER (should be clean again) ---'; cnmon 2>&1 | grep -E 'MiB/|No running processes' | head -12
echo '--- evidence files ---'; ls -l "$WD"/l2_*_$TS.out
echo "=== END step3 (TS=$TS) ==="
