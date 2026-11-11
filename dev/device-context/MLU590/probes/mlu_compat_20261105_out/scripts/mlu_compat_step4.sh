#!/usr/bin/env bash
# MLU590 兼容性核对 · Step4 逐算子独立判据 + 真实前向（vendor vs FlagGems，两个独立进程）
# 纪律：每个算子独立 try/except（一个失败不掩盖其余）；同 seed 同输入；记录 rc；证据只追加。
# 用法：ssh Mlu-1 'bash -s' < /tmp/mlu_compat_step4.sh
set -u
C=dc-mlu590-hliu553
DEV=2
WD=/tmp/mlu_compat_20261105
mkdir -p "$WD"
TS=$(date +%H%M%S)
PY=/flagos/bin/python3
ENVV="-e MLU_VISIBLE_DEVICES=$DEV -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/opt/triton"

echo "=== SELF_CHECK mlu_compat_step4 2026-11-05 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo '--- cnmon BEFORE ---'; cnmon 2>&1 | grep -cE '0 MiB/ 98304 MiB'
docker ps --filter "name=$C" --format '{{.Names}}|{{.Status}}'

cat > "$WD/op_parity.py" <<'PY'
import sys, os, json, traceback, torch, torch_mlu
MODE = os.environ["PARITY_MODE"]
print("=== OP PARITY mode=%s ===" % MODE, flush=True)
torch.mlu.set_device(0)
print("flag_gems_in_modules_before", "flag_gems" in sys.modules, flush=True)
if MODE == "gems":
    import triton, flag_gems
    flag_gems.enable()
    print("ENABLE_OK triton=%s flag_gems=%s" % (triton.__version__, getattr(flag_gems, "__version__", "?")), flush=True)
print("device_count", torch.mlu.device_count(), flush=True)

SPECS = [
    ("gemm64",     lambda: (torch.randn(64,64,generator=torch.Generator().manual_seed(11)).to("mlu:0") @ torch.randn(64,64,generator=torch.Generator().manual_seed(12)).to("mlu:0"))),
    ("matmul",     lambda: (torch.randn(32,64,generator=torch.Generator().manual_seed(13)).to("mlu:0") @ torch.randn(64,16,generator=torch.Generator().manual_seed(14)).to("mlu:0"))),
    ("add",        lambda: (torch.randn(32,64,generator=torch.Generator().manual_seed(15)).to("mlu:0") + torch.randn(32,64,generator=torch.Generator().manual_seed(16)).to("mlu:0"))),
    ("mul",        lambda: (torch.randn(32,64,generator=torch.Generator().manual_seed(17)).to("mlu:0") * torch.randn(32,64,generator=torch.Generator().manual_seed(18)).to("mlu:0"))),
    ("silu",       lambda: torch.nn.functional.silu(torch.randn(32,64,generator=torch.Generator().manual_seed(19)).to("mlu:0"))),
    ("relu",       lambda: torch.relu(torch.randn(32,64,generator=torch.Generator().manual_seed(20)).to("mlu:0"))),
    ("softmax",    lambda: torch.softmax(torch.randn(32,64,generator=torch.Generator().manual_seed(21)).to("mlu:0"), dim=-1)),
    ("layernorm",  lambda: torch.nn.functional.layer_norm(torch.randn(32,64,generator=torch.Generator().manual_seed(22)).to("mlu:0"), (64,))),
    ("triu",       lambda: torch.triu(torch.randn(32,64,generator=torch.Generator().manual_seed(23)).to("mlu:0"))),
]
for name, fn in SPECS:
    try:
        out = fn()
        torch.mlu.synchronize()
        c = out.detach().cpu()
        row = {"op": name, "ok": True, "head": [round(float(v),6) for v in c.flatten()[:8].tolist()],
               "sum": round(float(c.sum()),6), "nan": int(torch.isnan(c).sum()), "inf": int(torch.isinf(c).sum()), "err": ""}
    except Exception as e:
        tb = traceback.format_exc().splitlines()
        row = {"op": name, "ok": False, "head": [], "sum": None, "nan": None, "inf": None,
               "err": "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:180]),
               "last_frame": tb[-2].strip() if len(tb) >= 2 else ""}
    print("ROW:" + json.dumps(row, sort_keys=True), flush=True)
print("OP_PARITY_DONE", flush=True)
PY

cat > "$WD/forward_parity.py" <<'PY'
import sys, os, glob, json, traceback, torch, torch_mlu
MODE = os.environ["PARITY_MODE"]
print("=== FORWARD PARITY mode=%s ===" % MODE, flush=True)
torch.mlu.set_device(0)
if MODE == "gems":
    import triton, flag_gems
    flag_gems.enable()
    print("ENABLE_OK", flush=True)
from transformers import AutoModel, AutoTokenizer
snaps = glob.glob("/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/")
p = snaps[0] if snaps else ""
print("model_path", p, flush=True)
try:
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    m = AutoModel.from_pretrained(p, trust_remote_code=True).eval().to("mlu:0")
    batch = tok("设备执行上下文层把各家芯片的分歧显式化。", return_tensors="pt", truncation=True, max_length=128).to("mlu:0")
    with torch.no_grad():
        out = m(**batch)
    torch.mlu.synchronize()
    h = out.last_hidden_state[0, 0].detach().cpu()
    print("FORWARD_RESULT:" + json.dumps({"ok": True, "head": [round(float(v),6) for v in h[:8].tolist()],
        "sum": round(float(h.sum()),6), "nan": int(torch.isnan(h).sum()), "inf": int(torch.isinf(h).sum())}), flush=True)
except Exception as e:
    tb = traceback.format_exc().splitlines()
    print("FORWARD_RESULT:" + json.dumps({"ok": False,
        "err": "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:200]),
        "last_frame": tb[-2].strip() if len(tb) >= 2 else ""}), flush=True)
print("FORWARD_DONE", flush=True)
PY

for MODE in vendor gems; do
  echo "--- op parity: $MODE ---"
  timeout 300 docker exec -i $ENVV -e PARITY_MODE=$MODE "$C" $PY - < "$WD/op_parity.py" > "$WD/op_${MODE}_$TS.out" 2>&1
  echo "op_${MODE}_rc=$?"
  grep -aE '^(===|flag_gems_in_modules_before|ENABLE_OK|device_count|ROW:|OP_PARITY_DONE)' "$WD/op_${MODE}_$TS.out" | head -18
done

echo '--- forward parity: vendor / gems (timeout 300s each) ---'
for MODE in vendor gems; do
  timeout 300 docker exec -i $ENVV -e PARITY_MODE=$MODE "$C" $PY - < "$WD/forward_parity.py" > "$WD/fwd_${MODE}_$TS.out" 2>&1
  echo "fwd_${MODE}_rc=$?"
  grep -aE '^(===|ENABLE_OK|model_path|FORWARD_RESULT:|FORWARD_DONE)' "$WD/fwd_${MODE}_$TS.out" | head -8
done

echo '--- parity summary (host python3) ---'
python3 - "$WD/op_vendor_$TS.out" "$WD/op_gems_$TS.out" "$WD/fwd_vendor_$TS.out" "$WD/fwd_gems_$TS.out" <<'PY'
import sys, json
def rows(p, tag):
    d = {}
    for l in open(p, errors="replace"):
        if l.startswith(tag):
            try:
                r = json.loads(l[len(tag):])
                d[r["op"]] = r
            except Exception: pass
    return d
v, g = rows(sys.argv[1], "ROW:"), rows(sys.argv[2], "ROW:")
print("op          vendor  gems   max_abs_diff  verdict")
for k in sorted(set(v) | set(g)):
    a, b = v.get(k), g.get(k)
    if not a or not b:
        print("%-11s %-7s %-6s %-13s %s" % (k, "?" , "?", "-", "INCOMPLETE")); continue
    if a["ok"] and b["ok"]:
        d = max([abs(x-y) for x,y in zip(a["head"], b["head"])] or [0])
        bad = (a["nan"]+a["inf"]+b["nan"]+b["inf"]) > 0
        print("%-11s %-7s %-6s %-13.6f %s" % (k, "OK", "OK", d, "PASS" if d < 1e-3 and not bad else "NUM_FAIL"))
    else:
        print("%-11s %-7s %-6s %-13s %s" % (k, "OK" if a["ok"] else "FAIL", "OK" if b["ok"] else "FAIL", "-", "GEMS_FAIL" if not b["ok"] else "VENDOR_FAIL"))
def fwd(p):
    for l in open(p, errors="replace"):
        if l.startswith("FORWARD_RESULT:"):
            return json.loads(l[15:])
    return {"ok": False, "err": "no result line"}
f1, f2 = fwd(sys.argv[3]), fwd(sys.argv[4])
print("FORWARD vendor:", json.dumps({k: v for k, v in f1.items() if k in ("ok","err","head","sum")}, ensure_ascii=False)[:220])
print("FORWARD gems  :", json.dumps({k: v for k, v in f2.items() if k in ("ok","err","head","sum")}, ensure_ascii=False)[:220])
PY
echo '--- cnmon AFTER ---'; cnmon 2>&1 | grep -cE '0 MiB/ 98304 MiB'
echo "=== END step4 (TS=$TS) ==="
