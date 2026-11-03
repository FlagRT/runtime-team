#!/bin/bash
# P4：补丁挂上后跑我方验收（对照 10-26 归档读数：离线 118/0/1 · smoke 46/0 · 契约不变式 4/4 rc=0）
set -u
WH=/bmcp_lvm_fs/hliu553/weakref_check_20261103
WC=/workspace/weakref_check_20261103
P=/workspace/runtime-team/dev/device-context/prototype
O=$WH/out_p4
C=hliu553-dc-dev
mkdir -p "$O"
# 把补丁 so 与 sitecustomize 放进同一个注入目录（容器视角）
docker exec "$C" bash -lc "mkdir -p $WC/shim && cp $WC/wrfix*.so $WC/sitecustomize.py $WC/shim/ 2>/dev/null; ls $WC/shim"
run() {  # run <tag> <DCFIX> <cmd...>
  local tag=$1 fix=$2; shift 2
  docker exec -e CUDA_VISIBLE_DEVICES=0 -e DCFIX="$fix" -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH="$WC/shim:$P" "$C" bash -lc "cd $P && $*" > "$O/$tag.log" 2>&1
  echo $? > "$O/$tag.rc"
}
echo "=== [P4] 补丁臂（DCFIX=1）vs 对照臂（DCFIX=0），其余完全相同 ==="
for arm in fix ctl; do
  [ "$arm" = fix ] && F=1 || F=0
  run "offline_$arm" "$F" "python3 -u scripts/backend_offline_check.py --backend ppu"
  run "smoke_$arm"   "$F" "python3 -u runtime/smoke_runtime.py --backend ppu"
  run "inv_$arm"     "$F" "python3 -X faulthandler -u runtime/conformance/runner.py --backend ppu --cases contract_invariants --out $WC/out_p4/inv_$arm.json"
done
echo "=== 汇总（归档基线：离线 118/0/1 · smoke 46/0 · 契约不变式 4/4 且 rc=0）==="
python3 - "$O" <<'PY'
import os, sys, re
O = sys.argv[1]
def rc(t):
    p = os.path.join(O, t + ".rc")
    return open(p).read().strip() if os.path.exists(p) else "?"
def txt(t):
    p = os.path.join(O, t + ".log")
    return open(p, errors="replace").read() if os.path.exists(p) else ""
rows = []
for arm in ("fix", "ctl"):
    off = txt("offline_" + arm); sm = txt("smoke_" + arm); inv = txt("inv_" + arm)
    m_off = re.search(r"(\d+) 通过 / (\d+) 失败 / (\d+) 跳过", off) or re.search(r"(\d+) 通过 / (\d+) 失败", off)
    m_sm = re.search(r"(\d+) 通过 / (\d+) 失败", sm)
    m_inv = re.search(r"(\d+)/(\d+)", inv)
    verdict = "PASS" if "CONTRACT_INVARIANTS_PASS" in inv else ("FAIL" if "CONTRACT_INVARIANTS_FAIL" in inv else "?")
    rows.append((arm, m_off.groups() if m_off else ("?",)*3, m_sm.groups() if m_sm else ("?",)*2,
                 m_inv.group(0) if m_inv else "?", verdict,
                 rc("offline_"+arm), rc("smoke_"+arm), rc("inv_"+arm),
                 "wrfix LOADED" in off))
print(f"{'臂':6s} {'离线(通过/失败/跳过)':26s} {'smoke':10s} {'契约不变式':10s} {'verdict':22s} {'rc(off/smk/inv)':18s} so加载")
for a, o, s, i, v, r1, r2, r3, loaded in rows:
    print(f"{a:6s} {'/'.join(o):26s} {'/'.join(s):10s} {i:10s} {v:22s} {r1}/{r2}/{r3:<12s} {loaded}")
PY
