#!/bin/bash
# 910C(npu1-27) · 逐 die 全量探测（2026-10-16，**不早退**）
#   目的：为「FlagCX c10d 双卡复现」找出**两张**真正可 INIT 的 die。
#   判据：容器只挂该 die ⇒ ASCEND_RT_VISIBLE_DEVICES=0 ⇒ device_count>=1 且真算一个小算子。
#   ⚠️ 探测容器**逐个建、逐个删**，不留占卡容器（自己的探针容器也占名额）。
set -u

C=dc-dieprobe-910c-20261016
IMG=${IMG:-flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64}
PY=/usr/local/python3.11.15/bin/python3
CANDS=${CANDS:-"5 0 1 2 3 4 6 7 8 9 10 11 12 13 14 15"}
OUT=${OUT:-/mnt/raid/hliu553/dc_p9_20261016/probe_all_dies.log}
mkdir -p "$(dirname "$OUT")"; : > "$OUT"

echo "===== 逐 die 探测 $(date '+%F %T') =====" | tee -a "$OUT"
FREE=""
for d in $CANDS; do
    docker rm -f "$C" >/dev/null 2>&1 || true
    docker run -dit --name "$C" \
        --network host --ipc host --shm-size 64g \
        --security-opt label=disable --cap-add CAP_SYS_PTRACE \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:rw \
        --device "/dev/davinci$d" \
        --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
        "$IMG" bash >/dev/null
    sleep 2
    o=$(timeout 120 docker exec -e ASCEND_RT_VISIBLE_DEVICES=0 "$C" bash -lc "
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
$PY -c \"
import torch, torch_npu
n = torch.npu.device_count()
if n:
    a = torch.randn(64, 64, device='npu:0')
    print('FREE count=%d name=%s sum=%.1f' % (n, torch.npu.get_device_name(0), float((a@a).sum())))
else:
    print('BUSY count=0')
\"" 2>&1)
    if echo "$o" | grep -qa "FREE count=[1-9]"; then
        msg=$(echo "$o" | grep -a "FREE" | tail -1)
        printf "  die%-2s ✅ %s\n" "$d" "$msg" | tee -a "$OUT"
        FREE="$FREE $d"
    else
        why=$(echo "$o" | grep -aoE "Different containers share the same device|Failed to obtain the number of devices|Resource_Busy|can not use command" | head -1)
        printf "  die%-2s ❌ %s\n" "$d" "${why:-count=0}" | tee -a "$OUT"
    fi
done
docker rm -f "$C" >/dev/null 2>&1 || true
echo | tee -a "$OUT"
echo "===== 可用 die：${FREE:-（无）} =====" | tee -a "$OUT"
echo "$FREE" > /mnt/raid/hliu553/dc_p9_20261016/FREE_DIES
[ -n "$FREE" ] || exit 3
