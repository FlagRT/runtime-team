#!/bin/bash
# 910C · 依次试「成对 die」建容器 → 真做 ACL INIT + 两卡小算子（2026-10-16）
#   用途：FlagCX c10d 复现需要 **2 张卡**；本机纪律是只挂自己真正要用的 die，
#   且必须成对可用（单 die 可用 ≠ 成对可用）。
#   判据：容器内 ASCEND_RT_VISIBLE_DEVICES=0,1（容器内 0 基）能 INIT、device_count=2、
#         两卡各算一次小算子成功 ⇒ 该对可用，容器**保持运行**供后续使用。
set -u

C=${C:-dc-p9-910c-20261016}
IMG=${IMG:-flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64}
PY=${PY:-/usr/local/python3.11.15/bin/python3}
CANDS=${CANDS:-"4,5 5,6 2,3 8,9 12,13 0,1"}
OUT=${OUT:-/mnt/raid/hliu553/pick_pair_20261016.log}

: > "$OUT"
mk() {  # $1='a,b'
    IFS=, read -r A B <<< "$1"
    docker rm -f "$C" >/dev/null 2>&1 || true
    docker run -dit --name "$C" \
        --network host --ipc host --shm-size 512g \
        --security-opt label=disable --cap-add CAP_SYS_PTRACE \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:rw \
        -v /mnt/raid/hliu553:/mnt/raid/hliu553 \
        --device "/dev/davinci$A" --device "/dev/davinci$B" \
        --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
        "$IMG" bash >/dev/null
    sleep 2
}

for pair in $CANDS; do
    mk "$pair"
    o=$(timeout 180 docker exec -e ASCEND_RT_VISIBLE_DEVICES=0,1 "$C" bash -lc "
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
$PY -c \"
import torch, torch_npu
n = torch.npu.device_count()
print('COUNT=%d' % n)
if n >= 2:
    for d in (0, 1):
        a = torch.randn(64, 64, device='npu:%d' % d)
        print('DEV%d_OK name=%s sum=%.1f' % (d, torch.npu.get_device_name(d), float((a@a).sum())))
\"" 2>&1)
    rc=$?
    ok=$(echo "$o" | grep -ac "DEV0_OK\|DEV1_OK")
    line=$(echo "$o" | grep -aE "COUNT=|DEV[01]_OK|error code is|Resource_Busy|device is used|Two accelerators|Can't get|No such" | tail -2 | tr '\n' ' ')
    printf "  pair=%-7s rc=%-3s ok_dev=%s  %s\n" "$pair" "$rc" "$ok" "${line:0:150}" | tee -a "$OUT"
    if [ "$ok" = "2" ]; then
        echo "  ⇒ 选中 die 对 $pair（容器 $C 已按该对建好，保持运行）" | tee -a "$OUT"
        echo "$pair" > /mnt/raid/hliu553/PICKED_PAIR
        exit 0
    fi
done
echo "  !! 所有候选卡对均未能 INIT（整机繁忙）" | tee -a "$OUT"
exit 3
