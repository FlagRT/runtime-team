#!/bin/bash
# 910C · 逐 die 建容器试 INIT（2026-10-14 排查 + 选卡）
#  真因（本轮实测，与旧记忆不一致，按实测改）：
#   ① 映射**全部 16 个 die** ⇒ 容器内 `npu-smi` 直接失败：
#      `dcmi model initialized failed, because the device is used. ret is -8020`
#      + `Different containers share the same device` ⇒ **驱动层按"是否与他人容器共用同一设备"拒绝**，
#      映射全套 = 与所有人重叠 ⇒ 必然失败。**故"照抄历史容器映射全套"是错的**（那台当时独占整机）。
#   ② 只挂 1 个 die 才是对的，但**必须挑一个没有任何其他容器映射的 die**：
#      14:58 实测映射表：davinci1=evalx-910c · 2,3=kelvin-w4-lzy · 4=fl-profile-910c ·
#      6=flagos-batch-yxy-dev-910c · 7=rag-ljy-vllm-910c · 8-15=temp-cp-npu4567 ⇒ **davinci5 无人专用**。
#   ③ 首次失败正是因为我挑的是 davinci4（被 fl-profile-910c 占）。
#  ⇒ 逐候选建容器 → 真做一次 ACL INIT → 第一个成功者即本轮用卡（成功即占名额，不再多试）。
set -u

C=dc-t1re-910c-20261014
IMG=flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev-hostnet
PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python
CANDS=${CANDS:-"5 0 7 6 1 2 3 4"}
OUT=/mnt/raid/hliu553/dc_t1re_20261014/pick_die.log

: > "$OUT"
mk() {  # $1 = die 号
    docker rm -f "$C" >/dev/null 2>&1 || true
    docker run -dit --name "$C" \
        --network host --shm-size 512g \
        --security-opt label=disable --cap-add CAP_SYS_PTRACE \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:rw \
        -v /mnt/raid/hliu553:/mnt/raid/hliu553 \
        --device "/dev/davinci$1" \
        --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
        "$IMG" bash >/dev/null
    sleep 2
}

for d in $CANDS; do
    mk "$d"
    o=$(timeout 150 docker exec -e ASCEND_RT_VISIBLE_DEVICES=0 "$C" bash -lc "$PY -c \"
import torch, torch_npu
n = torch.npu.device_count()
name = torch.npu.get_device_name(0) if n else '-'
s = 0.0
if n:
    a = torch.randn(64, 64, device='npu:0'); s = float((a @ a).sum())
print('INIT_OK count=%d name=%s sum=%.1f' % (n, name, s))
\"" 2>&1)
    rc=$?
    line=$(echo "$o" | grep -aE "INIT_OK|error code is|Resource_Busy|device is used|can't|Can't get" | tail -1)
    printf "  die%-2s rc=%-3s %s\n" "$d" "$rc" "${line:0:160}" | tee -a "$OUT"
    if echo "$o" | grep -qa "INIT_OK count=[1-9]"; then
        echo "  ⇒ 选中 die=$d（容器 $C 已按该 die 建好，保持运行）" | tee -a "$OUT"
        echo "$d" > /mnt/raid/hliu553/dc_t1re_20261014/PICKED_DIE
        exit 0
    fi
done
echo "  !! 所有候选 die 均未能 INIT（可能整机繁忙）" | tee -a "$OUT"
exit 3
