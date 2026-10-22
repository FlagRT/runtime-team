#!/bin/bash
# 910C · 逐卡试 INIT（挑一个真正可用的卡位）—— 2026-10-14 排查用
#  背景：本轮首次建容器后 `aclInit` 报 **507899 Resource_Busy**（并提示"不同容器共享同一设备"）。
#        本机是**共享机**，多卡被他人容器映射（`davinci0` 被两个"全映射"容器覆盖），
#        而"卡位是否可用"不能只看 `npu-smi` 的进程表 —— **ACL 槽位**可能在 Init 后仍被占。
#  ⇒ 只能**逐卡试**：每张候选卡真做一次 ACL INIT + 一个小算子，记 rc。
#  ⚠️ 试成功即等于**占住该卡位** ⇒ 随即用它跑审计，不做无意义的多次试。
set -u

C=dc-t1re-910c-20261014
PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python
CANDS=${CANDS:-"5 4 0 1 6 2 3 7"}
OUT=/mnt/raid/hliu553/dc_t1re_20261014/probe_init.log

: > "$OUT"
echo "########## 逐卡试 ACL INIT（容器只映射设备节点，真正用哪张由 ASCEND_RT_VISIBLE_DEVICES 决定）##########"
docker ps --filter "name=^${C}$" --format '  容器: {{.Names}} {{.Status}}'
for d in $CANDS; do
    o=$(timeout 120 docker exec -e ASCEND_RT_VISIBLE_DEVICES=$d "$C" bash -lc \
        "$PY -c \"
import torch, torch_npu
n = torch.npu.device_count()
a = torch.randn(64, 64, device='npu:0') if n else None
print('OK count=%d name=%s sum=%.1f' % (n, torch.npu.get_device_name(0) if n else '-', float((a @ a).sum()) if n else 0.0))
\"" 2>&1)
    rc=$?
    line=$(echo "$o" | grep -aE "OK count=|error code is|Resource_Busy|Invalid device id" | tail -1)
    printf "  dev%-2s rc=%-3s %s\n" "$d" "$rc" "${line:0:150}" | tee -a "$OUT"
done
echo
echo "########## 可用的卡位 ##########"
grep -aE "OK count=" "$OUT" | head -3 | sed 's/^/  /'
