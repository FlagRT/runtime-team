#!/bin/bash
# P8 设备侧 reduce 稳定性循环：新容器 flagos-hliu553-dev-910c + tf-venv-integration
# 每轮：rank1(910C) 先起 → 8s 后 rank0(4090-1) → 检查 sum=3.0 + done
PASS=0; FAIL=0
for i in $(seq 1 10); do
  PORT=$((29960+i))
  ssh 910C "docker exec flagos-hliu553-dev-910c bash -c 'pkill -9 -f \"[t]est_ag_hetero\" 2>/dev/null'" 2>/dev/null
  ssh 4090-1 "pkill -9 -f '[t]est_ag_hetero'" 2>/dev/null
  sleep 1
  ssh 910C "docker exec flagos-hliu553-dev-910c bash -c 'cd /workspace && export LD_LIBRARY_PATH=/workspace/FlagCX/build/lib:\$LD_LIBRARY_PATH && export PYTHONPATH=/workspace/FlagCX/plugin/torch:\$PYTHONPATH && export HCCL_NPU_SOCKET_PORT_RANGE=16666,16676 && source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && export NODE_ROLE=npu RANK=1 WORLD_SIZE=2 MASTER_ADDR=10.123.4.21 MASTER_PORT=$PORT FLAGCX_USE_HETERO_COMM=1 FLAGCX_FORCE_NET_SOCKET=1 FLAGCX_SOCKET_IFNAME=bond4 FLAGCX_TOPO_DETECTION_DISABLE=1 FLAGCX_MEM_ENABLE=1 FLAGCX_VMM_ENABLE=0 FLAGCX_P2P_DISABLE=1 && nohup /root/tf-venv-integration/bin/python test_ag_hetero.py > /tmp/loop8_r1_$i.log 2>&1 & echo r1-ok'" 2>/dev/null
  sleep 8
  ssh 4090-1 "cd /home/data/hongbinliu && export LD_LIBRARY_PATH=\$HOME/FlagCX/build/lib:\$LD_LIBRARY_PATH && export NODE_ROLE=cuda RANK=0 WORLD_SIZE=2 MASTER_ADDR=10.123.4.21 MASTER_PORT=$PORT FLAGCX_USE_HETERO_COMM=1 FLAGCX_FORCE_NET_SOCKET=1 FLAGCX_SOCKET_IFNAME=enp33s0f1 FLAGCX_TOPO_DETECTION_DISABLE=1 FLAGCX_MEM_ENABLE=1 FLAGCX_VMM_ENABLE=0 FLAGCX_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1 && timeout 60 \$HOME/dvfs/.venv_4090/bin/python test_ag_hetero.py > /tmp/loop8_r0_$i.log 2>&1" 2>/dev/null
  sleep 12
  RES=$(ssh 4090-1 "echo -n 'sum='; grep -o 'sum=[0-9.]*' /tmp/loop8_r0_$i.log 2>/dev/null | head -1; echo -n ' done='; grep -c 'rank0 done' /tmp/loop8_r0_$i.log 2>/dev/null" 2>/dev/null | tr '\n' ' ')
  if echo "$RES" | grep -q "sum=3.0" && echo "$RES" | grep -q "done=1"; then
    PASS=$((PASS+1)); echo "round $i: PASS  ($RES)"
  else
    FAIL=$((FAIL+1)); echo "round $i: FAIL  ($RES)"
  fi
done
ssh 910C "docker exec flagos-hliu553-dev-910c bash -c 'pkill -9 -f \"[t]est_ag_hetero\" 2>/dev/null'" 2>/dev/null
ssh 4090-1 "pkill -9 -f '[t]est_ag_hetero'" 2>/dev/null
echo "=== FINAL: PASS=$PASS FAIL=$FAIL ==="
