#!/bin/bash
# 910C 服务化腿（2026-09-30 · (A) 方案落地轮）—— 覆盖 `runtime.create_stream()` 的**最后一个消费方**
# `runtime/proto/proto_infer_serve.py:89`（它需要**外部 vLLM 服务**，故单独一轮）。
#
# 两段：
#   [A] `serve_standard.sh`（STOP_AFTER=1）⇒ `SERVE_STANDARD_PASS`（标准形态不回归）
#   [B] 手动保持 vLLM 服务（STOP_AFTER=0）+ `proto_infer_serve.py --backend ascend`
#       ⇒ 重点看 [1b]「服务同卡上跨流计算」（正是本轮改动路径）
set -u
C=dc-serve-910c-20260930
P=/mnt/raid/hliu553/dc_legs_20260929/prototype
O=/mnt/raid/hliu553/dc_legs_20260929/out_serve_r9
M=/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B
# ⚠️ DEV 是**容器内**的设备索引（不是宿主 smi 索引）：本容器只挂 davinci7
#    ⇒ 容器内可见设备数=1，索引只能是 0。填 7 会得到 aclInit 107001「Invalid device ID」
#    （2026-09-30 踩到；与 P800「smi 索引 ≠ /dev/xpuN」同族的"索引口径"坑）。
DEV=0
mkdir -p "$O"

echo "===== [0] 容器与版本 ====="
docker ps --format '{{.Names}}|{{.Status}}' | grep "$C" || true
docker exec "$C" python3 -c "import torch, torch_npu, vllm; print('torch', torch.__version__, 'npu', torch_npu.__version__, 'vllm', vllm.__version__, 'dev', torch.npu.device_count())"
echo "版本核验（本层新代码到位）："
docker exec "$C" bash -lc "grep -c 'def _create_stream_raw' $P/runtime/backends/base.py; grep -c 'def stream_priority_readback' $P/runtime/__init__.py"

echo "===== [A] serve_standard.sh STOP_AFTER=1（标准形态复核）====="
docker exec -e DC_BACKEND=ascend -e DEV=$DEV -e MODEL=$M -e SERVE_FORM=embed -e STOP_AFTER=1 \
  -e DC_OUT_DIR=$O/serve_tp1_eager1 -e SMOKE_TIMEOUT=180 -e TP=1 -e EAGER=1 \
  "$C" bash -lc "cd $P && bash scripts/serve_standard.sh > /dev/null 2>&1"
echo "rc=$?"
grep -aE "verdict|服务就绪|冒烟|释放复查|用卡后" "$O/serve_tp1_eager1/serve_standard.log" 2>/dev/null | tail -8

echo "===== [B] 保持服务运行（STOP_AFTER=0）====="
docker exec -d -e DC_BACKEND=ascend -e DEV=$DEV -e MODEL=$M -e SERVE_FORM=embed -e STOP_AFTER=0 \
  -e DC_OUT_DIR=$O/serve_keep -e SMOKE_TIMEOUT=180 -e TP=1 -e EAGER=1 \
  "$C" bash -lc "cd $P && bash scripts/serve_standard.sh > /dev/null 2>&1"
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/models 2>/dev/null)
  [ "$code" = "200" ] && { echo "服务就绪（约 $((i*10)) s）"; break; }
  sleep 10
done
grep -aE "服务就绪|冒烟|verdict" "$O/serve_keep/serve_standard.log" 2>/dev/null | tail -4

echo "===== [B2] proto_infer_serve.py（本层消费方：服务同卡跨流计算）====="
docker exec -e DC_BACKEND=ascend -e DC_MODEL=$M -e DC_OUT_DIR=$O \
  "$C" bash -lc "cd $P && python3 runtime/proto/proto_infer_serve.py --backend ascend" \
  > "$O/proto_infer_serve.log" 2>&1
echo "rc=$?"
grep -aE "^\[1|\[1b|SERVE_LEG|verdict|PASS|不通过|失败" "$O/proto_infer_serve.log" | tail -12

echo "===== [C] 收尾：停服务 ====="
docker exec "$C" bash -lc 'pkill -f "[v]llm serve" 2>/dev/null; sleep 3; pkill -f "[E]ngineCore" 2>/dev/null; echo done'
echo "===== ALL DONE ====="
