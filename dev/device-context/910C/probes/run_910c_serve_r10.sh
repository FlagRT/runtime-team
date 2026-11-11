#!/bin/bash
# 910C 服务化腿（2026-10-08 · 第 10 轮 = D4）
# **在宿主上执行**（要调 docker）。
#
# 目的：覆盖 `create_stream()` 的**最后一个消费方** `runtime/proto/proto_infer_serve.py`
# （它需要**外部 vLLM 服务**，故单独一轮），并把服务化四形态（TP × EAGER）一并复跑。
#
# 前置：服务化容器已起（mount 2 张卡 ⇒ 容器内可见设备 = 0,1）。
# 用法：bash run_910c_serve_r10.sh <容器名>
set -u
C=${1:?用法: bash run_910c_serve_r10.sh <容器名>}
P=/mnt/raid/hliu553/dc_r10_20261008/prototype
O=/mnt/raid/hliu553/dc_r10_20261008/out_serve_r10
# ⚠️ 用**我方 scratch 的修复副本**：机上共享路径 `models/Qwen3-Embedding-0.6B/model.safetensors`
#    已被存储层静默损坏（详见 r10 报告）；修复副本 sha256 == HF blob 名 `0437e45c…`（位级原件）。
M=/mnt/raid/hliu553/dc_r10_20261008/models/Qwen3-Embedding-0.6B
# ⚠️ **首跑后修正（2026-10-08）**：就绪轮询必须用 `serve_standard.sh` 的实际端口 —— 它默认 `PORT=8100`，
#    而首跑脚本写的是 `8000` ⇒ 必然等满 `40 × 10 s`（服务其实 30 s 就绪、冒烟已通过）。
#    该轮询**不影响任何 verdict**（verdict 取自 `serve_standard.log` 与 `proto_infer_serve.py` 自身判定）；
#    修正前后均**未**为此放宽任何判据。详见 `../../docs/ASCEND_910C_R10_RERUN_20261008.md` §4。
SPORT=${SPORT:-8100}
# ⚠️ DEV 是**容器内**的设备索引（不是宿主 smi 索引）：本容器挂 2 张 ⇒ 可见 0,1
mkdir -p "$O"

echo "===== [0] 容器与版本 ====="
docker ps --format '{{.Names}}|{{.Status}}' | grep "$C" || true
docker exec "$C" python3 -c "import torch, torch_npu, vllm; print('torch', torch.__version__, 'npu', torch_npu.__version__, 'vllm', vllm.__version__, 'dev', torch.npu.device_count())"
echo "本层新代码到位核验（各期望 1）："
docker exec "$C" bash -lc "grep -c 'def owns_stream' $P/runtime/backends/base.py; grep -c 'def release_stream' $P/runtime/backends/base.py; grep -c 'def stream_priority_readback' $P/runtime/__init__.py"

run_form () {
  local tp="$1" eager="$2"
  local dev="0"; [ "$tp" = "2" ] && dev="0,1"
  local tag="tp${tp}_eager${eager}"
  echo "===== [A] SERVE_FORM=embed TP=$tp EAGER=$eager DEV=$dev ====="
  docker exec -e DC_BACKEND=ascend -e DEV="$dev" -e MODEL="$M" -e SERVE_FORM=embed -e STOP_AFTER=1 \
    -e DC_OUT_DIR="$O/$tag" -e SMOKE_TIMEOUT=240 -e TP="$tp" -e EAGER="$eager" \
    "$C" bash -lc "cd $P && bash scripts/serve_standard.sh > /dev/null 2>&1"
  echo "rc=$?"
  grep -aE "verdict|服务就绪|冒烟|释放复查" "$O/$tag/serve_standard.log" 2>/dev/null | tail -6
  echo "-- 形态真生效证据（不是「命令跑通了」）"
  if [ "$tp" = "2" ]; then
    grep -aoE "world_size=[0-9]+|Worker_TP[0-9]|backend=[a-z]+" "$O/$tag/vllm_serve.log" 2>/dev/null | sort -u | tr '\n' ' '; echo
  fi
  grep -aoE "enforce_eager=[A-Za-z]+|CUDAGraphMode\.[A-Z_]+|CompilationMode\.[A-Z_]+" \
    "$O/$tag/vllm_serve.log" 2>/dev/null | sort -u | tr '\n' ' '; echo
}

run_form 1 1
run_form 1 0
run_form 2 1
run_form 2 0

echo "===== [B] 保持服务运行（STOP_AFTER=0）+ proto_infer_serve.py（本层消费方）====="
docker exec -d -e DC_BACKEND=ascend -e DEV=0 -e MODEL="$M" -e SERVE_FORM=embed -e STOP_AFTER=0 \
  -e DC_OUT_DIR="$O/serve_keep" -e SMOKE_TIMEOUT=240 -e TP=1 -e EAGER=1 \
  "$C" bash -lc "cd $P && bash scripts/serve_standard.sh > /dev/null 2>&1"
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${SPORT}/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && { echo "服务就绪（约 $((i*10)) s）"; break; }
  sleep 10
done
grep -aE "服务就绪|冒烟|verdict" "$O/serve_keep/serve_standard.log" 2>/dev/null | tail -4

docker exec -e DC_BACKEND=ascend -e DC_MODEL="$M" -e DC_OUT_DIR="$O" \
  "$C" bash -lc "cd $P && python3 runtime/proto/proto_infer_serve.py --backend ascend" \
  > "$O/proto_infer_serve.log" 2>&1
echo "rc=$?"
grep -aE "^\[1|\[1b|SERVE_LEG|verdict|PASS|不通过|失败" "$O/proto_infer_serve.log" | tail -14

echo "===== [C] 收尾：停服务 ====="
docker exec "$C" bash -lc 'pkill -f "[v]llm serve" 2>/dev/null; sleep 3; pkill -f "[E]ngineCore" 2>/dev/null; echo done'
echo "===== ALL DONE ====="
date "+[%H:%M:%S]"
