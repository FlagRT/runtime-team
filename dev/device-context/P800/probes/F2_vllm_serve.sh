#!/bin/bash
# P800 阶段 3 补 · 推理腿（vLLM 服务化形态）
#
# 形态：vLLM OpenAI 兼容服务（`--runner pooling --convert embed`，与 910C 口径一致）
# 平台：vllm-plugin-FL（`VLLM_FL_PLATFORM=kunlunxin`）—— 容器内社区 vLLM 0.13.0 + FL 平台插件
#
# ⚠️ 两个环境要点（本次实测得出，接入手册级）：
#   1) **必须 `PYTHONPATH=/env/FlagGems/src`**：vllm_fl 在 import 时依赖
#      `flag_gems.runtime.backend.device.DeviceDetector`，而 site-packages 里的 flag_gems
#      安装不完整 → 不设该变量会报 "Failed to infer device type"（vllm 直接退出）。
#   2) 算子路径选 **vendor**（`VLLM_FL_PREFER=vendor` + `USE_FLAGGEMS=0`）：
#      避免引入 FlagGems 路径（其官方推荐变量 `XPU_EVENT_KL3_ENABLE=1` 与已知厂商缺陷相关），
#      与本方向锁定口径（不设该变量）保持一致。
#
# 用法（容器内）：  DEV=6 bash F2_vllm_serve.sh
set -u
# OUT 用于**多镜像等价性对照**：不同镜像跑同一脚本时把结果分开存放（默认行为不变）
OUT=${OUT:-/workspace/out_serve}
mkdir -p "$OUT"
exec > "$OUT/F2_vllm_serve.log" 2>&1
source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda

DEV=${DEV:-6}
PORT=${PORT:-8100}
BUDGET=${BUDGET:-240}                    # 服务就绪等待上限（秒）
MODEL=${MODEL:-/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}
#: EAGER=1（默认）→ 加 --enforce-eager（关图捕获）；EAGER=0 → 启用图捕获。
#: 这是"服务化 vs 前向"吞吐差的可对照单变量（见 docs 的延迟拆解）。
EAGER=${EAGER:-1}
[ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager" || EAGER_FLAG=""
TAG=${TAG:-eager$EAGER}
PROTO=/workspace/prototype/runtime/proto/proto_infer_serve.py

export PYTHONPATH=/env/FlagGems/src
export VLLM_FL_PLATFORM=kunlunxin VLLM_FL_PREFER=vendor USE_FLAGGEMS=0
export GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0
export CUDA_VISIBLE_DEVICES=$DEV DC_BACKEND=kunlun SERVE_PORT=$PORT SERVE_HOST=127.0.0.1

stop_server() {
  # ⚠️ 2026-09-20 实测教训：`vllm serve` 的 **EngineCore 子进程会残留并持续占卡**
  #    （只杀主进程后，卡 6 仍被占 73850 MiB / 96 GiB，导致下一次启动报
  #     "Free memory ... less than desired GPU memory utilization"）。
  #    与 910C 侧的「坑 A2：清理残留 EngineCore 子进程」同类 —— 必须一并 kill -9。
  for p in $(pgrep -f "VLLM::Engin"); do kill -9 $p 2>/dev/null; done
  for p in $(pgrep -f "vllm serve"); do kill -9 $p 2>/dev/null; done
  sleep 3
}

echo "=========== P800 阶段 3 补 · vLLM 服务化 开始 $(date) ==========="
echo "用卡: $DEV | 端口: $PORT | 模型: $MODEL"
echo "单变量对照: EAGER=$EAGER（$([ "$EAGER" = "1" ] && echo '关图捕获' || echo '启用图捕获')）| TAG=$TAG"
echo "--- 用卡前：各卡显存占用 ---"
xpu-smi 2>/dev/null | awk '/MiB \//{print}' | head -8

stop_server
echo
echo "############ 启动 vllm serve ############"
nohup vllm serve $MODEL --served-model-name qwen3-embedding-0.6b \
  --runner pooling --convert embed \
  --port $PORT --max-model-len 4096 --gpu-memory-utilization 0.25 $EAGER_FLAG \
  > $OUT/vllm_serve.log 2>&1 &
echo "  pid=$!  日志: $OUT/vllm_serve.log"

ready=0
for i in $(seq 1 $((BUDGET/5))); do
  sleep 5
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/v1/models 2>/dev/null)
  if [ "$code" = "200" ]; then ready=1; echo "  ✅ 服务就绪 t=$((i*5))s"; break; fi
  [ $((i % 6)) = 0 ] && echo "  ... 等待中 t=$((i*5))s http=$code"
done
[ $ready = 0 ] && { echo "  ❌ 服务未在 ${BUDGET}s 内就绪"; tail -20 $OUT/vllm_serve.log; }

if [ $ready = 1 ]; then
  echo
  echo "############ 服务化验证（proto_infer_serve.py）############"
  timeout 300 python3 $PROTO --backend kunlun --rounds 5 \
    --out $OUT/proto_infer_serve_result_kunlun_$TAG.json 2>&1 \
    | grep -vE "^INFO|^WARNING|XCCL|SYMBOL_REWRITE|^\(APIServer" | tail -20
  if grep -q "SERVE_LEG_PASS" $OUT/proto_infer_serve_result_kunlun_$TAG.json 2>/dev/null; then
    echo "  >>> ✅ 服务化形态 PASS（TAG=$TAG）"
  else
    echo "  >>> ⚠️/❌ 见结果 JSON（TAG=$TAG）"
  fi
fi

echo
echo "############ 停机（释放卡）############"
stop_server
echo "--- 用卡后复查 ---"
xpu-smi 2>/dev/null | awk '/MiB \//{print}' | head -8
echo "=========== 阶段 3 补 · vLLM 服务化 结束 $(date) ==========="
