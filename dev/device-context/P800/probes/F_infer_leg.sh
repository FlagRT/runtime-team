#!/bin/bash
# P800 阶段 3 · 推理腿（单卡前向 + 向量正确性 + 性能）
#
# 与 910C 口径对齐：维度 / 语义区分度 / 句每秒 / p50 时延 / 错误分级 + 业务继续
# 不受厂商 KL3 缺陷影响（不跑集合通信、不依赖 FlagGems ⇒ 不设该变量，同训练腿口径）
#
# 用法（容器内）：  bash /workspace/prototype/../../F_infer_leg.sh   # 或放入 /workspace 后执行
#   DEV=6 bash F_infer_leg.sh
set -u
mkdir -p /workspace/out_infer
exec > /workspace/out_infer/F_infer_leg.log 2>&1
source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda

DEV=${DEV:-6}                       # 单卡；用卡前先 xpu-smi 确认该卡空闲
PORT=${PORT:-30810}
BUDGET=${BUDGET:-420}               # 秒；超时即判挂死
PROTO=/workspace/prototype/runtime/proto/proto_infer_leg.py
MODEL=${MODEL:-/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B}

clean() { for p in $(pgrep -f proto_infer_leg); do kill -9 $p 2>/dev/null; done; sleep 2; }

echo "=========== P800 阶段 3 推理腿 开始 $(date) ==========="
echo "用卡: $DEV | 模型: $MODEL | 超时: ${BUDGET}s"
echo "--- 用卡前：各卡显存占用 ---"
xpu-smi 2>/dev/null | awk '/MiB \//{print}' | head -8

clean
echo
echo "############ 推理腿：单卡前向 + 向量正确性 ############"
env CUDA_VISIBLE_DEVICES=$DEV \
    DC_BACKEND=kunlun \
    DC_ROOT=/workspace/prototype \
    DC_MODEL=$MODEL \
    DC_OUT_DIR=/workspace/out_infer \
    FLAGCX_ADAPTOR=klx \
    timeout $BUDGET python3 $PROTO --backend kunlun --out proto_infer_leg_result.json
RC=$?

echo
if grep -q "INFER_LEG_PASS" /workspace/out_infer/proto_infer_leg_result.json 2>/dev/null; then
  echo "  >>> ✅ 推理腿 PASS"
elif grep -q "INFER_LEG_FAIL" /workspace/out_infer/proto_infer_leg_result.json 2>/dev/null; then
  echo "  >>> ⚠️ 推理腿 结果有 FAIL 项（见 json，非挂死）"
else
  echo "  >>> ❌ 未产出结果：rc=$RC（$([ $RC = 124 ] && echo 超时/挂死 || echo 其他失败)）"
fi

echo
echo "--- 用卡后复查（确认卡已释放）---"
xpu-smi 2>/dev/null | awk '/MiB \//{print}' | head -8
clean
echo "=========== 阶段 3 推理腿 结束 $(date) ==========="
