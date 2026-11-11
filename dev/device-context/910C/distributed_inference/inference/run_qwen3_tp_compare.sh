#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_qwen3_tp_compare.sh — O2+O3 一键执行（容器内运行，用户亲自执行）
# ═══════════════════════════════════════════════════════════════════════════════
# 【作用】串起：O2 单卡冒烟 → O3 TP=N 端到端 → 逐字对比，输出全部落 raid
# 【用法】docker exec -it flagos-infer-910c bash
#         bash /mnt/raid/hliu553/run_qwen3_tp_compare.sh [N] [greedy]
#         N 默认 2（可传 4）；第二参数传 greedy 用确定性解码（TP 对比正确对照方式）
# 【输出】results_qwen3_tp_compare/ 下：dense_qwen3_tp1[_greedy].json / dense_qwen3_tpN[_greedy].json
set -e
cd "$(dirname "$0")"

MODEL=/mnt/raid/hliu553/models/Qwen3-4B
OUT=/mnt/raid/hliu553/runtime-team/dev/device-context/benchmarks/inference/results_qwen3_tp_compare
mkdir -p "$OUT"
export DO_NOT_TRACK=1
# 注意：不设 VLLM_PLUGINS=fl —— P0 跑通时该变量为"(未设置)"（见 dense_infer_result.json）；
# vllm-plugin-FL 是 torch_fl 原生栈插件，A 线（torch_npu + vllm_ascend）设置后
# 会导致 current_platform.device_type 为空 → RuntimeError: Device string must not be empty
TPN="${1:-2}"
MODE="${2:-}"
EXT=""
GREEDY_ARG=""
if [ "$MODE" = "greedy" ]; then
    EXT="_greedy"
    GREEDY_ARG="--greedy"
    echo ">>> 模式：greedy 确定性解码（TP 逐字对比）"
fi

echo "[1/3] TP=1 单卡冒烟（O2）..."
python3 /mnt/raid/hliu553/runtime-team/dev/device-context/benchmarks/inference/dense_infer_qwen_1_5b_npu.py \
    --model "$MODEL" --tp 1 --seed 42 --preheat $GREEDY_ARG \
    --out "$OUT/dense_qwen3_tp1${EXT}.json"

echo "[2/3] TP=${TPN} 端到端（O3）..."
python3 /mnt/raid/hliu553/runtime-team/dev/device-context/benchmarks/inference/dense_infer_qwen_1_5b_npu.py \
    --model "$MODEL" --tp "$TPN" --seed 42 --preheat $GREEDY_ARG \
    --out "$OUT/dense_qwen3_tp${TPN}${EXT}.json"

echo "[3/3] 逐字对比（TP=1 vs TP=${TPN}）..."
python3 /mnt/raid/hliu553/runtime-team/dev/device-context/benchmarks/inference/compare_tp_outputs.py \
    "$OUT/dense_qwen3_tp1${EXT}.json" "$OUT/dense_qwen3_tp${TPN}${EXT}.json"

echo "════════════════════════════════════════════════════════════"
echo "结果目录: $OUT"
