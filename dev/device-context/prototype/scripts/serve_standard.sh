#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# serve_standard.sh — 统一推理服务启动脚本（**组内服务启动标准**）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 【这份脚本解决什么】
#   此前 910C 与 P800 各自维护一份启动脚本（910C 62 行 / P800 91 行），
#   环境变量、参数口径、停机清理各写各的 —— 下游各方向再各抄一份，就会分叉。
#   本脚本把"启动推理服务"收敛成**一条命令 + 一组环境变量**，跨芯片只改 DC_BACKEND。
#
# 【与两套旧脚本的关系（差异点均已在内部分支处理，不是删旧脚本）】
#   910C：`910C/distributed_inference/inference/start_vllm_serve_910c.sh`
#         · 坑 A5 禁用 vllm-plugin-FL（unset VLLM_PLUGINS）
#         · 坑 A3 DO_NOT_TRACK=1
#         · 坑 A2 清理残留 EngineCore 子进程
#         · 选卡 ASCEND_RT_VISIBLE_DEVICES
#         · 另有 D10/D11 集成（错误翻译包装器 + 设备状态监控）——见下方"集成层说明"
#   P800：`P800/probes/F2_vllm_serve.sh`
#         · PYTHONPATH=/env/FlagGems/src（硬前置）
#         · VLLM_FL_PLATFORM=kunlunxin / VLLM_FL_PREFER=vendor / USE_FLAGGEMS=0 /
#           GEMS_VENDOR=kunlunxin / KLX_USE_AUTOTUNE=0
#         · 选卡 CUDA_VISIBLE_DEVICES
#         · 停机必须连带清理 EngineCore 残留（实测占卡 73850 MiB / 96 GiB）
#
# 【统一的服务参数口径】（两实例同口径，便于横向比对）
#     --runner pooling --convert embed --max-model-len 4096 --port 8100
#   ⚠️ 该版本 vLLM **没有 --task 参数**；embedding 服务必须用 --runner pooling --convert embed
#      （写 --task embed 会报 `vllm: error: unrecognized arguments: --task embed`）。
#
# 【集成层说明（D10/D11）——如实标注当前范围】
#   910C 版脚本额外挂了两项集成：错误码翻译包装器（inject_error_translation.serve()）
#   与设备状态监控（device_state_monitor.py），资产在 910C 目录下。
#   本统一脚本**保留开关**（ERROR_TRANSLATION / MONITOR），但当前**仅在 910C 上可用**；
#   P800 侧需先把这两个资产上提到 prototype 侧才能启用 —— 见标准文档"待上提项"。
#   基础启动路径（起服务 / 就绪 / 健康检查 / 停机）在两个芯片上均已验证。
#
# 【用法】
#   # P800（容器内）
#   DC_BACKEND=kunlun DEV=6 MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B \
#     bash prototype/scripts/serve_standard.sh
#   # 910C（容器内）
#   DC_BACKEND=ascend MODEL=/mnt/raid/hliu553/models/Qwen3-4B SERVED_NAME=qwen3-4b \
#     bash prototype/scripts/serve_standard.sh
#
# 【环境变量】
#   DC_BACKEND      后端名（ascend | kunlun），默认 ascend
#   MODEL           模型路径（**须给到 snapshots/<hash>**，给缓存根目录会报 Unrecognized model）
#   SERVED_NAME     服务暴露的模型名（默认按后端给）
#   PORT            端口（8100）
#   DEV             选卡（ascend 用 ASCEND_RT_VISIBLE_DEVICES，kunlun 用 CUDA_VISIBLE_DEVICES）
#   TP              tensor parallel（1）
#   MAX_MODEL_LEN   （4096）
#   GPU_MEM_UTIL    （留空则不传；P800 实测共享机建议 0.25）
#   EAGER           1=加 --enforce-eager（默认 1）；0=启用图捕获
#                   （P800 实测：去掉 eager 反而更慢 p50 132ms，故默认保留）
#   DC_OUT_DIR      日志与 pid 目录（默认 /tmp/dc_serve）
#   STOP_AFTER      1=就绪并健康检查后立即停机（验证用）；0=保持运行（下游起服务用，默认 0）
#   READY_BUDGET    就绪等待上限秒数（180）
#   ERROR_TRANSLATION / MONITOR  集成层开关（默认 0；**当前仅 910C 可用**）
# ═══════════════════════════════════════════════════════════════════════════════
set -u

BACKEND=${DC_BACKEND:-ascend}
PORT=${PORT:-8100}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-}
EAGER=${EAGER:-1}
DEV=${DEV:-}
DC_OUT_DIR=${DC_OUT_DIR:-/tmp/dc_serve}
STOP_AFTER=${STOP_AFTER:-0}
READY_BUDGET=${READY_BUDGET:-180}
ERROR_TRANSLATION=${ERROR_TRANSLATION:-0}
MONITOR=${MONITOR:-0}
HOST=${HOST:-127.0.0.1}

mkdir -p "$DC_OUT_DIR"
LOG="$DC_OUT_DIR/serve_standard.log"
SRV_LOG="$DC_OUT_DIR/vllm_serve.log"
PIDFILE="$DC_OUT_DIR/serve_standard.pid"
exec > "$LOG" 2>&1

echo "=========== 统一推理服务启动 开始 $(date) ==========="
echo "backend=$BACKEND dev=${DEV:-<全部可见>} port=$PORT tp=$TP max_model_len=$MAX_MODEL_LEN eager=$EAGER"
echo "stop_after=$STOP_AFTER（1=验证后停机；0=保持运行）"

# ─────────────────────────────────────────────────────────────────────────────
# 1) 后端环境（两芯片的差异全部收敛在这里；其余流程完全共用）
# ─────────────────────────────────────────────────────────────────────────────
case "$BACKEND" in
  ascend)
    MODEL=${MODEL:-/mnt/raid/hliu553/models/Qwen3-4B}
    SERVED_NAME=${SERVED_NAME:-qwen3-4b}
    # 坑 A5：A 线禁用 vllm-plugin-FL（该插件无 ascend 后端，启用后 platform.device_type 变空
    #         → RuntimeError: Device string must not be empty）
    unset VLLM_PLUGINS
    # 坑 A3
    export DO_NOT_TRACK=1
    [ -n "$DEV" ] && export ASCEND_RT_VISIBLE_DEVICES="$DEV"
    SKIP_EXTRA=1        # 910C 既有脚本不传 --runner pooling（生成类服务）
    ;;
  kunlun)
    MODEL=${MODEL:-/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B}
    SERVED_NAME=${SERVED_NAME:-qwen3-embedding-0.6b}
    # 硬前置：site-packages 里的 flag_gems 子模块不完整，vllm_fl import 会失败
    #         → 缺它报 "Failed to infer device type"（看着像设备问题，其实是包问题）
    export PYTHONPATH=/env/FlagGems/src${PYTHONPATH:+:$PYTHONPATH}
    # 算子路径取 vendor：避开 FlagGems 路径及其 KL3 依赖（与本方向锁定口径一致）
    export VLLM_FL_PLATFORM=kunlunxin VLLM_FL_PREFER=vendor USE_FLAGGEMS=0
    export GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0
    [ -n "$DEV" ] && export CUDA_VISIBLE_DEVICES="$DEV"
    SKIP_EXTRA=0
    ;;
  *)
    echo "❌ 未知 DC_BACKEND=$BACKEND（支持 ascend | kunlun）"; exit 2 ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# 2) 停机清理（坑 A2 / P800 实测教训：EngineCore 子进程残留持续占卡）
#    变量拼接避免 pkill 命中自身（坑）
# ─────────────────────────────────────────────────────────────────────────────
cleanup() {
  P1=VLLM::Engin; P2="vllm serve"
  pkill -9 -f "${P1}eCore" 2>/dev/null || true
  pkill -9 -f "$P2" 2>/dev/null || true
  [ "$MONITOR" = "1" ] && { pkill -9 -f "device_state_monitor" 2>/dev/null || true; }
  sleep 3
}

# 用卡现状（便于前后对比；两芯片工具名不同）
card_snapshot() {
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>/dev/null | grep -E "^\| [0-9]+" | head -8
  elif command -v xpu-smi >/dev/null 2>&1; then
    xpu-smi 2>/dev/null | awk '/MiB \//{print}' | head -8
  else
    echo "  (无 npu-smi / xpu-smi)"
  fi
}

echo "--- 用卡前 ---"; card_snapshot
cleanup

# ─────────────────────────────────────────────────────────────────────────────
# 3) 启动（服务参数统一口径）
# ─────────────────────────────────────────────────────────────────────────────
ARGS=( "$MODEL" --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT"
       --max-model-len "$MAX_MODEL_LEN" --tensor-parallel-size "$TP" )
# embedding 服务形态（两实例同口径）：--runner pooling --convert embed
[ "$SKIP_EXTRA" = "0" ] && ARGS+=( --runner pooling --convert embed )
[ -n "$GPU_MEM_UTIL" ] && ARGS+=( --gpu-memory-utilization "$GPU_MEM_UTIL" )
[ "$EAGER" = "1" ] && ARGS+=( --enforce-eager )
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=( ${EXTRA_ARGS} )

echo
echo "############ 启动 vllm serve ############"
if [ "$ERROR_TRANSLATION" = "1" ]; then
  # D10 集成（当前仅 910C 可用）：serve 以包装器启动，未捕获异常先过 translate_error
  nohup python3 -c "import inject_error_translation; inject_error_translation.serve()" serve "${ARGS[@]}" \
    > "$SRV_LOG" 2>&1 &
else
  nohup vllm serve "${ARGS[@]}" > "$SRV_LOG" 2>&1 &
fi
SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "  pid=$SRV_PID  日志: $SRV_LOG"

# ─────────────────────────────────────────────────────────────────────────────
# 4) 就绪等待（判据：/v1/models 返回 200）
# ─────────────────────────────────────────────────────────────────────────────
READY=0
for i in $(seq 1 $((READY_BUDGET/5))); do
  sleep 5
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/v1/models" 2>/dev/null)
  if [ "$code" = "200" ]; then READY=1; echo "  ✅ 服务就绪 t=$((i*5))s"; break; fi
  kill -0 $SRV_PID 2>/dev/null || { echo "  ❌ 进程已退出（t=$((i*5))s）"; break; }
done
[ "$READY" = "0" ] && { echo "  ❌ 服务未在 ${READY_BUDGET}s 内就绪"; tail -20 "$SRV_LOG"; }

# 就绪后做一次 embedding 冒烟（仅在 embedding 形态下）
if [ "$READY" = "1" ] && [ "$SKIP_EXTRA" = "0" ]; then
  echo "--- 冒烟：/v1/embeddings ---"
  curl -s -X POST "http://$HOST:$PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_NAME\",\"input\":[\"服务启动标准冒烟\"]}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); v=d['data'][0]['embedding']; import math; n=math.sqrt(sum(x*x for x in v)); print(f'  维度={len(v)} 范数={n:.6f}')" 2>/dev/null \
    || echo "  ⚠️ 冒烟请求失败（见服务日志）"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5) 收尾：验证模式则停机；否则保持运行（并给出下游需要的三项信息）
# ─────────────────────────────────────────────────────────────────────────────
if [ "$STOP_AFTER" = "1" ]; then
  echo
  echo "############ 验证模式：停机 ############"
  cleanup
  echo "--- 用卡后复查（确认已释放）---"; card_snapshot
  echo "[verdict] $([ "$READY" = 1 ] && echo SERVE_STANDARD_PASS || echo SERVE_STANDARD_FAIL)"
else
  echo
  echo "############ 服务保持运行 ############"
  echo "  base_url   = http://$HOST:$PORT/v1"
  echo "  model      = $SERVED_NAME"
  echo "  停止服务   : STOP 时请用 cleanup 逻辑（kill -9 主进程 **与 EngineCore 子进程**），"
  echo "               或重跑本脚本并设 STOP_AFTER=1 以触发清理"
  echo "  pid        = $SRV_PID（$PIDFILE）"
fi
echo "=========== 结束 $(date) ==========="
