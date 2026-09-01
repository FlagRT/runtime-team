#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# start_vllm_serve_910c.sh — P3 服务化：vllm serve 长驻（Qwen3-4B，TP=1）
# ═══════════════════════════════════════════════════════════════════════════════
# 【作用】拉起 vLLM OpenAI 兼容服务，供 A8（EngineCore 子进程设备句柄）/
#         A9（错误码翻译）/ A10（四态监控 + 五段式恢复）三项 P3 验收使用
# 【用法】容器内：bash <本脚本路径> [port]      # port 默认 8100
# 【纪律】（均来自历史踩坑，勿删）
#   坑 A5：必须 unset VLLM_PLUGINS —— 设 fl 会破坏 platform 选择
#          → RuntimeError: Device string must not be empty
#   坑 A3：必须 DO_NOT_TRACK=1    —— 否则容器内解析 cpuinfo 失败
#   坑 A2：启动前 pkill EngineCore —— 残留子进程会占卡
#   坑 A1：首个请求 attention 极慢（可达 13+ 分钟），验收前必须先预热
# 【输出】日志与 pid 落 raid：/mnt/raid/hliu553/logs_inference/
# ═══════════════════════════════════════════════════════════════════════════════

PORT="${1:-8100}"
MODEL=/mnt/raid/hliu553/models/Qwen3-4B
TP="${TP:-1}"
LOGDIR=/mnt/raid/hliu553/logs_inference
mkdir -p "$LOGDIR"
LOG="$LOGDIR/vllm_serve_qwen3_4b_tp${TP}.log"
PIDFILE="$LOGDIR/vllm_serve_tp${TP}.pid"

# ── 坑 A5：A 线禁用 vllm-plugin-FL ──
unset VLLM_PLUGINS
# ── 坑 A3 ──
export DO_NOT_TRACK=1

# ── 坑 A2：清理残留 EngineCore 子进程（占卡）──
pkill -f "VLLM::EngineCore" 2>/dev/null || true
sleep 2

nohup vllm serve "$MODEL" \
    --served-model-name qwen3-4b \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 4096 \
    > "$LOG" 2>&1 &

SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[start] vllm serve pid=$SRV_PID tp=$TP port=$PORT"
echo "[start] log: $LOG"
echo "[start] 首次加载 + 预热较慢（坑 A1），就绪判定请用 probe_serve_health.py"
