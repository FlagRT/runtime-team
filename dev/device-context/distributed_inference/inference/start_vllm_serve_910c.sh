#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# start_vllm_serve_910c.sh — 推理服务启动（D10/D11 集成版，2026-09-02 升级）
# ═══════════════════════════════════════════════════════════════════════════════
# 【升级内容】
#   · D10 集成：serve 以 python -c 包装器启动，错误码翻译（conformance/errors.py）
#     挂接到 vLLM 兜底异常路径 —— 所有未捕获异常先过 translate_error 再走原逻辑
#   · D11 集成：并行启动 device_state_monitor 独立进程，周期探活 + 四态查询，
#     状态变化打印事件并落 JSON
# 【纪律】（均来自历史踩坑，勿删）
#   坑 A5：unset VLLM_PLUGINS；坑 A3：DO_NOT_TRACK=1；坑 A2：pkill EngineCore 残留
#   坑 A1：首个请求预热慢
# 【用法】容器内：bash <本脚本路径> [port]
# 【输出】日志与 pid 落 raid：/mnt/raid/hliu553/logs_inference/
# ═══════════════════════════════════════════════════════════════════════════════

PORT="${1:-8100}"
MODEL=/mnt/raid/hliu553/models/Qwen3-4B
TP="${TP:-1}"
INFER_DIR=/mnt/raid/hliu553/runtime-team/dev/device-context/benchmarks/inference
LOGDIR=/mnt/raid/hliu553/logs_inference
mkdir -p "$LOGDIR"
LOG="$LOGDIR/vllm_serve_qwen3_4b_tp${TP}.log"
PIDFILE="$LOGDIR/vllm_serve_tp${TP}.pid"
MONLOG="$LOGDIR/device_state_monitor.log"
MONOUT="$LOGDIR/device_state_monitor.json"

# ── 坑 A5：A 线禁用 vllm-plugin-FL ──
unset VLLM_PLUGINS
# ── 坑 A3 ──
export DO_NOT_TRACK=1
# ── 让 python -c 能找到 inject_error_translation / device_state_monitor ──
export PYTHONPATH="$INFER_DIR:$PYTHONPATH"

# ── 坑 A2：清理残留 EngineCore 子进程（占卡）；变量拼接避免 pkill 自杀 ──
P=VLLM::Engin
pkill -f "${P}eCore" 2>/dev/null || true
pkill -f "device_state_monitor" 2>/dev/null || true
sleep 2

# ── D10 集成：以包装器方式启动 serve ──
nohup python3 -c "import inject_error_translation; inject_error_translation.serve()" serve "$MODEL" \
    --served-model-name qwen3-4b \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 4096 \
    > "$LOG" 2>&1 &

SRV_PID=$!
echo "$SRV_PID" > "$PIDFILE"
echo "[start] vllm serve (D10 集成) pid=$SRV_PID tp=$TP port=$PORT"

# ── D11 集成：设备状态监控（并行）──
if [ "${MONITOR:-1}" = "1" ]; then
    nohup python3 "$INFER_DIR/device_state_monitor.py" \
        --ordinal 0 --interval 5 \
        --out "$MONOUT" \
        > "$MONLOG" 2>&1 &
    echo "[start] device state monitor pid=$! → $MONLOG"
fi

echo "[start] 首次加载 + 预热较慢（坑 A1），就绪判定请用 probe_serve_health.py"
