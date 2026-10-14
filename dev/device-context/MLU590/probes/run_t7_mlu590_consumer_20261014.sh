#!/bin/bash
# MLU590 · T7 消费方腿（重跑）—— 主腿 TP=1/EAGER=1 + 加跑 TP=2/EAGER=0 作为对照
#   在宿主（Mlu-1）执行。
#   ⚠️ 首跑的失败教训：**消费方的 stdout 必须重定向到证据文件** —— 首跑没重定向，
#      消费方那一段的现场（是服务崩了、还是消费方自己报错）随外层管道一起丢了，
#      只剩一个"没有结果文件"的空目录 ⇒ 无法定性。本脚本一律重定向。
set -u

C=${C:-dc-mlu590-vllm-hliu553}
P=/work/dc_t1re_20261014/prototype
O=/work/dc_t7_20261014
HO=/srv/hliu553/dc_t7_20261014
PORT=${PORT:-8100}

M=$(docker exec "$C" bash -lc 'ls -d /hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/ 2>/dev/null | head -1')
M=${M%$'\r'}
[ -z "$M" ] && { echo "❌ 容器内取不到模型路径 ⇒ 中止"; exit 2; }
echo "MODEL(容器内) = $M"

cleanup() {
    # 见 `run_t7_mlu590_serve_matrix_20261014.sh` 的同名说明：TP>1 的 **`VLLM::Worker_TP*`** 是大写进程名，
    # `pkill -f "[v]llm serve"` 大小写敏感、抓不到 ⇒ 必须按 `VLLM::` 前缀清，残留计数用 `grep -aiE`。
    docker exec "$C" bash -lc 'pkill -f "[s]erve_standard.sh" 2>/dev/null; sleep 1;
        pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null' || true
    local left=99
    for _ in $(seq 1 18); do
        sleep 5
        left=$(docker exec "$C" bash -lc 'ps -eo pid,stat,cmd | grep -aiE "[v]llm|[s]erve_standard" | grep -v defunct | wc -l' 2>/dev/null || echo 99)
        [ "$left" = "0" ] && break
    done
    echo "    残留进程数=$left"
}

leg() {   # $1=tag  $2=TP  $3=EAGER  $4=DEV
    local tag=$1 TPX=$2 EGX=$3 DEVX=$4
    local CD=$O/consumer_$tag HCD=$HO/consumer_$tag
    mkdir -p "$HCD"
    echo
    echo "===== 消费方腿 [$tag]：TP=$TPX EAGER=$EGX DEV=$DEVX @ $(date '+%H:%M:%S') ====="
    cleanup
    docker exec -d -e DC_BACKEND=cambricon -e SERVE_FORM=embed -e DEV="$DEVX" -e EAGER="$EGX" -e TP="$TPX" \
        -e STOP_AFTER=0 -e READY_BUDGET=900 -e PORT="$PORT" -e SMOKE_TIMEOUT=600 \
        -e MODEL="$M" -e SERVED_NAME=qwen3-embedding-0.6b -e DC_OUT_DIR="$CD" \
        "$C" bash -lc "cd $P && bash scripts/serve_standard.sh"

    local ready=0
    for i in $(seq 1 100); do
        sleep 10
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || echo 000)
        if [ "$code" = "200" ]; then ready=1; echo "  就绪：第 $((i*10)) s（http $code）"; break; fi
    done
    echo "  ready=$ready"

    if [ "$ready" = "1" ]; then
        docker exec -e DC_BACKEND=cambricon -e SERVE_PORT="$PORT" "$C" bash -lc \
            "cd $P && python3 -u runtime/proto/proto_infer_serve.py --rounds 5 --out $CD/proto_infer_serve_result.json" \
            > "$HCD/consumer_stdout.log" 2>&1
        echo "  consumer rc=$?"
        grep -aoE "SERVE_LEG_(PASS|FAIL)[^\"]*" "$HCD/consumer_stdout.log" "$HCD/proto_infer_serve_result.json" 2>/dev/null | tail -2 | sed 's/^/    /'
        grep -aE "\"passed\"|\"total\"|\"verdict\"" "$HCD/proto_infer_serve_result.json" 2>/dev/null | tail -3 | sed 's/^/    /'
        grep -aE "句/s|p50|区分度" "$HCD/consumer_stdout.log" 2>/dev/null | tail -4 | sed 's/^/    /'
        # ⚠️ 判据通过 ≠ 进程干净退出：记录消费方与服务的退出痕迹
        grep -acE "Segmentation|SIGSEGV|Fatal" "$HCD/consumer_stdout.log" 2>/dev/null | sed 's/^/    消费方崩溃痕迹行数=/'
        grep -acE "heartbeatMonitor|TCPStore" "$HCD/vllm_serve.log" 2>/dev/null | sed 's/^/    服务停机期 CNCL 心跳噪声行数=/'
    else
        echo "  !! 未在预算内就绪 —— 日志尾："
        tail -15 "$HCD/vllm_serve.log" 2>/dev/null | sed 's/^/    /'
    fi
    printf "%s TP=%s EAGER=%s ready=%s\n" "$tag" "$TPX" "$EGX" "$ready" >> "$HO/consumer_legs.log"
}

: > "$HO/consumer_legs.log"
leg "tp1_eager1" 1 1 "0"       # 主腿：与 10-08 同形态，且与另两家消费方腿同口径
leg "tp2_eager0" 2 0 "0,1"     # 对照：本轮新形态上再验一次

echo
echo "===== 停机与最终用卡 ====="
cleanup
docker exec -e MLU_VISIBLE_DEVICES=0,1 "$C" bash -lc 'python3 -c "
import torch
for i in range(torch.mlu.device_count()):
    f,t = torch.mlu.mem_get_info(i); print(\"    dev%d free=%.1f GiB\" % (i, f/2**30))
"' 2>&1 | tail -4
echo
echo "===== 汇总 ====="
cat "$HO/consumer_legs.log" | sed 's/^/  /'
echo "T7_CONSUMER_DONE"
date "+[%F %T]"
