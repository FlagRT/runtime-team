#!/bin/bash
# MLU590 · T7 消费方腿 **TP=2/EAGER=0 补跑**（客户端观测窗口放宽后定性）
#   背景：首次尝试在该形态上 `TimeoutError: timed out`（第 2 个 embedding 请求），
#         而服务侧 `/v1/models` = 200、同形态 `serve_standard.sh` 冒烟在 SMOKE_TIMEOUT=600 下 PASS
#         ⇒ 怀疑是**消费方客户端 120 s 硬编码观测窗口**不足（TP=1/EAGER=1 实测首请求已 85 s）。
#   本脚本**只放宽客户端观测窗口**（`SERVE_REQ_TIMEOUT=600`，服务侧判据一字未改）⇒ 定性：
#     · 若 PASS ⇒ 该形态服务侧无问题，属观测窗口不足（并把窗口需求写进报告）
#     · 若仍 FAIL ⇒ 该形态确有问题，如实登记
set -u

C=${C:-dc-mlu590-vllm-hliu553}
P=/work/dc_t1re_20261014/prototype
O=/work/dc_t7_20261014
HO=/srv/hliu553/dc_t7_20261014
PORT=${PORT:-8100}
TAG=consumer_tp2_eager0_retry

M=$(docker exec "$C" bash -lc 'ls -d /hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/ 2>/dev/null | head -1')
M=${M%$'\r'}
[ -z "$M" ] && { echo "❌ 容器内取不到模型路径 ⇒ 中止"; exit 2; }

CD=$O/$TAG; HCD=$HO/$TAG; mkdir -p "$HCD"

cleanup() {
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

echo "=== 客户端观测窗口自证 ==="
docker exec "$C" bash -lc "grep -n 'SERVE_REQ_TIMEOUT' $P/runtime/proto/proto_infer_serve.py | head -2"

echo
echo "=== 起服务 TP=2 EAGER=0（STOP_AFTER=0）@ $(date '+%H:%M:%S') ==="
cleanup
docker exec -d -e DC_BACKEND=cambricon -e SERVE_FORM=embed -e DEV=0,1 -e EAGER=0 -e TP=2 \
    -e STOP_AFTER=0 -e READY_BUDGET=900 -e PORT="$PORT" -e SMOKE_TIMEOUT=600 \
    -e MODEL="$M" -e SERVED_NAME=qwen3-embedding-0.6b -e DC_OUT_DIR="$CD" \
    "$C" bash -lc "cd $P && bash scripts/serve_standard.sh"

ready=0
for i in $(seq 1 100); do
    sleep 10
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then ready=1; echo "  就绪：第 $((i*10)) s（http $code）"; break; fi
done
echo "  ready=$ready"

if [ "$ready" = "1" ]; then
    date "+[%H:%M:%S] 跑消费方（SERVE_REQ_TIMEOUT=600）"
    docker exec -e DC_BACKEND=cambricon -e SERVE_PORT="$PORT" -e SERVE_REQ_TIMEOUT=600 "$C" bash -lc \
        "cd $P && python3 -u runtime/proto/proto_infer_serve.py --rounds 5 --out $CD/proto_infer_serve_result.json" \
        > "$HCD/consumer_stdout.log" 2>&1
    echo "  consumer rc=$?"
    grep -aoE "SERVE_LEG_(PASS|FAIL)[^\"]*" "$HCD/consumer_stdout.log" "$HCD/proto_infer_serve_result.json" 2>/dev/null | tail -2 | sed 's/^/    /'
    grep -aE "句/s|p50|区分度|TimeoutError" "$HCD/consumer_stdout.log" 2>/dev/null | tail -5 | sed 's/^/    /'
else
    echo "  !! 未就绪；vllm 日志尾："; tail -12 "$HCD/vllm_serve.log" 2>/dev/null | sed 's/^/    /'
fi

echo
echo "=== 停机与用卡 ==="
cleanup
for i in 1 2 3 4 5 6; do
    sleep 5
    f=$(docker exec -e MLU_VISIBLE_DEVICES=0,1 "$C" bash -lc 'python3 -c "
import torch
print(\"%.1f %.1f\" % (torch.mlu.mem_get_info(0)[0]/2**30, torch.mlu.mem_get_info(1)[0]/2**30))"' 2>/dev/null)
    echo "    t=$((i*5))s dev0/dev1 free(GiB)=$f"
done
# ⚠️ 这里原写成 `rc=$(grep -ac SERVE_LEG_PASS …)` ⇒ 把**命中行数**标成了 rc（日志里出现 `rc=1` 会被误读成失败）。
#    改为显式标签：判据以 `proto_infer_serve_result.json` 的 verdict 为准。
_verdict=$(grep -aoE "SERVE_LEG_(PASS|FAIL)" "$HCD/proto_infer_serve_result.json" 2>/dev/null | tail -1)
printf "%s ready=%s verdict=%s\n" "$TAG" "$ready" "${_verdict:-NA}" >> "$HO/consumer_legs.log"
echo "T7_CONSUMER_TP2_RETRY_DONE"
date "+[%F %T]"
