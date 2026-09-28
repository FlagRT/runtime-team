#!/bin/bash
# 生成式显存画像矩阵编排(910C)——按 PLAN_generative_profile_test_engineering_20260923.md
# 用法:容器内 bash gen-matrix_910c.sh <MODEL路径> <SERVED_NAME> [阶段]
#   阶段: smoke | A | CE | BD | F | all(默认)
# 前置:容器已起、卡已钉;宿主机另行跑 hbm-sampler(本脚本打印提醒)
set -u
MODEL=${1:?用法: gen-matrix_910c.sh <MODEL> <SERVED_NAME> [stage]}
NAME=${2:?}
STAGE=${3:-all}
PORT=8107
HERE=/workspace/runtime-team/dev/memory/probes/910c
OUT=/workspace/runtime-team/dev/memory/benchmarks/out/gen910c
mkdir -p $OUT

start_srv() {
  DC_BACKEND=ascend DEV=0 PORT=$PORT MODEL=$MODEL SERVED_NAME=$NAME GPU_MEM_UTIL=${GMU:-0.9}     bash /workspace/rt-kistich/dev/device-context/prototype/scripts/serve_standard.sh &
  for i in $(seq 1 60); do curl -sf localhost:$PORT/v1/models >/dev/null && break; sleep 5; done
  curl -sf localhost:$PORT/v1/models >/dev/null || { echo SERVER_NOT_READY; exit 1; }
}
stop_srv() { pkill -f "vllm serve" 2>/dev/null; sleep 8; }
load() { python3 $HERE/gen-load_910c.py --port $PORT --model $NAME --out $OUT "$@"; }

case $STAGE in
  smoke)
    start_srv
    load --tag ${NAME}-smoke --prompt-tokens 32 --max-tokens 8 --concurrency 1 --rounds 1
    stop_srv ;;
  A)
    start_srv
    load --tag ${NAME}-A-idle-probe --prompt-tokens 32 --max-tokens 16 --concurrency 1 --rounds 1 --warmup 1
    # 结算行由 vllm_serve.log 抓取;此处只留服务运行痕迹
    stop_srv ;;
  CE)
    start_srv
    # C: 单请求不同输出长度(KV 随 token 线性增长)
    for o in 64 256 1024; do
      load --tag ${NAME}-C-out$o --prompt-tokens 128 --max-tokens $o --concurrency 1 --rounds 1
    done
    # E: 完成×3轮 + 取消归还
    load --tag ${NAME}-E-done-r3 --prompt-tokens 128 --max-tokens 256 --concurrency 1 --rounds 3
    load --tag ${NAME}-E-cancel --prompt-tokens 128 --max-tokens 4096 --concurrency 2 --rounds 1 --cancel-after 5
    sleep 10
    load --tag ${NAME}-E-after-cancel-probe --prompt-tokens 32 --max-tokens 16 --concurrency 1 --rounds 1 --warmup 0
    stop_srv ;;
  BD)
    start_srv
    # B: 输入长度对照(输出固定)
    for i in 128 1024 2048 3584; do
      load --tag ${NAME}-B-in$i --prompt-tokens $i --max-tokens 64 --concurrency 1 --rounds 1
    done
    # D: 并发对照(输入输出固定)
    for c in 4 16 64; do
      load --tag ${NAME}-D-c$c --prompt-tokens 512 --max-tokens 256 --concurrency $c --rounds 1
    done
    stop_srv ;;
  F)
    # 配置对照:GMU 由外部循环传入,例如 GMU=0.4 bash ... F
    start_srv
    load --tag ${NAME}-F-gmu${GMU:-0.9} --prompt-tokens 512 --max-tokens 256 --concurrency 8 --rounds 1
    stop_srv ;;
  *) echo unknown stage $STAGE; exit 1 ;;
esac
echo "stage $STAGE done; outputs in $OUT"
