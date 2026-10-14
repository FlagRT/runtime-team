#!/bin/bash
# MLU590 · T7 服务化 **四形态补全**（TP∈{1,2} × EAGER∈{1,0}）+ 本层消费方腿
#   2026-10-14 ｜ 在宿主（Mlu-1）执行；对应执行单 T7 与审计 A3
#   与 910C（09-29 已补齐四形态）对称：本实例此前只有 `TP=1 + eager` 单形态。
#
# 三条硬前提（都是历史踩坑固化下来的）：
#   ① 服务化必须用 **vLLM 应用镜像容器**（运行时镜像不含 vLLM）⇒ `dc-mlu590-vllm-hliu553`
#   ② 判据只读 `$DC_OUT_DIR/serve_standard.log`（脚本开头 `exec > serve_standard.log`，
#      外层重定向得到 0 字节空文件）
#   ③ MODEL 必须在**容器内**解析（宿主是 /srv/data/hf_cache，容器是 /hf_cache）+ **非空断言**
set -u

C=${C:-dc-mlu590-vllm-hliu553}
P=/work/dc_t1re_20261014/prototype          # 容器内 = 宿主 /srv/hliu553/dc_t1re_20261014/prototype（当前版本）
O=/work/dc_t7_20261014                       # 容器视角
HO=/srv/hliu553/dc_t7_20261014               # 宿主视角（由挂载 /srv/hliu553 → /work 推出，**不得手写**）
PORT=${PORT:-8100}
SMOKE_TIMEOUT=${SMOKE_TIMEOUT:-600}
READY_BUDGET=${READY_BUDGET:-900}

cleanup() {
    # ⚠️ 2026-10-14 实测教训（T7 首跑把两卡各占 89 GiB 没释放）：
    #    TP>1 时 vLLM 会起 **`VLLM::Worker_TP0` / `VLLM::Worker_TP1`** 子进程，**进程名是大写 `VLLM::`**，
    #    而 `pkill -f "[v]llm serve"` 是**大小写敏感**的 ⇒ 只杀掉主进程与 EngineCore，
    #    **Worker 会成为孤儿继续持有设备内存**（cnmon 实测：卡 0/1 各 89351 / 89353 MiB）。
    #    ⇒ 停机必须按 **`VLLM::` 前缀**清，残留计数也必须**大小写不敏感**（`grep -aiE`）。
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

echo "########## [0] 容器启动 / 版本 / 模型解析（容器内）##########"
date "+[%F %T]"
docker ps --filter "name=^${C}$" --format "  {{.Names}}  {{.Status}}"
docker start "$C" >/dev/null 2>&1 && echo "  docker start OK（本容器为我方资产，Exited 5 days）"
sleep 4
docker exec "$C" bash -lc 'python3 -c "import vllm,torch,torch_mlu;print(\"  vllm\",vllm.__version__,\"torch\",torch.__version__,\"torch_mlu\",torch_mlu.__version__)"'
M=$(docker exec "$C" bash -lc 'ls -d /hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/ 2>/dev/null | head -1')
M=${M%$'\r'}
if [ -z "$M" ]; then
    echo "❌ 容器内取不到模型路径 ⇒ 中止（不得退到默认值）"
    docker exec "$C" bash -lc 'ls -l /hf_cache/hub/ 2>&1 | head'
    exit 2
fi
echo "  MODEL(容器内) = $M"
docker exec "$C" bash -lc "test -f '$M/config.json' && echo '  config.json ✓' || echo '  ✗ 缺 config.json'"
echo "  原型版本自证：$(docker exec "$C" bash -lc "sha256sum $P/runtime/backends/base.py | cut -c1-16")（本机 4dca3469d99ebf81）"

echo
echo "########## [0b] 选卡可用性（TP=2 需要两卡；MLU_VISIBLE_DEVICES=0,1）##########"
docker exec -e MLU_VISIBLE_DEVICES=0,1 "$C" bash -lc 'python3 -c "
import torch, torch_mlu
n = torch.mlu.device_count()
print(\"  可见设备数 =\", n)
for i in range(n):
    f, t = torch.mlu.mem_get_info(i)
    print(\"  dev%d free=%.1f GiB / total %.1f GiB\" % (i, f/2**30, t/2**30))
"' 2>&1 | tail -5

mkdir -p "$HO"
: > "$HO/form_matrix.log"

echo
echo "########## [1] 四形态矩阵 ##########"
for form in "1 1" "1 0" "2 1" "2 0"; do
    set -- $form
    TPX=$1; EGX=$2
    if [ "$TPX" = "2" ]; then DEVX="0,1"; else DEVX="0"; fi
    OUTD="$O/tp${TPX}_eager${EGX}"
    HOUTD="$HO/tp${TPX}_eager${EGX}"
    rm -rf "$HOUTD"; mkdir -p "$HOUTD"

    echo
    echo "----- 形态 TP=$TPX EAGER=$EGX (DEV=$DEVX) @ $(date '+%H:%M:%S') -----"
    cleanup
    date "+[%H:%M:%S] start"
    docker exec -e DC_BACKEND=cambricon -e SERVE_FORM=embed -e DEV="$DEVX" -e EAGER="$EGX" \
        -e TP="$TPX" -e STOP_AFTER=1 -e READY_BUDGET="$READY_BUDGET" -e PORT="$PORT" \
        -e SMOKE_TIMEOUT="$SMOKE_TIMEOUT" -e MODEL="$M" -e SERVED_NAME=qwen3-embedding-0.6b \
        -e DC_OUT_DIR="$OUTD" \
        "$C" bash -lc "cd $P && bash scripts/serve_standard.sh" > "$HOUTD/outer.log" 2>&1
    rc=$?
    date "+[%H:%M:%S] end（外层 rc=$rc）"
    V=$(grep -aoE "SERVE_STANDARD_(PASS|FAIL)[^\"]*" "$HOUTD/serve_standard.log" 2>/dev/null | tail -1)
    echo "  verdict = ${V:-（未取到）}"
    grep -aE "维度|范数|就绪|耗时|smoke|ready" "$HOUTD/serve_standard.log" 2>/dev/null | tail -6 | sed 's/^/    /'
    grep -aE "non-default args|tensor-parallel-size|enforce-eager" "$HOUTD/vllm_serve.log" 2>/dev/null | tail -2 | sed 's/^/    [vllm] /'
    printf "TP=%s EAGER=%s rc=%s verdict=%s\n" "$TPX" "$EGX" "$rc" "${V:-NA}" >> "$HO/form_matrix.log"
    if [ -z "${V:-}" ]; then
        echo "  ---- 未通过时的异常尾 ----"
        grep -aE "ERROR|Error|Traceback|RuntimeError|not supported|assert" "$HOUTD/vllm_serve.log" 2>/dev/null | grep -avE "NCCL INFO" | tail -8 | sed 's/^/    /'
    fi
done

echo
echo "########## [2] 本层消费方腿（取 TP=2 + EAGER=0 这个**全新的**形态）##########"
cleanup
CD="$O/consumer"; HCD="$HO/consumer"; mkdir -p "$HCD"
date "+[%H:%M:%S] start serve (detached, TP=2 EAGER=0)"
docker exec -d -e DC_BACKEND=cambricon -e SERVE_FORM=embed -e DEV=0,1 -e EAGER=0 -e TP=2 \
    -e STOP_AFTER=0 -e READY_BUDGET="$READY_BUDGET" -e PORT="$PORT" -e SMOKE_TIMEOUT="$SMOKE_TIMEOUT" \
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
    docker exec -e DC_BACKEND=cambricon -e SERVE_PORT="$PORT" "$C" bash -lc \
        "cd $P && python3 runtime/proto/proto_infer_serve.py --rounds 5 --out $CD/proto_infer_serve_result.json"
    echo "  consumer rc=$?"
    grep -aE "SERVE_LEG_PASS|SERVE_LEG_FAIL|句/s|p50|\"total\"" "$HCD/proto_infer_serve_result.json" 2>/dev/null | tail -6 | sed 's/^/    /'
    grep -aE "句/s|p50|verdict|passed" "$HCD/vllm_serve.log" "$HCD/proto_infer_serve_result.json" 2>/dev/null | tail -6 | sed 's/^/    /'
else
    echo "  !! 服务未在预算内就绪 —— vllm 日志尾："
    tail -20 "$HCD/vllm_serve.log" 2>/dev/null | sed 's/^/    /'
fi

echo
echo "########## [3] 停机与用卡复查 ##########"
cleanup
echo "  用卡（复跑后应回落）："
docker exec -e MLU_VISIBLE_DEVICES=0,1 "$C" bash -lc 'python3 -c "
import torch
for i in range(torch.mlu.device_count()):
    f, t = torch.mlu.mem_get_info(i); print(\"    dev%d free=%.1f GiB\" % (i, f/2**30))
"' 2>&1 | tail -4

echo
echo "########## [4] 四形态汇总 ##########"
cat "$HO/form_matrix.log" | sed 's/^/  /'
echo "T7_MLU590_DONE"
date "+[%F %T]"
