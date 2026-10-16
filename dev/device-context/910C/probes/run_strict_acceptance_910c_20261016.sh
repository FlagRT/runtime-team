#!/usr/bin/env bash
# 910C（npu1-27）· 以 up.sh --strict 完成一次真实验收，并复现/证明「image.id 跨机」这条机制级障碍。
#
# 步骤：
#   [1] 两腿 check 现状
#   [2] 复现：baseline.py export 会把 image.id 抄回**制造机** id ⇒ check FAIL（证明这是机制行为，不是我们写错）
#   [3] 对齐：pin_local_image_id.sh 把该字段对齐到**本机** id ⇒ check PASS（幂等、回读校验）
#   [4] check 两腿（应全 PASS）
#   [5] up.sh <train> --strict（应 rc=0 起容器）
#   [6] 容器内跑单入口验收 run_910c_p9_acceptance.sh（应 rc=0 / overall PASS）
#   [7] down（须带 WORKSPACE_ROOT + die 号；ENV-SPEC §4.3 只写了起容器，实测 down 也要）
#   [8] 收尾核验：容器 0 / 残留进程 0 / die 释放
#
# 用法（在 npu1-27 上，仓根执行）：
#   DC_DIE_A=5 DC_DIE_B=0 bash run_strict_acceptance_910c_20261016.sh > <log> 2>&1
set -u

WS=${WS:-/mnt/raid/hliu553/dc-ws}
HO=${HO:-/mnt/raid/hliu553/dc_env_20261016/strict}
ENV_TRAIN=dev/device-context/910C/env/train
ENV_INFER=dev/device-context/910C/env/infer
PIN=dev/device-context/910C/env/pin_local_image_id.sh
C=flagos-dc-train-910c-dev
ACC=/workspace/dev/device-context/910C/probes/run_910c_p9_acceptance.sh
DIE_A=${DC_DIE_A:-5}
DIE_B=${DC_DIE_B:-0}

mkdir -p "$HO/log" "$HO/out"
cd "$WS" || exit 2

# up.sh 与 compose 取同一 WORKSPACE_ROOT；绕开 up.sh 用 docker compose 时须自行导出（ENV-SPEC §4.3）
export WORKSPACE_ROOT="$WS" FLAGRT_WORKSPACE_ROOT="$WS"
export DC_DIE_A="$DIE_A" DC_DIE_B="$DIE_B"

echo "############ 0. 环境自证 ############"
echo "  时间       : $(date '+%F %T %Z')"
echo "  机器       : $(hostname)  WS=$WS"
echo "  compose    : $(docker compose version 2>&1 | head -1)"
echo "  python3    : $(python3 -V 2>&1) / yaml=$(python3 -c 'import yaml;print(yaml.__version__)' 2>&1)"
echo "  die        : A=$DIE_A B=$DIE_B"
echo "  FlagCX HEAD: $(git -C "$WS/FlagCX" rev-parse HEAD 2>/dev/null)"
echo "  挂载镜像 id : $(docker image inspect flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64 --format '{{.Id}}' 2>&1)"
echo

echo "############ 1. 两腿 check（现状）############"
for d in "$ENV_TRAIN" "$ENV_INFER"; do
    echo "----- check $d -----"
    python3 dev/lib/baseline.py check "$d"; echo "  rc=$?"
done
echo

echo "############ 2. 复现：export 会把 image.id 抄回制造机 id ############"
echo "----- export $ENV_TRAIN --require-clean -----"
python3 dev/lib/baseline.py export "$ENV_TRAIN" --require-clean; echo "  export rc=$?"
echo "  export 后 env 基线里的 image.id = $(grep -m1 '^  id: ' $ENV_TRAIN/env.baseline.yaml)"
echo "----- check（期望 FAIL：与制造机 id 比对）-----"
python3 dev/lib/baseline.py check "$ENV_TRAIN"; echo "  rc=$?（1=不一致，符合预期）"
echo

echo "############ 3. 对齐：把 image.id 对齐到本机 id ############"
bash "$PIN" "$ENV_TRAIN"; echo "  pin rc=$?"
echo "  对齐后 env 基线里的 image.id = $(grep -m1 '^  id: ' $ENV_TRAIN/env.baseline.yaml)"
echo "  幂等复跑：" ; bash "$PIN" "$ENV_TRAIN" | tail -3; echo "  pin(2nd) rc=$?"
echo

echo "############ 4. check 两腿（应全 PASS）############"
for d in "$ENV_TRAIN" "$ENV_INFER"; do
    echo "----- check $d -----"
    python3 dev/lib/baseline.py check "$d"; echo "  rc=$?"
done
echo

echo "############ 5. up.sh <train> --strict ############"
echo "--- 起前：清我方同名残留（compose 遇同名会 Conflict 起不来）---"
docker ps -a --format '{{.Names}}|{{.Status}}' | grep -E "^${C}$" || echo "  无同名容器"
docker rm -f "$C" >/dev/null 2>&1 && echo "  已删旧 $C" || true
echo "--- 起前：本机带卡容器映射（昇腾驱动对带卡容器独占）---"
for cc in $(docker ps --format '{{.Names}}'); do
    d=$(docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$cc" 2>/dev/null | tr ' ' '\n' | grep -oE 'davinci[0-9]+$' | sed 's/davinci//' | sort -n | tr '\n' ',')
    [ -n "$d" ] && printf '  %-34s die=%s\n' "$cc" "$d"
done
echo "--- 启动 ---"
FLAGRT_MODELS_DIR="$WS/models" dev/lib/up.sh "$ENV_TRAIN" --strict
echo "  up.sh --strict rc=$?"
sleep 8
# ⚠️ 判据要名副其实：docker ps 输出行是「名字|状态」，用 `-x`（整行匹配）会永不命中
if docker ps --format '{{.Names}}' | grep -qx "${C}"; then
    echo "  [OK] 容器在运行: $(docker ps --format '{{.Names}}|{{.Status}}' | grep -E "^${C}\\|")"
else
    echo "  [FAIL] 容器未在运行（up.sh --strict 之后容器应处于 Up）"
fi
echo

echo "############ 6. 等挂载安装并跑单入口验收 ############"
for i in $(seq 1 60); do
    sleep 10
    if docker logs "$C" 2>&1 | grep -qE "Successfully installed|ERROR:|error: |failed with error"; then break; fi
    printf "  t=%ss 仍在安装/构建…\n" "$((i*10))"
done
echo "  --- compose 实际 command（自证挂载生效方式）---"
docker inspect -f '  cmd={{json .Config.Cmd}}' "$C"
echo "  --- 安装自证 ---"
docker logs "$C" 2>&1 | grep -aE "Successfully installed|error: |undefined symbol" | tail -4
echo "  --- 加载位置自证 ---"
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -lc 'python3 -c "import flagcx,os;print(\"   flagcx ->\", os.path.dirname(flagcx.__file__))"' 2>&1 | tail -2
echo "  --- 单入口验收（两项）---"
docker exec -e ASCEND_RT_VISIBLE_DEVICES=0,1 -e PYTHONDONTWRITEBYTECODE=1 "$C" \
    bash -lc "cd /workspace && bash $ACC" > "$HO/out/acceptance_in_env.log" 2>&1
echo "  acceptance rc=$?"
grep -aE '^=+ |^  [a-z_]+(\+[a-z]+)? +(PASS|FAIL)|^    rc=|^  overall:|^  [a-z_]+ +与' "$HO/out/acceptance_in_env.log" | tail -22
echo

echo "############ 7. down ############"
docker compose -f "$ENV_TRAIN/docker-compose.yml" down
echo "  down rc=$?"
sleep 4
echo

echo "############ 8. 收尾核验 ############"
echo "  我方运行中容器: $(docker ps --format '{{.Names}}' | grep -cE '^flagos-dc-|^dc-' || true)"
echo "  我方残留进程  : $(ps -eo pid,cmd | grep -cE '[t]orchrun|[d]dp_native|[f]lagcx_sync_test' || true)"
echo "  die $DIE_A/$DIE_B 上的进程表（⚠️ 该机为共享机：表里可能有**他人**容器的进程，"
echo "     本方向的判据是「我方容器已 Removed + 我方进程计数 0」，不按卡归因）:"
npu-smi info 2>/dev/null | sed -n '/Process id/,$p' | head -8 | sed 's/^/    /'
echo "  我方容器是否还在: $(docker ps -a --format '{{.Names}}' | grep -cx "$C" || true)（0=已 Removed）"
echo
echo "完成 $(date '+%F %T')"
