#!/bin/bash
# device-context 方向 · 910C 训练腿：新机制端到端实测（2026-10-16）
#
# 目标：在 npu1-27 上把总组的环境基线机制**真正跑一遍**，而不是用等价 docker run 替代。
#   ① 工具自证（docker compose v2 + pyyaml，均为用户级安装、无 sudo）
#   ② baseline.py export --require-clean（生成 env.baseline.yaml）
#   ③ baseline.py show / check
#   ④ dev/lib/up.sh <env> --strict（机制规定的验收启动方式）
#   ⑤ 容器起来后：挂载覆盖的 pip editable 安装自证 → 跑两项验收（单入口）
#
# ⚠️ 已知会命中：本机镜像 id 与基线记录的不一致（跨机事实，image_list.md 已登记
#    "npu1-27 侧为 7028028bb62c"），故 ④ 预期**拒绝启动** —— 这正是要如实登记的结果。
set -u

WS=/mnt/raid/hliu553/dc-ws
HO=/mnt/raid/hliu553/dc_env_20261016
L=$HO/mechanism
ENV_TRAIN=dev/device-context/910C/env/train
ENV_INFER=dev/device-context/910C/env/infer
C=flagos-dc-train-910c-dev
ACC=/workspace/dev/device-context/910C/probes/run_910c_p9_acceptance.sh
export FLAGRT_WORKSPACE_ROOT=$WS
export DC_DIE_A=${DC_DIE_A:-5}
export DC_DIE_B=${DC_DIE_B:-0}
mkdir -p "$L"
cd "$WS" || exit 1

echo "############################################################"
echo "# 910C 新机制端到端实测  $(date '+%F %T')  @$(hostname)"
echo "# WORKSPACE_ROOT=$WS   DC_DIE_A=$DC_DIE_A DC_DIE_B=$DC_DIE_B"
echo "############################################################"

echo
echo "===== [0] 工具自证（用户级安装，无 sudo、不改宿主包）====="
echo "  docker     : $(docker --version)"
echo "  compose    : $(docker compose version 2>&1 | head -1)"
echo "  plugin 位置: $(ls -l ~/.docker/cli-plugins/docker-compose 2>/dev/null | awk '{print $NF}')"
python3 -c "import yaml,sys; print('  pyyaml     :', yaml.__version__, '->', yaml.__file__); print('  python     :', sys.version.split()[0])"
echo "  sudo 未使用: $(sudo -n true 2>&1 | head -1)"

echo
echo "===== [1] export --require-clean（训练腿）====="
python3 dev/lib/baseline.py export "$ENV_TRAIN" --require-clean
echo "  rc=$?"

echo
echo "===== [1b] export --require-clean（推理腿）====="
python3 dev/lib/baseline.py export "$ENV_INFER" --require-clean
echo "  rc=$?"

echo
echo "===== [1c] 生成的 env.baseline.yaml（训练腿，全文）====="
cat "$ENV_TRAIN/env.baseline.yaml"

echo
echo "===== [2] show（训练腿）====="
python3 dev/lib/baseline.py show "$ENV_TRAIN"

echo
echo "===== [3] check（训练腿）====="
python3 dev/lib/baseline.py check "$ENV_TRAIN"
echo "  check rc=$?"

echo
echo "===== [4] up.sh --strict（机制规定的验收启动方式）====="
echo "  本地镜像 id : $(docker inspect -f '{{.Id}}' flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64 2>/dev/null)"
echo "  基线记录 id : $(grep -m1 '^  id:' "$ENV_TRAIN/env.baseline.yaml" 2>/dev/null | sed 's/^  id: *//')"
FLAGRT_MODELS_DIR=$WS/models dev/lib/up.sh "$ENV_TRAIN" --strict
echo "  up --strict rc=$?"

echo
echo "===== [5] 容器状态（--strict 拒绝后应无我方容器）====="
docker ps -a --format '{{.Names}}|{{.Status}}' | grep -E "^${C}$" || echo "  （无 ${C} —— 与 --strict 拒绝启动一致）"

echo
echo "===== [6] 用同一 env 目录、不加 --strict 起容器（只报告不阻止）====="
FLAGRT_MODELS_DIR=$WS/models dev/lib/up.sh "$ENV_TRAIN"
echo "  up rc=$?"
sleep 8
docker ps --format '{{.Names}}|{{.Status}}' | grep -E "^${C}$" || { echo "  ⚠️ 容器未起来，后续步骤不可用"; }

echo
echo "===== [7] 等挂载覆盖的 pip editable 安装完成（compose 的 command 第一步）====="
for i in $(seq 1 60); do
    sleep 10
    if docker logs "$C" 2>&1 | grep -qE "Successfully installed|ERROR:|error: |failed with error"; then break; fi
    printf "  t=%ss 仍在构建…\n" "$((i*10))"
done
docker logs "$C" 2>&1 | grep -aE "Successfully installed|error: |ERROR:|fatal error|undefined symbol" | tail -8 | sed 's/^/    /'
echo "  --- 加载位置自证（应为 /workspace/FlagCX/...）---"
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -lc 'python3 -c "import flagcx,os;print(\"   flagcx ->\", os.path.dirname(flagcx.__file__))"' 2>&1 | tail -2

echo
echo "===== [8] 两项验收（单入口，跑在本机制起的容器里）====="
docker exec -e ASCEND_RT_VISIBLE_DEVICES=0,1 -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -lc "cd /workspace && bash $ACC"
echo "  acceptance rc=$?"

echo
echo "===== [9] .last-check.json（check 的留痕）====="
cat "$ENV_TRAIN/.last-check.json" 2>/dev/null | sed 's/^/  /'

echo
echo "===== [10] 收尾：停止容器（释放 die）====="
cd "$WS" && docker compose -f "$ENV_TRAIN/docker-compose.yml" down 2>&1 | tail -3
echo "  我方运行中容器: $(docker ps --format '{{.Names}}' | grep -cE '^flagos-dc-' || true)"
echo "  我方残留进程  : $(ps -eo pid,cmd | grep -cE '[t]orchrun|[d]dp_native|[f]lagcx_sync_test' || true)"
echo
echo "完成 $(date '+%F %T')"
