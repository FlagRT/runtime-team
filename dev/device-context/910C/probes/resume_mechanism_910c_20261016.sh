#!/bin/bash
# 910C · 新机制路径 [5b]–[10] 重跑并全程落盘（2026-10-16）
# 起因：上一轮 [6]–[9] 的输出只回显未落盘，且 `docker compose down` 因该 shell 未设
#       DC_DIE_A 而失败（compose 在解析阶段就要插值 die 号）⇒ 容器残留、die 未释放。
set -u

WS=/mnt/raid/hliu553/dc-ws
HO=/mnt/raid/hliu553/dc_env_20261016
ENV_TRAIN=dev/device-context/910C/env/train
C=flagos-dc-train-910c-dev
ACC=/workspace/dev/device-context/910C/probes/run_910c_p9_acceptance.sh
export FLAGRT_WORKSPACE_ROOT=$WS
export DC_DIE_A=${DC_DIE_A:-5}
export DC_DIE_B=${DC_DIE_B:-0}
L=$HO/mechanism/resume.log
mkdir -p "$HO/mechanism"
cd "$WS" || exit 1
exec > >(tee -a "$L") 2>&1

echo "############################################################"
echo "# 新机制路径 [5b]–[10] 重跑  $(date '+%F %T')  @$(hostname)"
echo "# DC_DIE_A=$DC_DIE_A DC_DIE_B=$DC_DIE_B   WORKSPACE_ROOT=$WS"
echo "############################################################"

echo
echo "===== [5b] 清我方同名残留容器（上一轮只 stop 未删 + 本轮 down 未成功）====="
docker ps -a --format '{{.Names}}|{{.Status}}' | grep -E "^${C}$" | sed 's/^/  /' || true
docker rm -f "$C" >/dev/null 2>&1 && echo "  已删 $C" || echo "  无 $C（无需删）"

echo
echo "===== [6] up.sh <env-dir>（不加 --strict：spec 明示默认只报告、不阻止启动）====="
FLAGRT_MODELS_DIR=$WS/models dev/lib/up.sh "$ENV_TRAIN"
rc=$?
echo "  up rc=$rc"
sleep 8
docker ps --format '{{.Names}}|{{.Status}}' | grep -E "^${C}$" | sed 's/^/  /'

echo
echo "===== [7] 等挂载覆盖的 pip editable 安装（compose 的 command 第一步）====="
for i in $(seq 1 60); do
    sleep 10
    if docker logs "$C" 2>&1 | grep -qE "Successfully installed|ERROR:|error: |failed with error"; then break; fi
    printf "  t=%ss 仍在构建…\n" "$((i*10))"
done
echo "  --- compose 实际使用的 command（自证「挂载生效方式」就是声明的 pip-editable）---"
docker inspect -f '{{json .Config.Cmd}}' "$C" | sed 's/^/  /'
echo "  --- 安装与构建 ---"
docker logs "$C" 2>&1 | grep -aE "Successfully installed|Running setup.py|error: |undefined symbol|fatal error" | tail -6 | sed 's/^/    /'
echo "  --- 加载位置自证（应为 /workspace/FlagCX/...，不是镜像内 site-packages）---"
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -lc \
  'python3 -c "import flagcx,os,glob,hashlib
p=os.path.dirname(flagcx.__file__); print(\"   flagcx ->\", p)
[print(\"    \", os.path.basename(f), hashlib.sha256(open(f,\"rb\").read()).hexdigest()[:16]) for f in sorted(glob.glob(p+\"/*.so\"))]"' 2>&1 | tail -4

echo
echo "===== [8] 两项验收（单入口，跑在本机制起的容器里）====="
docker exec -e ASCEND_RT_VISIBLE_DEVICES=0,1 -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -lc "cd /workspace && bash $ACC"
echo "  acceptance rc=$?"

echo
echo "===== [9] .last-check.json（check 的留痕）====="
cat "$ENV_TRAIN/.last-check.json" 2>/dev/null | sed 's/^/  /'

echo
echo "===== [10] 收尾：docker compose down（本 shell 已设 DC_DIE_A/B）====="
docker compose -f "$ENV_TRAIN/docker-compose.yml" down
echo "  down rc=$?"
sleep 5
echo "  我方运行中容器: $(docker ps --format '{{.Names}}' | grep -cE '^flagos-dc-|^dc-')"
echo "  我方残留进程  : $(ps -eo pid,cmd | grep -cE '[t]orchrun|[d]dp_native|[f]lagcx_sync_test')"
echo
echo "完成 $(date '+%F %T')"
