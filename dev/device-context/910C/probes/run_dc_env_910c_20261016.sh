#!/bin/bash
# device-context 方向 · 910C 训练腿运行环境的「实质验证」（2026-10-16）
#
# 目的：证明 dev/device-context/910C/env/train/docker-compose.yml 声明的**挂载覆盖**路径在真机上成立：
#   按 compose 的**逐项等价**参数起容器 → 容器启动时用官方命令做 pip editable 安装 →
#   装出来的必须是 /workspace/FlagCX（宿主挂载的那份），而不是镜像内固化版 →
#   然后在同一容器内跑 issue 的两项修复验收，并与镜像固化版做 A/B 对照。
#
# ⚠️ 本脚本**不能替代** `dev/lib/up.sh <env-dir> --strict`：
#   npu1-27 缺 docker compose v2 插件与 pyyaml（见 910C/env/README.md §2），
#   baseline.py / up.sh 在该机上跑不了。本脚本用 docker run 复刻 compose 的等价参数，
#   只用于验证设备映射、挂载生效方式与判据，不产出 env.baseline.yaml。
#
# ⚠️ 与 compose 的**唯一偏差**：额外挂了 $HO/baked:/opt/baked-flagcx:ro，
#   仅在 A 臂（镜像固化版影子）用；compose 里没有这个挂载。
set -u

HO=/mnt/raid/hliu553/dc_env_20261016
WS=/mnt/raid/hliu553/dc-ws
SRC=/mnt/raid/hliu553/dc_p9_20261016            # 10-16 留下的复现脚本
C=flagos-dc-train-910c-dev
IMG=flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64
A=${DC_DIE_A:-5}
B=${DC_DIE_B:-0}
PY=/usr/local/python3.11.15/bin/python3
ACC=/workspace/dev/device-context/910C/probes/run_910c_p9_acceptance.sh

mkdir -p "$HO/baked" "$HO/log" "$HO/out" "$WS/models"
exec > >(tee -a "$HO/log/dc_env_run.log") 2>&1

echo "############################################################"
echo "# device-context 910C 训练腿运行环境 · 实质验证  $(date '+%F %T')"
echo "# die: A=$A B=$B   容器名: $C"
echo "############################################################"

echo
echo "===== [0] 起前查残留 ====="
docker ps -a --format '{{.Names}}|{{.Status}}' | grep -E "^${C}$" && { echo "  删掉同名旧容器"; docker rm -f "$C" >/dev/null; } || echo "  无同名容器"
docker exec -e ASCEND_RT_VISIBLE_DEVICES=0,1 "$C" bash -lc 'ps -eo pid,cmd | grep -E "[v]llm|[t]orchrun|[p]ython3 .*test" | head -3' 2>/dev/null || true
echo "  本机带卡容器（对照，防与他人相交）:"
for c in $(docker ps --format '{{.Names}}'); do
  d=$(docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$c" 2>/dev/null | tr ' ' '\n' | grep -oE 'davinci[0-9]+$' | sed 's/davinci//' | sort -n | tr '\n' ',')
  [ -n "$d" ] && printf "    %-32s die=%s\n" "$c" "$d"
done

echo
echo "===== [1] 放置复现脚本（镜像侧 assets 里那份，逐字不复制）====="
mkdir -p "$WS/dev/images/ascend-train-comm/v3/assets"
cp -f "$SRC/flagcx_sync_test.py" "$WS/dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py"
sha256sum "$WS/dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py"

echo
echo "===== [2] 取出镜像自带 FlagCX（A 臂影子）====="
docker run --rm --entrypoint bash -v "$HO/baked":/out "$IMG" -lc \
  'rm -rf /out/flagcx && cp -a /usr/local/python3.11.15/lib/python3.11/site-packages/flagcx /out/flagcx && ls /out/flagcx | head -5' 2>&1 | tail -6

echo
echo "===== [3] 被测源码自证（宿主挂载的那份）====="
git -C "$WS/FlagCX" rev-parse HEAD | sed 's/^/  FlagCX HEAD = /'
printf "  FlagCX 未提交项 = %s\n" "$(git -C "$WS/FlagCX" status --porcelain | wc -l)"
printf "  默认 DEVICE_LINK 里 cust_opapi 命中 = %s（应为 0）\n" "$(grep -E '^DEVICE_LINK  :=' "$WS/FlagCX/makefiles/ascend.mk" | grep -c cust_opapi)"
printf "  默认 ADAPTOR_FLAG 里 DAG 宏命中 = %s（应为 0）\n" "$(grep -E '^ADAPTOR_FLAG :=' "$WS/FlagCX/makefiles/ascend.mk" | grep -c FLAGCX_USE_ASCEND_DAG)"

echo
echo "===== [4] 起容器（逐项对齐 train/docker-compose.yml）====="
docker run -dit --name "$C" \
  --network host --ipc host --shm-size 512g \
  --cap-add SYS_PTRACE \
  -v "$WS":/workspace \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v "$WS/models":/workspace/models:ro \
  -v "$HO/baked":/opt/baked-flagcx:ro \
  --device "/dev/davinci$A" --device "/dev/davinci$B" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e HCCL_WHITELIST_DISABLE=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e GIT_OPTIONAL_LOCKS=0 \
  -e WORKSPACE_ROOT="$WS" \
  "$IMG" bash -c 'USE_ASCEND=1 python3 -m pip install --no-deps --no-build-isolation -e /workspace/FlagCX && exec sleep infinity'
echo "  容器状态: $(docker inspect -f '{{.State.Status}}' "$C")"

echo
echo "===== [5] 等 pip editable 安装完成（compose 的 command 第一步）====="
ok=0
for i in $(seq 1 100); do
  sleep 15
  if docker logs "$C" 2>&1 | grep -qE "Successfully installed|ERROR:|error: |failed with error"; then ok=1; break; fi
  printf "  t=%ss 仍在构建…\n" "$((i*15))"
done
echo "  --- 安装结果 ---"
docker logs "$C" 2>&1 | grep -aE "Successfully installed|Running setup.py develop|error: |ERROR:|fatal error|undefined symbol|Failed building" | tail -12 | sed 's/^/    /'
echo "  --- 已装位置自证（应为 /workspace/FlagCX/...）---"
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -lc "$PY -c \"import flagcx, os; print('   flagcx ->', os.path.dirname(flagcx.__file__))\"" 2>&1 | tail -2
echo "  --- 与镜像固化版指纹对照 ---"
docker exec "$C" bash -lc "sha256sum /usr/local/python3.11.15/lib/python3.11/site-packages/flagcx/_C*.so /opt/baked-flagcx/flagcx/_C*.so 2>/dev/null; ls -l /usr/local/python3.11.15/lib/python3.11/site-packages/flagcx | head -3" 2>&1 | tail -5

[ "$ok" = "1" ] || { echo; echo "  ⚠️ 安装未在超时内完成或失败 —— 后续判据不可用"; }

# ⚠️ 证据目录必须在**容器可见**的挂载内。第一版把 OUT 传成宿主路径
#    （$HO/out/<arm>）⇒ 容器里创建了同名但属另一文件系统的目录，宿主侧看到的 out/ 是空的
#    （宿主/容器路径混用的老坑，本次为第 10 次）。改用验收脚本的默认 OUT
#    （落在 /workspace/dev/device-context/910C/probes/acceptance_out），再拷到 $WS/acc/<arm>。
run_arm() {  # $1=臂名(B|A)
  local arm=$1 extra_env=""
  [ "$arm" = "A" ] && extra_env="-e PYTHONPATH=/opt/baked-flagcx"
  echo
  echo "===== $arm 臂（$([ "$arm" = B ] && echo '本环境：宿主挂载的 FlagCX（editable）' || echo '镜像固化版影子：PYTHONPATH=/opt/baked-flagcx')）====="
  docker exec "$C" bash -lc "rm -rf /workspace/dev/device-context/910C/probes/acceptance_out"
  # shellcheck disable=SC2086
  docker exec -e ASCEND_RT_VISIBLE_DEVICES=0,1 -e PYTHONDONTWRITEBYTECODE=1 $extra_env \
    "$C" bash -lc "cd /workspace && bash $ACC" > "$HO/log/arm_$arm.log" 2>&1
  local rc=$?
  echo "  rc=$rc"
  mkdir -p "$WS/acc/$arm"
  docker exec "$C" bash -lc "cp -a /workspace/dev/device-context/910C/probes/acceptance_out/. /workspace/acc/$arm/ && rm -rf /workspace/dev/device-context/910C/probes/acceptance_out"
  grep -aE "^  [a-z_]+(\+[a-z]+)? +(PASS|FAIL)|^overall:|flagcx 包:|device_count" "$HO/log/arm_$arm.log" | sed 's/^/    /'
}
run_arm B
run_arm A

echo
echo "===== [8] 汇总（判据行取自 acceptance_out，不经日志截断）====="
mkdir -p "$WS/acc"
for arm in B A; do
  printf "  %-3s sync_test : %s   ddp_native: %s\n" "$arm" \
    "$(grep -aoE 'overall: (PASS|FAIL)' "$WS/acc/$arm/sync_test.out" 2>/dev/null | tail -1)" \
    "$(grep -aoE 'overall: (PASS|FAIL|ERROR)' "$WS/acc/$arm/ddp_test.out" 2>/dev/null | tail -1)"
  printf "  %-3s 加载的 flagcx: %s\n" "$arm" \
    "$(grep -a -m1 'flagcx 包' "$HO/log/arm_$arm.log" 2>/dev/null | sed 's/^ *//')"
done
echo
echo "完成 $(date '+%F %T')  证据在 $HO/out/ 与 $HO/log/dc_env_run.log"
