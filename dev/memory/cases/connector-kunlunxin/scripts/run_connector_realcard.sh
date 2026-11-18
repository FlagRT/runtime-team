#!/usr/bin/env bash
# connector-kunlunxin 容器侧执行体：xpu KV connector 判据三件（RUNBOOK 口径）。
# 由 run.sh 经 docker exec 调用。运行口径 = FM 仓 tests/tiering/test_xpu_tier_manager_realcard.py
# 文件头与 xpu RUNBOOK 的已验证命令形态：python -m pytest -m realcard -p no:cacheprovider
# （镜像 conda env python310_torch29_cuda 自带 pytest 9.x，无需网络/pip；PYTHONPATH 生效
# FlagMemory，勿 pip install——env/v1 compose 头注已登记该环境事实）。
# 三件分开跑、失败不短路（全量证据落 /tmp/case-evidence/，由 run.sh 经 docker cp 中转）。
# 本脚本由 run.sh 经 docker exec -i bash -s 从 stdin 注入（容器不挂仓根——仓根挂 /workspace 会遮蔽镜像自带 vllm-plugin-FL 源树，见 env/v1 compose 头注）。
set -uo pipefail
cd /workspace/FlagMemory
OUT=/tmp/case-evidence   # 容器内暂存，完毕由 run.sh docker cp 回宿主日志目录
mkdir -p "$OUT"
# 镜像 PATH 不含 conda env 的 bin（`python` 会落到 miniconda base，无 pytest/torch）；
# 判据一律用 env 绝对路径解释器（同 2026-11-15 probe 与 RUNBOOK 实证口径）。
P=/root/miniconda/envs/python310_torch29_cuda/bin/python
$P -c "import flagmemory; print('FM_VERSION', flagmemory.__version__)" | tee "$OUT/fm-version.txt"

run_one() {  # $1=slug，余参 = pytest 参数
  local slug="$1"; shift
  $P -m pytest -m realcard -p no:cacheprovider "$@" -v \
    --junitxml="$OUT/junit-$slug.xml" > "$OUT/pytest-$slug.log" 2>&1
  local rc=$?
  echo "[INFO] $slug: pytest rc=$rc"
  tail -4 "$OUT/pytest-$slug.log"
  return "$rc"
}

rc=0
run_one offload  tests/tiering/test_offload_realcard.py -k xpu   || rc=1
run_one backend  tests/tiering/test_xpu_backend_realcard.py      || rc=1
run_one tiermgr  tests/tiering/test_xpu_tier_manager_realcard.py || rc=1
exit "$rc"
