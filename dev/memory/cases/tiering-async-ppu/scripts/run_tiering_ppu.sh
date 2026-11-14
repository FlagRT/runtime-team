#!/usr/bin/env bash
# tiering-async-ppu 容器侧执行体：FlagMemory PPU 分层存储异步轨 E2E 真卡用例
# tests/tiering/test_ppu_tier_manager_realcard.py（3 用例：T-8 负例 / 异步轨全链 / destroy 排空）。
# 由 run.sh 经 docker exec 调用；口径按 FM 仓该文件头部与 tests/README：
#   · 显式指到本文件（模块守卫要求 argv 含文件名）
#   · FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS（run.sh 注入）
#   · `-o addopts= -m realcard`（默认 addopts 会整体 deselect 真卡用例）
#   · 卡由 CUDA_VISIBLE_DEVICES 授予（run.sh 注入；PPU 复用 torch.cuda 命名空间）
# SDK 环境变量由镜像 ENV 固化（thead-ppu-runtime v2），无须 envsetup。
# 外部依赖：pytest 缺失时按需 pip 安装（默认 PyPI；已在样例 run.sh 头部声明）。
set -euo pipefail
cd /workspace/FlagMemory
OUT=/workspace/evidence/tiering-async-ppu
mkdir -p "$OUT"
if ! python3 -m pip show pytest >/dev/null 2>&1; then
  echo "[INFO] 容器内无 pytest，pip 安装（run.sh 已声明的外部依赖）"
  python3 -m pip install pytest
fi
python3 -m pytest tests/tiering/test_ppu_tier_manager_realcard.py -o addopts= -m realcard \
  -v --junitxml="$OUT/junit.xml" 2>&1 | tee "$OUT/pytest.log"
rc=${PIPESTATUS[0]}
echo "pytest rc=$rc"
exit "$rc"
