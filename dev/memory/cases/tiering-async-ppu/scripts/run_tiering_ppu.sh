#!/usr/bin/env bash
# tiering-async-ppu 容器侧执行体：FlagMemory PPU 分层存储异步轨 E2E 真卡用例
# tests/tiering/test_ppu_tier_manager_realcard.py（3 用例：T-8 负例 / 异步轨全链 / destroy 排空）。
# 由 run.sh 经 docker exec 调用；口径按 FM 仓该文件头部与 tests/README：
#   · 显式指到本文件（模块守卫要求 argv 含文件名）
#   · FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS（run.sh 注入）
#   · `-o addopts= -m realcard`（默认 addopts 会整体 deselect 真卡用例）
#   · 卡由 CUDA_VISIBLE_DEVICES 授予（run.sh 注入；PPU 复用 torch.cuda 命名空间）
# SDK 三件套（FM RUNBOOK §1）：宿主镜像 flagrelease vllm0.24 定档镜像的 SDK 环境
# 不固化在镜像 ENV —— 本脚本 source /usr/local/PPU_SDK/envsetup.sh 并置
# UMD_PLATFORM_TYPE=1、HGGC_DRIVER_CANDIDATE=UMD（compose environment 已带，此处
# 显式声明使脚本自含、口径与 RUNBOOK/FM 文件头一致）。
# 外部依赖：pytest 镜像内自带（7.2.0）；缺失时按需 pip 安装（默认 PyPI；run.sh 头部已声明）。
# 注意：envsetup.sh 无参调用（MODE=CUDA，与 FM 文件头/RUNBOOK 口径一致），其内部引用
# $1，须在 `set -u` 生效前 source（否则 unbound variable 即退）。
set -eo pipefail
cd /workspace/FlagMemory
source /usr/local/PPU_SDK/envsetup.sh
set -u
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
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
