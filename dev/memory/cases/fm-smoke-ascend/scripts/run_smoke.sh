#!/usr/bin/env bash
# fm-smoke-ascend 容器侧执行体：FlagMemory 池真卡冒烟 3 用例
# （tests/pool/test_pool_ascend_realcard.py：真分配与跨流依赖 / 复用拒绝旧租约 / 借用方保护）。
#
# 运行口径（FM 仓 tests/pool 该文件头的既定纪律，本脚本如实执行）：
#   · 有界预算安全契约——FLAGMEMORY_ASCEND_* 四变量缺一不可（显式使能 + 设备号 +
#     累计分配上限 + 非阻塞查询已验证）；本样例取 BYTE_LIMIT=64MiB（用例要求 ≥2×8MiB）。
#   · `-o addopts= -m realcard`（FM 默认 addopts 会整体 deselect 真卡用例）。
#   · 设备 = 容器内 0 号 = 宿主 TRAIN_DIE_A（run.sh 选卡纪律承担）。
# 外部依赖：pytest 缺失时按需 pip 安装（默认 PyPI；已在样例 run.sh 头部声明）。
set -uo pipefail
cd /workspace/FlagMemory || exit 1
if ! python3 -m pip show pytest >/dev/null 2>&1; then
  echo "[INFO] 容器内无 pytest，pip 安装（run.sh 已声明的外部依赖）"
  python3 -m pip install pytest
fi
export FLAGMEMORY_ASCEND_ENABLE=approved-mapping-and-bounds
export FLAGMEMORY_ASCEND_DEVICE=0
export FLAGMEMORY_ASCEND_BYTE_LIMIT=67108864
export FLAGMEMORY_ASCEND_QUERY_NONBLOCKING=verified
mkdir -p /tmp/fm-smoke-evidence
python3 -c 'import torch_npu, torch_npu.contrib.transfer_to_npu, sys, pytest; sys.exit(pytest.console_main())' \
  tests/pool/test_pool_ascend_realcard.py -o addopts= -m realcard \
  -q --junitxml=/tmp/fm-smoke-evidence/junit.xml 2>&1 | tee /tmp/fm-smoke-evidence/pytest.log
rc=${PIPESTATUS[0]}
echo "pytest rc=$rc"
exit "$rc"
