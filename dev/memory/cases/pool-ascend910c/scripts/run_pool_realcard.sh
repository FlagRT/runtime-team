#!/usr/bin/env bash
# pool-ascend910c 容器侧执行体：FlagMemory 池真卡套件 tests/pool/test_pool_realcard.py（11 用例）。
# 由 run.sh 经 docker exec 调用；运行口径按 FM 仓 tests/README 与该文件头部要求：
#   · 显式指到本文件（模块守卫要求 argv 含文件名）
#   · FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS（由 run.sh 经 docker exec -e 注入）
#   · `-o addopts= -m realcard`（FM 默认 addopts 会整体 deselect 真卡用例）
#   · 设备 = 容器内 0 号 die（= TRAIN_DIE_A，FLAGMEMORY_REALCARD_DEVICE=0）
# torch_npu 接管 torch.cuda 兼容面，套件原样可跑（文件头部 npu 先例说明），无需 STACK 变量。
# 外部依赖：pytest 缺失时按需 pip 安装（默认 PyPI；已在样例 run.sh 头部声明）。
set -euo pipefail
cd /workspace/FlagMemory
OUT=/workspace/evidence/pool-ascend910c
mkdir -p "$OUT"
if ! python3 -m pip show pytest >/dev/null 2>&1; then
  echo "[INFO] 容器内无 pytest，pip 安装（run.sh 已声明的外部依赖）"
  python3 -m pip install pytest
fi
# npu torch.cuda compat: bare torch_npu import does NOT take over torch.cuda (torch 2.10);
# fixture is_available()=False -> all 11 skip. Pre-import torch_npu.contrib.transfer_to_npu instead.
python3 -c 'import torch_npu, torch_npu.contrib.transfer_to_npu, sys, pytest; sys.exit(pytest.console_main())' \
  tests/pool/test_pool_realcard.py -o addopts= -m realcard \
  -q --junitxml="$OUT/junit.xml" 2>&1 | tee "$OUT/pytest.log"
rc=${PIPESTATUS[0]}
echo "pytest rc=$rc"
exit "$rc"
