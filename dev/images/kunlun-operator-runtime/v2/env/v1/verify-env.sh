#!/bin/bash
# env/v1 装置自证（kunlun-operator-runtime v2 · 昆仑芯 GEMM compat）
# 在容器内执行：bash /workspace/dev/images/kunlun-operator-runtime/v2/env/v1/verify-env.sh
# 无需 GPU。四项全 PASS 才算环境就绪：
#   1. enable 生效（环境变量 + import 路径）
#   2. compiler 版本锁（安装的 compiler.py SHA == 验收基线 7133710c…）
#   3. diagnostic 副本生成并进程内挂载（安装目录未改动）
#   4. import flag_gems 取到挂载的补丁版源树（/workspace/FlagGems/src）
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source "$HERE/enable-gemm-compat.sh"

PY=${PY:-/root/miniconda/envs/python310_torch29_cuda/bin/python}
"$PY" - <<'EOF'
import os, sys

fails = []

# 1. enable 的环境变量
for k in ("KLX_USE_AUTOTUNE", "XMLIR_MATMUL_FAST_MODE", "XPU_SDNN_PRESERVE_TTXIR_ERRORS"):
    print(f"[1] {k}={os.environ.get(k, '<未设>')}")
    if os.environ.get(k) != "0" and k == "KLX_USE_AUTOTUNE":
        fails.append(k)
    if os.environ.get(k) not in ("0", "1"):
        fails.append(k)

# 2+3. import gemm_compat（内部完成版本锁校验 + 副本生成 + 进程内挂载，
#       SHA 漂移会抛 RuntimeError）
import gemm_compat

# 4. flag_gems 取到补丁版
import flag_gems
where = flag_gems.__file__
print(f"[4] flag_gems -> {where}")
if not where.startswith("/workspace/FlagGems/src"):
    fails.append(f"flag_gems 路径异常: {where}")

if fails:
    print(f"[FAIL] {fails}")
    sys.exit(1)
print("[PASS] env/v1 装置自证 4 项全过（enable / 版本锁 / 副本挂载 / 补丁版 FlagGems）")
EOF
