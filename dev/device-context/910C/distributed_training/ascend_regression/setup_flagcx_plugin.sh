#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# setup_flagcx_plugin.sh — FlagCX torch 插件一键安装（.pth 注册法）
# ═══════════════════════════════════════════════════════════════════════════════
# 用法：bash setup_flagcx_plugin.sh [/path/to/venv]
#   默认 venv：/root/tf-venv-integration（910C A 线容器惯例）
#   可选 env：FLAGCX_DIR=/workspace/FlagCX（插件与 libflagcx.so 根目录）
# 原理：不 pip install（pyproject wheel 构建会失败），用 .pth 注册插件包路径，
#       libflagcx.so 通过 LD_LIBRARY_PATH 在运行时解析。
# 文档：docs/flagcx_plugin_setup.md

set -euo pipefail

VENV="${1:-/root/tf-venv-integration}"
FLAGCX_DIR="${FLAGCX_DIR:-/workspace/FlagCX}"
PLUGIN_DIR="$FLAGCX_DIR/plugin/torch"
LIB_DIR="$FLAGCX_DIR/build/lib"

echo "== FlagCX 插件安装（.pth 注册法）=="
echo "venv        : $VENV"
echo "plugin dir  : $PLUGIN_DIR"
echo "lib dir     : $LIB_DIR"

# 1) 前置检查
[ -d "$PLUGIN_DIR/flagcx" ] || { echo "FATAL: 插件目录不存在 $PLUGIN_DIR/flagcx"; exit 1; }
[ -f "$LIB_DIR/libflagcx.so" ] || { echo "FATAL: libflagcx.so 不存在 $LIB_DIR/libflagcx.so（先 make USE_ASCEND=1 编译 FlagCX）"; exit 1; }
SO=$(ls "$PLUGIN_DIR/flagcx"/_C.cpython-*.so 2>/dev/null | head -1)
[ -n "$SO" ] || { echo "FATAL: 插件 Python 扩展 _C.cpython-*.so 缺失（需在插件目录编译）"; exit 1; }
echo "py 扩展    : $(basename "$SO")"

# 2) 定位 venv site-packages（.pth 写入位置）
PY_BIN="$VENV/bin/python"
[ -x "$PY_BIN" ] || { echo "FATAL: venv python 不存在 $PY_BIN"; exit 1; }
SITE_PKG=$("$PY_BIN" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
echo "site-packages: $SITE_PKG"

# 3) 写 .pth
PTH="$SITE_PKG/flagcx.pth"
echo "$PLUGIN_DIR" > "$PTH"
echo ".pth 已写入 : $PTH"

# 4) 输出验证命令
cat <<EOF

== 安装完成。运行前必须 export LD_LIBRARY_PATH，然后验证： ==

export LD_LIBRARY_PATH=$LIB_DIR:\$LD_LIBRARY_PATH
cd /tmp && $PY_BIN -c "
import flagcx
import torch.distributed as dist
print('flagcx import ok:', flagcx.__file__)
print('backend registered:', hasattr(dist.Backend, 'FLAGCX') or 'flagcx' in (dist.Backend.backend_list if hasattr(dist.Backend, 'backend_list') else []))
"
# 期望输出：flagcx import ok + backend registered True
EOF
echo "== 完成。详细说明见 docs/flagcx_plugin_setup.md =="
