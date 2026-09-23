#!/usr/bin/env bash
# 910C FlagOS 开发环境一键初始化（在容器内执行）
# 用法：docker exec -it <容器名> bash /workspace/scripts/setup_910c.sh
set -euo pipefail

echo "==== 1/4 环境检查 ===="
python3 -V
python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null || echo "[WARN] torch 未装，请先确认基础镜像"

echo "==== 2/4 torch_npu 移植 ===="
# torch_npu 2.10.0 来源：宿主 /tmp/py311_mirror 或同版本 torch_npu wheel
# 若镜像内已自带则跳过；否则从宿主拷贝：
#   docker cp /tmp/py311_mirror/lib/python3.11/site-packages/torch_npu <容器>:/usr/local/python3.11.15/lib/python3.11/site-packages/
#   docker cp /tmp/py311_mirror/lib/python3.11/site-packages/torch_npu-2.10.0.dist-info <容器>:<同上>
if python3 -c "import torch_npu" 2>/dev/null; then
  echo "torch_npu 已存在：$(python3 -c 'import torch_npu; print(torch_npu.__version__)')"
else
  echo "[SKIP] 容器内无 torch_npu——请按 README 用 docker cp 从宿主移植（镜像内不自带）"
fi

echo "==== 3/4 FlagCX 编译 ===="
cd /workspace/FlagCX
if [ -f build/lib/libflagcx.so ]; then
  echo "FlagCX core 已编译：$(ls -lh build/lib/libflagcx.so | awk '{print $5}')"
else
  make USE_ASCEND=1 -j"$(nproc)"
  echo "FlagCX core 编译完成"
fi

echo "==== 4/4 FlagCX torch 插件安装 ===="
cd /workspace/FlagCX/plugin/torch
FLAGCX_ADAPTOR=ascend pip install -e . --no-build-isolation 2>&1 | tail -1
python3 -c "import flagcx, torch.distributed as dist; assert 'flagcx' in dist.Backend.backend_capability; print('flagcx backend OK')"

echo "==== 全部就绪 ===="
echo "验证：torchrun --nproc_per_node=2 /workspace/dev/benchmarks/test_ag_npu.py"
