#!/usr/bin/env bash
# 重建 kunlun-operator-runtime v1 —— 单层 pip 安装，无需回收资产，直接联网构建。
#
# 前置：
#   - 可拉取 harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base
#   - 可访问 resource.flagos.net（flagtree wheel，~3.3GB，建议不经代理直连）
#     与 pypi.tuna.tsinghua.edu.cn（FlagGems 源码构建依赖已在基座内，仅需索引可达）
#   - docker build --network=host（容器内联网安装）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOCKER_BUILDKIT=0 docker build --network=host \
  -f "$HERE/Dockerfile.repro" \
  --build-arg BASE_IMAGE=harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base \
  -t flagrt/kunlun-operator-runtime:1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64 \
  "$HERE"

echo "构建完成。真机验证（需 P800 设备 + XPU 驱动）："
echo "  docker run --rm --net=host --privileged \\"
echo "    --ulimit stack=67108864 --ulimit memlock=-1 --ulimit nofile=120000 --shm-size=32g \\"
echo "    --group-add video --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \\"
echo "    --device=/dev/xpuN --device=/dev/xpuctrl \\"
echo "    -e CUDA_VISIBLE_DEVICES=<空闲卡号,不要设 XPU_EVENT_KL3_ENABLE,见 lock.yaml known_issues> \\"
echo "    flagrt/kunlun-operator-runtime:1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64 \\"
echo "    bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda && \\"
echo "      python3 -c \"import torch,flag_gems,triton;print(torch.__version__,flag_gems.__version__,triton.__version__)\" && \\"
echo "      cd /env/FlagGems/tests && python3 -m pytest -q --mode quick test_norm_ops.py::test_accuracy_rmsnorm'"
echo "详见 REBUILD.md「验证」一节（含 GEMM 已知崩溃、XPU_EVENT_KL3_ENABLE 挂死已知问题）。"
