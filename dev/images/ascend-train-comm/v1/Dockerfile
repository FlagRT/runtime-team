# flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64
#
# 来源：从本机镜像 docker history 还原（id 3b9e08f231d0，构建于 2026-09-03T09:36:48Z，
#       Docker 20.10.8 / arm64）。assets/docker-history.txt 为原始逐层记录。
# 角色：本月训练腿锁定基座，release.stage=candidate。torch_fl 例外线（厂商官方主线是华为昇腾的 torch_npu）。
# 边界：父镜像为 custom（见 ../../ascend-operator-runtime/v1）；本文件仅叠加 FlagCX 一层，全部 custom。
# byte 级重建还需：父镜像、assets/、flagcx wheel、按 commit+patch 构建的 flagcx。缺口见 lock.yaml:gaps。

ARG PARENT_IMAGE=flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64
FROM ${PARENT_IMAGE}

ARG FLAGCX_VERSION=0.13.0
ARG FLAGCX_COMMIT=55eb2ffff6988ae1db5e6ecb325472aecc93d238
ARG FLAGCX_PATCH_SHA256=bc84ea26eee3783085ac140e8792455433299fb0bb2330ec9d2680b6d2b7736d
ARG FLAGCX_EVENT_PATCH_SHA256=524654689fdfe896c2f935f89e884bca4c57e50cd263cc145659e3b8b8ac7f0a
ARG VERIFIER_SHA256=8055c276f531032b8cc5e84fd9bbf0ef992a388771c3e39ac97a451fd5bbb6e5
ARG PARENT_IMAGE_ID=sha256:d948410966b0dfdfaf4f9c95b9b14bdb7c4279cb85ccd7f9157f0fb5d1c6d397

LABEL org.opencontainers.image.title="FlagRT Ascend communication runtime" \
      org.opencontainers.image.version="0.1.3" \
      org.opencontainers.image.base.name="flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64" \
      io.flagrt.parent.image.id="${PARENT_IMAGE_ID}" \
      io.flagrt.cann.version="9.0.0" \
      io.flagrt.torch.version="2.10.0+cpu" \
      io.flagrt.flagcx.version="${FLAGCX_VERSION}" \
      io.flagrt.flagcx.commit="${FLAGCX_COMMIT}" \
      io.flagrt.flagcx.patch.sha256="${FLAGCX_PATCH_SHA256}" \
      io.flagrt.flagcx.event.patch.sha256="${FLAGCX_EVENT_PATCH_SHA256}" \
      io.flagrt.flagcx.adaptor="ascend" \
      io.flagrt.flagcx.torch_backend="flagos" \
      io.flagrt.qualification.verifier.sha256="${VERIFIER_SHA256}" \
      io.flagrt.release.stage="candidate"

# torch_fl 专用运行时开关：禁止 torch_npu 自动加载，通信走 flagos 后端
ENV FLAGCX_TORCH_BACKEND=flagos \
    HCCL_WHITELIST_DISABLE=1 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# FlagCX wheel（离线）+ 校验清单 + fail-closed 自检脚本
COPY assets/wheels/ /opt/flagrt/wheels/
COPY assets/flagcx-wheel.sha256 /opt/flagrt/flagcx-wheel.sha256
COPY assets/flagcx-source-artifacts.sha256 /opt/flagrt/flagcx-source-artifacts.sha256
COPY assets/verify_flagcx_runtime.py /opt/flagrt/verify_flagcx_runtime.py
COPY assets/verify_flagcx_p2p.py /opt/flagrt/verify_flagcx_p2p.py

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && python3 -m pip install --no-index --no-deps /opt/flagrt/wheels/flagcx-${FLAGCX_VERSION}-*.whl \
    && ! python3 -m pip list --format=freeze | grep -E '^torch[-_]npu==' \
    && python3 /opt/flagrt/verify_flagcx_runtime.py --static \
    && rm -rf /opt/flagrt/wheels /root/.cache/pip

WORKDIR /workspace/FlagPerf/base
