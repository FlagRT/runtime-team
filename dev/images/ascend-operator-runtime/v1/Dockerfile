# flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64
#
# 来源：从本机镜像 docker history 还原（id d948410966b0，构建于 2026-09-03T09:29:49Z，
#       Docker 20.10.8 / arm64）。assets/docker-history.txt 为原始逐层记录。
# 边界：BASE_IMAGE 为 official（昇腾 CANN 9.0.0 基座，见 lock.yaml）；本文件内新增层为 custom。
# byte 级重建还需：BASE_IMAGE 可拉取、assets/、wheelhouse/、按 lock.yaml commit 检出的
#       Torch-FL / FlagGems 源码、mpich-4.1.3.tar.gz。缺口见 lock.yaml:gaps。

# --- official：CANN 9.0.0 基座（ubuntu22.04 + python3.12.13 + /usr/local/Ascend + 构建工具 + ATB/ASCEND ENV + ENTRYPOINT） ---
# 注：docker history 不保留 FROM 的镜像名，只余 digest；此处填回已知的真实来源（见 lock.yaml:parent_image / image_list.md）。
ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl@sha256:a36a302204e282411dd46bd3f1edd64f7fce7190020bb4511e8b409ac8d4ba29
FROM ${BASE_IMAGE}

# --- custom：FlagRT operator runtime 层 ---
ARG TORCH_FL_COMMIT=af50297463d59ca4bb3aca59f51724afb5f6723a
ARG FLAG_GEMS_COMMIT=f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4
ARG MPICH_VERSION=4.1.3

LABEL org.opencontainers.image.title="FlagRT Ascend operator runtime" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.base.digest="sha256:a36a302204e282411dd46bd3f1edd64f7fce7190020bb4511e8b409ac8d4ba29" \
      io.flagrt.cann.version="9.0.0" \
      io.flagrt.python.version="3.11.15" \
      io.flagrt.torch.version="2.10.0+cpu" \
      io.flagrt.triton_ascend.version="3.2.1" \
      io.flagrt.mpi.implementation="MPICH" \
      io.flagrt.mpi.version="4.1.3" \
      io.flagrt.hccl_test.collective="all_reduce" \
      io.flagrt.torch_fl.commit="${TORCH_FL_COMMIT}" \
      io.flagrt.flag_gems.commit="${FLAG_GEMS_COMMIT}"

# 运行期用 python3.11.15（基座自带的 3.12.13 保留在 PATH 末尾）
COPY rootfs/usr/local/python3.11.15 /usr/local/python3.11.15
ENV PATH=/usr/local/python3.11.15/bin:/usr/local/mpich/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/python3.11.15/lib:/usr/local/mpich/lib:${LD_LIBRARY_PATH} \
    GEMS_VENDOR=ascend \
    TRITON_ENABLE_TASKQUEUE=false \
    DO_NOT_TRACK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 离线 wheel 源 + 运行时依赖清单（全 pin，见 assets/requirements-runtime.txt）
COPY assets/wheelhouse/ /opt/flagrt/wheelhouse/
COPY assets/requirements-runtime.txt /opt/flagrt/requirements-runtime.txt

# official：MPICH 4.1.3 源码编译（--disable-fortran）
COPY assets/mpich-${MPICH_VERSION}.tar.gz /opt/flagrt/src/mpich-${MPICH_VERSION}.tar.gz
RUN mkdir -p /opt/flagrt/src/mpich \
    && tar -xzf /opt/flagrt/src/mpich-${MPICH_VERSION}.tar.gz -C /opt/flagrt/src/mpich --strip-components=1 \
    && cd /opt/flagrt/src/mpich \
    && ./configure --prefix=/usr/local/mpich --disable-fortran \
    && make -j8 && make install \
    && rm -rf /opt/flagrt/src/mpich /opt/flagrt/src/mpich-${MPICH_VERSION}.tar.gz

# 运行时依赖（torch 2.10.0+cpu 等）；断言绝不含 torch_npu
RUN python3 -m ensurepip --upgrade \
    && python3 -m pip install --no-index --find-links=/opt/flagrt/wheelhouse -r /opt/flagrt/requirements-runtime.txt \
    && ! python3 -m pip list --format=freeze | grep -E '^torch[-_]npu=='

# custom：Torch-FL + FlagGems 源码装（昇腾后端），并打 triton 补丁
COPY src/FlagGems/ /opt/flagrt/src/FlagGems/
COPY src/Torch-FL/ /opt/flagrt/src/Torch-FL/
COPY assets/FlagGems-DSA-__init__.py /opt/flagrt/src/FlagGems/src/flag_gems/fused/DSA/__init__.py
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=0.0.0+${FLAG_GEMS_COMMIT:0:12} \
       python3 -m pip install --no-index --find-links=/opt/flagrt/wheelhouse --no-build-isolation --no-deps /opt/flagrt/src/FlagGems \
    && ACCELERATOR=ascend ASCEND_KERNEL=1 CUDA_KERNEL=0 FLAGGEMS_KERNEL=0 FLAGGEMS_PYTHON=1 FLAGOS_BUILD_JOBS=8 \
       python3 -m pip install --no-index --find-links=/opt/flagrt/wheelhouse --no-build-isolation --no-deps /opt/flagrt/src/Torch-FL \
    && python3 /opt/flagrt/src/Torch-FL/scripts/patch_triton_ascend.py \
    && rm -rf /root/.cache/pip /root/.triton/cache /opt/flagrt/src

# custom：triton-ascend 3.2.1 二次补丁 + hccl_test 编译 + 静态自检（fail-closed）
COPY assets/verify_runtime.py /opt/flagrt/verify_runtime.py
COPY assets/patch_triton_ascend_3_2_1.py /opt/flagrt/patch_triton_ascend_3_2_1.py
RUN python3 /opt/flagrt/patch_triton_ascend_3_2_1.py --triton-path /usr/local/python3.11.15/lib/python3.11/site-packages/triton \
    && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && make -C /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test MPI_HOME=/usr/local/mpich ASCEND_DIR=/usr/local/Ascend/ascend-toolkit/latest \
    && python3 /opt/flagrt/verify_runtime.py --static

WORKDIR /workspace/FlagPerf/base
