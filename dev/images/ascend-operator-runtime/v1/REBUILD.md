# 重建 ascend-operator-runtime v1

## 方式

`build.sh <构建上下文目录>` 驱动 `Dockerfile.repro`。产物 `flagrt/ascend-operator-runtime:0.2.0-reproB`，功能等价（非逐字节）。

## 构建上下文需备齐

| 项 | 来源 |
|---|---|
| `Dockerfile` | = 本目录 `Dockerfile.repro`（build.sh 会拷） |
| `wheelhouse/`（36 wheel, 574M） | 从原镜像 `/opt/flagrt/wheelhouse` 回收；校验 `assets/wheelhouse.sha256`；实体见 `ARCHIVE.md` |
| `mpich-4.1.3.tar.gz` | 官方发行版，sha256 见 `assets/mpich-4.1.3.tar.gz.sha256` |
| `src/FlagGems/` | `git archive` 自 `FlagGems @ f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4` |
| `src/Torch-FL/` | `git archive` 自 `PyTorch-Plugin-FL @ af50297463d59ca4bb3aca59f51724afb5f6723a` |
| `FlagGems-DSA-__init__.py` / `requirements-runtime.txt` / `verify_runtime.py` / `patch_triton_ascend_3_2_1.py` | 本目录 `assets/` |
| CPython 3.11.15 | `Dockerfile.repro` 构建期从 python.org 拉取（需 `docker build --network=host`） |
| 基座镜像 | `harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev`（本机 / harbor / 离线包） |

## 与原镜像的差异

- CPython：原镜像是 owner 私有构建上下文的 opaque COPY；此处改为 python.org 源码编译（功能等价，非 bit 一致）。
- 未 squash，多 ~22 层，14.5GB vs 14.4GB。

## 踩坑

1. CPython `make install` 默认 `compileall -j0`（=CPU 数）在高负载共享机死锁 → `Dockerfile.repro` 锁 `COMPILEALL_OPTS="-j 4"`，`make -j8`。
2. 走限流网络下载 `mpich-4.1.3.tar.gz` 易截断 → 在 `docker build --network=host`（容器网络不受限）里拉，或先校验 tar 完整。

## 验证（2026-09-10 实测）

| 判据 | 结果 |
|---|---|
| `pip freeze` vs 原镜像 | **逐行一致**（62 包） |
| torch / torch_fl / flag_gems(`f7ae8e6b`) / triton / triton_ascend 版本 | 全一致 |
| `verify_runtime.py --static` | 通过 |
| 裸容器 import 行为 | 与原镜像一致（`torch.npu` 不可用、`libascend_hal.so` 缺失均为预期，须带 `--device` 才全绿） |
| 用时 | 684s |

基线文件：`assets/provenance/GOLDEN-pipfreeze-operator-runtime.txt`、`reproB-pipfreeze.txt`、`reproB-build-steps.txt`。
