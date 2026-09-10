# 镜像索引

> 所有归档基座镜像的一览。详细 pin 清单以各 `<name>/v<N>/lock.yaml` 为准；本表只做索引。
> "用途 / 使用约束" 见消费方文档（`dev/stack.lock.910c.v1.yaml`、`docs/` 阶段目标文档），本表不涉及。
> 本文档里"官方"必须带主体名，三个主体（**华为昇腾官方** / **BAAI·FlagTree 官方** / **BAAI 内部（非发布物）**）的定义见 `README.md` §4.0。

## 版本线：FlagTree ascend3.5

> **BAAI·FlagTree 官方指南**（非华为）：**FlagTree ascend 用户手册** <https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ascend>（分 ascend3.5 / ascend3.2 两条线，各自的 CANN / torch / triton / 基座镜像 / FlagTree 分支见手册）。

本目录归档的三个系列整套栈均落在该手册的 **ascend3.5** 线：

| 轴 | 本组镜像 | FlagTree ascend3.5 | FlagTree ascend3.2 |
|---|---|---|---|
| CANN | 9.0.0 | 9.0.0 | 8.5.0 |
| Python | 3.11 | 3.11 | 3.11 |
| torch (910C) | 2.10.0 | 2.10.0 | 2.6.0 |
| triton | 3.5.0 | 3.5.x | 3.2.x |
| vLLM | 0.20.2 | 0.20.2 | — |

- `triton_ascend 3.2.1`（下表各系列都列了）是**华为昇腾 Triton 插件** `triton-ascend` 的版本号，与 `triton` 是两个独立包。它的 wheel 从 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（即 ascend3.5 / vLLM 0.20.2 镜像）拷出，`import` 时自报 `3.2.0`（已知版本串差异）。**它不随 `triton` 或 FlagTree 线号推进，不代表 ascend3.2 线。**
- **未按 BAAI·FlagTree 官方手册的构建路径**：手册要求用 FlagTree 3.5 线基座镜像 + checkout `triton_v3.5.x` + `FLAGTREE_BACKEND=ascend` 自编译。本组镜像不含 FlagTree 构建物 —— 推理腿直接用**华为昇腾官方** `vllm-ascend`；训练腿用 **BAAI 内部** CANN 9.0.0 底座 + pip 装上游 `triton==3.5.0` + **华为昇腾官方** `triton-ascend==3.2.1`（wheel）+ FlagOS 组件（Torch-FL / FlagGems / FlagCX）。版本纪元对齐 ascend3.5，构建方式自成一路。
- 全组通用 dev 底座（`VERSIONS.md` §1 记为 FlagTree 分支 `triton_v3.2.x`）是另一套东西，不用于原型结论性验证。**FlagTree fork 分支本原型不需要变更**（原型镜像不含 FlagTree 构建物）；日后若要在 ascend3.5 镜像里引入 FlagTree 构建物，其 fork 须先从 `triton_v3.2.x` 切到 `triton_v3.5.x`（手册规定 3.5 线配该分支），由编译层 / FlagTree 维护者决定。

## ascend-operator-runtime 系列（昇腾工具链 + FlagOS 组件底座）

| 层 | tag / 标识 | 血统 | 关键内置版本 | repro_status | 归档目录 |
|---|---|---|---|---|---|
| 基座 | `harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev` `@sha256:a36a3022…` | ubuntu 22.04 | CANN 9.0.0（**华为昇腾官方** toolkit 版本）· Python 3.12.13 · 昇腾工具链全套 | 🟢 已归档（镜像本体为 **BAAI 内部**手工构建底座，非华为发布物；BAAI harbor 可拉 + 本机 + 离线包，digest 锁定） | `ascend-operator-runtime/v1/ARCHIVE.md` |
| 基座变体 | `flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev-hostnet` | 同上（同 IMAGE ID，hostnet 变体） | 同上 | 🟢 同上 | 同上 |
| 运行时 | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64` (id `d948410966b0`) | ← 上面的基座 | Python 3.11.15 · torch 2.10.0+cpu · triton 3.5.0 · triton_ascend 3.2.1 · MPICH 4.1.3 · Torch-FL 0.1.0 `@af50297` · FlagGems `@f7ae8e6b` | 🟢 functional-repro | `ascend-operator-runtime/v1/` |

## ascend-operator-runtime-comm 系列

| tag | 血统 | 关键内置版本（在父层之上） | repro_status | 归档目录 |
|---|---|---|---|---|
| `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` (id `3b9e08f231d0`) | ← `ascend-operator-runtime:0.2.0` | FlagCX 0.13.0 `@55eb2ff` +2 patch(`bc84ea26…`/`524654689…`) · ENV `FLAGCX_TORCH_BACKEND=flagos` / `TORCH_DEVICE_BACKEND_AUTOLOAD=0` | 🟢 functional-repro（flagcx `.so` 逐字节一致） | `ascend-train-comm/v1/` |

## ascend-infer-vllm 系列

| tag | 血统 | 关键内置版本 | repro_status | 归档目录 |
|---|---|---|---|---|
| `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` `@sha256:5cf8a2b6…` | **华为昇腾官方**镜像 | vLLM 0.20.2 · torch_npu · CANN · vllm_ascend | 🟢 华为昇腾官方（公共 registry，digest 锁定，已验证可拉） | `ascend-infer-vllm/v1/` |

## 当前生效版本

| 系列 | 生效 tag | 目录 |
|---|---|---|
| ascend-operator-runtime | `flagrt/ascend-operator-runtime:0.2.0-…` | `ascend-operator-runtime/v1/` |
| ascend-operator-runtime-comm | `flagrt/ascend-operator-runtime-comm:0.1.3-…` | `ascend-train-comm/v1/` |
| ascend-infer-vllm | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | `ascend-infer-vllm/v1/` |
