# 镜像索引

> 所有归档基座镜像的一览。详细 pin 清单以各 `<name>/v<N>/lock.yaml` 为准；本表只做索引。
> "用途 / 使用约束" 见消费方文档（`dev/stack.lock.910c.v1.yaml`、`docs/` 阶段目标文档），本表不涉及。

## ascend-operator-runtime 系列（昇腾工具链 + FlagOS 组件底座）

| 层 | tag / 标识 | 血统 | 关键内置版本 | repro_status | 归档目录 |
|---|---|---|---|---|---|
| 基座 | `harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev` `@sha256:a36a3022…` | ubuntu 22.04 | CANN 9.0.0 · Python 3.12.13 · 昇腾工具链全套 | 🟢 官方/上游（BAAI harbor 可拉 + 本机 + 离线包） | `ascend-operator-runtime/v1/ARCHIVE.md` |
| 基座变体 | `flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev-hostnet` | 同上（同 IMAGE ID，hostnet 变体） | 同上 | 🟢 同上 | 同上 |
| 运行时 | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64` (id `d948410966b0`) | ← 上面的基座 | Python 3.11.15 · torch 2.10.0+cpu · triton 3.5.0 · triton_ascend 3.2.1 · MPICH 4.1.3 · Torch-FL 0.1.0 `@af50297` · FlagGems `@f7ae8e6b` | 🟢 functional-repro | `ascend-operator-runtime/v1/` |

## ascend-operator-runtime-comm 系列

| tag | 血统 | 关键内置版本（在父层之上） | repro_status | 归档目录 |
|---|---|---|---|---|
| `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` (id `3b9e08f231d0`) | ← `ascend-operator-runtime:0.2.0` | FlagCX 0.13.0 `@55eb2ff` +2 patch(`bc84ea26…`/`524654689…`) · ENV `FLAGCX_TORCH_BACKEND=flagos` / `TORCH_DEVICE_BACKEND_AUTOLOAD=0` | 🟢 functional-repro（flagcx `.so` 逐字节一致） | `ascend-train-comm/v1/` |

## ascend-infer-vllm 系列

| tag | 血统 | 关键内置版本 | repro_status | 归档目录 |
|---|---|---|---|---|
| `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` `@sha256:5cf8a2b6…` | 华为官方镜像 | vLLM 0.20.2 · torch_npu · CANN · vllm_ascend | 🟢 官方（公共 registry，digest 锁定，已验证可拉） | `ascend-infer-vllm/v1/` |

## 当前生效版本

| 系列 | 生效 tag | 目录 |
|---|---|---|
| ascend-operator-runtime | `flagrt/ascend-operator-runtime:0.2.0-…` | `ascend-operator-runtime/v1/` |
| ascend-operator-runtime-comm | `flagrt/ascend-operator-runtime-comm:0.1.3-…` | `ascend-train-comm/v1/` |
| ascend-infer-vllm | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | `ascend-infer-vllm/v1/` |
