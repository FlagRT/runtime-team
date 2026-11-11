# 场景对齐模型索引

> 规范见 [`README.md`](README.md)。以下均对应
> `docs/运行时层原型验证-战略目标-910C.v1.md` 更新说明（2026-09-21）：
> **推荐项 / 预研记录，本轮阶段目标不作硬性约束，大概率成为下一轮验收标准**。

| 场景 | 模型 | 类型 | 当前版本 / 来源 | 状态 | 备注 |
|---|---|---|---|---|---|
| 生成式 LLM 场景 | OneRec-8B | 预训练 checkpoint（HuggingFace） | `OpenOneRec/OneRec-8B`，commit `29f95b3da63a4bbc00a1cd3aeee948efb2ff40ff` | **已在 npu1-27 完整下载**（2026-09-21，`hf` CLI 拉取，18/18 文件，含内置完整性校验）；tag 与架构元数据已核实（`Qwen3ForCausalLM`，apache-2.0） | 无自定义代码，直接按 tag 拉取；不建目录 |
| 传统深度模型场景 | DeepFM | 参考模型定义 + 构建代码 | [`deepfm/v1/`](deepfm/v1/) | 已交付：架构定义 + 随机张量 smoke test 实测通过（CPU + torch 2.8.0，421,452 参数，3 步前向反向无 NaN/Inf）；**真实芯片已验证**：910C 单卡（davinci0），训练腿 v2 候选镜像 `flagrt/ascend-operator-runtime-comm:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`，`device=npu`，同样 421,452 参数、3 步前向反向 PASS，未改代码 | 复用主流开源方案 [`rixwew/pytorch-fm`](https://github.com/rixwew/pytorch-fm)（MIT）并做零依赖/设备无关的现代化调整；纯 `torch.nn.Module`，未接入 device-context 运行时抽象；真实数据集数值验证仍为后续项，见 [`deepfm/v1/README.md`](deepfm/v1/README.md) 与 [验证记录](../memory/docs/goals/proto-910c-202609/note_DeepFM_v2候选镜像smoke验证_910c.md) |

## OneRec-8B 拉取方式

直连 `huggingface.co` 在部分机器（如总组协调用的开发机）会被出网代理截断大文件传输；
已验证在 npu1-27 上用 **hf-mirror.com 镜像端点 + 官方 `hf` CLI**（`huggingface_hub` 新版
命令，等价于旧版 `huggingface-cli`）可以正常拉取，自带完整性校验，无需再手工核对 sha256：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download OpenOneRec/OneRec-8B
```

## 已知本地副本（按机器记录，避免重复下载 17GB）

| 机器 | 路径 | 拉取方式 | 拉取人 / 日期 |
|---|---|---|---|
| npu1-27 | `/home/xliu969/.cache/huggingface/hub/models--OpenOneRec--OneRec-8B/snapshots/29f95b3da63a4bbc00a1cd3aeee948efb2ff40ff` | 上方命令 | xliu969 / 2026-09-21 |

> 该路径在 `xliu969` 的用户级 HF 缓存下（非团队共享目录），其他用户若要在
> npu1-27 上直接复用，需先确认自己对该路径有读权限；没有权限或在其他机器上，
> 按上方命令自行拉取一份（镜像下载很快，不必等 owner 开权限）。
> 后续如果哪个方向把它落到团队共享盘（参考 `rag_ljy` 用的 `/mnt/raid/jliu171/models`
> 那种约定），在此表补一行即可，不需要新开文档。

## 后续跟踪项

- DeepFM：**已在 910C 单卡跑通** `build.py`（2026-09-21，见
  [验证记录](../memory/docs/goals/proto-910c-202609/note_DeepFM_v2候选镜像smoke验证_910c.md)）
