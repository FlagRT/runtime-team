# DeepFM · v1

> 场景：传统深度模型场景对齐模型（与生成式 LLM 场景的 `OneRec-8B` 并列，见 `docs/运行时层原型验证-战略目标-910C.v1.md` 更新说明与 `dev/model-defs/model_list.md`）——**推荐项，本轮阶段目标不作硬性约束，大概率成为下一轮验收标准**。

## 这是什么

标准 DeepFM（Guo et al., IJCAI 2017《DeepFM: A Factorization-Machine based
Neural Network for CTR Prediction》）参考实现：wide（线性一阶项）+ FM（二阶交叉项）
+ deep（MLP 塔）三部分共享一张 embedding 表，sigmoid 输出 CTR 概率。

**代码来源**：不是从零写的，直接复用了主流开源社区方案——
[`rixwew/pytorch-fm`](https://github.com/rixwew/pytorch-fm)（MIT License，1000+
star，非 archived，最近一次更新 2024-04）的 `DeepFactorizationMachineModel` 与
`FeaturesLinear` / `FeaturesEmbedding` / `FactorizationMachine` /
`MultiLayerPerceptron` 四个组件（仅取 DeepFM 用到的部分，上游库还包含
xDeepFM/AFM/DCN/PNN 等其他模型，本目录不需要）。License 见同目录 `LICENSE`。

相对上游的改动（均为行为保持不变的现代化修正，非架构改动）：
- 字段偏移量（field offset）改用 `register_buffer`，而非上游每次 forward 用
  `x.new_tensor(...)` 重新构造 —— 同样的数学结果，且随模型 `.to(device)` 一起搬。
- 去掉了 `numpy` 依赖（上游用 `np.long`，NumPy ≥ 1.24 已移除该别名，会直接报错）。

## 环境无关设计（重点）

这是"其他方向直接拿来用"的前提，写死在设计里，不是承诺：

- **只用稳定的 `torch.nn` 原语**（`Embedding` / `Linear` / `BatchNorm1d` /
  `Dropout` / `Sequential`），不用任何新/实验性 API。
- **不硬编码设备**：模型代码里没有任何 `.cuda()` / `.npu()` / `device="..."`
  字样。设备选择完全是调用方的事——`build.py:pick_device()` 只用
  `hasattr` 探测已导入的 torch 暴露了什么（`torch.cuda` / `torch.npu` /
  `torch.backends.mps`），本文件从不 `import torch_npu` 等厂商包。
  如果调用方在自己的容器里先 `import torch_npu`，`pick_device()` 能顺手
  用上，但 `dev/model-defs/deepfm/` 目录本身对任何芯片 SDK 零依赖。
- **零外部依赖**：除 `torch` 本身，不需要 `numpy`、不需要数据集、不需要联网。
- **可整目录拷走**：`v1/` 下这几个文件互相之间只用同目录 import
  （`build.py` 会把自己所在目录加进 `sys.path`），拷到别的仓库/容器原样能跑。

## 文件

| 文件 | 内容 |
|---|---|
| `model.py` | 模型定义（`DeepFM` 及四个子组件），改自 pytorch-fm，见上方来源说明 |
| `config.py` | `DeepFMConfig` 构建配置（field_dims / embed_dim / mlp_dims / dropout） |
| `build.py` | `build_model()` 构建入口 + `smoke_test()` 随机张量前向反向验证 |
| `requirements.txt` | 唯一依赖：`torch`（宽松版本下限，不锁具体版本） |
| `LICENSE` | 上游 MIT License 原文 + 本目录文件的授权范围说明 |

## 使用方式

```python
import sys
sys.path.insert(0, "dev/model-defs/deepfm/v1")  # 或直接把整个 v1/ 目录拷进自己项目

from config import DeepFMConfig
from model import DeepFM

# field_dims：每个稀疏类别特征的取值个数，按自己的真实特征编码表来填
config = DeepFMConfig(field_dims=[你的_field_dims], embed_dim=16, mlp_dims=(400, 400, 400), dropout=0.2)
model = DeepFM(config.field_dims, config.embed_dim, config.mlp_dims, config.dropout)

# 输入约定：x 是 LongTensor (batch_size, num_fields)，每一列已经是该字段内部的
# 0-indexed 类别 id（不需要自己算跨字段偏移，模型内部处理）
```

或者直接跑最小验证：

```bash
pip install -r dev/model-defs/deepfm/v1/requirements.txt
python dev/model-defs/deepfm/v1/build.py
```

## 已完成的验证（诚实标注范围）

**已验证**：`build.py` 在纯 CPU + torch 2.8.0（`pip install torch`，无
NPU/CUDA 环境）下实测通过——模型可构建（421,452 参数，示例
`field_dims=[1000,2000,500,300,50]`）、前向输出形状正确、无 NaN/Inf、
反向传播与优化器 step 正常执行，3 步 smoke test 全部 PASS。

**未验证 / 明确不在本次范围**：
- 真实数据集上的数值准确性（如 Criteo 数据集的 AUC/LogLoss）——当前
  smoke test 用随机标签，loss 曲线不代表任何拟合效果，纯粹是"跑得通"的证据。
- 昇腾/昆仑芯等具体芯片后端上的实测——本文件设计上应该能直接跑，但
  "设计上能跑"不等于"已经在那台机器上跑过"，请在自己的锁定环境里补一次
  `build.py` 再作为验收证据。
- 与 device-context 统一运行时（`RuntimeBackend`）的接入——按总组决策，
  本模型定义刻意不接入，保持纯 `torch.nn.Module`。

## 版本化

遵循仓库 `dev/images/` 同款约定：`v1/` 目录不可变，架构如有变更（而非措辞修正）
新开 `v2/`，本目录首行加 `superseded_by: v2`。当前生效版本见
`dev/model-defs/model_list.md`。
