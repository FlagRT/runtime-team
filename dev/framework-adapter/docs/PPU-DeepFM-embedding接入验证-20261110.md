# PPU DeepFM：embedding接入与安全准入

日期：2026-11-10。顾宬方向本地候选，尚未提交或推送。

## 这次做什么

在已有SiLU、sigmoid、ReLU之外，新增embedding接入。先用DeepFM两个真实embedding形状验证，再在原公共模型中仅替换embedding；其余算子保持原生。

embedding需要同时检查“权重矩阵”和“索引张量”，不能沿用一元算子只审核一个shape的方式。这次新增两输入资格记录、调用前条件检查、前反向桥接和退出恢复。DeepFM求和产生的展开梯度先作值不变的连续化，再调用已有内核；没有修改算子实现或精度阈值。

## 验证结果

| 检查 | 覆盖 | 结果 |
|---|---|---|
| 独立算子资格 | 两组FP32权重×唯一/重复/全相同/首尾边界索引×3重复 | 24组前反向通过；前向差0，反向相对PPU原生最大差9.54e-6，CPU对照也通过 |
| 单算子接入与回退 | 两组形状×关闭/开启/缺资格 | 最终包6组通过；开启真实命中目标前反向，其余0调用；退出后恢复原生 |
| DeepFM原生参考与重复 | 各3步，含Adam更新 | 重复294项逐值一致 |
| DeepFM独立候选 | 仅替embedding，3步 | 294项通过，最大绝对差1.49e-8；6次目标前向、6次反向，每步1次梯度布局复制 |
| DeepFM关闭 | 3步 | 294项逐值一致，目标0调用 |
| DeepFM缺模型审核 | 3步 | 294项逐值一致，目标0调用 |
| DeepFM审核后开启 | 3步 | 294项通过，最大绝对差7.45e-9；目标前向6次、反向6次 |
| 本地控制 | 安装包控制 / case检查器 | 166项 / 3项通过 |

模型比较包括预测、loss、17个参数梯度、28个model state及Adam state。使用固定合成类别输入和二分类标签、随机初始化、batch32，原公共Dropout/BatchNorm不变。目标路径容差为既有`atol=1e-5, rtol=1e-4`；原生重复、关闭、缺审核要求逐值一致。短训练用于验证接入，不评估CTR业务效果。

## 接入边界与实现选择

- DeepFM权重为FP32 `[3850,16]`、`[3850,1]`；同卡int64索引`[32,5]`。暂限无padding、无max_norm、无频次缩放、稠密一阶梯度。越界索引不会送入不带边界检查的目标内核。
- 使用FlagGems公开`embedding`及`embedding_backward`函数，作用域拦截`F.embedding`，覆盖`nn.Embedding`。目标调用异常停止，不原地重算。未准入且原生合法时保留原生。
- 固定FlagOP版本`2591d081`的PPU实现是原生`index_select`/`embedding_backward`委托。本轮选择独立FlagGems内核，未使用FlagOP的全局注册入口。
- 模型开关由case配置与既有WorkloadReview绑定。多输入接口尚未并入公共一元算子CLI；后续统一配置时保留这套权重、索引和语义约束。
- 本次仅替embedding，不改变此前PPU sigmoid/SiLU模型失败记录。Qwen和OneRec需按各自embedding形状、精度重新资格验证后扩展。

## 环境、制品与复现

- PPU ZW810E，公共v2镜像ID `sha256:1ba7ead15378160ba9aa1dafcfb77686de6b16732590ffab062bab19c60c32b8`；驱动1.5.1-1d747a，torch2.10.0，FlagTree0.7.0+ppu.gite223f42c，Triton3.6.0。冻结公共runtime0.4.0。
- FlagGems wheel SHA `4b0ff0534346170950bda66dc74c48d9473ae00420dfb981c6aa1777ce21512f`；最终适配包dev4 wheel SHA `228763f46b2e0d10ee91b14ee4f92a9b2e8b378bf68245efde1ec5722353d12a`。
- [case、审核及执行说明](../cases/embedding/README.md)；完整原件与不可变输入位置见[HANDOFF](../HANDOFF.md)。每轮保存baseline/demo、输入SHA、真实命中、数值比较和容器停止回执。
- 资格、模型审核均由看完证据后的独立命令产生，不由训练runner自行批准。input04相对input03只增加模型审核证据，执行源码和wheel一致。

收尾：[9轮证据核验摘要](evidence-embedding-20261110/summary.json)。113份轻量原件逐文件SHA核验、294份原生张量备份核验通过；9个本人容器全部停止，Pid0、OOMfalse。没有修改共享镜像、算子源码或其他人的容器。
