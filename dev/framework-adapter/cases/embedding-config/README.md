# embedding 统一配置回归

本case沿用11-10公共DeepFM模型原文、环境、数据种子与容差，改用功能包的schema4、`prepare_policies`和单作用域`FrameworkAdapter`。当前执行结果看STATUS，路径看HANDOFF。

本页包括DeepFM接入/数值定位，以及后续[Qwen embedding验证](#qwen-embedding独立资格及模型三路)。实际[baseline/demo](demo/README.md)与轻量摘要随case保留。

## 配置与审核

`admission.py`生成三种配置：关闭、开启、缺模型审核。权重与索引形状和类型分别记录；原有一元算子仍用`shapes/dtypes`。算子资格用`embedding_records`，模型审核用`workload_reviews`，缺任一资格均不能启用目标；严格开启要求启动审核和实际前反向命中。

模型审核绑定整个已安装适配包的内容身份、公共模型源码和训练设置。不能把旧模型审核复制到新代码上。`candidate`仅是隔离的实验路径，用来生成待审核证据，没有对应的生产配置，也不自动批准。

## 执行

1. 用独立仓`tools/build_local.py`构建并通过安装后测试。
2. `prepare.py --source <11-10 embedding-input04> --wheel <新wheel> --output <新目录>`核原输入SHA，冻结衍生输入。保留有效算子资格，移除衍生目录中的旧模型审核，原输入不动。
3. 重新核VPN、本人权限、存储、PPU v2镜像、设备占用。复用冻结`run.py`，先原生/重复/独立候选，再关闭与缺审核；每轮限时、空卡检查及finally停止容器。
4. 人工核原生重复和候选完整结果，再用`../embedding/review_model.py`生成新模型审核。另建冻结输入，加审核后跑开启；不能原地改已运行的输入。
5. 保留公共baseline/demo、模型和算子命中、全部294项张量比较、容器停止记录。复用`../ppu-models/collect.py`采集及逐文件SHA核验。

完成DeepFM接入检查及数值归属定位后，再独立审核Qwen的`[151669,1024]`权重与`[1,32]`索引。Qwen需要自己的算子资格与模型审核，不复用DeepFM形状资格；其他算子保持原生。DeepFM目标实现尚未获准时继续原生，不以修改上游算子为后续工作的前置条件。

## 11月11日执行结果

统一配置/启动审核代码完成，安装后178项包控制和4项case控制通过。使用PPU v2固定镜像及同一FlagGems，公共DeepFM模型原文不改。

| 路径 | 模型结果 | 目标调用 |
|---|---|---|
| 原生及原生重复 | 各3步，重复294项逐值一致 | 0 |
| 关闭适配 | 3步，294项逐值一致 | 0 |
| 启用但缺模型审核 | 3步，294项逐值一致，workload_unverified | 0 |
| 独立候选 | 第1步通过，第2次参数更新后偏置超差，停止 | 前向4次、反向4次 |

候选首次超阈值项为`step2/state/mlp.mlp.0.bias`，331/400元素超差，最大绝对差0.0011403188109397888；固定容差仍为atol=1e-5、rtol=1e-4。第1步embedding梯度已有3.73e-9的微小差别；第2步该偏置梯度最大差1.19e-7，更新后偏置差变大。这说明仅看loss或单步梯度不足以批准训练组合。差异可能经优化器放大，尚不能据此判定是适配调用、内核还是编译器根因。

两步实际路由均正确：embedding各2次目标前反向，ReLU 3次和sigmoid 1次原生，退出时dispatcher恢复。11-10独立接口的通过记录保留，但不能用其审核覆盖改动后的统一入口。本轮已生成失败模型审核；用实际失败记录回放普通/严格启动，分别得到workload_denied和BLOCKED，2项CPU控制通过，未签发passed审核或运行已批准开启。

5个容器均Exited/Pid0/OOMfalse；63份原件和294份参考张量本地SHA核验通过。[核验摘要](../../docs/evidence-embedding-config-20261111/summary.json)，精确构建、运行、归档路径见[HANDOFF](../../HANDOFF.md#11-11-embedding统一配置回归)。Qwen移植脚本和输入已准备，但未上卡；先做同输入直接调用/独立适配器/统一适配器的最小对照，再决定是否扩大模型。

## 11月11日同输入数值定位

结论：在本次样本中没有发现适配层改变传给算子的数值、类型或选项；不经过适配层的FlagGems直接调用已能复现反向结果波动。失败模型审核继续保留，尚未批准该DeepFM目标组合。

### 实验与结果

1. 在相同原生训练中记录两个Embedding模块的权重、索引和输出梯度；仍使用公共模型原文、相同种子和Adam设置。采样后的3步训练与原冻结参考294项逐值一致。
2. 取前2步×2个表的4组真实输入。分别走直接FlagGems、独立`EmbeddingAdapter`、统一`FrameworkAdapter`，每条路径5次、轮换执行顺序，共60组前反向。每次检查传入provider的权重/索引/梯度值及全部调用选项；线性表的展开梯度连续化后值相同。三路径均复用同一dev4实测wheel，不是加载两个版本的包。
3. 另用完全不导入适配包和公共runtime的[直接复现脚本](direct_replay.py)，对4组样本分别测试原始索引与人工唯一索引，每种做5次原生和5次直接调用，检查反向重复性。

| 检查项 | 结果 |
|---|---|
| 60组目标前向与原生 | 全部逐值一致 |
| 60组provider输入与选项 | 全部一致；展开梯度只做值不变连续化 |
| 三路径反向重复性 | 直接、独立、统一路径都观察到微小波动；最大1.4901161193847656e-8 |
| 独立最小复现：4组原始索引 | 每组直接FlagGems的5次反向均出现非逐值一致；原生5次均一致 |
| 独立最小复现：4组唯一索引 | 两条路径各5次均稳定，且目标与原生逐值一致 |

两个训练步骤的160个索引分别包含18/12个重复项。实际目标来自`flag_gems.ops.embedding`，源码SHA `58ac369551bf30ba6bdc82145985a10914e5f033d67fa5ede06ac7e23b148e9a`，反向使用`tl.atomic_add`累加。这与重复索引下浮点累加顺序影响低位的解释相符；并不直接证明编译器错误。运行已开启PyTorch确定性模式，但该自定义目标仍出现上述波动。

单算子波动约1e-8，本身在原单算子容差内；前轮整模型第2步参数差约1e-3。优化器可能放大小梯度差异，但本轮没有重放完整候选训练的因果链，不能把两个量级直接等同，也不能据此宣布模型精度已修复。11-10单次成功仍是历史事实，后续模型审核需关注重复训练稳定性及更新后参数，而非只看单算子通过。

### 给算子组的确认事项

> PPU v2、FlagGems 4.2.1rc0下，DeepFM实际embedding输入含重复索引时，同输入直接调用`embedding_backward`重复5次有约1.49e-8以内的差异；原生重复稳定，改为唯一索引后目标也稳定。直接调用与两种适配入口均观察到该现象，传入参数和值已核对一致。想确认该实现的反向确定性支持范围，以及训练场景建议采用的精度/重复性验收标准。整模型已有参数超差记录，目前仍保留原生，未修改内核或容差。

### 复现与证据

- 完整诊断：[diagnose.py](diagnose.py)；冻结：[prepare_diagnostic.py](prepare_diagnostic.py)；归档及离线检查：[collect_diagnostic.py](collect_diagnostic.py)、[verify_diagnostic.py](verify_diagnostic.py)。结果摘要：[diagnostic-summary.json](../../docs/evidence-embedding-config-20261111/diagnostic-summary.json)。
- 最小复现只需`direct_replay.py`和实测`captured.pt`，在相同PPU环境、单卡可见且安装固定FlagGems后运行：`GEMS_VENDOR=nvidia python3 direct_replay.py --samples captured.pt --out result.json`。不要在未经核验的共享容器内直接执行；实际baseline/demo和启动命令见[HANDOFF](../../HANDOFF.md#11-11-embedding数值定位)。
- 原件400份（含294份原生张量、60份三路径张量和8份最小复现张量集合）已逐文件大小/SHA核验，并从张量独立重算重复差异。3个本人容器均停止，Pid0、OOMfalse。首版最小脚本因镜像无独立`triton`发行包元数据而在版本读取阶段退出；改读模块版本后用新目录完成，失败原件保留。
- 本轮只新增诊断材料，沿用既有包；4项case CPU控制通过。未改内核、训练容差或模型审核，未提交推送。

## Qwen embedding独立资格及模型三路

2026-11-11：PPU上Qwen3-Embedding-0.6B的embedding限定接入验证完成。复用同一dev4实测wheel，未修改功能包或算子内核。只启用embedding，28层SiLU及其余算子保持原生。

### 模型与接入范围

- 使用完整预训练0.6B权重，595,776,512参数、310个参数张量；PyTorch/Transformers eager、FP32、单卡、batch1、seq32。
- 复用公共训练腿的合成causal-LM验证方式：固定种子生成token序列，AdamW、lr=1e-5、eps=1e-8、weight_decay=0.01、梯度裁剪1.0，每轮3步。权重和config SHA、训练设置、代码、已安装包身份共同纳入模型审核。
- 输入embedding权重`[151669,1024]`、索引`[1,32]`，框架padding_idx=None，contiguous、稠密一阶反向，无max_norm和frequency scaling。其他输入或训练配置需要对应审核。
- 三步合成输入均为32个唯一索引；独立资格另外覆盖重复、全重复和边界索引。此前DeepFM重复索引的数值记录继续保留，不用本轮Qwen结果替代其审核。

### 结果

| 验证 | 结果 | 目标调用 |
|---|---|---|
| 独立前反向：4类索引×3重复 | 12组通过；前向最大差0，反向最大差4.7684e-6，固定atol=1e-5/rtol=1e-4 | 各12次前向、反向 |
| 原生训练及原生重复 | 各3步；重复1863项逐值一致 | 0 |
| 两轮独立候选训练 | 每轮3步、1863项逐值一致；审核后才生成开启配置所需的模型记录 | 每轮前向3次、反向3次 |
| 关闭 | 3步、1863项逐值一致；experimental_disabled | 0 |
| 开启且满足资格 | 3步、1863项逐值一致；通过算子及模型两层审核 | 前向3次、反向3次 |
| 开启但缺模型审核 | 3步、1863项逐值一致；workload_unverified，保留原生 | 0 |

每步对照1项loss、310份梯度和310份更新后参数，共621项×3步=1863项；除原生用于生成参考外，其余6轮均完成完整对照，最大差0。所有作用域退出均恢复dispatcher；监测到实际参数更新，非仅前向调用。

### 复现与材料

1. [prepare_qwen.py](prepare_qwen.py)从冻结的既有Qwen训练探针生成embedding-only入口，不另写模型实现；检查实际权重形状及padding契约。
2. [qwen_batch.py](qwen_batch.py)按qualification → native → candidates → controls → enabled分阶段串行运行，任一失败即停止；每轮重新核空卡、输入SHA、固定镜像并严格启动，结束停止本人容器。
3. [qwen_review.py](qwen_review.py)由人工核结果后执行；算子资格限定本次形状，模型审核要求原生重复及两轮完整候选通过。[freeze_review.py](freeze_review.py)把审核材料放入新的只读输入，不覆盖已执行快照。两轮候选报告内容SHA相同，但来自不同容器和启动时间，均有独立运行记录。
4. [test_qwen.py](test_qwen.py)5项CPU控制通过；[verify_qwen.py](verify_qwen.py)另回放3条真实记录的启动决策，并核验本轮106份轻量原件。资格与模型结果见[摘要](../../docs/evidence-qwen-embedding-20261111/summary.json)，实际[baseline/demo](demo/README.md)，精确路径见[HANDOFF](../../HANDOFF.md#11-11-qwen-embedding资格及模型三路)。

8个本人容器已停止，Pid0、OOMfalse。1863份原生参考张量共14,302,794,504字节保留远端，本地备有SHA清单；轻量原件、审核材料、固定输入均已本地保存。本轮未改容差、未提交推送。
