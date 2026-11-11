# framework-adapter 当前状态

更新：2026-11-11，PPU Qwen0.6B embedding独立资格及关闭/开启/缺审核三路完成，三路各3步、1863项逐值一致。DeepFM数值失败审核继续保留；下一步准备softmax契约及验证。本期发布dev4源码候选78358ab及个人分支证据，公共dev-1.0本次未更新。顾宬 / cgu135。

周报与对外证据统一见[阶段进展与验证入口](docs/阶段进展与验证入口-20261111.md)。发布核验：两次构建各178项包控制、36项集成控制及3条实际记录启动回放通过，16个运行时文件与Qwen实测wheel相同。版本锁见package.lock.json；以下日期条目保留当时实验收尾状态，“未推送/待测”等历史字样不替代本页当前发布结论。

## 本期完成

- 11-11 Qwen embedding接入：PPU完整0.6B FP32、batch1/seq32，独立前反向12组通过；原生重复、两轮候选及关闭/开启/缺模型审核各3步、各1863项逐值一致。开启目标前反向各3次，其余两路目标0调用；其他算子保持原生，所有退出恢复。5项CPU接线控制及3条实际记录启动决策通过。106份轻量原件核验，1863份原生张量保留远端并备SHA清单，8容器停止。见[配置、方法与结果](cases/embedding-config/README.md#qwen-embedding独立资格及模型三路)、[摘要](docs/evidence-qwen-embedding-20261111/summary.json)、[baseline/demo](cases/embedding-config/demo/README.md)。沿用实测dev4包，未改内核/容差，未推送。

- 11-11 embedding数值定位：采样后原生训练294项仍逐值一致；4组真实输入×3路径×5次共60组，前向一致、provider输入/选项一致，直接调用与两种适配入口都观察到反向微小波动，最大1.49e-8。无适配包依赖的最小复现确认：原始重复索引下直接目标波动、原生稳定；唯一索引下双方稳定且一致。与目标原子累加实现相符，尚不强判编译器根因或完整模型放大因果链。保留DeepFM失败审核，不改内核/容差。400份原件核验，3个容器停止。见[定位与对接说明](cases/embedding-config/README.md#11月11日同输入数值定位)、[摘要](docs/evidence-embedding-config-20261111/diagnostic-summary.json)；本地未推。

- 11-11 统一embedding入口：新增schema4多输入配置、单作用域FrameworkAdapter及统一训练报告，旧接口兼容；178项安装包控制、4项case控制通过。PPU DeepFM原生/重复/关闭/缺模型审核各3步，后三组各294项逐值一致、目标0调用；独立候选实际命中4次前向/4次反向，在第2次更新后mlp.mlp.0.bias的331/400元素超差，最大1.1403e-3，停止并登记失败审核。失败审核的普通/严格启动2项CPU控制通过；未批准开启。63份原件及294份参考张量已校验备份，5个容器停止。见[实现、结果与后续](cases/embedding-config/README.md)、[核验摘要](docs/evidence-embedding-config-20261111/summary.json)。Qwen输入仅本地准备，待先区分调用路径影响与目标数值重复性；未推送。

- 11-10 embedding扩展：新增权重/索引联合资格、F.embedding作用域库调用和稠密反向桥接；支持DeepFM展开梯度的值不变连续化。PPU两个真实权重形状×4类索引×3重复，24组独立前反向通过；最终包关闭/开启/缺算子资格6组通过。DeepFM原生、重复、候选、关闭、缺模型审核、审核后开启各3步；5组对照各294项通过，候选/开启最大差1.49e-8/7.45e-9，各6次真实目标前向/反向，关闭和缺审核目标0调用。166项包控制、3项case控制通过；9个容器均停止，113份原件及294份原生张量已校验备份。见[报告](docs/PPU-DeepFM-embedding接入验证-20261110.md)、[case](cases/embedding/README.md)。多输入配置暂在独立case，待并入公共配置；本地未提交推送。

- 11-10 三模型清单与ReLU扩展：PPU DeepFM/Qwen0.6B/OneRec-8B各完成1步原生训练采集，前向实际ATen入口19/43/44种，对照固定FlagGems与FlagOP源码。新增ReLU前反向接入；单算子资格3次及关闭/开启/缺资格3路通过。DeepFM原生、重复、候选、关闭、开启、缺模型审核各3步；非原生参考的5组各294项比较，最大差0，候选/开启各9次目标前向和9次反向，另外两路目标0调用，sigmoid保持原生。安装包152项CPU控制、清单/检查器7项通过。首次检查器误将原生sigmoid判错，修复后用新输入重建参考，失败轨迹保留；14个本人容器均停止，182文件证据校验通过。见[报告](docs/三模型算子清单与ReLU接入验证-20261110.md)、[完整清单](docs/evidence-operator-inventory-20261110/算子清单-20261110.md)。本地未提交推送。

- 11-10 PPU Qwen/OneRec补测完成：两模型SiLU形状资格各3次通过；原生/重复/关闭/缺审核各3步完成，后三路分别1863/2397项逐值一致、目标0调用。候选第1步真实命中SiLU前反向各28/36次，分别在down_proj.weight和embedding梯度超差（12/28460元素，整个失败张量最大绝对差3.3569e-4/0.28125），均在更新参数前停止，退出恢复。两份失败审核及各2项CPU拒绝控制已记录，不批准这两个模型配置使用目标实现。见[报告](docs/PPU两模型接入与安全准入验证-20261110.md)、[case](cases/ppu-models/README.md)。未改内核或容差，未推送。

- 11-10 PPU DeepFM：新增sigmoid FP32 shape32独立资格3次通过；原生训练3步，重复/关闭/缺模型审核各3步、294项逐值一致，后两路目标0调用。直接FlagGems候选第1次参数更新后偏置超差（最大1.76e-3，317/400），未批准模型目标路径；失败审核及严格拒绝CPU控制已记录。9项case CPU控制通过，7个本人容器已停止，381文件及完整原生张量备份。见[模型报告](docs/PPU-DeepFM训练与安全准入验证-20261110.md)及[摘要](docs/evidence-ppu-deepfm-20261110/summary.json)。未改内核/容差，本地未推。

- 11-10 PPU适配增量：功能包新增PPU环境准入及4项跨平台记录隔离测试，安装包146项控制通过。v2基座离线构建固定公开FlagGems；SiLU/sigmoid×三精度×3重复，18组独立前反向对照通过。审核后关闭、开启、缺审核各6组，共18组通过；开启前反向各6次真实命中，其余目标0调用，18组退出恢复通过。全部本人容器停止、证据备份，见[报告与证据](docs/PPU算子接入与安全回退验证-20261110.md)。未改内核或阈值，本地未推。

- 11-10早前PPU原生阶段：同一v2镜像、物理7号卡、shape32×64，SiLU/sigmoid×三精度共6组CPU对照通过。原镜像未装FlagGems，上条已推进到本方向独立wheel安装及三路验证。首次探针非叶子.grad问题及两轮资源拦截保留，见[原生复现](HANDOFF.md#11-10-ppu原生最小验证)。

- 11-10当前可用性复测：152及81分别使用SMI7→PCI0000:bb:00.0→/dev/xpu6→逻辑0，冻结11-04输入、v2镜像及dev2 wheel，先核空闲再启动。两次inspect均在300秒期限后退出137，未完成原生检查，未启动资格或模型三路；容器均Exited、Pid=0、OOMKilled=false。152驱动在10:04:55记录的KL_XID_KERNEL_EXCEPTION对应本次probe进程3790310。证据与复现路径见[HANDOFF](HANDOFF.md#11-10-p800当前可用性复测)。11-09记录中的“非代码/整机故障”属于过强推断，应以当前结论为准：底层异常伴随原生检查失败，尚未定位硬件损坏、驱动状态或软件兼容性中的具体原因。11-04成功证据保留。

- 11-09 910C官方镜像dev3回归（ZCode会话）：复用11-04冻结输入与既有资格记录，unary阶段12组目标（12前向+12反向真实GEMS命中，前向最大差1.19e-7、梯度4.88e-4，与10-18历史一致）及12组回退（6原因×2算子，目标0调用）全部通过。首试芯片4被他人容器namespace占用致aclInit 507899，换芯片2通过；失败证据保留。输出：11机/home/cgu135/framework-unary-20261109/reg910-unary-1109-02。XPU两台新机当日设备初始化均挂死（152卡4/6 KL3 kernel exception、dmesg XPUW自11:37起2.4万条；81全机多卡KL_XID_C2C_EVENT），同一冻结input 11-04曾通过，属机器/驱动状态问题，待总组复位后复跑；input已复制至81:/data2/cgu135/framework-regression-20261104。

- 11-06训练v3补包审核与模型回归：cann-shmem 1.6.0的版本、wheel SHA及安装内容纳入环境身份；保留未修复环境的失败拦截，重新审核SiLU和模型。Qwen0.6B双rank关闭、开启、缺模型审核各3步通过；开启每rank84次前向/84次反向，其余两路目标调用0。原生重复3726项逐值一致，候选3726项通过、最大差9.54e-7。142项包测试通过（新增14项），见[报告](docs/训练v3补包审核与Qwen回归-20261106.md)、[摘要](docs/evidence-overlay-regression-20261106/summary.json)。本人容器与SSH已关闭，本地未推。
- 11-06最小修复验证：cann-shmem 1.6.0使原shmem-missing最小循环及SiLU三形状前反向6项通过；默认1.7.0下载哈希与官方索引不符，未安装。见[修复验证](docs/cann-shmem修复验证-20261106.md)。
- 新增版本化训练回调候选：公共启动器传实际模型/配置及训练函数，适配包统一做准入、退出恢复和报告；旧CLI保留。安装包128项CPU控制通过（新增18项），三个无卡示例通过。见[接口对接与验证](docs/公共训练回调接口对接-20261104.md)。
- 公共训练入口接齐模型身份及审核：实际权重、配置、源码、依赖版本与训练设置共同匹配，通过审核后才启用目标算子。
- 910C官方镜像：Qwen0.6B双rank关闭、开启、缺模型审核三路各3步通过；开启每rank84次SiLU前向及84次反向，其余两路保留原生。
- 独立资格：原生重复3726项逐值一致；候选3726项通过，最大差9.54e-7。
- 新P800：DeepFM FP32 batch32三路各3步，294项对照一致；本轮P800使用dev2。
- dev3固定提交构建及独立重建各110项包控制通过，集成脚本控制42项通过；11个运行时文件与实测wheel一致。

本期功能提交：78358ab；集成材料发布在个人分支cgu135/framework-adapter，固定证据以本页所在提交定位。package.lock更新到dev4。上期dev3为40864f9、材料695cc32/476d274；公共STATUS仍为ea25e9f，本次未合并公共分支。

- 公共入口安全检查4个双rank场景通过：学习率变更保留原生3步；学习率错配、缺审核、运行时代码身份变化三种严格路径均在训练前拒绝，零步骤/零目标调用。证据检查器14项CPU控制通过，报告见[安全检查](docs/公共训练入口安全检查-20261104.md)，[摘要](docs/evidence-public-safety-20261104/summary.json)。

## 功能与待办

| 功能 | 当前实现 | 待完成 |
|---|---|---|
| 算子接入 | PyTorch eager SiLU、sigmoid、tanh及本地ReLU、限定embedding前反向；schema4统一多输入配置、开关、逐调用准入与退出恢复；PPU Qwen embedding三路通过 | softmax契约/验证，OneRec BF16 embedding契约，自定义/融合及其他框架入口；DeepFM数值材料待算子组确认 |
| 模型审核 | 固定公共双rank入口及P800 case采集真实身份；新增训练回调本地候选，匹配模型与算子审核 | 与device-context确认回调位置、身份传递、多rank职责，再做硬件联调 |
| 安全处置 | 条件不满足保留合法原生；严格模式拒绝；执行异常停止 | 多候选实现选择、任务级恢复 |
| 双rank联动 | 固定入口验证更新前共同阻断；正常完成与恢复有记录 | 更新后回滚、跨主机恢复由相关方向协同 |
| 三种接入方式 | 算子注册、库替换、版本补丁的方案及接口约定已交评审稿 | 跨组确认与逐项实现 |

## 平台证据

| 平台/场景 | 结果与问题 | 证据 |
|---|---|---|
| PPU v2 / Qwen0.6B / embedding FP32 seq32 | 12组独立前反向通过；模型关闭/开启/缺模型审核各3步、1863项逐值一致；开启前反向各3次，其余两路0调用 | [本轮结果](cases/embedding-config/README.md#qwen-embedding独立资格及模型三路)、[摘要](docs/evidence-qwen-embedding-20261111/summary.json) |
| PPU v2 / DeepFM embedding / 11-11统一入口 | 原生重复/关闭/缺审核各3步、294项逐值一致；候选第2步参数超差，失败审核生效，保留原生 | [本轮case](cases/embedding-config/README.md)、[核验摘要](docs/evidence-embedding-config-20261111/summary.json) |
| PPU v2 / 公共DeepFM / ReLU FP32 32×400 | 新增ReLU；关闭/开启/缺模型审核各3步、各294项对照最大差0，开启前反向各9次命中；sigmoid原生 | [11-10清单与扩展报告](docs/三模型算子清单与ReLU接入验证-20261110.md) |
| PPU v2 / 公共DeepFM / embedding FP32 | 权重3850×16及3850×1、索引32×5；关闭/开启/缺模型审核各3步，各294项通过；开启前反向各6次命中，最大差7.45e-9 | [11-10 embedding报告](docs/PPU-DeepFM-embedding接入验证-20261110.md) |
| PPU v2 / Qwen0.6B / SiLU FP32 seq32 | 原生/重复/关闭/缺审核各3步；候选第1步梯度超差、更新前停止，保留原生 | [11-10两模型报告](docs/PPU两模型接入与安全准入验证-20261110.md) |
| PPU v2 / OneRec-8B / SiLU BF16 seq16 | 完整8B原生/重复/关闭/缺审核各3步；候选第1步梯度超差、更新前停止，保留原生 | [11-10两模型报告](docs/PPU两模型接入与安全准入验证-20261110.md) |
| PPU v2 / 公共DeepFM / sigmoid | 原生/重复/关闭/缺审核3步通过；sigmoid目标候选第1步参数超差，继续原生 | [11-10模型报告](docs/PPU-DeepFM训练与安全准入验证-20261110.md) |
| PPU v2 / 两单算子 | SiLU/sigmoid FP32/FP16/BF16，shape32×64；18组资格、18组三路通过 | [11-10报告](docs/PPU算子接入与安全回退验证-20261110.md) |
| 910C官方镜像 / 公共719de8c入口 | Qwen0.6B FP32 batch1 seq32，双rank三路通过 | [11-04报告](docs/公共训练入口模型审核接入-20261104.md)、[摘要](docs/evidence-public-entry-20261104/summary.json) |
| 新152 P800 / v2 / GEMM补丁关闭 | DeepFM FP32 batch32三路通过，sigmoid资格6组×3次通过 | [11-04报告](docs/P800新机三路回归-20261104.md)、[摘要](docs/evidence-p800-regression-20261104/summary.json) |
| 910C历史模型 | Qwen有限训推、OneRec seq16通过；OneRec seq32梯度及DeepFM参数超差，保留原生 | [10-23报告](docs/环境准入与算子扩展验证-20261023.md) |
| P800历史模型 | 三单算子通过；DeepFM通过；Qwen FP32 seq32与OneRec BF16 seq16梯度超差，审核拒绝 | [10-29报告](docs/P800接入与安全回退验证-20261029.md) |
| 910C双rank故障注入 | 前向/反向异常均双方0次优化器更新 | [10-28报告](docs/双rank步骤门控验证-20261028.md) |
| 910C训练v3 + cann-shmem 1.6.0 | 补包身份审核、SiLU三路及Qwen FP32 batch1 seq32双rank三路各3步通过 | [11-06模型回归](docs/训练v3补包审核与Qwen回归-20261106.md)、[最小修复](docs/cann-shmem修复验证-20261106.md) |

模型回归使用合成输入短训练；数值资格由独立探针验证，本方向不改内核或放宽阈值。

## 发布与复现

| 材料 | 固定版本或入口 |
|---|---|
| 功能源码 | [dev4@78358ab](https://github.com/FlagRT/framework-adapter/tree/78358ab05da16878e91fa5775b3859ba0ba479d2) |
| 制品版本锁 | [package.lock.json](package.lock.json)，精确源码、实测wheel和两次重建SHA |
| 发布审查 | [阶段报告](docs/阶段进展与验证入口-20261111.md)、[dev4核验摘要与脚本](docs/evidence-dev4-release-20261111/)；[历史dev3](docs/dev3发布审查-20261104.md) |
| 三通道方案/接口 | [方案](https://github.com/FlagRT/framework-adapter/blob/40864f9ac266163a80e323ca1abe082bb6b8a2f9/docs/三通道接入方案.md)、[接口约定](https://github.com/FlagRT/framework-adapter/blob/40864f9ac266163a80e323ca1abe082bb6b8a2f9/docs/接入接口约定.md) |
| 本方向材料 | 本人分支cgu135/framework-adapter；公共dev-1.0只同步STATUS及固定证据指针 |
| 操作与索引 | [HANDOFF](HANDOFF.md)、[报告目录](docs/README.md) |

源码候选按固定提交离线构建；未上传PyPI。新P800旧机迁移已完成，9395文件逐文件SHA核验通过。11-04实测结束后11机7个及27机1个本人容器均已停止。

## 下一步

下一动作统一见workspace[执行计划](/Users/empty/PhD/Workspace/DomesticRuntime/协作/执行计划.md#近期按顺序交付)。PPU历史失败模型配置仍保持原生；package.lock锁定dev4最终实测源码，不将旧模型审核自动移到新代码。

本页至少每两周更新，重要交付同步证据链接。历史过程见[归档](docs/archive/session-20261103/README.md)。
