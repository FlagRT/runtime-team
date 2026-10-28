# 10-23环境准入与算子扩展证据

日期为北京时间2026-10-23；Docker及部分日志使用UTC 2026-10-22。候选包0.3.0.dev0未提交发布，不能使用0.2.0的固定链接冒充本轮代码。完整解释及复现入口见[报告](../环境准入与算子扩展验证-20261023.md)。

## 结果与原始文件

本地持久根目录：`Project/DomesticRuntime/outputs/framework-runs/20261023-expansion`；27备份在`27-results/`，11在`framework-startup-result-20261023-01/`。每组保留baseline、demo、安装位置、源码manifest和命令日志。原件也保留在本人服务器home目录，不依赖临时缓存。

| 原始结果（相对本地根） | 结论 | result.json SHA256 |
|---|---|---|
| 27-results/qualification-01 | 3算子×3dtype，9/9前反向和注册恢复通过；只审核FP32/32×64小模型资格 | 3d60ff2156e0440b83a48ee2240f385df7d13cf76f7d015704e7f60396619c81 |
| 27-results/models-02 | 2合成小模型×4路径×3步=24步；目标24次真实前反向内核命中 | 9895c0cf83ac777f4d8c74c4d53bb4301db7522a60a6356ff584fc72082068d5 |
| 27-results/failures-01 | 3独立worker故障注入均按预期停止；0次optimizer更新/原生重试 | 82332469f7331ec932ee912dccda85df86740ca2e6fb3d29b79a27e7b6757779 |
| 27-results/qwen-01 | Qwen3-Embedding-0.6B推理及3步SiLU训练回归通过；每步28前向/28反向、310梯度/参数张量对照 | 4bcb0ba409f6929756f2410f82ba2333befa4d7adcecc9ef0d03ec9f83e74141 |
| 27-results/sdpa-01 | FlagOP限定FP16方形、无mask、causal开/关2组直接调用通过，非全局注册 | 0ea025fe02333113bf30e3bb6a744324293ecf3e4b85bf8bbc57993ea1dc686d |
| 27-results/flagop-cpu-2.10 | 4正常数学反向对照通过；全遮罩/非方形causal失败；默认调用/撤销缺口复现，命令预期退出1 | 93993baf680bd90b9aad03e1a9d7cfef57d395ad96b53e13014ea7dbea1ca46c |
| framework-startup-result-20261023-01 | v3 off/degrade各3步原生通过、strict启动前BLOCKED，目标0、未导入Gems | fca75ff57e6890afcc4cf58eed50595cf6dae07efc7d0822546f3e35a08d8dbd |

27传输归档`framework-expansion-20261023-evidence.tar.gz`两端SHA256均为`f6130acf45167ea2e766d8e800fe91a887374acf8557fc9af719d9254620697f`；只排除installed及编译cache，未删除失败证据。`models-01`是同日较早版本的通过记录，不重复累加。无卡环境审计见根目录的`official-no-card-audit.json`和`v3-no-card-audit.json`；最终CPU复现脚本在本机2.12另运行，见`flagop-cpu-final-2.12.json`。

## 构建身份与验证边界

### OneRec 多步与第二形状

本地根Project/outputs/framework-runs/20261023-multistep/all-results；11原件/mnt/raid/cgu135/framework-multistep-20261023，轻量归档另存独立/home。最终all-evidence.tar.gz两端SHA256为6b2afda43b1e2fe96d75cd23de7ebb93ef3112228b196831d842fe2b36e47ef9。9组原始结果保留；seq16-on模型PASS但外层清理ERROR已人工停止，seq32-on模型FAIL不覆盖。

| 关键result.json | SHA256 |
|---|---|
| s16-on-01 | 28a1901fc0efc55a17f3486842c990bde69ff5859328cbef9188f7580e01a2da |
| s32-qualification-01 | 543f7b3547f56399724e36a4867f42cb25547fc33023f51d375aa7da9b462429 |
| s32-on-01 | 61240c3f31ee47b6fe7c772b607913c65a9764198ea0d2be9f08eae7be24b3a5 |

seq16四路3步、三对照各2397项最大差0；seq32原生/关闭/缺资格3步通过，目标首步Embedding梯度20938元素超差、最大0.125，0参数更新。拒绝记录records-denied-s32.json及CPU拒绝检查另存本地根；原records-s32.json只是复现失败的实验输入。188GiB基线张量仅在11大盘，不在轻量归档。9容器已停、SSH退出。

精确执行源码快照source.tar.gz在本地根；探针SHA445170579a1d9096f79c3d4399de6b63f4ebeedc0ca7188e4367572be6e703eb，case入口SHA2e52ebd8bea87cd35c29a4e979968919f30ef4731e8640a1e1464c2ecefbfe74，配置入口SHA477e2c5eef4b33d8744fa4bdb5fa0f8ab5043d1faddfcdfa82fc5eff2bdb68e7。包仍实测c5b9796…；本地重建与当前源码8个文件一致，57CPU控制通过，另5配置/6case控制通过。所有成绩及限制见报告多步节。

### 同日后续 OneRec

独立本地根Project/outputs/framework-runs/20261023-onerec/11-results；完整解释见报告OneRec节。六运行summary全部PASS，源码使用最终c5b9796… wheel。完整8B五轨迹各单步，原生重复/关闭/缺资格/开启各799项与原生一致，目标真实36前向+36反向。初次qualification-01因容器重名启动前拒绝，保留ERROR，不计入六次实测。

| result.json目录 | SHA256 |
|---|---|
| onerec-qualification-02 | 473f6a8f4b31faed2731320bd5e625c4568c63d471cdb09864b89aeff2817a1e |
| onerec-native-01 | d437b1b1007c30237c7744624592f74983efbf17874925a83f0c52659befa1bd |
| onerec-repeat-01 | 77497a78558a955a27cce6949d1dc296144b7fdd9c4a63ef2c4d7ffcb27d0b63 |
| onerec-disabled-01 | b81d98ec810a96e0f7369060999f692efb38621917ee08f0a9d6a76a149e35f4 |
| onerec-unqualified-01 | b1bd4303881e3f10d131c81facb30a0d0b5a1268c426da087d7a903b723399dd |
| onerec-target-01 | 0d2fdcfdc2b466f65e6bfb160f3ceb62b7d396810da9030f7b12f1209b73bb51 |

轻量归档framework-onerec-20261023-evidence.tar.gz两端SHA256为9191be6802b7e0838646cacd4a98fe6585e2cac0b3bb5a6091bfb4e9dab03ca9；含baseline/demo、源码manifest、日志、结果与审核输入，排除installed/cache/32GB原生张量。完整张量和模型留11独立/home，未声称全量本地备份；源快照另在本地source.tar.gz。27本人旧仓备份SHA8843b829…，范围另见本地raid-backup.md，不代替全机备份确认。

### 同日后续 DeepFM

独立证据根为Project/outputs/framework-runs/20261023-deepfm，不与先前合成模型合计。`11-results/qualification-01`为6组单算子PASS；`model-capture32-01`和`model-capture16-01`各含原生/重复/关闭/缺资格3步PASS及目标首步FAIL。`model-01`是batch32首次失败，不累计。报告详见[DeepFM章节](../环境准入与算子扩展验证-20261023.md#deepfm-公共模型训练回归)。

完整归档framework-deepfm-20261023-evidence.tar.gz两端SHA256为`62fb35f8018f48807dac90c2a869f97ae8001538a31a3a9b11465dba9a5f29b0`；包含四次运行baseline/demo、命令、结果、失败张量和历史审核记录，仅排除installed/cache。单算子result.json SHA256为`ec7838be0a1e016c07a00ba8dde8dd064bc81c8e48bf62ee281973d821ab3af0`。本地diagnosis32/16.json由保存张量生成；当前records-final.json为空，DeepFM保持原生，历史passed仅是失败复现输入。使用与先前相同的最终c5b9796… wheel，无包源码增量。

### 先前扩展构建

- 最终NPU实测wheel：`c5b9796c2db41320092d10bcc3e8d82228693acd89f3863e95a2f70ae8b295d1`，位于Project/outputs/framework-package/20261023-final。
- 独立源码重建wheel：`02d433b04a37cb9b37dfba58bd89b262a49a2b51b4b317e859314700d0311378`，位于同根20261023-rebuild；两个构建均安装后57项CPU方法通过，不累计成114项。二者与当前源码8个包文件逐字节一致，归档hash不同。
- 资格实验/SDPA直接候选使用较早wheel：`b0d196e8f5292f7377b965ee3540d40d36ed043b89ba417a0f8e7f76d6004caa`；models-02、failures-01、qwen-01及v3启动使用最终wheel。SDPA没有使用包内通用SDPA适配器（尚未实现）。
- 额外case CPU控制：官方6项、v3 11项通过。整目录test发现53通过/1个历史硬件脚本因本机无torch_npu导入失败，不作为全绿CPU套件，错误摘要保留。
- 上游核对：runtime-team dev-1.0=1d39898，device-context=17d8b90（prototype未改）；FlagOP=79b1b1aa867bd4dedd90eff69e6e9d717104e1fb，register.py SHA256=eb9ae87eb2e01e70d02dc16501c59d64b133e2683381be17b778e74afdde15f5。

没有修复shmem，没有完成SDPA模型级开关/回退、双rank失败联动或跨芯片验收。CPU包装层缺陷不冒充NPU内核失败；小模型合成数据成绩不冒充业务模型覆盖。package.lock.json仍锁定已发布0.2.0。
