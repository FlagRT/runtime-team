# framework-adapter · 开发交接

更新：2026-10-28。当前成果/目标只看[STATUS](STATUS.md)，协作流程看[规则](/Users/empty/PhD/Workspace/DomesticRuntime/协作/协作与交接.md)。

## 接管前

发布提交：功能仓85a9896537a7d88f758d36ae20465e19470985e2；协调仓代码/证据快照8963353653ff8610d44b1f6a89be43ea680b77cf（个人分支，当前文件是其后续文档更新）。周报固定链接见STATUS顶部。不要将准备阶段旧HEAD/未发布字样继续当作最新事实；本轮未合并dev-1.0。

10-28源码候选交付入口：[交付记录](docs/候选交付记录-20261028.md)。功能commit=85a9896537a7d88f758d36ae20465e19470985e2，锁定版本0.3.0.dev0，提交后构建Project/outputs/framework-package/20261028-publish通过，69项包测试+30项集成控制、9文件与NPU包一致。本轮不连接服务器；协调仓只提交个人分支，不合并dev-1.0。双rank代码保留为联调参考，正式任务级容错需另对齐责任。下方准备阶段“待授权/未提交/旧锁”等已被本条替代；后续接管仍用git核远端状态。

10-28 18:27 P800补记：SSH/sudo/Docker已只读核实可用；cgu135属sudo组，不属docker组，使用sudo即可，本轮未改组。8卡均有进程，卡1占72MiB且0%不是无人使用；驱动5.0.21.47。昆仑芯v2本机镜像ID cac5ddae5f62b356622858dd4f0ea0f82d489cf0cb71e7230f0f97797c25605a，仍需与公共baseline核对。根盘余6.5GB、/data2余3.2TB；未创建个人目录/容器或安装，未进入他人容器。SSH22839退出。历史“P800权限挂起”已过时，现阻塞转为选卡/环境准入，未验证本方向算子。

10-28发布准备已收尾：先读[候选清单](docs/候选发布准备-20261028.md)。功能仓新增RELEASE、修正文档测试数和范围，build_local源码包携带发布说明；运行时未改。Project/outputs/framework-package/20261028-release-prep构建PASS、69项包测试+30项协调仓控制通过，9个包源码文件与ea060217…NPU产物一致，新wheel哈希0f3089fe…。仓内summary.json提供可发布精简证据。未创建Git提交/标签、未推送/改锁/占卡；两仓dirty仍须保留。下一次发布须获授权，用真实功能仓提交SHA更新协调仓锁，不能拿main旧HEAD代表候选。

最新10-28后续已收尾，可接管：独立包step_guard.py及train_cli.py已实现单机两rank更新前门控；最终安装包CPU69项、协调仓8+6项通过。27物理0/1正常各3步、故障两组均双方0次optimizer调用，三组PASS，详见[步骤门控报告](docs/双rank步骤门控验证-20261028.md)。最终wheel在Project/outputs/framework-package/20261028-step-guard-03，SHA ea060217…；不要用初版02（正常回归暴露包装问题）。证据Project/outputs/framework-runs/20261028-step-guard/evidence，归档7a767f62…两端一致；远端/home/cgu135/framework-step-guard-20261028。4本人guard容器均停/OOMfalse/Pid0，SSH95275/31415及SCP12440退出，无后台任务。两仓dirty保留、未提交推送；package.lock仍发布版。下一步对齐device-context步骤边界，更新放行后失败仍需恢复协议。下方为历史同日修补前FAIL，不是当前三组结果；原失败证据不可删。

10-28已收尾可接管：27 RAID和官方镜像恢复可用；0/5失败有udevid5 namespace冲突日志，0/1原生检查和双rank各3步通过。故障两组均在对端进入optimizer后终止，严格FAIL，勿写“联动通过”或“权重确定已更新”。报告[1028](docs/双rank失败联动验证-20261028.md)，本地outputs/framework-runs/20261028-rank-failure含全部轻量结果/源码，归档8dd9c472…；远端/home/cgu135/framework-rank-failure-20261028。run.sh control 1 02 / forward 1 01 / backward 1 01已执行，新输出编号才能重跑。7本人容器均停/OOMfalse/Pid0，SSH92557/81422和SCP均退出，无后台任务。功能包未改，所有旧dirty保留，未提交推送。下一步先设计训练步更新前全rank确认及独立故障通知，与device-context对齐后在独立包实现。不要再次盲跑0/5，也不要把10-24的11初始化失败根因直接套用；资源使用前仍重核。

10-24双rank测试已收尾可接管：新增rank_failure_worker/test_rank_failure_npu及7项CPU判定，case6项通过；功能包未改。11正常对照双rank初始化资源忙，故障组未运行，不能按已通过续报。见[报告](docs/双rank失败联动验证-20261024.md)。本地outputs/framework-runs/20261024-rank-failure含完整命令、初次源码和轻量证据，SHA f6cbd639…；远端/home/cgu135/framework-rank-failure-20261024。容器cgu135-unary-rank-control-01停止/OOMfalse/Pid0，SSH67299/82393及传输均结束。源码快照后本地只增强判定器错误信息，重做CPU7项，未NPU重跑。下一步重新核两个设备可初始化，再control/forward/backward；输出用新编号。数值失败只记录，不改算子，不推送。

多步续跑已收尾，可接管：SSH68033/52027及旧断线会话均结束，9容器Running=false/OOMfalse/Pid0，docker ps为空。11输出/mnt/raid/cgu135/framework-multistep-20261023，源码及配置/home/cgu135/framework-multistep-20261023。seq16四路模型3步PASS（on外层清理ERROR保留，已人工停）；seq32原生/关闭/缺资格3步PASS，开启首步梯度FAIL、0更新。不要重复硬跑或改内核/容差。全部轻量结果本地Project/outputs/framework-runs/20261023-multistep/all-results，归档SHA6b2afda4…；188GiB原生张量仅11大盘。records-s32.json是历史实验准入，不作当前放行；本地records-denied-s32.json引用失败证据并已CPU验证拒绝。下一步对齐数值准入/模型范围，或独立推进双rank失败联动；发布另授权。不要覆盖单步目录、不要动27。

最新OneRec已收尾：11 `/home/cgu135/framework-onerec-20261023`保留完整权重、原生32GB梯度/参数快照及六组PASS；五轨迹各单步，四组各799张量对照最大差0，开启36前向+36反向。精确wheel沿用c5b9796…，新增test_onerec_adapter_npu.py和官方case onerec/onerec-qualification阶段，未改独立包/内核。详见10-23报告OneRec节。轻量归档备份Project/outputs/framework-runs/20261023-onerec，SHA9191be68…；完整模型/32GB张量仅在11独立/home，不宣称本地全量备份。6容器停止/OOMfalse/Pid0，SSH75077/26870/35138及传输均结束。首个同名容器预检ERROR保留，未覆盖旧对象。

27计划15点重构/mnt/raid：本人该盘1.3MB旧仓已完整归档到独立/home并取回本地，SHA8843b829…；近期实验挂载在/home，Docker镜像/可写层仍受影响。本人27容器已停，本轮没在27新增实验。备份范围见本地onerec/raid-backup.md，不代表其他人数据安全。不要在维护窗口使用27。

同日后续DeepFM已收尾：公共1d39898模型不改架构，11官方镜像最终c5b9796… wheel；6组单算子通过，原生/关闭/缺资格18步及原生重复6步一致；两batch开启首步Adam参数验收FAIL，保持原生。详见10-23报告DeepFM节。新增test_deepfm_adapter_npu.py及官方case两stage，功能包本轮未再改。

原始证据11 `/home/cgu135/framework-deepfm-20261023`，本地Project/outputs/framework-runs/20261023-deepfm/11-results；两端归档哈希62fb35f8…一致。source-capture保存失败张量；review为历史实验资格，不是模型准入，本地records-final.json为空。四个本人容器均停止，最后本人前缀docker ps为空，SSH2640/16361及SCP已退出。27后续RAID已挂载但官方镜像overlay路径缺失，不能照先前环境描述启动；未修共享Docker。两仓未提交/推送、OneDrive未动。

10-23实施已完成并收尾。当前先读[报告](docs/环境准入与算子扩展验证-20261023.md)和[证据摘要](docs/evidence-expansion-20261023/README.md)，不要重做本轮9组算子/24步小模型/0.6B回归。功能包0.3.0.dev0在独立仓main上未提交；协调仓个人分支未提交baseline/demo、探针、报告和状态。已发布HEAD仍为新仓1befe4d、协调仓a31e559，package.lock锁已发布0.2.0，未合并/推送/写OneDrive。

### 本轮代码与精确产物

- 功能代码：Workspace/framework-adapter/src/flagrt_adapter，新增startup.py、严格准入与tanh，修改配置/桥接/训练入口；新增2组CPU测试文件及strict配置。57项安装包CPU控制通过；协调仓case另17项通过。
- 最终实测包：Project/outputs/framework-package/20261023-final/wheels/flagrt_framework_adapter-0.3.0.dev0-py3-none-any.whl，SHA c5b9796c2db41320092d10bcc3e8d82228693acd89f3863e95a2f70ae8b295d1。20261023-rebuild是独立源码重建；20261023-expansion是较早56项版本，不要混称最终版。8个包文件已核与当前源码完全一致。
- 集成测试：本方向cases/ascend-official-unary新增expanded-qualification、expanded-models、model-failures、sdpa-candidate；cases/ascend-train-v3/assets/run_startup.py为v3启动准入单入口。当前报告/Issue在仓内仅本地，不能用旧发布URL声称本轮已上线。

### 本轮环境与结果（已结束，不是资源预约）

- 27 SSH及Docker正常；/mnt/raid/docker当时实际落在根分区，非独立RAID挂载，根盘约45GB余量；未迁移存储。使用官方image 2e56022ae5b3…、物理0→逻辑0，实际Triton模块3.2.0/发行triton-ascend3.2.1。
- 27源快照/结果：/home/cgu135/framework-expansion-20261023。source-candidate为最终功能探针快照，source-final为较早快照；FlagOS-OP固定79b1b1a只读。模型/FlagGems沿用本人历史已核目录，不是共享/model。运行参数见各summary/runner.log、case README，勿手工补猜命令。
- 11使用精确训练v3 e13a15d0d02b…、物理0→逻辑0，安装最终wheel。/home/cgu135/framework-startup-result-20261023-01保留off/degrade/strict结果；/home/cgu135/framework-startup-20261023保留本轮源码。sudo Docker包装需本人认证，凭证不写记录。
- 本地备份：Project/outputs/framework-runs/20261023-expansion/27-results与framework-startup-result-20261023-01。27归档排除installed/cache，两端哈希一致；原始失败/控制结果均保留。无卡环境审计JSON及最终CPU SDPA复现另放同根。
- 27本轮8个本人容器均停止；11本轮3个本人容器已核Running=false/OOMKilled=false/Pid=0；SSH93566、72135已退出。本轮无后台任务，未创建goal。下一次执行仍须实时重核资源。

### 下一条动作

按STATUS推进：OneRec多步/第二形状已完成有界验证，seq32失败保持不准入；先对齐与DeepFM的数值判据或推进公共双rank失败联动，不改内核。SDPA稳定接口仍待确认。shmem未修复，P800/PPU暂挂；发布需新授权和真实commit/锁更新。27维护后需重核镜像，不能沿用旧容器状态。

## 历史接管记录（以下“最新/未推送”等只指对应日期）

10-21会议纪要已读取（Workspace/周会纪要/*1021.txt，元宝摘要非逐字稿），行动项见STATUS：优先可用镜像，补v3版本/路径差异与迁移成本；FlagOP需继续到实际代码而不仅文档，核对注册/原生退路/训练支持。A1与vllm-ascend兼容为跨组待核项，未实施。保持总组统一发镜像，不据纪要自动推送合并。本轮仅更新本地记录，无新增实验。

10-21只读核对：origin/dev-1.0=72411ea、origin/kistich/device-context=17d8b90，未merge；设备prototype与2926d1c一致。新增文件为设备目录910C/docs/INFERENCE_VLLM_ADAPTER_CHOICE_20261020.md及STORAGE_MIGRATION_20261021.md。27 Docker位于待重建RAID这一信息来自后者，尚未实机核验，勿复用旧“镜像存在”的结论。FlagOP HEAD79b1b1a，贡献规范读取成功，具体catalog读取失败；后续优先确认训练支持条件、原生回退入口、注册生命周期。详细分析在STATUS的10-21节；本轮无新服务器任务或发布。

10-20拆仓：包开发转到Workspace/framework-adapter（FlagRT/framework-adapter）；runtime-team保留测试/文档/环境。当前package.lock.json锁定93daa5e；源码/bootstrap已拆除，先在新仓构建安装再跑协调仓测试。官方case所有stage显式--wheel，训练v3 case未做拆仓后的硬件验收。最新跨仓58项控制通过，新旧包payload7文件一致。不要按下方旧日期指令在runtime-team里寻找src/build_local。

最新安装包实测：27 /home/cgu135/framework-wheel-20261020下wheel-off-01、wheel-on-01、wheel-unqualified-01全部PASS，三容器停止、SSH退出。容器不挂/code，实际从/results/installed加载0.2.0；wheel SHA b3f1fffe6e16b75b4b6736dc6186dc8e80fbcd8fc0b2325d9d502978336ddde7。本地完整备份Project/outputs/framework-runs/20261020-wheel；1020报告末节含复现参数。case新增--wheel支持，不改镜像依赖。已完成安装方式三路，不需要再测同一组合；未推送。

同日包收口：训练逻辑移入src/flagrt_adapter/train_cli.py，旧tools脚本保留兼容。Project/outputs/framework-package/20261020-training-delivery是最终交付；20261020-training-entry是中间版，20261020-training-rebuild是独立重建证据。41项安装包CPU测试与后续独立NPU证据分开；未提交/推送。

10-20最新：上游训练入口配置集成三组全部PASS，详细见[报告](docs/训练入口配置接入验证-20261020.md)。27 /home/cgu135/framework-training-config-20261020保存source/upstream/records及off-01(ERROR)、off-02/on-01/unqualified-01(PASS)；四容器已停。实际设备0/5，模型和Gems沿用只读历史目录；不能将此次旧官方成功写成训练v3通过。新增tools/train_with_adapter.py、src配置/训练模块、demo/training-configs，未提交推送。本地备份Project/outputs/framework-runs/20261020-training-config。接管不重跑已完成三组，先看授权和当前资源。

最新增量：官方环境统一入口在cases/ascend-official-unary，宿主27 /home/cgu135/framework-demo-20261018；工具隔离在tools，源码source，公共工具public。qual-01、model-01(FAIL)、native-repeat-01(FAIL)、model-deterministic-01(PASS)、unary-01(PASS)全部已结束，容器已停，SSH退出。本地完整备份Project/outputs/framework-runs/20261018-sigmoid，含初版及修改版源码快照。看1018报告末节再续接，不把失败目录删掉。

小模型复现必须使用--deterministic，判据不变。demo/env.baseline.yaml是已执行样本，实际运行由runner重新生成独立demo；不要直接启动带SET_BY_RUNNER的模板。qwen选项本轮未以新入口重跑，不把旧0.6B结果计成新demo全阶段验收。

1. 确认旧执行会话已停止，读取[会话登记](/Users/empty/PhD/Workspace/DomesticRuntime/协作/会话登记.md)。
2. git status --short、git branch --show-current，保留所有未提交改动。
3. 不自动切分支、merge、push或创建worktree；尤其不能用HEAD覆盖10-16未提交case。
4. 本地控制优先。硬件任务先核主机授权、空闲设备、容器归属、镜像身份和容量。
5. 代码/数据/环境身份必须对应报告。旧绝对路径只用于查找，不保证仍存在。

## 环境登记（检查日期不是资源预约）

最新10-18：11/27均可SSH，选择27旧官方镜像完成SiLU/sigmoid及0.6B回归。本人容器cgu135-framework-hw-20261018已停止、OOMKilled=false，SSH已退出。宿主/home/cgu135/framework-hardware-20261018保留source/results；本地备份Project/outputs/framework-runs/20261018-npu。27驱动26.1.1，具体身份、命令、哈希见[实测报告](docs/910C公共适配器实测-20261018.md)。下表旧条目仅用于历史定位。

| 环境 | 最近实查与限制 | 持久位置 |
|---|---|---|
| 910C 11：10.120.72.11 | 10-16 cgu135可SSH、sudo操作Docker；精确训练v3镜像e13a15d0d02b…；本人单物理2→容器逻辑0，结束已停止 | /home/cgu135/framework-v3-20261016 |
| 910C 27：10.120.72.27 | 10-14旧官方环境实测通过；10-16只读复制本人模型。用户反馈盘维修，优先不用；同源训练镜像7028028bb62c…不等于锁定e13a | /home/cgu135/framework-adapter-910c/acceptance-20260916 |
| P800：43.180.254.67:26008 | 10-28 SSH/sudo/Docker已可用；8卡各96GiB均有进程。根盘余6.5GB，/data2余3.2TB | 本方向尚无确认的容器/数据目录；见本页顶部权限复核 |
| 本地CPU | 既有tablerag Python，torch2.12.0；只用于控制测试 | /opt/homebrew/Caskroom/miniconda/base/envs/tablerag/bin/python |

其他主机不自动沿用旧五机授权；用户10-16允许11/27空卡使用，不授权进入他人容器。P800权限不能沿用910C的sudo假设。凭证不写入这些文件。

## 11新基座复现

- 宿主目录内repo是本人代码快照、public为公共74197bf工具导出、tools为个人Compose/PyYAML；不改全局包。
- tools/docker包装使用sudo，认证可能过期；Compose2.29.7配置位于config/。宿主PYTHONPATH只加tools，不注入容器。
- 本人容器cgu135-framework-adapter-train-v3；模型只读挂/case-model，代码只读挂/case-workspace，结果挂/case-output。
- 正式入口为[case README](cases/ascend-train-v3/README.md)，prepare生成baseline，up.sh --strict会真正启动容器，不是只读检查。
- 物理设备号由FLAGRT_DEVICE及--device传入；入口验证单卡映射，容器内部使用逻辑0，不直接使用物理2。
- 模型：model/model-verified-20261014；model.safetensors SHA256为0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd。
- 锁定runtime提交0c59b9549d45d55b959294869f605a30dc3789c9，由run.py导出，不直接用本地个人分支旧子树。
- 完整结果：results/20261016T063504772692Z；初次预检错误在20261016T063302816232Z。summary=FAIL，fallback=PASS，operators=FAIL，model未执行。
- 本地/tmp/framework-v3-20261016-1uTFbb只是缓存，可能消失；重建方法见case，不把软链复制成独立交付包。
- 7份结果JSON/log另有[本地持久备份](/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261016-v3-063504/README.md)，10-18复制核哈希一致，不是新增实测。
- 当前缺shmem尚未解决，不反复重跑同一失败路径。退出已停止容器、SSH已退出；完整日志留宿主。

## 旧环境与证据

旧官方镜像及10-14运行命令见[1014报告](docs/910C-v03推理与训练验证-20261014.md)。
本人27目录内model-verified-20261014才是核验过的源权重；旧共享/model实测含非有限值，不可复用。微调检查点不是源模型。复用前重新核哈希和有限性。

## 本地测试

最终10-18交付：Project/outputs/framework-package/20261018-delivery，35项安装包测试及CPU demo PASS；case11项另行通过，总46项。最终安装包测试含5种拒绝路径0目标调用，以及known_compile_failure覆盖。最终源码、命令、依赖、wheel SHA详见该目录local.baseline.json。资格边界见docs/资格验证与审核流程.md。本轮已完成本地目标；接管后不要重做打包，先按用户下一任务决定是否做新环境资格/硬件扩展。

上述本地构建之后，10-18另完成真实硬件任务，见本页顶部实测报告；该任务容器已停止，无本会话遗留测试进程。服务器实时状态仍需重查，不能由此推断别的会话没占卡。下方本地测试最终口径为46，不与硬件用例混计。

安装包单入口：在方向目录用既有Python执行`tools/build_local.py --output <不存在的绝对目录>`。源码阶段的中间结果与最终证据映射见[公共包记录](docs/公共适配包开发记录-20261018.md)。

公共实现位于src/flagrt_adapter，历史probes入口只做兼容导入；case源码哈希已包含src，不能只部署probes。启动器--admission-records缺合格记录返回BLOCKED（宿主退出3），不会作为PASS。不能从actual直接生成passed。旧报告/证据原样保留。

从当前代码仓根运行（其他agent可用等价已有Python环境，不默认安装）：

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/tablerag/bin/python -B -m unittest discover -s dev/framework-adapter/probes -p test_silu_policy.py -v
/opt/homebrew/Caskroom/miniconda/base/envs/tablerag/bin/python -B -m unittest discover -s dev/framework-adapter/probes -p test_silu_training_adapter.py -v
/opt/homebrew/Caskroom/miniconda/base/envs/tablerag/bin/python -B -m unittest discover -s dev/framework-adapter/probes -p test_admission_records.py -v
/opt/homebrew/Caskroom/miniconda/base/envs/tablerag/bin/python -B -m unittest discover -s dev/framework-adapter/probes -p test_multi_operator.py -v
/opt/homebrew/Caskroom/miniconda/base/envs/tablerag/bin/python -B -m unittest discover -s dev/framework-adapter/cases/ascend-train-v3/assets -p test_runner.py -v
```

以上是控制测试，不证明FlagGems内核通过。收尾记录命令/返回码/结果位置、源码版本和未提交差异；只有实际结束才标完成。
历史路径映射和全过程见[整理前HANDOFF](docs/archive/session-20261018/HANDOFF.md)，按需查，不作为当前环境保证。
