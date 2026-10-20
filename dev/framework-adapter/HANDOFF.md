# framework-adapter · 开发交接

更新：2026-10-20。当前成果/目标只看[STATUS](STATUS.md)，协作流程看[规则](/Users/empty/PhD/Workspace/DomesticRuntime/协作/协作与交接.md)。

## 接管前

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
| P800：43.180.254.67:26008 | 10-18 SSH通，8卡各96GB、当时全空闲；无Docker/sudo权限，暂挂起。根盘仅3.7GB；/data2余4TB，但本人目录未创建 | 本方向尚无确认的容器/数据目录 |
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
