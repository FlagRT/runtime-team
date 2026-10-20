# framework-adapter · 开发续接

10-18用户决定：P800权限暂不处理，不重复SSH尝试；先按STATUS当前执行顺序推进环境准入、公共机制与本地包。服务器复跑另核资源；shmem修复不由本方向接管。此轮仅整理，未开始这些新开发。

更新：2026-10-16。先读[STATUS](STATUS.md)，只在需要重现时读日期报告。

10-18 P800：SSH cgu135@43.180.254.67端口26008成功，8卡检查时全空闲，但账号既无Docker组权限也无sudo权限；未能核实际镜像或运行算子，SSH已退出。公共推荐kunlun-operator-runtime/v2（期望ID cac5ddae5f62…）仅文档已查，不代表服务器已验证。需管理员解决访问授权；不要复用910C的sudo假设。根分区仅3.7G，后续持久数据宜申请/data2本人目录（目前不存在），不写其他人目录。

14:36当前续接：11已完成真实回归，覆盖下段“等待资源协调”。本人单卡容器cgu135-framework-adapter-train-v3已停止；宿主/home/cgu135/framework-v3-20261016包含repo/public/tools/model/results。strict通过，5类原生回退通过，目标SiLU编译缺shmem失败、模型三步未跑。物理2在容器内为逻辑0；runner现在检查物理挂载后设置可见0。包发行名flagtree、导入名triton，已修正预检；只读仓挂/case-workspace避免镜像工作目录冲突。未改镜像/他人容器/内核，未推送。下一步协调基座或编译方向的shmem支持，不反复排查空卡；详见1016报告。用户当前授权11/27空卡可用，但27盘维修优先避开。重跑仍需实时查占用，工具在本人目录、sudo可能需重新认证。

10-16当前续接：训练v3 case已在`cases/ascend-train-v3/`，尚未NPU执行，等待资源协调。11有精确e13a15d0d02b镜像且sudo能读Docker，但他人算子容器运行中；27有7028028bb62c同源不同ID镜像、多个带卡容器。两机默认Compose/PyYAML缺失，不能直接up.sh。未启停容器，SSH退出。公共工具源码临时导出`/tmp/framework-v3-20261016-1uTFbb`，本地demo/public-baseline是指向该目录的忽略软链；换机器先按case README prepare，不能复制软链当可复现包。env baseline记录dirty:true，未提交推送。run.py自动从本仓Git对象导出10-14设备runtime，记录固定哈希，避免个人旧子树；不进入其他owner容器。下一步先协调时段/权限，补工具环境，再strict启动、跑单入口，别把当前状态读成已完成v3回归。

19:18最新续接：27密码登录成功，前段密钥认证失败不是服务器不可用。新适配器实机路径`/work/scoped-adapter-20261014/`；训练成功用`test_silu_training_npu-v3.py`和修复引擎线程后的`silu_training_adapter.py`，结果`scoped-silu-training-v4.*`；准入/推理成功用`test_silu_training_npu-v5.py`（当前本地探针），结果`scoped-silu-admission.*`，均在原results-20261014目录。容器停止`2026-10-14T11:18:15.445429876Z`，SSH退出。5类原生回退＋3dtype＋三步模型梯度/权重对照通过；更长轨迹和正式准入仍未完成。控制测试目前11＋5项，不再是10＋5；启动方填写实际芯片名`Ascend910_9382`，不得把示例默认Ascend910误当设备实测名。

机制开发续接：新`silu_policy.py`/`silu_training_adapter.py`尚未上NPU，不能用旧脚本结果证明它通过。CPU控制可用本机`/opt/homebrew/Caskroom/miniconda/base/envs/tablerag/bin/python`（torch2.12.0）执行`-m unittest discover -s dev/framework-adapter/probes -p 'test_silu_policy.py' -v`和`-p 'test_silu_training_adapter.py' -v`，5＋10项通过。硬件回归前须测量实际环境身份、显式限定shape、使用实验开关；前向和backward在同一作用域，失败立即放弃该step，不捕获后继续optimizer.step。注册是进程全局临时修改，只允许本人独占单线程worker。旧Qwen实例适配器未改、仍仅推理。27本轮BatchMode SSH认证失败，未用密码、未启动容器；这不等于设备不可用，后续用授权连接材料核验后继续。

18:48最终状态：同权重10轮及真实NFCorpus Embedding两组各10步已完成，详见本日报告末节。容器最后停止时间18:48:44。新脚本`qwen_embedding_train_test.py`（服务器训练快照）和`qwen_embedding_checkpoint_verify.py`（当前本地脚本，增加eval-only）；数据在同级`nfcorpus.zip`，SHA256见报告，未进Git。两份检查点为`/work/nfcorpus-native-20261014`、`/work/nfcorpus-gems-20261014`，各2.3GB、仅实验候选，重载得分一致，但检索指标下降，不默认替换源模型。结果同一`results-20261014`目录，均已下载。下方18:35与17:56为此前批次收尾，不是最新状态。

10-14晚间续测：`test_silu_training_npu.py`独立进程ATen SiLU前/反向替换，单算子三dtype通过，模型第4步梯度超差中止；native-control 10步梯度/权重差0。证据仍同日结果目录，别误报全通过。最后容器停止时间更新为18:35:57，SSH退出。原型/模型/镜像未换，旧QwenScopedGems保持原样；注册作用域进程级，不得当模型实例隔离机制使用。详见本日报告末节。

10-14新增实测入口：[v0.3推理/训练报告](docs/910C-v03推理与训练验证-20261014.md)。27本人容器已停止，17:56:48；持久目录仍是下表，但新runtime在`/work/framework-v03-NwoBpG/dev/device-context/prototype`（Git 0c59b95），新探针在同级probes，结果`/work/results-20261014`。模型必须用`/work/model-verified-20261014`，SHA256见报告；旧`/model`挂载本轮证实含非有限权重，不能复用。未更换镜像，未merge或push。下方09-22环境/0916副本路径仅历史。训练原生已跑通，FlagGems训练替换仍待做；下一轮别重复当成未开始。

10-10：三份对外文档已推送个人分支`2c73f48`，远端已核验；不再称这三份未发布。公共分支未合并，其余本地修改/旧稿未上传；此次无新增实验，后续环境仍需重新核查。

10-09远端已fetch但未merge：公共dev-1.0为`160ce4e`，设备个人分支为`3d72ef3`；新设备组件v0.3.0尚未全部进入公共设备子树。本人工作树保留未提交文档。部署前先与总组/设备方向确认组件与镜像组合，再更新服务器；本轮未SSH。旧共享模型路径被设备10-08报告记录为权重损坏，须核对可信来源哈希及原生有限性，不能直接复用旧挂载。当前对外三份材料见[文档导航](docs/README.md)，旧底稿保留。

## 工作区与协作

- 本人分支：`cgu135/framework-adapter`；09-23用户授权将09-22原型与证据按团队流程提交、合并推送。用户`.DS_Store`文件保留、不提交。
- 09-23发布前fetch公共分支到`6af89c4`，含`8cc9dbe`设备原型去torch_fl更新；发布同步不改变下表服务器0916运行副本，也不替代新基座回归。具体提交及合并位置查看Git历史，执行前重新fetch核对。
- 09-17历史成果已通过merge/push进入公共`dev-1.0@e9fd2ff`；PR #16关闭，不走原PR流程。后续按[根README](../../README.md)标准流程，在个人分支开发、同步解决冲突后合并；是否提交/推送以当轮授权为准，禁止强推共享分支。
- runtime-team保留原型、探针、部署和证据；正式子库代码独立维护。旧`vllm-plugin-FL`个人分支`cgu135/safe-op-fallback@635ff6d`不会随本仓推送，也不自动装入当前镜像。

## 最近使用的环境（运行前重新核查）

| 项目 | 记录 |
|---|---|
| 当前允许主机 | 27 / 10.120.72.11，账号cgu135；不要沿用旧五机列表授权 |
| 实测主机 | npu1-27 / 10.120.72.27，驱动26.1.1；11的Docker权限尚不足 |
| 本人容器 | flagos-proto-infer-910c，owner=cgu135，非privileged，单davinci0→npu:0 |
| 最后状态 | 09-22 18:17:35 stopped，SSH退出；[记录](docs/evidence-model-20260922/rms-final-state-20260922.txt) |
| 持久工作目录 | 宿主/home/cgu135/framework-adapter-910c/acceptance-20260916 → /work |
| 模型只读挂载 | /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B → /model |
| 初始化runtime | /work/dev/device-context/prototype，仍为0916保存副本，并非整个最新公共代码 |
| 依赖 | /work/FlagGems-f7ae8e6-20260922/src；/work/deps-20260922；仅目录覆盖，未改公共镜像 |

完整镜像digest、包版本及已知差异只维护在[实验报告§2](docs/910C模型接入与安全回退-20260922.md)与[证据](docs/evidence-model-20260922/README.md)。未引入FlagTree构建物。凭证不写记录，需要时读取用户连接材料。

## 接着运行

1. 重新检查资源、分配规则、容器归属、设备映射和Git状态；不能因上次空闲直接启动。
2. 在本人已准备好的容器内使用报告§5的环境变量；模型与回退测试看§5，RMSNorm诊断看§8。本地统一入口及代码分工见[README](README.md)。
3. 使用新结果文件名和有界超时；目标执行错误不吞掉、不自动重试/重置设备。不得停止他人任务。
4. 完成后下载原始结果及源码哈希、更新同主题日期报告，再更新STATUS。本人容器按实际使用收尾停止，记录状态；不擅自上传OneDrive或Git。

服务器文件名映射（不要覆盖旧失败快照）：

| 仓库probes/文件 | 容器/work文件 |
|---|---|
| qwen_embedding_baseline.py | qwen_embedding_baseline-v3.py |
| qwen_gems_validation.py | qwen_gems_validation-v3.py |
| qwen_scoped_adapter.py / test_qwen_scoped_adapter.py | 同名 |
| qwen_rms_shadow.py / qwen_rms_ablation.py | 同名 |

rms-shadow的completed只表示统计收集成功；rms-ablation要另看within_tolerance。模型最终向量通过不能抵消隐藏状态失败；不扩大默认准入，不放宽阈值掩盖问题。

## 历史与维护

下轮vLLM服务对接优先复用统一启动脚本与规范。09-22审查的是[个人分支脚本固定版本](https://github.com/FlagRT/runtime-team/blob/bdefeeea61f9d0f951f026e72990f672f0da5257/dev/device-context/prototype/scripts/serve_standard.sh)和[规范](https://github.com/FlagRT/runtime-team/blob/bdefeeea61f9d0f951f026e72990f672f0da5257/dev/device-context/prototype/docs/SERVICE_STARTUP_STANDARD_20260920.md)；09-23公共6af89c4已有prototype同步，后续以公共最新版本重新核对。已审查版本支持SERVE_FORM=embed；需明确MODEL/SERVED_NAME/DEV及日志目录。其清理按进程名范围匹配，不得直接在共享服务环境使用；本方向尚未运行。FAIL可能仍退出0，核验必须看ready/smoke/verdict。厂商服务已跑通不等于我们的FlagGems接入已完成。

旧环境、旧阻塞和此前步骤保存在[整理前快照](docs/历史入口快照-20260922.md)及[日期报告索引](README.md)，不在此追加流水账。新结论写STATUS，技术细节写报告，环境发生变化才改本文件。
