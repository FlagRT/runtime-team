# 官方910C镜像：公共算子接入回归

## 用途与现状

消费官方镜像`quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`，精确ID见image.baseline.yaml；用于单卡PyTorch eager下SiLU/sigmoid接入、执行前回退和短训练回归，不调用vLLM服务。不是训练v3验收，也不修复shmem。

镜像基线逐字引用runtime-team公共863315d的`dev/images/ascend-infer-vllm/v1/image.baseline.yaml`；本方向不重新构建该镜像。runtime、依赖与方向代码为只读快照，runner记录文件哈希；FlagGems挂载仓由公共baseline检查提交。真实使用前必须重新确认空卡。

支持边界：独立单线程eager进程、单卡、一阶梯度；默认保留原生，只有显式审核记录匹配才启用目标。未支持compile、融合、自定义算子、多卡或长期训练准入；执行后异常停止，不重试原生。

## 复现

| 前置项 | 来源 / 约束 |
|---|---|
| 镜像 | 上述官方镜像精确ID；本例27本地已有 |
| 公共工具 | runtime-team提交863315d0389e6262d26e7880ec7fbc62e4b55919的dev/lib，原样导出 |
| 宿主工具 | Python3＋PyYAML、Docker Compose v2；本轮工具仅放本人目录，非系统安装 |
| FlagGems | f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4干净源码仓；挂/gems |
| runtime | device-context原型快照，挂/runtime；当前历史快照内容在snapshot-manifest.json记录 |
| 额外Python依赖 | 本人deps-20260922只读挂/deps；逐文件hash记录，不假称镜像内置 |
| 模型/数据 | sigmoid为随机初始化16→64→2 MLP，固定种子合成数据，不下载权重；qwen阶段才需源模型及其审核记录 |
| 设备 | 1个已协调的空闲物理设备；runner内部逻辑0 |

在27本人目录，公共工具与源码准备好后，一条命令执行资格阶段（下列路径是本次落地路径，其他机器需按实际设置）：

```bash
DOCKER_CONFIG=/home/cgu135/framework-demo-20261018/tools PYTHONPATH=/home/cgu135/framework-demo-20261018/tools python3 /home/cgu135/framework-demo-20261018/source/cases/ascend-official-unary/assets/run.py --public-root /home/cgu135/framework-demo-20261018/public --source /home/cgu135/framework-demo-20261018/source --gems /home/cgu135/framework-adapter-910c/acceptance-20260916/FlagGems-f7ae8e6-20260922 --deps /home/cgu135/framework-adapter-910c/acceptance-20260916/deps-20260922 --runtime /home/cgu135/framework-adapter-910c/acceptance-20260916/framework-v03-NwoBpG/dev/device-context/prototype --device 0 --stage qualification --output /home/cgu135/framework-demo-20261018/qual-01
```

runner生成该次demo/env.baseline.yaml，执行镜像check及公共up.sh --strict，然后限时运行并自动停本人容器。原始结果不覆盖，换输出目录才能重跑。模板中的SET_BY_RUNNER由入口替换，不直接执行模板Compose。

资格实验与正常调用分开：qualification仅输出sigmoid(32,64)三种dtype前反向证据；执行者阅读证据后单独审核，填写与证据同目录的records.json。不能由runner自动签发passed。

## 验证

同一入口按`--stage`选择阶段，其余环境参数相同：

| stage | 类别 | 额外参数 | 判据 |
|---|---|---|---|
| qualification | 验证 | 无 | 独立候选3种dtype输出/梯度及注册恢复 |
| sigmoid | 验证 | --records 审核记录绝对路径 | 四条各5步轨迹：原生、目标、关闭、无资格；全部梯度/权重/重载预测对照 |
| unary | 验证 | --records 上轮两算子的审核记录 | 12组目标前反向＋12组回退 |
| qwen | 验证 | --records 上轮审核记录 --model 已核源权重目录 | 0.6B推理及3步SiLU训练回归 |
| dc-train | 上游训练入口集成 | --second-device 第二个空闲物理设备 --training-config off/on/unqualified --model 已核源权重 --records 审核记录 | 双进程各3步、上游各6项检查、每进程84次路由及真实前反向内核计数 |

每次`--output`必须是新的目录。PASS退出0；结果不符FAIL退出1；环境不满足ERROR退出2。summary.json、result.json、基线检查及test.log保留在该目录；完整日志不入仓。容器停止失败也不能判PASS。

sigmoid模型正式回归增加`--deterministic`，开启PyTorch确定性执行而不改容差。诊断模式`--native-repeat-only`只跑两条无适配器的原生轨迹，用来区分原生非确定性与接入问题；它不是目标实现通过的证据。10-18初次默认模式及原生重复对照均在loss处出现1.1921e-7差异，失败证据保留，不能把默认模式写成逐值复现通过。

模型对照容差atol=1e-5、rtol=1e-4，回退轨迹要求逐值相同；所有4组参数张量均比较，不抽样。目标轨迹每步应命中1次sigmoid前向和反向，两种回退轨迹目标命中必须为0。只验证机制，不以loss下降或分类准确率作为交付指标。

最近实测结果及证据位置见[报告](../../docs/910C公共适配器实测-20261018.md)。未推送的快照不等于已发布package；本例没有镜像重建配方，继承公共官方基线。

dc-train使用[训练配置](../../demo/training-configs/README.md)，继承同一精确官方镜像并生成独立baseline/demo；映射两个设备，host网络/IPC、48GB内存上限，用于HCCL。不是训练v3镜像验收，也不是所有模型通用入口。必须先确认两设备当前空闲且可初始化。开启组仅对审核通过组合放行；不满足条件组不给记录并检查明确的native原因，不能以仅跑通模型代替路由验证。

安装包验收增加`--wheel /绝对路径/flagrt_framework_adapter-0.2.0-py3-none-any.whl`（仅dc-train）。此模式不挂载/code源码，只挂载wheel和配置；在每个新容器中离线、无依赖安装到/results/installed，不改镜像原有torch。随后检查实际导入位置、以torchrun --module调用包入口，继续使用相同的双rank路由/内核/训练判据。install.log、installed-import.log和summary里的wheel SHA256均需保留。source参数在此模式仅供宿主case与配置定位，不提供容器中的Python实现。

10-18实际结果：qualification、确定性sigmoid、unary三个stage PASS；默认sigmoid与原生重复诊断FAIL并保留[判定截选](assets/evidence/20261018-summary.md)。qwen选项尚未通过本入口重跑，0.6B既有证据不重复计入此次demo验收。demo/env.baseline.yaml是已执行样本，实际运行时由runner重生成。
