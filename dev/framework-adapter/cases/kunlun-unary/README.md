# P800单算子与模型接入验证

## 用途与现状

发布范围（2026-11-03）：这是10-29实测的集成材料与CPU控制测试。功能包dev2源码现已固定为[6ee9129](https://github.com/FlagRT/framework-adapter/tree/6ee91294e7ab028f2e804225482f58d24c8ccd5a)，按[依赖锁](../../package.lock.json)构建；不包含下载即用的wheel、权重或基座镜像。新构建的10个运行时文件与历史实测wheel逐字节一致，但wheel归档哈希不同；复跑必须记录实际制品、重新核环境资格，不宣称已在新env/v1通过。见[发布审查](../../docs/dev2发布审查-20261103.md)。

用昆仑芯公共v2镜像验证本方向适配包的SiLU、sigmoid、tanh前反向，以及关闭、开启和缺资格保留原生；模型入口支持Qwen3-Embedding-0.6B、公共DeepFM与OneRec-8B。每次仅挂一张空闲P800，复用device-context的kunlun后端，不修改算子或公共环境。安装包必须离线安装，不能用宿主功能源码代替wheel。

本case不启用env/v1中的GEMM兼容补丁，只使用镜像固化的FlagGems和FlagTree。三种单算子及DeepFM两个batch短训练均通过；Qwen0.6B原生/off/unqualified通过，但on首步梯度超差，仍保留原生。不能将一个模型的通过推广到其他模型或完整三级回退。最新实测见[P800报告](../../docs/P800接入与安全回退验证-20261029.md)。

## 复现

| 前置项 | 要求 |
|---|---|
| 镜像 | `flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`，ID `sha256:cac5ddae5f62b356622858dd4f0ea0f82d489cf0cb71e7230f0f97797c25605a`；宿主已缓存，不自动拉取 |
| 公共材料 | runtime-team提交`71197e73d13021556de04ff04c42c89b3bfc8f73`的dev/lib、dev/images/kunlun-operator-runtime/v2、dev/device-context/prototype；使用独立checkout或该提交快照，不切换共享仓 |
| 设备 | 1张P800，`--device`指xpu-smi显示编号，启动前须0MiB、0%、无进程；按PCI地址匹配/sys/dev/char与实际/dev/xpuN后挂载，不能假设两个编号相同；容器内逻辑号0 |
| 功能包 | 当前模型入口使用0.3.0.dev2候选wheel，新增schema 3模型审核；历史单算子与模型成绩使用dev1，不混称同一实测包。精确SHA和结果见日期报告 |
| 宿主工具 | Python3、PyYAML、Docker Compose v2及当前用户的Docker使用权限；不自动改用户组 |
| 存储 | 结果目录所在盘至少2GiB可用；不在满系统盘下载模型或编译缓存 |
| 模型与数据 | 单算子不需要模型；模型阶段需下述只读权重/公共源码及固定合成数据，不下载业务数据 |

从本case目录执行，输出目录每次必须不存在；路径替换为本人的实际路径。若Docker需sudo，使用sudo执行同一入口，临时目录放数据盘。

```bash
python3 assets/run.py --public-root /data2/USER/public-snapshot \
  --wheel /data2/USER/adapter.whl --output /data2/USER/inspect-01 \
  --device 5 --stage inspect
```

后续首次资格实验将stage改为`qualification`；它只产出结果，不签发passed记录。审核实际命中、数值和恢复证据后，按独立包的资格流程填写记录。

单算子默认`--operator silu`；改为`sigmoid`或`tanh`时固定测试形状16、32，FP32/FP16/BF16、各重复3次。sigmoid和tanh的反向接口接收前向输出，而SiLU反向接收原始输入；探针和适配包分别核对该约定，不能直接改个函数名复用输入。

正常机制验证使用`--stage routes --mode off`、`--mode unqualified`或`--mode on --records /绝对路径/records.json`。开启必须显式提供已审核证据；缺资格模式禁止带记录。所有阶段都由入口生成demo和env.baseline.yaml，再经公共`up.sh --strict`启动；不要直接复用历史输出目录中的compose。

仓内[demo样本](demo/docker-compose.yml)与[基线](demo/env.baseline.yaml)原样保留10-29开启路径的实际启动记录，含当时宿主路径，不能直接当成他人机器的推荐配置。复跑以单入口生成的新目录为准。

## 验证

### 模型审核配置

当前模型适配路径必须传`--adapter-config`，不能仅凭单算子`--records`开启。审核同时匹配实际环境、模型文件、训练配置（含batch/步数/优化器/种子）、公共代码及所选算子组合；通过后仍检查单算子前反向资格。原生/native_repeat不用审核文件。训练CLI与该单卡case是不同入口，本case不扩大昇腾双rank CLI的范围。

[review-20261029](review-20261029/model-reviews.json)是执行者核对既有实验后填写的限定审核包：DeepFM batch32/FP32/3步可实验启用，Qwen FP32/seq32/3步已失败。附带两个精确FP32算子资格及原始结果SHA；不是自动批准或生产验收。模型设置不同则默认无审核，不能修改摘要伪造通过。

| 配置文件及mode | 预期行为 |
|---|---|
| deepfm-on.json / on | 模型审核和sigmoid单算子资格都匹配，严格目标 |
| deepfm-unknown.json / unknown | 故意只有Qwen审核，DeepFM保留原生，workload_unverified |
| qwen-denied.json / denied | 已知失败，训练走原生，workload_denied |
| qwen-strict.json / strict | 同一失败记录，但要求目标，模型加载及optimizer创建前阻断 |

配置中的require_target不是精度检测开关。strict测试进程返回3、result.status=blocked，外层只有确认原因和零模型训练后才把负向用例标为PASS；不能写成“目标训练成功”。修改审核文件、环境或执行条件后须重新核验，旧原始日志/记录不覆盖。

### 公共DeepFM短训练

使用公共`71197e7`的`dev/model-defs/deepfm/v1`只读快照，三份源码SHA在`assets/deepfm_probe.py`固定。保留421,452参数的完整参考架构、BatchNorm、0.2 Dropout及Adam学习率1e-3；随机初始化，使用固定合成分类特征与二值标签，不下载业务数据、不评价CTR质量。

```bash
python3 assets/run.py --public-root /data2/USER/public-snapshot \
  --wheel /data2/USER/adapter.whl --output /data2/USER/deepfm-b32-native-01 \
  --device 1 --stage deepfm --mode native \
  --model /data2/USER/models/dev/model-defs/deepfm/v1 --batch-size 32 --steps 3
```

随后使用新输出目录，指定`--mode on --adapter-config /data2/USER/review-20261029/deepfm-on.json --reference /data2/USER/deepfm-b32-native-01`；错误审核对照使用`--mode unknown`和deepfm-unknown.json。两者除配置/模式/输出外保持参数相同。batch16历史测试通过，但本轮提供的模型审核仅含batch32，不能直接用于16；不同batch须独立基线和审核。开启应各命中1次目标前反向/步，unknown应命中1次workload_unverified原生/步；其他算子保留原生。

每步比较prediction、loss、17个梯度、28个完整状态张量（包括BatchNorm缓冲区）及51个Adam状态，共98项，3步294项。对照要求逐值一致；目标沿用DeepFM的atol=1e-5、rtol=1e-4。预测、loss或梯度失败不执行该步更新；更新后的状态失败则停止后续步骤，并如实记录已经发生的optimizer次数，不宣称回滚。原生重复不一致时，不依据后续目标差异判断算子问题。每次最多900秒，结果盘至少2GiB，不更改共享模型源码。

### 0.6B单卡短训练

单算子资格通过后，用同一入口的`--stage model`做完整Qwen3-Embedding-0.6B短训练。复用公共`runtime.use('kunlun')`与`model_path.require_model`，训练方式沿用公共训练腿的FP32因果语言模型损失和AdamW；单卡不运行通信验收。固定batch=1、seq=32、合成token、确定性模式，最多3步。权重SHA256须为`0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`，输出盘需至少24GiB；模型只读挂载，不下载、不改共享缓存。

```bash
python3 assets/run.py --public-root /data2/USER/public-snapshot \
  --wheel /data2/USER/adapter.whl --output /data2/USER/qwen-native-01 \
  --device 5 --stage model --mode native --model /data2/USER/model-cache --steps 3
```

原生完成后，用新输出目录和`--mode denied --adapter-config /data2/USER/review-20261029/qwen-denied.json --reference /data2/USER/qwen-native-01`验证已知失败保留原生。严格阻断用`--mode strict --adapter-config /data2/USER/review-20261029/qwen-strict.json`，不传reference，因为不得进入模型训练。其余参数与原生一致。模型可传普通目录或单模型HF缓存根；HF根需连同blobs/snapshots一起只读挂载，由公共函数解析。比较路径核对环境、权重、初始参数、工作负载及公共代码身份，不能串用不同基线。

10-29该模型on已两次数值失败，本轮不重跑该已知失败组合、不改内核或阈值。旧[单算子格式拒绝记录](../../docs/evidence-p800-20261029/records-model-denied.json)作为历史证据保留，当前用新增模型审核表达它的实际作用范围。denied与strict不是数值问题修复，也不是执行后重试。

每步对比loss、全部310组梯度及更新后310组参数，共621项；3步为1863项。原生重复、关闭和缺资格要求逐值一致；开启沿用0.6B模型阈值atol=1e-5、rtol=1e-4，不为结果调宽。开启每步要求真实28前向+28反向，关闭/缺资格每步要求28次正确原因的原生调用，退出恢复必须通过。loss/梯度失败时不执行该步optimizer；更新后参数失败则停止后续步骤，不宣称已做参数回滚。原生张量保留在输出的tensors/，仓内只放摘要；单轮进程上限900秒。

### OneRec完整权重短训练

使用公共清单中的`OpenOneRec/OneRec-8B`，固定revision `29f95b3da63a4bbc00a1cd3aeee948efb2ff40ff`。`onerec_probe.py`核验3个权重分片SHA、Qwen3架构、完整8,391,013,376参数与399参数张量。BF16、batch1/seq16、非重入梯度检查点、SGD lr=0.01无momentum，1—3步合成token因果语言模型训练；其余算子保持厂商原生。沿用既有OneRec目标判据atol=1e-3、rtol=1e-2，原生路径要求逐值一致。

```bash
python3 assets/run.py --public-root /data2/USER/public-snapshot \
  --wheel /data2/USER/adapter.whl --output /data2/USER/onerec-native-01 \
  --device 1 --stage onerec --mode native --steps 3 --model /data2/USER/OneRec-8B
```

原生每步保存loss、399项梯度和399项更新后参数，共799项；三步2397项。完整基线张量约94GiB，入口要求输出盘至少140GiB可用（3步），容器RAM上限96GiB，一张96GiB P800；输出、缓存及临时目录全部落本人数据盘。权重与reference只读，每次最长1800秒，退出停止本人容器。不要把短序列三步通过写成长训练或业务质量达标。

首次模型审核分两步，不预先给新模型填写passed：

1. `--stage qualification --operator silu --shape-profile onerec16`先测1×16×12288的FP32/FP16/BF16前反向。人工核对命中、数值与退出恢复后，提供本实验单算子记录。
2. `--stage onerec --mode candidate --records /data2/USER/review/records.json --reference /data2/USER/onerec-native-01`是显式模型资格实验，必须有原生对照；它用既有单算子资格执行候选，但不产生模型批准。loss/梯度超差时停止、该步不更新，更新后参数超差则停止后续步，不宣称回滚。

普通`on`路径仍只接受schema 3的`--adapter-config`；必须另审核候选实测结果并填写精确模型记录后才能开启，不能用candidate当成部署开关。先用同一reference做native_repeat确认原生重复性。所有路径使用新输出目录；资格失败不放宽判据、不改内核。

本次候选首步Embedding梯度超差，模型审核为failed，没有提供passed/on配置。后续使用[off](review-onerec-20261029/off.json)与[denied](review-onerec-20261029/denied.json)，分别以mode off/denied和同一reference验证关闭/已知失败保留原生；[strict](review-onerec-20261029/strict.json)使用mode strict、不传reference，要求加载模型和导入FlagGems前阻断。[unknown](review-onerec-20261029/unknown.json)是可选无审核示例，不计作本轮已执行硬件路径。精确失败及执行状态见日期报告。

### 检查项

| 脚本 | 类别 | 通过判据 |
|---|---|---|
| assets/run.py | 复现和验证 | 严格镜像检查、执行包身份、输入快照未变、probe通过、本人容器确已停止 |
| assets/probe.py | 验证 | 原生CPU对照；目标真实前反向命中；原定容差内；关闭/缺资格0目标调用且与原生逐值一致；每次注册退出恢复 |
| assets/test_runner.py | 验证 | 5项CPU控制：空卡、占用/进程拒绝、芯片识别及预装构建产物白名单；不是硬件成绩 |
| assets/run.py中的PCI匹配 / test_runner.py | 资源保护 | 另3项CPU控制：监控号与minor号不同、映射缺失/歧义拒绝、映射匹配仍拒绝忙卡；实机映射写入device-mapping.json |
| assets/run.py中的workload_labels / test_runner.py | 记录准确性 | 另2项CPU控制：模型/单算子的算子名和batch标签按真实任务填写，不沿用DeepFM默认batch32；test_runner共10项 |
| assets/model_probe.py | 验证 | 固定权重完整模型，3步逐loss/梯度/参数比较，真实目标命中/原生原因及退出恢复；模型数值失败不转同进程原生重试 |
| assets/test_model_probe.py | 验证 | 7项CPU控制：逐值/容差比较、非有限值和类型拒绝、基线身份与状态检查、有限条差异样本、等值快路径与原判据等价；不是硬件成绩 |
| assets/deepfm_probe.py | 验证 | 公共DeepFM、固定初始化/Dropout/数据、单sigmoid替换、逐预测/梯度/模型及Adam状态对照 |
| assets/test_expansion.py | 验证 | 5项CPU控制：三种反向参数语义、算子日志隔离、源码拒绝、固定合法数据与BatchNorm缓冲区身份 |
| assets/model_admission.py / test_model_admission.py | 集成与控制 | 从实际模型/配置采集身份；5项CPU测试覆盖身份变化、模式不颁发资格、严格拒绝、晚拒绝不能算通过及旧schema不能开启模型 |
| assets/onerec_probe.py / test_onerec.py | 集成与控制 | 完整8B身份、BF16及限定形状；7项CPU控制覆盖原架构/阈值、候选实验边界、普通on仍需模型审核和dtype身份绑定 |

单算子SiLU阈值沿用已有记录：FP32为1e-5、FP16为5e-3、BF16为3e-2，同时用于atol和rtol；模型阈值见上节，均不因失败放宽。单算子每轮最多300秒，Qwen/DeepFM最多900秒、OneRec最多1800秒；执行中异常停止，不在同进程切回原生重试。

目标环境有两个需显式记录的兼容细节：PyTorch设备名为`GPU`，不能靠它判断厂商，需同时核宿主P800、torch_xmlir和昆仑驱动；镜像FlagGems仓根自带4个未跟踪构建产物，源码无diff。仅对该精确镜像和4项精确路径允许采用实际包内容哈希，不将其宣称为干净Git身份，也不允许新增源码改动。默认功能包Git检查仍拒绝脏仓库。

完整结果、执行命令、镜像/设备映射和源码哈希在各次输出目录；仓内只保留日期报告和必要证据。本人容器保持停止供追溯，不自动删除其他对象。
