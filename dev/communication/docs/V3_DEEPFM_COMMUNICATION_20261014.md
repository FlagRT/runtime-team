# DeepFM：v3 基座梯度聚合接入与重复验证

_本方向负责梯度聚合、训练一致性和通信组生命周期；复用设备原型，不重复申报设备 conformance。_

---

## 📋 与组内对齐

公共分支以 `c81479eda2dad3a1ddb847e9467f96bc97154cb0` 为起点，核查基座发版、总组裁定及近期 PR（性能方向 #22 已合入）。
设备原型实际固定在 `3c62d59f0becaa503036ce118e65ced62ae40aad`；测试期间新增 `2c0434e` 主要修订 PPU/MLU 归因和证据留存，不修改本次消费的 runtime 实现。
继续使用 `use/device_count/set_device/current/synchronize`，有界同步传入 `timeout_ms=30000`。设备恢复成功不代表 c10d 通信组恢复；本实现失败后封闭通信组并上抛，不在单 Rank 内重试。

[v3 锁定声明](../../stack.lock.910c.v3.yaml)现要求 Route A + FlagCX，并明确原生 DDP、列表 all_gather 及退出清理的已知问题。因此原定“原生/观测 DDP 各三轮”调整为手动逐张量/合并梯度对照。未把旧 torch 2.11/HCCL 环境继续当作新版基座，也未修改公共镜像锁。

## 🔧 本周实现

在已有 DeepFM 入口增加 `manual`、`fused` 两种显式聚合方式：初始化和 BN buffer 从 Rank 0 广播，等长数据分片，梯度 SUM 后除以 world size。合并路径把同设备、同 dtype 的稠密梯度拼接后一次 AllReduce，再拆回各参数；不支持 unused/sparse 梯度或不等长输入，不宣称通信计算重叠。

新增集合通信前后设备同步，分别保证输入生产完成、结果可读；同步异常原样上抛并禁止继续使用该组。通信计数只统计梯度归约，不包括 buffer 广播、有限值检查和参数核查；字节数是每 Rank 输入张量载荷，不是网络实际流量。

重复入口 [run_repeated.py](../training/run_repeated.py)交替执行两模式各三轮，固定随机种子、模型和 batch，每轮独立进程组，5 步预热 + 50 步计量。启动超时只终止本次启动器进程组，任何非零退出停止后续重复。结果在清理前持久化，再记录清理结果；即使模型返回 PASS，启动器非零或组未关闭也不能被汇总器判为整体通过。

## 📊 实验口径与证据

共同配置：公共 `dev/model-defs/deepfm/v1`，421,452 个参数，FP32，global batch 64 / local batch 32，seed 923。梯度参考使用全局 batch CPU eval（冻结 BN、关闭 dropout），训练使用 train 模式、本地 BN 和 dropout=0.2。合成数据仅验证通信与数值一致性，不用于 CTR 指标或业务收敛结论。

CPU/Gloo 六轮已通过，[原始结果与汇总](../results/20261014-host/summary.json)独立归档，不代替 NPU 验收。
主机单测共 32 项通过（既有 14、原型消费侧 5、手动聚合 7、证据负向校验 6），见[测试日志](../results/20261014-checks/)。新增 13 项，不把原型内部 conformance 计入本方向。

910C 使用 npu1-11 的统一 v3 完整镜像 `e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15`，静态核验 FlagCX `0.13.0@4e0e0cb`。实验前无运行容器、无 NPU 计算进程。仅映射物理 0、1，独立网络/IPC、1 GiB shm；相对公共 demo 的全卡与 host 配置收缩权限，镜像、驱动和软件组件不变。仅运行本人容器，结果写共享存储并同步 Mac。

### FlagCX：未通过，不掩盖退出失败

初始尝试触发 SIGABRT，清理前证据尚未持久化；增加持久化后重跑，确认训练末尾参数检查本身也失败。再加通信前同步做定位对照，仍有最终跨 Rank 参数差 `0.02475881576538086`。两端 17/17 梯度参考通过（最大误差 `4.190951585769653e-9`），50 步 loss 和参数均有限；不能因此忽略参数不一致。

[前后同步定位证据](../results/20261014-flagcx-presync/)包含两 Rank JSON、启动日志与退出码；[首次尝试](../results/20261014-flagcx-initial/)和[清理前落盘定位尝试](../results/20261014-flagcx-diagnostic/)也保留。另测合并梯度路径，50 步后参数差 `0.016498330980539322`；将两端 421,452 个参数分别导出，再用不调用 c10d 的 NumPy 比对，确认不是广播自检误报，见[独立参数比对](../results/20261014-flagcx-hostcheck/01-fused/independent_comparison.json)与同目录原始 `.npy`。

四次启动均未通过，非性能采样，不拿失败轮计算收益。清理触发 `free(): invalid pointer`，与发布说明 A-3 现象一致；参数差的根因尚未确定，不能仅据此断言是某个 FlagCX 内核错误。HCCL 对照通过将排查范围缩至当前 FlagCX 消费路径及其交互，仍须进一步找出首次分歧步骤。

追加逐步诊断：新增 [trace_deepfm.py](../training/trace_deepfm.py)，在初始化、反向后、归约后、更新后分别保存完整逐参数主机数组。[十步独立 NumPy 比对](../results/20261014-deepfm-trace10/comparison.json)中，10/10 步的两端归约梯度差、参数差、相对于两端本地梯度主机均值的误差均为 0；清理仍 SIGABRT、启动器退出 1。先行三步诊断得到同样现象，摘要保留在 `results/20261014-model-extension/trace3-preliminary-comparison.json`，它的启动退出信息仅在交互记录中，不作为完整验收证据。十步有完整启动日志及退出记录。

这不是原问题修复：主机复制/落盘改变了执行时序，且该最小入口没有旧入口前置 CPU 梯度 oracle。当前只能确认更强的主机观察下未复现分歧，下一步要分别控制前置 oracle 和主机复制两个变量。不能把零误差覆盖原 50 步未插桩失败证据。原始十步数组保存在共享测试目录 `trace-flagcx-fused10/` 并同步 Mac 周报证据目录；公共摘要包含每个原始文件 SHA256 和逐参数差值，可复跑脚本重新生成。

### HCCL：同镜像诊断对照

[六轮实测汇总](../results/20261014-hccl-control/summary.json)：两模式各三轮全部通过，启动器均退出 0、通信组 CLOSED，每端 17/17 梯度对照通过；最大梯度误差 `4.19e-9`、更新误差 `3.73e-9`，最终跨 Rank 参数差 0，两模式 loss 曲线最大差 `7.15e-7`。

| 指标 | 逐张量 | 合并梯度 |
| --- | ---: | ---: |
| 每端每 50 步梯度 AllReduce | 850 | 50 |
| 每端输入张量载荷 | 84,290,400 B | 84,290,400 B |
| 每轮平均 step 耗时的均值 | 11.425 ms | 7.223 ms |
| 三轮平均耗时的样本标准差 | 0.163 ms | 0.556 ms |

调用次数减少 94.1%，载荷不变。耗时统计单位是独立运行，不把两个 Rank 或相关的 50 个 step 当成独立样本。step 包含前后向、buffer 广播、梯度打包/拆包、同步和更新，不含数据生成与 step 后校验；本配置仅三轮，不能推广成业务端到端性能收益。**HCCL 是同镜像诊断对照，不是默认 FlagCX 路线通过。**

## ⚠️ 运行准备与待确认

27 号服务器已能连接，共享目录挂载正常；未对 RAID 旧数据做删除或搬移。因运行中的容器不符合本次 v3 基座且存在多容器设备映射，改用用户提供的 11 号机器，没有停止他人容器。

P800 初次核查账号没有 Docker/sudo 权限，用户协调恢复后重新登录确认已加入两组。检查时 8 卡均 0 MiB/0% 使用；Docker 所在盘约 84 GiB 可用（99% 使用），根盘约 5.4 GiB 可用。使用已发布 kunlun v2 `cac5ddae5f62`，仅授予同 NUMA 物理卡 2/3 与 xpuctrl，容器内重编号 0/1。环境检查退出 0：Python 3.10.18、torch 2.9.0+cu129、FlagTree 0.6.1+xpu3.6、FlagCX 包自报 0.10.0，flagcx 后端已注册、可见卡数 2，见[准备证据](../results/20261014-p800-preflight/)。不以昇腾 FlagCX 0.13.0 版本号替代昆仑实测值。

迁移保留 K-1～K-4 限制：不启用 `XPU_EVENT_KL3_ENABLE`，不重装 FlagTree/torch，不启用有缺陷的 FlagGems GEMM。临时环境检查容器使用独立无网络配置，已自动删除。本周仅完成环境准备，**未执行 P800 集合通信或训练验收**。

下一步优先缩小 FlagCX 参数不一致的最小复现，向通信库方向提供当前消费侧证据；P800 执行同模型迁移。整组重建和检查点恢复仍需设备/调度联合确认，目前没有收到下游确认，不计作完成。整体仍按原任务表 5/37=13.5%，月度仍按已验收交付 1/4=25%；910C 新基座测试已执行，但默认 FlagCX 路线未闭环，不用 HCCL 成功顶替。

## 🔗 复现

设备原型从上述固定提交取 `dev/device-context/prototype`，不使用未核验的同名目录。GPU/NPU 检查前确认镜像及空闲卡。以容器 `/workspace` 为例：

```bash
python dev/communication/training/run_repeated.py \
  --backend ascend --collective flagcx \
  --runtime-root /workspace/dev/device-context/prototype \
  --prototype-revision 3c62d59f0becaa503036ce118e65ced62ae40aad \
  --model-root /workspace/dev/model-defs/deepfm/v1 \
  --output /workspace/new-flagcx-results --launch-timeout 240
# HCCL 诊断对照显式改 --collective hccl，并使用不同的 output。
python dev/communication/training/summarize_repeated.py /workspace/new-results
```

汇总器拒绝失败启动器、缺少 Rank、环境不一致、不完整步数、错误分片/通信计数或两路径 loss 不一致。FlagCX 已知失败目录不应得到 PASS。证据目录必须全新，禁止覆盖后挑选成功结果。
