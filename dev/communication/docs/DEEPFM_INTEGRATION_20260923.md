# DeepFM 数据并行训练接入

_本轮开发记录：主机与910C双卡集成验证通过；下游确认和公共基座口径收口未完成。_

---

## 📋 范围与来源

复用 [公共 DeepFM v1](../../model-defs/deepfm/v1/README.md)，模型代码零改动，421,452 参数。
新增代码位于 [training/](../training/)，不复制或修改设备原型，不复跑 Qwen 作为新增成果。
设备 API 按 `kistich/device-context@bdefeee` 的 `runtime.use/device_count/set_device/synchronize/translate_error` 接入。
当前公共基线为 `dev-1.0@0ae2198`。用户已确认采用设备方向最新Route A；
本次复用其独立解释器，不修改公共镜像锁、不将镜像默认torch_fl解释器当成Route A。

采用成熟 PyTorch DDP 管理梯度桶及同步。观测钩子执行 SUM 后除以 world size，
并显式等待设备完成；这是同步诊断路径，不宣称异步重叠或性能优化。DDP 自定义通信钩子负责其自己的归约与平均语义。[^1]

## 📊 已取得的证据

### 910C 双卡集成

结果见 [NPU摘要](../results/20260923-npu/summary.json)、
[环境与启动器退出记录](../results/20260923-npu/execution.json)及同目录两模式逐Rank JSON。
实际运行代码为 `54652c0`，共享模型和设备原型零修改。
原训练镜像搭配只读挂载的 `venv-infer-a`：torch `2.11.0+cu130`、torch_npu `2.11.0`，
通过统一API选择 `ascend`，实际进程组为 `hccl`，pyACL可用。

| 检查 | 观测DDP | 原生DDP |
| --- | ---: | ---: |
| 两端训练步数 | 各50，另5预热 | 各50，另5预热 |
| 每端梯度参考检查 | 17/17 | 17/17 |
| 梯度最大绝对误差 | 4.66e-9 | 4.66e-9 |
| 一步更新最大绝对误差 | 3.73e-9 | 3.73e-9 |
| 最终跨Rank参数差 | 0 | 0 |
| 启动器退出码 | 0 | 0 |
| 单次计量吞吐（samples/s） | 9513.5 | 10510.1 |

两模式50步loss曲线逐Rank比较，最大绝对差均为 `2.384185791015625e-7`，
满足预设 `<1e-6` 对照阈值。观测模式每Rank记录100次梯度AllReduce、
84,290,400字节输入载荷，钩子累计耗时分别25.04/31.35 ms。
这是同步诊断钩子的单次运行数据，不是纯HCCL内核耗时、链路带宽或优化收益。
吞吐仅使用下文定义的计量区间，不是完整epoch端到端吞吐；未做重复运行统计或单卡加速比。

每端50个计量batch的样本ID区间已逐步检查：全局64、本地32，完整覆盖且不重复。
训练为合成数据通信集成检查，loss不单调，不据此声称模型业务收敛。

### 主机预验证

结果见 [主机摘要](../results/20260923-host/summary.json) 与同目录两模式逐 Rank JSON。
运行环境为本机 torch 2.4.1、CPU/Gloo；两个启动器均正常退出0，不代表 Ascend/HCCL 可用。

| 检查 | 结果 | 口径 |
| --- | --- | --- |
| 新增单测 | 14/14 | 分片、均值、后端拒绝降级、失败状态、销毁等主机测试 |
| 通信目录回归 | 39/39 | 含原有25项，不与设备测试相加 |
| 梯度参考 | 每端17组通过 | 同一全局batch、FP32、BN冻结、Dropout关闭的受控检查 |
| 参数更新参考 | 最大绝对差1.862645149230957e-9 | 一步SGD，相对CPU全局batch参考 |
| 正常训练模式 | 两端各50步完成 | 另有5步预热；Adam lr=1e-3，全局batch64、本地32 |
| 跨Rank最终参数差 | 0 | 检查可训练参数，不要求本地BN运行统计相等 |
| 原生/观测DDP对照 | 两端loss曲线最大差0 | 同种子和输入，分别运行；不是速度提升结论 |
| 梯度通信 | 每端100次，84,290,400字节 | 仅50个计量步；payload为输入张量大小，不是实际网络流量 |

上述39项为个人分支的完整回归记录。本次dev开发态交付只选取DeepFM新增14项，
未带入历史25项及其实现；在本批公共目录运行下面的单测命令应得到14/14，不是39项。

训练数据为固定种子生成的合成类别特征和确定性标签，不是Criteo或真实业务数据。
不以loss必须单调下降作为通过判据，不报告AUC或业务收敛。
正常训练开启模型原有BN/Dropout；其local BN统计与单进程全局batch统计不等价，
因此与受控梯度等价性检查分开，避免把BN差异误判成通信错误。

计量步耗时包含输入传输、前反向、梯度归约、优化器更新和结束同步，不含合成数据生成及
每步计量之后的额外有限值检查。钩子耗时含归约、除法和设备同步，不等于纯HCCL内核时间。
原生DDP模式未采集通信次数，字段明确标记未观测，不以0代替未知。

## 🔧 复现入口

在仓库根目录，使用已安装torch的解释器：

```bash
python -m unittest discover -s dev/communication/tests -v
python dev/communication/training/run_host_checks.py \
  --model-root dev/model-defs/deepfm/v1 \
  --output /path/to/new-host-result
```

输出目录必须不存在；监督器对每次双进程运行设120秒上限，检查进程退出、两端结果、
数据分片完整且无重复，以及原生/观测模式loss对照。CPU计时不可推导NPU性能。

交付前在精简目录复跑时，本机动态rendezvous发生建连重置，训练尚未开始。
仅主机监督器改为显式 `127.0.0.1` 的static rendezvous后，两模式各50步再次通过、loss差0；
NPU训练代码不变。端口由本机临时分配，若发生端口竞争则明确失败，不掩盖为通过。

以下为本次NPU运行方式；在获准的双卡容器内，使用上述Route A解释器。
`COMM_TEST_DEVICES`为容器内逻辑编号，本次物理14/15映射为逻辑0/1。
分别添加 `--comm-mode observed` 和 `--comm-mode native`，输出到不同新目录：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_RT_VISIBLE_DEVICES="${COMM_TEST_DEVICES:?先设置获准的双卡编号}" \
  timeout -k 10 300 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  dev/communication/training/deepfm_train.py \
  --backend ascend --runtime-root /path/to/approved/prototype \
  --prototype-revision bdefeeea61f9d0f951f026e72990f672f0da5257 \
  --model-root dev/model-defs/deepfm/v1 --steps 50 \
  --output /path/to/new-npu-result
```

必须用Route A解释器；程序经统一设备API触发厂商注册，再初始化HCCL，禁止CPU静默兜底。
不修改公共DeepFM模型目录，也不把另一个方向的整个分支直接带入公共分支。

等待其他成员的推理容器自然退出后才启动测试，未停止他人任务。
测试容器仅映射物理14/15，虚拟环境和宿主驱动只读挂载；每次启动器外层设300秒超时、
10秒强制终止宽限。两次均正常结束，停止测试容器前 `docker top` 仅剩保活sleep；
停止后UDA已无本次双卡独占归属。没有修改驱动、设备共享开关或公共容器。

本地结果核验入口为 `training/summarize_npu.py`，读取逐Rank记录与退出记录，
检查身份、分片、梯度及更新结果和两模式loss对照后生成摘要，拒绝覆盖现有摘要。

## ⚠️ 未关闭项

- 新原型已移除flagos后端，但公共训练锁仍是torch_fl例外；本次Route A已获用户确认并实跑，公共锁的修订仍须总组收口。
- NPU DDP钩子、HCCL同步和统一设备API已在本模型正常路径通过，不据此宣称全部通信算子/异步时序均已覆盖。
- 已记录末尾显存和单次计量吞吐，未形成峰值画像、多轮统计或单/双卡性能基线。
- 日志提示HCCL执行超时默认值大于进程组60秒超时；本次正常完成未验证故障路径的超时/恢复行为。
- 超时依赖进程组设置与外部监督器；不承诺底层卡死可以由进程内wait可靠中断。
- 错误按统一API翻译后上抛，未实现自动重放、设备重置或检查点恢复。
- 至少一个下游方向的确认仍待取得；不将主机通过或个人提交等同于dev合入验收完成。

本批仅发布开发态代码与可复现证据，不修改公共基座或宣称下游验收通过。
复现需要单独取得设备方向 `bdefeeea61f9d0f951f026e72990f672f0da5257` 的prototype，
并通过 `--runtime-root` 指定；本批不复制其代码，也不合并设备方向整分支。

[^1]: PyTorch. DistributedDataParallel.register_comm_hook. https://docs.pytorch.org/docs/2.4/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook
