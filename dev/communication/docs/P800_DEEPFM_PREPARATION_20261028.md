# P800 DeepFM消费侧适配与单卡准备验证

_多卡通信方向增量：复用设备原型，补齐模型通信入口与可审计的选卡、执行、结果校验；尚不计双卡迁移验收。_

---

## 📋 当前结论

在公共Kunlun v2锁定镜像中，物理卡5完成DeepFM逐张量/合并梯度两条路径，各预热5步、训练50步，两个启动器退出0、通信组CLOSED。每次17/17组梯度参考通过，梯度与更新最大误差均为`3.73e-9`，两模式loss逐步完全一致、最终参数SHA256相同。见[严格汇总](../results/20261028-p800/deepfm-single-v2/summary.json)及[原始Rank记录](../results/20261028-p800/deepfm-single-v2/)。

这是`world_size=1`的消费侧迁移准备：证明模型、原型、FlagCX入口和正常销毁路径可以在该P800环境运行，不能证明跨卡通信、跨Rank一致性、扩展效率或故障恢复。整体交付仍7/37=18.9%，月度2/4=50%；双卡迁移与下游确认未闭环。

## 🔧 复用边界与环境

执行前再次fetch两个仓库，runtime-team公共版本为`fbac57b6df456f052db83fa47ea423cc0e8ec8a1`，FlagCX公共版本为`89dfaa0dbaed78d263b94696f6e7bfb6f1034abb`，未发现新的公共契约变更。设备原型、DeepFM v1模型定义和公共基线工具保持原样；通信侧只增加`--backend kunlun`、设备身份断言、选卡入口及证据校验，不复制设备conformance或Qwen训练腿。

按[昆仑芯锁](../../stack.lock.kunlunxin.v1.yaml)使用[Kunlun v2镜像基线](../../images/kunlun-operator-runtime/v2/image.baseline.yaml)，镜像ID为`sha256:cac5ddae5f62b356622858dd4f0ea0f82d489cf0cb71e7230f0f97797c25605a`。实际解释器为`/root/miniconda/envs/python310_torch29_cuda/bin/python3`，Python3.10.18、torch2.9.0+cu129、镜像固化FlagCX0.10.0（`3b43bdb237fc93736eddf3e118edf32f664b7861`），不是910C挂载覆盖的FlagCX版本。未向P800套用昇腾补丁、改公共镜像或安装宿主依赖。

`runtime.use('kunlun')`后必须同时满足厂商名`kunlun`、设备命名空间`cuda`及足够的可见卡数，不允许CPU或其他厂商回退。遵循锁定规避，不设置`XPU_EVENT_KL3_ENABLE`、不启用FlagGems。旧原型顶部仍会输出“建议设置KL3”的警告，本轮以公共锁与其结构化known_issues为准，没有修改设备方向源码。

启动前只选显存0 MiB且利用率0%的卡，查询实际minor和UUID，再在容器内核对完整可见UUID集合。卡5实际是`/dev/xpu7`，UUID为`GPU-f396486d-9850-50e4-81c4-50f2e9f6ca87`，不能把物理卡号当设备节点后缀。单卡[小算子选卡诊断](../results/20261028-p800/health-selection/probe.json)的填充读回、32²矩阵乘和Event完成检查均通过，耗时1.44 s；它不是重复申报设备验收。

独立测试容器仅映射选中设备和`/dev/xpuctrl`、私有IPC、2核/8 GiB内存，按`baseline.py export`及`up.sh --strict`启动。[实际环境核对](../results/20261028-p800/env-single/.last-check.json)通过。两路径结束后已停止自有容器，没有停止或重置他人容器/卡。测试文件保存在P800的独立`/data2/lianzhongyou/runtime-tests/20261028-communication/`目录；代码下载、提交和推送均在Mac。

## 📊 单卡模型实测

复用421,452参数DeepFM、FP32、全局/本地batch64、种子923和合成标签。参考阶段冻结BN/dropout，与CPU全局batch梯度及SGD更新对照；正式训练恢复本地BN和dropout，5步预热后记录50步。

| 指标 | 逐张量 | 合并梯度 |
| --- | ---: | ---: |
| 启动器退出0、组CLOSED | 1/1 | 1/1 |
| 梯度参考通过 | 17/17 | 17/17 |
| 梯度最大误差 | 3.73e-9 | 3.73e-9 |
| 更新最大误差 | 3.73e-9 | 3.73e-9 |
| 第1→50个实测步loss | 1.251576→1.096676 | 1.251576→1.096676 |
| 50步入口AllReduce调用 | 850 | 50 |
| 输入张量字节数 | 84,290,400 | 84,290,400 |
| 50步平均耗时 | 12.370 ms | 8.509 ms |

两条loss轨迹最大差0、最终参数哈希相同。单Rank的AllReduce调用不是跨卡传输，输入字节也不是链路流量。两模式各仅一次启动、共享机无排他租约，其他任务会动态启动；耗时仅作诊断，不宣称通信加速或业务收益。`memory_stats_at_end.used_mb`是整卡快照且有共享任务影响，不能当本模型峰值；本进程结束allocated为9 MiB。

## ⚠️ 失败记录与卡异常判断

首轮用了`network_mode: none`，BKCL报`no ipv4/ipv6 net card found`、`get_unique_id: init_root failed`，随后FlagCX报未完整初始化，模型未开始训练。[失败启动和Rank记录](../results/20261028-p800/deepfm-single-v1/)保留为FAIL。改为既有Docker bridge、无宿主网络/端口映射后，同卡同镜像两路径通过；这是测试环境配置修正，不是新发现或修复的FlagCX库缺陷。

sudo已实际验证可用，权限不是当前阻塞。只读内核审计确认物理卡1（PCI`16:00`，`/dev/xpu2`）当天多次`task timeout`、`KL_XID_KERNEL_EXCEPTION`，出错任务包括`randn_kernel`、`host2device_async`和`constant`。卡上仍有其他容器的活进程/显存，故本轮不选卡1，也不擅自清理或reset。日志可以证明发生设备执行异常，不能单独证明物理损坏或永久不可用。

同日卡5也存在`assertion_kernel`、`asserted_copy`、`asserted_dot`及vLLM时钟内核异常记录，但本轮两条模型路径实际通过；19:17后的assert异常晚于自有容器19:16:24停止时刻。见[任务名/PCI/时间审计](../results/20261028-p800/kernel-task-audit.log)。这进一步说明不能将所有kernel exception等同于坏卡，也不能将他人后续任务错误算到本模型结果里。

## 📍 双卡出口与复现

主机回归[72/72](../results/20261028-p800/host-tests.txt)，新增厂商/设备身份、完全空闲判定、单卡证据边界与错误结果拒绝检查。主机测试只验代码，不替代硬件结果。双卡入口只接受同NUMA连续健康空闲卡对`(4,5)/(6,7)/(2,3)`；满足条件后执行两模式各三轮×50步，要求每端17/17参考、参数差0、launcher退出0、通信组CLOSED，再由严格汇总器验收。[首个十分钟选卡窗口](../results/20261028-p800/deepfm-pair-v1/selection-execution.json)未出现合格卡对，记`NOT_RUN`，未启动双卡容器或生成训练PASS。

在已同步本仓测试文件的P800宿主上运行；输出名必须全新：

```bash
python3 dev/communication/training/run_p800_selected.py \
  --workspace /data2/lianzhongyou/runtime-tests/20261028-communication \
  --mode pair --wait-seconds 600 \
  --prototype-revision fbac57b6df456f052db83fa47ea423cc0e8ec8a1 \
  --output-name new-deepfm-pair
```

同一时间只运行一个选卡驱动，新版入口用文件锁排除自身并发，也拒绝触碰已运行或非本项目的同名容器。延长监测可设置`--wait-seconds 21600 --poll-seconds 15`（最多六小时，每15秒检查；没有卡就不创建带卡容器）。观测时限届满且没有合格双卡则退出3；测试异常退出后也只停止明确命名的自有容器。单卡准备记录可用`training/summarize_p800_single.py`重算，严格汇总拒绝非零启动器、错误UUID、缺失梯度检查、错误world_size、非有限loss与未关闭的通信组。双卡窗口完成前不提高月度完成率。
