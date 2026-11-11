<!-- 归档说明（device-context 方向维护）
来源：对外问题反馈《昆仑芯 P800 单卡设备同步挂死问题反馈》（文件名 `P800-卡1同步挂死Issue(1).md`，
     2026-09-23 收），**以下正文逐字归档，未做改写**，仅在本说明后追加我方交叉核对结果。
归档原因：该报告是我方 2026-09-29「P800 卡 1 冒烟挂死」结论的**上游依据**；
          本层必须能自证"这不是本方向代码缺陷"，故随仓库保留。
交叉核对（2026-09-29，我方独立完成）：
  · 卡身份**逐项一致**：Serial `02K15K0263D002KN` / XPU UUID `GPU-b3509946-0bc6-5744-bd7c-a0b87dadef02`
    / Minor `2` / Bus `00000000:16:00.0`；容器内 `CUDA_VISIBLE_DEVICES=1` 自报
    uuid `b3509946-0bc6-5744-bd7c-a0b87dadef02` ⇒ **确认是同一张物理卡**（对照组卡 4 自报
    `b7942319-362e-5a53-a8eb-a2d06fbe84f6`，与本文档第五节卡 4 输出一致）。
  · 我方观测到的症状与本文档一致但**更靠前**：卡 1 上 `Event.record + wait_host` 即失败、
    `probe_device` 卡死（`stdlib` 事件层就不通）；**控制面查询正常**（`mem_get_info` 返回
    `(103045660672, 103079215104)`），与本文档第七节「只有计算内核不完成」一致。
  · 我方判别证据见 `../probes/DIAG_kunlun_card1_event_hang_20260929.log`。
结论：**该卡存在设备执行通路层面的硬件级故障，与本方向代码无关，且非我方负载引入**
（本文档 §二：20 次同类异常中 12 次来自其他用户互不相关的作业，最早一次 2026-09-17 早于我方首次使用该卡）。
-->

# 昆仑芯 P800 单卡设备同步挂死问题反馈

## 一、问题描述

地址为 43.180.254.67 的 P800 服务器上的**物理卡 1**（PCI `0000:16:00.0`）上，**任何设备计算内核都无法完成执行**：

- 进程停在 `torch.cuda.synchronize()` **永不返回**，发送信号也无法中断，只能靠外层 `timeout` 强杀；
- 同一时刻内核日志出现：
  - `KLRM: Xid (PCI:0000:16:00): 0, pid=..., name=python, KL_XID_KERNEL_EXCEPTION, ch=0`
  - `KLRM: (PCI:0000:16:00): [XPUW] kl3_wait_for_noc_idle() timeout, val= 00000007 mask= ... reg0228= 007ff000`
  - `KLRM: (PCI:0000:16:00): [XPUW] cluster[N]: ..reason[29] task timeout`
- 驱动把该会话标记为 `XPUERR_KEXCEPTION`，其余会话标记为 `XPUERR_EVENTWAIT`；
- 驱动会直接打印出错的内核符号 —— 卡 1 上出错的是一个**普通的 fill（常量填充）内核**（见第六节）。

**同一条命令、同一个容器、同样参数在卡 4 和卡 6 上 3 秒内正常返回**，只有卡 1 挂死。挂死时卡 1 完全空闲（`Memory_used = 0 MiB`，`util = 0%`，`fuser` 无句柄持有者）。

## 二、影响范围：不止我们的程序

当前内核日志保留窗口内，卡 1 共有 **20 次 `KL_XID_KERNEL_EXCEPTION`**。其中只有 8 次来自我方测试，**其余 12 次来自其他用户互不相关的作业**，且失败算子各不相同：

| 时间（CST） | 进程 | 失败的内核（`c++filt` 解析后） | 来源 |
|---|---|---|---|
| 09-20 19:07:23 | `python` | `xpukernel_xpu3::constant<float>` | 我方测试 |
| 09-20 19:10:38 | `python` | `xpukernel_xpu3::constant<float>` | 我方测试 |
| 09-20 19:14:14 | `python` | `xpukernel_xpu3::constant<float>` | 我方测试 |
| 09-20 20:04:06 | `python` | `xpukernel_xpu3::constant<float>` | 我方测试 |
| 09-20 20:07:01 | `python` | `xpukernel_xpu3::constant<float>` | 我方测试 |
| 09-20 20:29:35 | `python` | `xpukernel_xpu3::constant<float>` | 我方测试 |
| **09-21 15:50:31** | **`VLLM::EngineCor`** | `xpukernel_xpu3::constant<int>` | **其他用户的 vLLM 推理** |
| 09-22 12:18:14 | `python3` | `get_cluster_clock` | 其他用户 |
| 09-22 12:20:28 | `python3` | `xpukernel_xpu3::transpose_0213_cluster<float16>` | 其他用户 |
| 09-22 13:25:55 | `python3` | `xpukernel_xpu3::normal_philox4x32_10_pytorch<float>` | 其他用户 |
| 09-22 13:46:10 | `python3` | `xpukernel_xpu3::host2device_async<256>` | 其他用户 |
| 09-22 14:08:12 | `python3` | `xpukernel_xpu3::host2device_async<256>` | 其他用户 |
| 09-22 15:26:52 | `python` | `xpukernel_xpu3::host2device_async<3072>` | 其他用户 |
| 09-22 15:29:15 | `python` | `xpukernel_xpu3::host2device_async<3072>` | 其他用户 |
| 09-22 18:17:19 | `python3` | `xpukernel_xpu3::host2device_async<512>` | 其他用户 |
| 09-22 18:22:04 | `python3` | `xpukernel_xpu3::host2device_async<512>` | 其他用户 |
| 09-22 18:27:13 | `python3` | `xpukernel_xpu3::host2device_async<512>` | 其他用户 |
| 09-22 19:26:29 | `python3` | `get_cluster_clock` | 其他用户 |
| **09-23 10:44:48** | `python` | `xpukernel_xpu3::constant<float>` | **本次复现** |
| **09-23 10:49:41** | `python` | `xpukernel_xpu3::constant<float>` | **本次复现** |

同一窗口内的其他计数：

| 消息 | 次数 |
|---|---|
| `KL_XID_KERNEL_EXCEPTION` | 20 |
| `kl3_wait_for_noc_idle() timeout` | 163 |
| `cluster[N]: ..reason[29] task timeout` | 163 |
| `[INFO] vstream[0] sess[1] tainted` | 19 |

**失败的内核涵盖 fill（`constant`）、随机数（`normal_philox`）、H2D 拷贝（`host2device_async`）、转置（`transpose`）、`get_cluster_clock` 等**（更早的日志中还有 AllReduce `kl3_all_reduce`），触发进程包括我们的探针、其他用户的 `python3` 作业、以及一个 **vLLM 推理引擎**。我们据此判断这**不是某个应用或算子的 bug，而是该卡执行通路层面的故障**。

> 补充：更早的日志（已随环形缓冲区回绕丢失，见当时保存的证据）显示，该卡最早一次同类异常出现在 **2026-09-17 23:57:45**（`normal_philox4x32_10_pytorch<float>`），**早于我们首次使用该卡**，说明问题并非由我们的负载引入。

## 三、软件与硬件环境

| 项目 | 值 |
|---|---|
| 宿主 OS / 内核 | Ubuntu 24.04.2 LTS / `6.8.0-87-generic` |
| 驱动版本 | `5.0.21.47` |
| XPU-RT（`xpu-smi` 报告） | `10.2` |
| 卡固件 | PBL `1.0` / PCIE `2.43` / SBL `1.86` / CPLD `5.3`（ALL `1.0.2.43.1.86`） |
| 卡型号 | `P800 OAM`，KL3 架构，96 GiB HBM |
| **目标卡** | 物理卡 1 / PCI `0000:16:00.0` / `/dev/xpu2`（minor 2） |
| 目标卡 UUID / SN | `GPU-b3509946-0bc6-5744-bd7c-a0b87dadef02` / `02K15K0263D002KN` |
| 对照卡 | 卡 4（`0000:84:00.0` → `/dev/xpu4`）、卡 6（`0000:B6:00.0` → `/dev/xpu5`） |
| 容器镜像 | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`<br>ID `sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975` |
| Python / PyTorch | `3.10.18` / `2.9.0+cu129` |
| XPYTORCH / XHPC | `XMLIR--bc1b1dc6f-dev+2026032411` / `20770323` |
| XTDK / XRE / XCCL / CUPTI | `3.6.0.1` / `5.13.0.0` / `20260209_9` / `1.1.0.0` |
| 环境变量 | `CUDA_VISIBLE_DEVICES=0`；`XPU_EVENT_KL3_ENABLE`、`XPU_VISIBLE_DEVICES` **确认未设置（unset，非 0）**；`USE_FLAGGEMS=0` |

## 四、最小复现代码与步骤

复现**不依赖任何框架代码**，只需要昆仑芯自身的 PyTorch 运行时。

### 步骤

```bash
# 1) 确认卡 1 空闲（第 18 列 Memory_used = 0，第 20 列 util = 0）
xpu-smi -m

# 2) 解析卡 1 的设备节点（注意：smi 索引 ≠ /dev/xpuN 后缀）
xpu-smi -i 1 -q | grep -E "XPU UUID|Minor Number"
#    -> Minor Number: 2   =>   /dev/xpu2

# 3) 执行最小用例（宿主机需 root 仅为调用 docker；容器内非 privileged）
sudo docker run --rm --network=none \
  --device /dev/xpu2:/dev/xpu2 \
  --device /dev/xpuctrl:/dev/xpuctrl \
  --env CUDA_VISIBLE_DEVICES=0 \
  --entrypoint /bin/bash \
  flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608 \
  -lc 'source /root/miniconda/bin/activate python310_torch29_cuda && \
       timeout --signal=KILL 120 python -X faulthandler -c "
import torch
t = torch.ones(4, device=\"cuda\")
torch.cuda.synchronize()
print(\"PASS\", t.cpu().tolist())
"'
```

- **预期**：`PASS [1.0, 1.0, 1.0, 1.0]`，约 3 秒返回，退出码 `0`。
- **实际（卡 1）**：`torch.cuda.synchronize()` 永不返回，120 秒后被 `timeout` 强杀，**退出码 `124`**。

把命令中的 `/dev/xpu2` 换成 `/dev/xpu4`（卡 4）或 `/dev/xpu5`（卡 6），同一命令**正常通过**。

### 完整复现脚本

```python
import faulthandler, torch

faulthandler.dump_traceback_later(40, repeat=True)   # 挂死时定期打印调用栈

print("torch        :", torch.__version__, flush=True)
print("device_count :", torch.cuda.device_count(), flush=True)
print("device_uuid  :", torch.cuda.get_device_properties(0).uuid, flush=True)
print("mem_get_info :", torch.cuda.mem_get_info(0), flush=True)   # 控制面查询

print("checkpoint: create four-element tensor", flush=True)
tensor = torch.ones(4, device="cuda")          # 只启动一个 fill 内核

print("checkpoint: synchronize device", flush=True)
torch.cuda.synchronize()                       # <-- 卡 1 停在这里，永不返回

print("checkpoint: tensor readback", tensor.cpu().tolist(), flush=True)
print("RESULT: PASS", flush=True)
```

## 五、原始输出

### 卡 1（故障）— stdout

```
torch 2.9.0+cu129 uuid b3509946-0bc6-5744-bd7c-a0b87dadef02
```

（此后不再有任何输出）

### 卡 1（故障）— stderr

```
XCCL /root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/torch_xmlir/xccl//so/libbkcl.so loaded
SYMBOL_REWRITE torch success
Timeout (0:00:40)!
Thread 0x00007ae8056b0480 (most recent call first):
  File "/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/torch/cuda/__init__.py", line 1083 in synchronize
  File "<string>", line 6 in <module>
Timeout (0:00:40)!
Thread 0x00007ae8056b0480 (most recent call first):
  File "/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/torch/cuda/__init__.py", line 1083 in synchronize
  File "<string>", line 6 in <module>
```

退出码 `124`，耗时 `124s`（40 秒一次的栈转储表明进程始终卡在 `torch.cuda.synchronize`）。

### 对照卡 4（正常）— stdout

```
python       : 3.10.18
torch        : 2.9.0+cu129
torch_xmlir  : OrderedDict([('XPYTORCH', 'XMLIR--bc1b1dc6f-dev+2026032411'), ('XHPC', '20770323'), ('XTDK', '3.6.0.1'), ('XRE', '5.13.0.0'), ('CUPTI', '1.1.0.0'), ('XCCL', '20260209_9'), ('compile_date', '20260324-11:53:49'), ('arch', 'x86_64')])
CUDA_VISIBLE_DEVICES: '0'
XPU_VISIBLE_DEVICES: None
XPU_EVENT_KL3_ENABLE: None
USE_FLAGGEMS : '0'
device_count : 1
device_name  : GPU
device_uuid  : b7942319-362e-5a53-a8eb-a2d06fbe84f6
mem_get_info : (103045660672, 103079215104)
checkpoint: set_device
checkpoint: create four-element tensor
checkpoint: synchronize device
checkpoint: tensor readback [1.0, 1.0, 1.0, 1.0]
RESULT: PASS
```

退出码 `0`，耗时 `3s`。卡 4、卡 6 的**内核日志均为 0 行**（无任何该卡相关记录）。

> 补充：执行卡 6 对照时，该卡上还有另一位用户约 6.5 GiB 的显存占用，仍然 3 秒通过；执行卡 4 对照时该卡完全空闲。两例都通过，说明卡 1 的失败与设备是否独占无关。

## 六、内核日志（关键原文）

下面是 **2026-09-23 10:49:41** 这次复现产生的完整异常过程（`pid 4029205` 为复现进程），按时间顺序整理；完整寄存器转储见附件 `card1.kernel-latest-window.txt`：

```text
# --- 阶段 1：NOC idle 等待超时，每约 0.5 秒三条，mask 在 08/10/20 间轮转 ---
[Wed Sep 23 10:49:38 2026] KLRM: (PCI:0000:16:00): [INFO] vstream[0] sess[1] tainted
[Wed Sep 23 10:49:38 2026] KLRM: (PCI:0000:16:00): [XPUW] kl3_wait_for_noc_idle() timeout, val= 00000007 mask= 00000008 reg0228= 007ff000
[Wed Sep 23 10:49:38 2026] KLRM: (PCI:0000:16:00): [XPUW] kl3_wait_for_noc_idle() timeout, val= 00000007 mask= 00000010 reg0228= 007ff000
[Wed Sep 23 10:49:39 2026] KLRM: (PCI:0000:16:00): [XPUW] kl3_wait_for_noc_idle() timeout, val= 00000007 mask= 00000020 reg0228= 007ff000
    ... （共 12 条，重复至 10:49:40）

# --- 阶段 2：驱动标记会话错误 ---
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [INFO] mark sess error(errno= XPUERR_KEXCEPTION), sess= 1, pid= 4029205, comm= python ...
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [INFO] mark sess error(errno= XPUERR_EVENTWAIT), sess= 2, pid= 4029205, comm= python ...
    ... （sess= 3 .. 8 同样为 XPUERR_EVENTWAIT）

# --- 阶段 3：打印出错内核 ---
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] ------------[ cut here  ]------------
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] err task, pid=4029205, comm=python, sess=1, hwq=0, .tk=5 .name=_ZN14xpukernel_xpu38constantIfEEvPclT_ .ncl=12 .nco=64 .addr=400a200000 .ksz=560
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] ..param[0]= 02000000
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] ..param[1]= 00000040
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] ..param[2]= 00000004
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] ..param[3]= 00000000
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] ..param[4]= 3f800000

# --- 阶段 4：逐个 cluster 报 task timeout 并转储寄存器（cluster 0 .. 11） ---
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] cluster[0]: cl_excp_st= 20000000
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] cluster[0]: ..reason[29] task timeout
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] cluster[p0, v0]: [0010]= 00000000
    ...
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] cluster[p0, v0]: [0038]= 02087f0c
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] cluster[p0, v0]: [003c]= 592e1fe2
    ...
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] cluster[p11, v0]: [016c]= 00000000
[Wed Sep 23 10:49:41 2026] KLRM: (PCI:0000:16:00): [XPUW] ------------[ cut here  ]------------
[Wed Sep 23 10:49:41 2026] KLRM: Xid (PCI:0000:16:00): 0, pid=4029205, name=python, KL_XID_KERNEL_EXCEPTION, ch=0
```

### 出错内核符号

```text
.name=_ZN14xpukernel_xpu38constantIfEEvPclT_
```

经 `c++filt` 解析：

```cpp
void xpukernel_xpu3::constant<float>(char*, long, float)
```

即**一个普通的 fill（常量填充）内核**，对应 `torch.ones(4, device="cuda")`。其参数 `param[4] = 3f800000` 正是 IEEE-754 的 `1.0f`，即要填充的值。

两次独立复现（10:44:48 与 10:49:41）得到的内核算子描述**完全一致**，具有确定性：

```text
.tk=5   .ncl=12   .nco=64   .addr=400a200000   .ksz=560
```

## 七、我们已排除的可能

| 假设 | 验证结果 |
|---|---|
| 卡被占用 / 显存不足 | 卡 1 复现前后均为 `0 MiB` 占用、`0%` 利用率，`fuser` 无句柄持有者 |
| 我方容器或启动序列问题 | 同镜像、同命令、同容器参数在卡 4、卡 6 上 3 秒通过；去掉 `-S`/`site.main()` 的纯一行版本在卡 1 上同样挂死 |
| 监控采样干扰 | 早前测试中监控开 / 关两种情况下卡 1 均同步超时 |
| KL3 开关问题 | `XPU_EVENT_KL3_ENABLE` 与 `XPU_VISIBLE_DEVICES` 均为 **unset**，进程内已打印确认 |
| 特定算子 bug | 失败内核涵盖 fill、随机数、H2D、转置、`get_cluster_clock`（更早还有 AllReduce），并跨越我们的探针、其他用户的 `python3` 与一个 vLLM 引擎 |
| 控制面 / 驱动通信故障 | 卡 1 上 `get_device_properties`、`mem_get_info` 等**查询类调用正常返回**，只有**计算内核不完成** |
| 显存 ECC 故障 | 该卡健康查询显示 `0` uncorrectable ECC、无 pending remap |
| 驱动 / 固件版本不匹配 | 同一驱动 `5.0.21.47` + XPU-RT `10.2` 在卡 4、卡 6 上工作正常 |

> 
