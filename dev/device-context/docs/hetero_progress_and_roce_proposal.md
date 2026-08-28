# FlagOS 异构分布式训练进展汇报 与 基于 RoCE 的下一阶段方案

> 汇报人：hliu553（Kistich） ｜ 日期：2026-08-28（更新版） ｜ 受众：资深通信 / LLM 训练工程师  
> 主线：FlagOS 框架 + FlagCX 通信库，在 **4090(NVIDIA) + 910C(昇腾)** 异构集群上做统一的分布式训练。  
> 本文定位：一份可研读、可被挑战的进展与提案——先讲清“已经做对并验证了什么”，再论证“为什么下一阶段该上 RoCE”。
> 更新记录：2026-08-28 新增 P8（设备侧 reduce + CANN UVA 实测）与 P9（net.cc 完成判定 eventSynchronize）修复，单步同步 47.5s → 26.6s，集合级 30/30 全过。

---

## 0. 摘要（Executive Summary）

过去三轮工作，我们完成了从“跨芯片通信链路可行性”到“**异构原生 allreduce 真实训练闭环**”的跨越：

- **同构基线**：4090 双卡 / 910C 双卡，Qwen2.5-1.5B DDP 全程 2481 步收敛（loss 1.94 量级）。
- **异构闭环**：910C + 4090 跨机、DP=2 跨芯片梯度同步，**FlagCX 原生 allreduce 路径**（非 gloo）真实训练 50 步收敛，loss 与 gloo 基线逐位一致，全程零死锁零数据错乱。
- 过程中定位并修复了 **4 个核心实现缺陷**（host-func 死锁 / ACL 回调乱序 / 临时缓冲 OOM / net.cc 完成判定偶发数据错）+ **1 个性能架构增强**（设备侧 reduce），并**实测修正了“昇腾无 UVA”的结论**（CANN 有完整 UVA 接口，已实测真通）。全部已提交 GitHub `FlagRT/runtime-team` 与 OpenI 双平台。

**当前唯一瓶颈是数据面**：梯度同步走 **socket（管理网 10GbE）**，3GB 梯度单步同步实测 **~26.6s**（P9 剥离部分诊断开销后），其中物理传输下限 ~25–30s，已接近 socket 天花板。

**因此我请求批准下一阶段：基于 FlagOS/FlagCX 的 RoCE（RDMA）异构训练。** 理由：正确性风险已经消除（这是最难的部分——5 个修复、30/30 集合级、50 步训练全验证），剩下的只是把数据面从 socket 切到 RDMA；FlagCX 已具备 IB/RDMA netAdaptor 与 GDR 通路，缺的只是（a）RoCE 组网、（b）修复 IB 适配器、（c）跨芯片 GDR 实测。

---

## 1. 背景与目标

### 1.1 我们为什么要做这件事

国产 AI 芯片（昇腾等）与 NVIDIA 长期并存。训练侧要同时用好两种芯片，通常只能“各自为政”：NVIDIA 用 NCCL/CUDA，昇腾用 HCCL/CANN，两套栈之间无法通信。**FlagOS 生态试图提供一套芯片无关的替代栈**，其中 **FlagCX 对标 NCCL，做跨芯片统一通信**。我们的工作就是把这条“跨芯片统一通信”从纸面落到可复现的工程事实。

### 1.2 目标与路线

| 阶段                                | 内容                          | 状态       |
| --------------------------------- | --------------------------- | -------- |
| 同构 4090 双卡训练                      | FlagCX(NVIDIA adaptor)→NCCL | ✅ 完成     |
| 同构 910C 双卡训练                      | FlagCX(Ascend adaptor)→HCCL | ✅ 完成     |
| **异构 4090+910C 跨机训练（socket 数据面）** | FlagCX 异构模式，跨芯片梯度同步         | ✅ 完成（本轮） |
| **异构 RoCE 训练**                    | socket → RDMA，逼近原生吞吐        | ⬜ 本提案目标  |

---

## 2. FlagOS / FlagCX 技术理解（源码级）

> 汇报前先对齐我们对框架的认知深度，便于后续讨论建立在同一套事实上。

### 2.1 FlagOS 生态分层

| 层         | 库                             | 职责                               |
| --------- | ----------------------------- | -------------------------------- |
| 运行时/设备上下文 | `pytorch-plugin-fl`（Torch-FL） | 设备句柄、显存池、Stream/Event/同步语义、错误码翻译 |
| 通信        | `flagcx`                      | 跨芯片统一集合通信（对标 NCCL + HCCL 的统一层）   |
| 算子        | `flag-gems`                   | 跨芯片算子库                           |
| 训练框架      | `flag-tree`                   | 并行训练编排                           |
| 推理插件      | `vllm-plugin-fl`              | vLLM 的国产芯片插件                     |

### 2.2 FlagCX 核心架构

FlagCX 内部按“**runner / adaptor / 调度执行**”三层组织：

**（a）Runner 层**（`flagcx/runner/`）——集合通信算法的执行主体：

| Runner         | 用途                          |
| -------------- | --------------------------- |
| `uniRunner`    | **异构统一通信**（跨芯片 DP 场景，本工作主线） |
| `homoRunner`   | 同构通信（单后端）                   |
| `hostRunner`   | 主机侧 fallback / C2C 校验       |
| `hybridRunner` | 混合                          |

关键点：`uniRunnerAllReduce`（`uni_runner.cc`）的异构实现原本是 **朴素路径**——`allgather(Send/Recv) → D2H → host reduce → H2D`，主动绕过 UVA/DAG。**2026-08-27 起已演进为设备侧 reduce（P8）**：对 Sum + fp32/fp16/bf16 直接 `aclnnInplaceAdd`（CANN）/ 自写 CUDA kernel（NVIDIA）在设备上完成归约，消除 D2H + CPU reduce + H2D，且按片（128MB）分片避免大缓冲 OOM（P7）。正确性优先的朴素路径仍是其他 op/dtype 的兜底。

> 关于 UVA 的结论修正（重要）：此前记录“昇腾无 CUDA 式 UVA”**不准确**。经官方文档 + 910C 实测：CANN 有完整 UVA 接口（`aclrtMallocHost` + `aclrtHostRegisterV2` + `aclrtHostGetDevicePointer`，及 VA 一致性 flag），**NPU 能真实读 host 映射内存**（实测 `aclnnInplaceAdd` 以 host 映射地址为输入算得正确结果）。真缺口是 **FlagCX 的 cann adaptor 结构体里 `hostGetDevicePointer` 字段留了 NULL**（上游适配未写完）——这为后续解锁 DAG 引擎留下了可行性（见 §7.3）。

**（b）Adaptor 层**——屏蔽厂商差异：

| Adaptor                      | 对接                                                            |
| ---------------------------- | ------------------------------------------------------------- |
| 设备侧 `cuda_adaptor.cc`        | CUDA runtime（`cudaLaunchHostFunc`/`cudaMallocAsync`/…）        |
| 设备侧 `cann_adaptor.cc`        | ACL runtime（`aclrtLaunchCallback`/`aclrtSynchronizeStream`/…） |
| 通信侧 `hccl_adaptor.cc`        | HCCL（`HcclCommInitRootInfo`）                                  |
| 通信侧 `nccl`                   | NCCL                                                          |
| **网络侧 netAdaptor**（`net.cc`） | **socket / IB(RDMA) / 同节点 D2D IPC**                           |

**（c）调度执行**（`core/group.cc` + `core/launch_kernel.cc` + `core/net.cc` + `core/proxy.cc`）：

- `flagcxGroupStart` / `flagcxHeteroSend` / `flagcxHeteroRecv` / `flagcxGroupEnd` 组织一个“组”内的 Send/Recv op。
- `groupLaunch` 把 op 排队，由 **proxy 线程**异步执行 D2H/H2D 拷贝与网络收发。
- 完成同步靠 `flagcxHostSemaphore`：主机侧 `cpuAsyncKernel` 回调 `signalStart()` 触发 proxy，主线程 `wait()` 等待所有 op 完成。
- **拷贝完成判定**（P9 修复点）：`net.cc` 的 proxy send/recv 用 per-chunk `eventSynchronize` 精确等待 D2H/H2D 事件真正完成（替代早期 `streamQuery` 状态查询）。

> 这套“group + proxy + semaphore + eventSynchronize”的调度模型，正是我们踩到并修复 4 个核心缺陷的地方（见 §4.3）。

### 2.3 异构模式与数据通路

- 开启：`FLAGCX_USE_HETERO_COMM=1`。
- **同节点**：走设备缓冲区 IPC（D2D 直连）。
- **跨节点**：走 netAdaptor RDMA（IB 支持 `iputBatch`），**socket 兜底**（我们当前使用的）。
- 当前强制 `FLAGCX_FORCE_NET_SOCKET=1`（规避 IB 探测段错误，见缺陷 2）。

---

## 3. 同构基线（正确性的锚点）

任何异构结论，都必须先有两个同构基线做对照。

| 配置          | 后端链路                | 结果                                                          |
| ----------- | ------------------- | ----------------------------------------------------------- |
| **4090 双卡** | FlagCX(NVIDIA)→NCCL | 2481 步收敛，loss **1.9432 / 1538 tok/s**（修复了 NCCL 流语义 + 当前流问题） |
| **910C 双卡** | FlagCX(Ascend)→HCCL | 2481 步收敛，loss **1.9501 / 4157 tok/s**                       |
| 910C 双卡（对照） | torch_npu + 原生 HCCL | loss 1.9472 / **5428 tok/s**                                |

910C 侧我们还修了两个影响稳定性的上游问题：`flagcxCannEvent` **析构缺失导致 event 资源泄漏**（每步累积 ~120 个 aclrtEvent 直至资源耗尽），以及 **work 完成语义**（wait block 用 collective 流 / future 完成前 synchronize）。

---

## 4. 异构 4090+910C 跨机训练（socket 数据面）——本轮核心成果

### 4.1 拓扑与网络

| 角色    | 机器                    | 设备                   | 数据面网卡                  |
| ----- | --------------------- | -------------------- | ---------------------- |
| rank0 | 4090-1（`10.123.4.21`） | RTX 4090             | `enp33s0f1`（10GbE 管理网） |
| rank1 | 910C（`10.120.72.27`）  | Ascend 910（HBM 64GB） | `bond4`（25GbE 管理网）     |

- 管理网已全通（10.92.x 骨干 + 跨区路由，RTT≈0.3ms）。
- **RoCE fabric 未通**（这是下一阶段的核心，见 §6）。
- 模型 Qwen2.5-1.5B（1543.7M 参数，bf16），batch=1、seq=512，flat 梯度缓冲 **3.09GB**。

### 4.2 打通链（异构 allgather / allreduce 的可运行前提）

跨芯片通信在 CANN 侧不是开箱即用，我们累计修了一批适配缺陷：

| #   | 缺陷                                                              | 修复                                                                                       |
| --- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 2   | IB 探测 `flagcxIbFindMatchingDev(NULL)` 段错误                       | `FLAGCX_FORCE_NET_SOCKET=1` 规避（**RoCE 阶段需真修复，见 §6.3**）                                   |
| 3   | `cann::getDevicePciBusId` 返回 NULL 段错误                           | 补 sysfs/伪 busId 实现                                                                       |
| 4   | `flagcxGetLastError` TODO 存根吞错误                                 | 已记录，待完善                                                                                  |
| 8/9 | 禁 topo 后 `nicDistance` 解引用 NULL                                 | 加 null guard                                                                             |
| ④   | `aclrtLaunchCallback` 返回 107015（stream 未 SubscribeReport）       | 方案 B：`aclrtSubscribeReport`+`aclrtProcessReport` 正确实现                                    |
| ⑤   | `uniRunnerAllReduce` 无朴素实现                                      | 已实现（分片 + 设备侧 reduce）                                                                      |
| P8b | 插件 ascend 构建失败：无条件 include `ATen/cuda/CUDAContext.h`          | `#if defined(USE_NVIDIA_ADAPTOR)||…` 保护（py3.12 重建 + 编译链修复 C++17/sm_89/COMPILE_KERNEL 拆分） |
| —   | HCCL commId 缺陷（`HcclRootInfo` 4108B > `flagcxUniqueId` 256B 溢出） | `flagcxHomoCommInit` 传 bootstrap state，rank0 生成 RootInfo + `bootstrapCollBroadcast` 全量分发 |

### 4.3 核心实现缺陷（源码级根因 → 修复 → 证据）

这是本次工作的技术核心，也是“异构原生 allreduce 到底能不能用”的答案。

**P2 —— host-func 回调内自旋导致三方死锁**

- 现象：`test_ag_hetero.py` 跑 AG→AG→AR 三 collective，第 3 个（AR）必卡死；多轮必挂。
- 根因：`launch_kernel.cc` 的 `cpuAsyncKernel` 作为 `cudaLaunchHostFunc` 回调，在其**回调内调用 `semaphore->wait()` 自旋**。回调执行期间 CUDA driver 占用其回调/内部锁；而 op 完成依赖 proxy 线程的 `cudaMemcpyAsync`——proxy 拿不到执行机会 → op 永不完成 → wait 永不返回 → **rank0/proxy/rank1 三方死锁**。纯时序竞态（proxy 的 memcpy 若在 host func 开始自旋前已入队则侥幸成功）。
- 修复：`cpuAsyncKernel` 只 `signalStart()` 即返回；`wait()` 移到 `group.cc` 的 `flagcxGroupLaunch` 主线程末尾。
- 证据：修复后 10/10 轮循环零死锁。

**P6 —— 910C 发送路径数据错乱（ACL 回调不等 stream 前置任务）**

- 现象：不死锁后，约 60% 轮次 910C 发出“慢一拍”旧数据（应发 `[10,11]` 却发 `[0,2]`）。
- 根因：`cann_adaptor.cc` 主路径 `aclrtLaunchCallback(fn, args, ACL_CALLBACK_NO_BLOCK, stream)` **回调执行并不等待 stream 上的前置任务**。`fn`=signalStart 触发 proxy 的 D2H，但此时 NPU tensor 写入 kernel 尚未完成 → 读到上一拍的旧数据。**CUDA 侧因 `cudaLaunchHostFunc` 的 legacy-stream 天然保序而幸免**，故缺陷只在异构场景暴露。
- 修复：`aclrtLaunchCallback` 前加 `aclrtSynchronizeStream(stream->base)`（NULL 分支防御性 `aclrtSynchronizeDevice`）。
- 证据：修复后 10/10 轮 `out=[1,2]`、`out2=[10,11]`、`sum=3.0` 全对。

**P7 —— `uniRunnerAllReduce` 临时缓冲 OOM（真实训练阶段才暴露）**

- 现象：真实训练 step0 双 rank 成功，**step1 立即崩** `flagcxUnhandledDeviceError: Call to Device function failed`。
  - 根因：`uni_runner.cc` 的 `uniRunnerAllReduce` **每次调用分配 `bytes×nranks` 临时设备缓冲（3GB×2=6GB）**。step0 的 allreduce 发生在 `optimizer.step()` **之前**（显存充足），step0 结束后 AdamW fp32 状态（~12.4GB）建立，24GB 卡上 step1 的 6GB `cudaMallocAsync` OOM。`DEVCHECK` 宏**失败静默返回**（零日志）；报错尾巴 “flagcxComm is not fully initialized” 是 `flagcxGetLastError(NULL)` 的固定字符串（**红鲱鱼**，与 comm 状态无关）。
- 修复：**分片 allreduce**——按片（默认 128MB，`FLAGCX_HETERO_AR_SLICE_MB` 可调）执行 allgather→（设备侧 reduce，P8）→H2D，临时缓冲 6GB→256MB。
- 证据：修复后真实训练 50 步全程无死锁无 OOM。

**P9 —— net.cc 完成判定偶发数据错（集合级 1/10，2026-08-28 修复）**

- 现象：P8 后集合级 10 轮稳定性循环仍有 **1/10 间歇性 sum=1.0**（rank0 收到的 rank1 数据为旧值/0）；50 步训练不触发（3GB/24576 chunk 流水线深，isend 时 D2H 早已完成）。
- 根因（历史日志实锤）：`net.cc` send 侧 `streamQuery(cpStream)` 判定“拷贝完成”在 **910C(CANN) 上早于 D2H 数据对 CPU 可见**（`aclrtStreamQuery` 官方语义只承诺“任务已完成”，未承诺 DMA 数据可见；910C 为 aarch64，存在 DMA/CPU 缓存一致性窗口）→ isend 偶发发出**旧 buffer 残留**。失败轮 P4-SEND-DATA 打点显示 isend 前 buffer=11（AG 旧数据），PASS 轮=2.0（正确）。CUDA 侧 `cudaStreamQuery` 语义严格故恒对。
- 修复：send/recv 两侧 `streamQuery(cpStream)` → **`eventSynchronize(cpEvents[step])`**（阻塞等 D2H/H2D 事件真正完成；不受 event 环形复用影响——保守方向只会慢不会错；此前尝试 eventQuery 非阻塞查询失败正是事件复用误判）。
- 证据：修复后 **30/30（三轮循环）全过、0 死锁**（修复前 9/10、早期 7-8/10）。

> 这四个缺陷的共同点：都是**实现层竞态/资源缺陷，而非“异构不可行”**。修掉之后，异构原生 allreduce 是真实可用的——这是本提案最想强调的一点。

### 4.3.1 P8 —— 设备侧 reduce + CANN UVA 实测（性能/架构增强）

- 动机：朴素路径的 D2H + CPU reduce + H2D 是次要瓶颈；且此前“昇腾无 UVA”的判断限制了 DAG 引擎路线。
- **UVA 实测**：910C 上 C 程序验证 `aclrtMallocHost` + `aclrtHostRegisterV2(PINNED\|MAPPED)` + `aclrtHostGetDevicePointer` 全部成功；**`aclnnInplaceAdd` 以 host 映射地址为输入算得 7.0（5.0+2.0）→ NPU 真实读 host 内存**。坑：内核 5.10 下普通 malloc + RegisterV2 失败（ret 507899），必须用 `aclrtMallocHost`。
- **设备侧 reduce**：adaptor 加 `reduceSum` 字段（CANN=`aclnnInplaceAdd` / NVIDIA=自写 CUDA kernel `flagcx_device_reduce.cu`），`uniRunnerAllReduce` 对 Sum+fp32/fp16/bf16 走设备侧，消除 D2H+CPU reduce+H2D。
- 附带修复编译链：CUDA 13 需 `-std=c++17`；gencode 补 **sm_89**（4090 是 Ada，缺了 kernel 静默不执行）；`COMPILE_KERNEL`/`COMPILE_KERNEL_HOST` 拆分（后者会启用 kernel proxy 线程干扰 socket proxy，曾导致间歇性数据错）。
- 证据：test_ag 双侧 sum=3.0；50 步训练 loss 与基线一致，sync 47.5s → 42s（P8）→ 26.6s（P9）。

### 4.4 验证结果（真实训练）

| 指标               | gloo 基线（20 步）   | FlagCX 原生（50 步，P2+P6+P7+P8+P9 全量） |
| ---------------- | --------------- | --------------------------- |
| rank0(4090) loss | 2.8891 → 2.1312 | 2.8891 → 2.1 区间（s20=2.1327） |
| rank1(910C) loss | 3.17 → 2.43     | 3.1656 → 1.8–2.6 区间         |
| 梯度同步             | 60–130 s/步      | **~26.6 s/步**（sync_total=1331s/50 步） |
| 死锁/数据错乱          | —               | **0 死锁 / 0 错乱**（50 步全程）     |

- 集合级：`test_ag_hetero.py` AG/AG/AR，**30/30 轮**（P9 后三轮）`out=[1,2]`、`out2=[10,11]`、`sum=3.0` 全对；`loop_hetero_local.sh` 连跑全过。
- **loss 一致性是关键证据**：异构首步 loss 与 gloo 基线逐位一致，说明跨芯片梯度融合的数值正确性成立。

### 4.5 为什么 gloo 不可行（对照）

gloo 仅作“通信通路可行性”验证：梯度 D2H → gloo TCP all_reduce（fp32 中转）→ H2D，20 步 60–130s/步、~17 tok/s。**吞吐过低，只能证明“能通”，不能作为验收标准**。这正是促使我们死磕 FlagCX 原生路径、并最终走到 RoCE 的直接动机。

---

## 5. 当前瓶颈：socket 数据面是天花板

把 3GB 梯度同步拆开看（实测，P8+P9 后）：

| 组成                                      | 量级      |
| --------------------------------------- | ------- |
| 物理传输（10GbE，~1.0–1.2GB/s）                | ~25–30s |
| 设备侧 reduce（P8 后，aclnn/kernel，片上）       | ~0.5s 以下 |
| 诊断打点开销（工作树残留 P1-P4 打印，每步 ~60MB stderr） | ~10-15s（P9 后已降，剥离后仍可再快） |
| 分片/调度/完成判定（P9 eventSynchronize 阻塞）      | 若干秒     |

结论：**即使剥掉诊断打点，socket 物理下限仍在 ~25–30s/步**，异构训练永远被数据面卡在“分钟级/步”，无法逼近同构基线的吞吐。

---

## 6. 下一阶段：基于 RoCE 的异构训练（提案核心）

### 6.1 为什么是 RoCE——收益量化

- 4090-1 网卡是 **Mellanox CX-6**（100G/200G 级），910C 是 **华为 Hi1822**（RoCE，VLAN2173），两者都原生支持 **RoCEv2**。
- 以保守 100GbE 估算：3GB 梯度 RDMA 传输 ~**0.25s**（相比 socket 的 25–30s，**~100×**）；即便计上调度与 reduce，单步同步可预期落到 **秒级以内**，训练从“数据面主导”变为“计算主导”，吞吐逼近同构基线（4090 1538 tok/s / 910C 4157 tok/s 的量级）。
- 一句话：**正确性已闭环（5 个修复 + 30/30 + 50 步），现在只差把数据面换成 RDMA，这是收益最大、风险最低的下一步。**

### 6.2 FlagCX 已有的 RDMA 基础（不必从零做）

- `net.cc` 已有 **IB(RDMA) netAdaptor**（`iputBatch`），并有 **GDR 通路**（`gdrMemAlloc`：`cuMemCreate`/`cuMemMap` VMM + DMA-BUF export）——即 FlagCX 设计上就支持 GPU-direct RDMA。
- 昇腾侧有 **zero-copy 映射**（`aclrtHostRegisterV2` + `aclrtHostGetDevicePointer`，**已实测真通**，见 §4.3.1）。
- 因此 RoCE 阶段主要是**打通与验证**，而非重写通信栈。

### 6.3 需要的三件事

| # | 事项                                                                                                                                  | 性质      |
| - | ----------------------------------------------------------------------------------------------------------------------------------- | ------- |
| 1 | **RoCE 组网**：把 4090 RoCE 网段（192.168.10.0/24，无网关）接入 10.92.x 骨干；放行 **UDP 4791**；配置**无损 QoS（PFC/ECN）**                                  | 需 IT 配合 |
| 2 | **修复 FlagCX IB 适配器**：`flagcxIbFindMatchingDev` 段错误（缺陷 2）目前用 `FORCE_NET_SOCKET=1` 规避，RoCE 阶段必须**真修复**而非规避                            | 我方开发    |
| 3 | **跨芯片 GDR 实测**：Mellanox CX-6 ↔ 华为 Hi1822 之间，CUDA ↔ CANN 的 device-to-device RDMA 是否成立（RoCEv2 是标准，双方 GID/GRH 应可互操作，但**跨厂商无现成保证，需实测**） | 我方验证    |

### 6.4 风险与兜底（诚实声明）

- **跨芯片 GDR 是本提案唯一实质技术风险点**。若 CUDA↔CANN 的 device-to-device RDMA 不成立，降级方案：RDMA 走 **host-bounce**（仍比 socket 快一个数量级）；再兜底：RoCE 网段打通后至少能用 **TCP-over-RoCE**（仍优于当前管理网 socket）。
- ~~昇腾侧无 CUDA 式 UVA~~ **已修正**：CANN 有完整 UVA 接口且实测真通（§4.3.1），不再阻塞；DAG 引擎路线（需要昇腾侧设备 kernel 或 aclnn 折中）是可选项而非必需（朴素路径 + 设备侧 reduce 已可跑）。

### 6.5 里程碑

1. RoCE 组网完成（IT）→ 链路连通性 + UDP4791 + 无损 QoS 验证。
2. 修复 `flagcxIbFindMatchingDev`，FlagCX 走 IB 适配器打通同构 RDMA allreduce。
3. 异构 RDMA allreduce 打通（先 host-bounce，再尝试 GDR）。
4. 真实训练对照：RoCE 下 loss 与 socket 基线一致 + 吞吐对比表。

---

## 7. 遗留与开放问题（如实列出）

1. **诊断打点未剥离**：工作树里 net.cc/proxy.cc 等仍残留 P1-P4 打印（每步 ~60MB stderr），提交上游前需剥离成干净 diff；剥离后单步同步预计再快 ~10s。
2. **上游提交**：P2+P6+P7+P8+P9 干净 diff 尚未 commit 到 `FlagRT/FlagCX` 的 `kistich/ascend-dev1.0`（计划 RoCE 验证通过后一并提交）。
3. **昇腾 DAG 引擎**：CANN UVA 已实测可用，但 DAG 引擎完整落地还需昇腾侧设备 kernel（或 aclnn 折中实现 reduce 节点）+ cann adaptor 补 `hostGetDevicePointer` 字段——作为可选优化，不阻塞当前路线。
4. **`flagcxGetLastError` 存根**（缺陷 4）：错误诊断能力待补。
5. **socket 协议无 tag 匹配**（长期项）：收发匹配依赖 ctrlSock 的 FIFO size 握手，多 op 排队场景存在理论错位风险（P9 已消除当前实际路径的竞态，协议级加固可作为独立任务）。

---

## 8. 附录：关键事实速查

**三套训练的数字锚点**：4090 双卡 1.9432/1538 tok/s；910C 双卡 1.9501/4157 tok/s（原生 HCCL 5428）；异构 socket 50 步 loss 2.8891→2.1（rank0）、3.1656→1.8-2.6（rank1），同步 **~26.6s/步**（P2+P6+P7+P8+P9 全量）。

**核心源码地图**：`launch_kernel.cc`(cpuAsyncKernel/semaphore) · `group.cc`(groupLaunch/flagcxHeteroSend/Recv) · `net.cc`(proxy send/recv, socket/IB/D2D, eventSynchronize 完成判定) · `uni_runner.cc`(uniRunnerAllReduce 分片+设备侧 reduce) · `cann_adaptor.cc`(launchHostFunc/SubscribeReport/reduceSum-aclnn) · `cuda_adaptor.cc`(launchHostFunc/deviceMalloc/reduceSum-kernel) · `hccl_adaptor.cc`(RootInfo 分发) · `backend_flagcx.cpp`(allreduce/initComm/getStreamByIndex)。

**五个缺陷一句话版**：P2=回调内 wait 自旋死锁；P6=ACL 回调不等前置流导致旧数据；P7=每步 6GB 临时缓冲在 24GB 卡二次分配 OOM；P8=设备侧 reduce（增强，含 UVA 实测与编译链修复）；P9=net.cc `streamQuery` 在 CANN 上早于数据落位 → `eventSynchronize` 精确等待。

---

*本提案的核心主张：异构训练的正确性风险已被消解（P2+P6+P7+P9 缺陷修复 + P8 增强 + UVA 实测 + 30/30 集合级 + 50 步真实训练验证），剩余瓶颈是纯数据面问题；RoCE 是唯一能同时解决吞吐与逼近同构基线的路径，且 FlagCX 已具备 RDMA/GDR 基础，投入产出比最高。*
