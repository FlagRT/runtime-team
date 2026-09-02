# 昇腾 910C · 分布式训练/推理 × 设备上下文 × 多流 Stream 职责映射与完成进度

> **版本**：v1.0（2026-09-02）
> **作者**：Kistich ｜ **路线**：A 线（torch_npu 原生，不走 torch_fl/FlagCX 插件路径）
> **定位**：本方向（设备执行上下文）在**昇腾 910C** 上，面向**分布式训练**与**分布式推理**两个场景的
> 职责映射全景 + 每项职责的**原理 / 实现过程 / 完成判据 / 当前进度**。
> **关联文档**：
> - `DEVICE_CONTEXT_INFERENCE_MAPPING_20260831.md`（推理侧验收依据，A1-A11）
> - `DEVICE_CONTEXT_TRAINING_MAPPING_20260831.md`（训练侧验收依据，B1-B11）
> - `ACL_ERROR_MAP_20260901.md`（D10 错误码映射专题）
> - `PR_DEV_1_0_20260901.md`（PR 描述草稿）

---

## 0. 文档定位与阅读指引

**为什么需要这份文档**：设备执行上下文的职责（统一设备/内存/执行句柄、Stream 语义、异步传输、
双缓冲、同步语义、错误码翻译、状态恢复）在训练与推理两个场景下的**验证载体完全不同**，
容易导致"训练侧验了、推理侧没验"或"两边都标了 ✅ 但判据不一致"。本文档把 11 项职责
（D1-D11）与 16 项多流 Stream 子项（S-1~S-16）在**两个场景下逐一对照**，并对每项给出：

| 要素 | 含义 |
|---|---|
| **原理** | 这项职责解决什么问题、底层机制是什么、为什么昇腾 910C 上需要特别处理 |
| **实现过程** | 用什么资产/脚本验证或落地、关键代码路径、踩过哪些坑 |
| **完成判据** | 什么条件下判定"完成"——必须是**可观测、可复现、可证伪**的客观标准 |
| **进度** | 当前状态 + 证据（结果文件/日志/提交号） |

> **判据纪律**：本文档所有"完成"判定均遵循——**功能与性能分开标注**；
> **不接受"机制通过"替代"性能达标"**；**不接受不可证伪的归因**（如"物理限制"）。

---

## 1. 验证环境与载体

### 1.1 硬件与软件栈

| 项 | 配置 |
|---|---|
| 主机 | 昇腾 910C（10.120.72.27，SSH 别名 `910C`），8 卡 × 2 chip = **16 个逻辑 NPU 设备** |
| 容器（推理/A 线主线） | `flagos-infer-910c` |
| 框架 | torch 2.10.0+cpu + **torch_npu 2.10.0** + transformers 5.5.3 + vLLM（源码安装 `/vllm-workspace/vllm`） |
| CANN | `/usr/local/Ascend/ascend-toolkit/latest`（pyACL 可用） |
| 模型 | 训练 `Qwen2.5-1.5B`（1.54B bf16）｜推理 `Qwen3-4B` |
| 数据 | wikitext-2-raw-v1（缓存于 raid，软链至 `/workspace/data`） |

**关键环境坑（已归档）**：
- 训练容器 `flagos-910c-train-850` 的 torch_npu 为 2.6.0rc1（旧版只认 1 卡），**不可用于双卡训练**
- infer 容器缺 `datasets` 模块（已装 5.0.1）；容器内无 `/workspace/models`，需软链 raid 模型目录
- 容器 1 号进程为 `sleep infinity`（非真 init）→ **不回收子进程**，任务结束留 zombie（无害，占 PID）

### 1.2 训练侧载体（分布式训练）

| 项 | 配置 |
|---|---|
| 并行方式 | **DDP**（DistributedSampler + `dist.init_process_group`） |
| 后端 | `BACKEND=hccl`（当前环境）｜历史基准含 `flagcx`（FlagCX torch 插件，HCCL adaptor） |
| 启动 | `torchrun --nproc_per_node=2 --master_port=29522 train_qwen_1_5b_npu.py` |
| 规模 | BATCH_SIZE=1，MAX_LEN=512，bf16，AdamW |
| 历史基准 | 2481 步 loss 1.9501 / 4245 tok/s（flagcx）、5428 tok/s（hccl） |

### 1.3 推理侧载体（分布式推理）

| 项 | 配置 |
|---|---|
| 引擎 | vLLM（vllm-ascend 官方镜像路线），`vllm serve` 长驻服务 |
| 并行方式 | **TP**（Tensor Parallel），已验 TP=1 / TP=2 / TP=4 |
| 通信 | HCCL（torch_npu 原生） |
| 集成 | D10 错误码翻译挂接 vLLM 异常路径 + D11 设备状态监控并行运行（见 §3 D10/D11） |
| 基准 | 16 prompt × 256 token 批量：Qwen3-4B **285.2 tok/s**、Qwen2.5-1.5B **283.0 tok/s** |

### 1.4 术语与口径约定

| 术语 | 含义 |
|---|---|
| **A 线** | torch_npu 原生路线（本方向主线），**不走** torch_fl / vllm-plugin-FL |
| **重叠率** | `(无重叠基准 − 流水线实测) / 无重叠基准`；**分母必须是「同步拷贝 + 纯计算」实测**，不能用另一种实现的耗时互作分母 |
| **无重叠基准** | 同步拷贝 + 纯计算串行执行的总耗时（理论下界参照） |
| **L1-L4** | 错误分级：L1 资源（可重试）/ L2 参数（上抛调用方）/ L3 执行（同上下文重放）/ L4 致命（设备恢复） |
| **mapped / graded_by** | F5 可观测字段：分级是否命中映射表、分级来源（code_map / message_hint / default） |

---

## 2. 职责映射总览

### 2.1 设备上下文 11 项职责（D1-D11）× 训练 × 推理

| # | 职责 | 训练侧覆盖 | 推理侧覆盖 | 进度 |
|---|---|---|---|---|
| D1 | 封装 Runtime 接口 | torch_npu + FlagCX/HCCL 接入，训练闭环 | vllm-ascend + torch_npu，dense/TP 推理 | ✅ |
| D2 | 设备句柄 + 生命周期 | 双卡枚举、conformance 设备无关化 | i1 加载后上下文 + **A8 子进程句柄取证** | ✅ |
| D3 | 内存句柄 + 生命周期 | 六探针（pinned_pool / ai_cpu_core） | i3/i4/i5 + **serve 21 分钟长驻零增长** | ✅ |
| D4 | 执行句柄 | S1-S4 / E1-E3 用例 | i2 多轮前向同流顺序（误差 0.00e+00） | ✅ |
| D5 | **统一 Stream 语义** | S1-S4（流内顺序/依赖/可见性/传递） | **16 项子项全量核查 PASS**（§4） | ✅ |
| D6 | Host/Device 异步传输 | T1/T2/T3 | i4 + 双缓冲；**性能依赖 D8** | ⚠️ 部分 |
| D7 | 页锁定内存 | T1 + pinned_pool 探针 | 双缓冲全程 pin_memory；**训练侧已回补** | ✅ |
| D8 | 双缓冲流水线 | 训练脚本已启用 pin_memory+non_blocking | **v2 四模式 + 按负载选型，DBUF2_PASS** | ✅ |
| D9 | 同步语义 | 2481 步 DDP 稳定；event 泄漏修复 | TP_COMM_PASS 4/4 + F 场景跨流不阻塞 | ✅ |
| D10 | 错误码翻译 | F1 用例（161002 → L2_PARAM） | **64.8% 覆盖 + timeout 真实触发** | ✅ |
| D11 | 设备状态恢复 | R1-R5 用例 | **8/8 + 真实重建落地 + 多进程联调** | ✅ |

**进度：11 项中 10 项 ✅，1 项（D6）⚠️ 部分达标**（功能 ✅，性能项依赖 D8 的重叠率，
在 n≤512 小负载下为负——已由 D8 的"按负载选型"给出可操作结论）。

### 2.2 多流 Stream 16 项子项（S-1~S-16）× 训练 × 推理

> 这 16 项是 D5「统一 Stream 语义」的细化拆解。2026-09-02 全量核查前，
> **S-8 ~ S-13 共 6 项从未覆盖**，且 S-1/S-2 的既有用例为"框架层近似"（未真正创建流）。

| # | Stream 子项 | 训练侧 | 推理侧 | 进度 |
|---|---|---|---|---|
| S-1 | 流内顺序性（FIFO） | S1（近似）→ 补强 | i2 + 补强真创建流 | ✅ |
| S-2 | 显式依赖 / 无隐式同步 | S2（近似）→ 补强 | 补强真创建两流 | ✅ |
| S-3 | 跨流可见性（event） | S3（真实） | i3 KV 跨流可见性 | ✅ |
| S-4 | wait_stream 传递 | S4（真实） | — | ✅ |
| S-5 | 多流并发重叠 | 多流流水线 | 双缓冲 v2 + TP F 场景 | ✅ |
| S-6 | 集合通信与流绑定 | DDP 梯度同步 | TP 流绑定 4/4 + 跨流不阻塞 | ✅ |
| S-7 | 图捕获（graph capture）流语义 | — | GRAPH_CAPTURE_PASS 5/5 | ✅ |
| S-8 | 默认流 vs 命名流 + 跨流分配器安全 | ❌→本次补 | ❌→本次补 | ✅ |
| S-9 | 流错误隔离（分层） | ❌→本次补 | ❌→本次补 | ✅（API 级）/ ⚠️（设备级推断） |
| S-10 | 流/事件生命周期与配额 | 训练侧 event 泄漏已修 | 500 次创建销毁无泄漏 | ✅ |
| S-11 | 跨流内存分配 | ❌→本次补 | ❌→本次补 | ✅ |
| S-12 | 流优先级 | ❌→本次补 | ❌→本次补 | ✅ |
| S-13 | 多设备流绑定 | ❌→本次补 | ❌→本次补 | ✅ |
| S-14 | 流同步超时语义 | — | 507046 真实触发 | ✅ |
| S-15 | 跨进程流共享（IPC） | — | — | ⬜ 不适用（大规模分布式/上游） |
| S-16 | 流数量配额 | — | 2000 流创建成功 | ✅ |

**进度：16 项中 15 项已核查通过，1 项（S-15 IPC）标注不适用。**

### 2.3 总体进度

| 层 | 状态 |
|---|---|
| 职责 D1-D11 | 10 ✅ + 1 ⚠️（D6 性能项，已有可操作结论） |
| Stream 子项 S-1~S-16 | 15 ✅ + 1 ⬜（不适用） |
| 集成层面 | D10/D11 已挂接真实 vLLM 服务；D8 已在推理侧落地 + 训练侧回补 |
| 回归 | conformance **13/13** + infer **6/6** PASS（多轮） |
| 三方一致 | mac / GitHub / 910C 同步（提交 `75b5ebe`） |

---

## 3. 设备上下文职责详解（D1-D11）

### D1 封装不同芯片 Runtime 接口

**【原理】**
不同芯片（昇腾 NPU / NVIDIA GPU）的 Runtime API 完全不同（ACL/aclrt vs CUDA/cuBLAS）。
职责要求提供**统一封装层**，使上层框架（vLLM / 训练脚本）无需感知芯片差异。昇腾侧的技术
选型关键在于：**vLLM 官方无 ascend platform**，必须走华为/vllm-project 的 `vllm-ascend` 插件。

**【实现过程】**
1. 尝试 `vllm-plugin-FL`（FlagOS 插件）→ **失败**：`vllm_fl/dispatch/backends/vendor/` 只有
   metax/musa/sunrise/thead/txda，**无 ascend 后端**；且 setup.py 注明 "currently CUDA only"
   （已归档为**坑 A5**）
2. 改用 `vllm-ascend` 官方镜像路线 + torch_npu → dense 推理与 TP 推理均跑通
3. 训练侧：torch_npu + `dist.init_process_group(backend=BACKEND)`，BACKEND 支持 `flagcx` / `hccl` 切换

**【完成判据】**
- dense 单卡推理输出正确（`DENSE_INFER_PASS`）
- TP=1/2/4 均能正常加载并产出（不要求跨 TP 逐字一致，见 A5 修正）
- 设备可用：`torch.npu.device_count()` > 0 且可 set_device

**【现状】✅** 2026-09-01 PASS。⚠️ 注：早期记录的 19.5 tok/s 已被证伪（预热不足污染），
同口径重测为 **283.0 tok/s**（见 §5 纠错历程）。

---

### D2 统一设备句柄 + 生命周期管理

**【原理】**
设备句柄封装 device 的枚举、初始化、上下文（Context）创建与销毁。生命周期管理的核心是
**防止句柄泄漏**——尤其在多进程（vLLM 的 EngineCore 是 spawn 子进程）场景下，
子进程持有的设备句柄在退出时必须释放，否则会持续占用 NPU 名额。

**【实现过程】**
1. conformance **i1**：模型加载后设备上下文可用
2. **A8 探针**（`probe_enginecore_handle_detail.py`）：对 vLLM EngineCore 子进程做 `/proc/<pid>` 取证
   - 运行中：spawn 子进程持有 **7 个 davinci fd**（均指向 `/dev/davinci_manager`）、
     设备内存映射 **davinci 78 + CANN 库 524 + 文件 3010（总 3619）**、RSS 5952MB
   - 停止后：EngineCore / Worker 残留均为 **0**，`released: True`
   - **口径修正**：早期记录"fd=1"实为去重后的路径数、"124 处映射"为过滤子集，已订正
3. **SIGKILL 验证**：`kill -9` EngineCore 后进程残留 0、HBM（~46GB）完全释放回落基线、
   NPU 进程表为空 → **坑 A2 的"SIGKILL 残留占位"在当前环境未复现**

**【完成判据】**
- i1 PASS（加载后上下文可用）
- 子进程句柄可取证（pid/ppid + fd + 映射）
- **正常停止与 SIGKILL 两种路径均无进程/句柄残留**

**【现状】✅** 含 A8 明细修正与 SIGKILL 释放验证。

---

### D3 统一内存句柄 + 生命周期

**【原理】**
显存句柄封装分配/释放。推理场景的特殊性在于 **KV cache 长驻**——服务运行数小时后
显存不能持续增长（泄漏）。训练场景则是每步反复分配释放，需验证分配器稳定性。

**【实现过程】**
1. **i3** KV 模拟缓冲跨流可见性、**i4** D2H 采样回传、**i5** 长驻 20 轮无 NaN/Inf
2. **serve 长驻观察**（P1-⑤）：Qwen3-4B TP=1 serve 持续运行，多点采样
   - T1（加载后 11:06）：HBM 61129MB / EngineCore RSS 5855MB
   - T2（11:08）：**61129MB / 5855MB（零增长）**
   - T3（12 条请求后）：HBM +107MB / RSS +81MB（vLLM KV cache 正常缓存）
   - T4（21 分钟后 11:26）：HBM 61237MB / RSS 5936MB（**与 T3 一致，零增长**）
3. 训练侧：六探针（pinned_pool / ai_cpu_core）

**【完成判据】**
- 分配访问正确（i3/i4）
- **长驻无泄漏**：多点采样下 HBM 与 RSS 不持续增长（允许请求后的 KV cache 一次性抬升后走平）

**【现状】✅** 21 分钟窗口零增长，无泄漏。

---

### D4 统一执行句柄

**【原理】**
执行句柄抽象执行队列（Stream/Queue）的提交通道。判据侧重**执行的确定性**——
同一输入在同一执行通道上多次前向，结果应完全一致。

**【实现过程】**
- **i2**：固定输入做 3 轮前向（模拟 3 轮 decode），逐轮结果一致，**误差 0.00e+00**
- 训练侧 S1-S4 / E1-E3 用例

**【完成判据】**
- 多轮前向结果一致（相对误差 < 1e-3，实测 0.00e+00）
- 执行通道无异常中断

**【现状】✅**（2026-09-02 校准：原标 🔄 属状态滞后）

---

### D5 统一 Stream 语义

**【原理】** 见 **§4**（16 项子项详解）——这是本文档的重点章节。

**【实现过程】**
- 训练侧：S1-S4 + E1-E3 conformance 用例
- 推理侧：i6 双缓冲 + TP 通信同步 + graph capture + **`probe_stream_semantics_full.py` 全量核查**

**【完成判据】**
- 16 项子项中 15 项通过（S-15 IPC 标注不适用）
- `STREAM_SEMANTICS_PASS 8/8`（本次核查的 8 项）

**【现状】✅**

---

### D6 Host/Device 异步传输

**【原理】**
H2D（输入拷贝）/ D2H（结果回传）若走同步拷贝，会阻塞主机与设备流水线。
异步拷贝（`non_blocking=True` + pin_memory）允许拷贝与计算重叠，隐藏传输延迟。

**【实现过程】**
1. **T1** pinned → device non_blocking 拷贝数据一致
2. **T2** 在途保护（拷贝中释放源缓冲的边界）
3. **T3** 跨设备传输（拓扑如实标注：torch_npu 未暴露统一拓扑查询）
4. 推理侧 i4（D2H 采样回传）+ 双缓冲流水线

**【完成判据】**
- **① 功能**：异步拷贝数据一致（✅ 已达标）
- **② 性能**：与计算重叠 —— **依赖 D8 的重叠率结论**：n≤512 为负、n≥1024 为正

**【现状】⚠️ 部分达标**（功能 ✅，性能随 D8 选型结论条件性达标）

---

### D7 页锁定内存

**【原理】**
异步拷贝（DMA）要求主机内存**物理页固定**，否则操作系统可能换页导致传输数据错乱。
`pin_memory=True` 是 `non_blocking=True` 能真正异步的**前提条件**——这是最容易被忽略的
强依赖关系（**坑 B4**：pin_memory 需设备预热后才生效）。

**【实现过程】**
1. **T1**：pinned → device non_blocking 拷贝数据一致 PASS
2. 双缓冲探针全程使用 pin_memory
3. **训练侧回补**（2026-09-02）：`DataLoader(..., pin_memory=True)` +
   `batch.to(dev, non_blocking=True)`（semantic 等价、数值不变；num_workers 保持 0，
   容器内 fork 子进程有风险）

**【完成判据】**
- pinned → device non_blocking 拷贝数据一致（T1 PASS）
- 训练侧已启用（参数可见 + 完整复测 100 步无崩溃）

**【现状】✅**

---

### D8 双缓冲流水线

**【原理】**
双缓冲（buf[0]/buf[1] 交替）+ 多流（传输流/计算流/回传流）+ Event 依赖，
目标是让**第 i+1 批的 H2D 与第 i 批的计算重叠**，从而隐藏传输延迟。
理论重叠上限 = `min(拷贝, 计算) / (拷贝 + 计算)`。

**关键认知**：重叠能否为正，取决于**计算粒度 vs 同步开销**的比例——
昇腾 `EVENT_WAIT` 单次约 **280μs**（910C 实测 3.37ms/12 次），而 512² matmul 仅约 10μs 量级。
小负载下同步开销远超计算，流水线必然"亏本"。

**【实现过程】**
1. **v1**（2026-08-31）：仅 V0 单实现（每批 2 record + 2 wait），重叠率为负 → DBUF2_PARTIAL
2. **O2 求解实验**（2026-09-02）：四变体对比 + 规模扫描，定位转折点与方案适用边界
3. **v2 落地**（2026-09-02）：`test_double_buffer_pipeline.py` 升级为四模式真实现 + 按负载自动选型

**四模式**：

| 模式 | 结构 | 每批同步点 | 适用 |
|---|---|---|---|
| V0 基线 | H2D→计算→D2H 完整 event 链 | 2 wait（12 次/6 批） | 大负载（保留批间流水） |
| V1 精简链 | 只保留 H2D→计算，D2H 末尾统一 wait_stream | 1 wait（6 次） | 中等负载（综合最稳健） |
| V4 批量提交 | 全部 H2D→1 次同步→全部计算→1 次同步→全部 D2H | 2 次（牺牲批间流水） | 小负载（同步开销 -84%） |
| V5 同流顺序 | 同一流顺序执行（无流水线） | 0 | 极小负载（理论下界） |

**三档实测（rounds=5 中位，分母 = 同步拷贝+纯计算基准）**：

| n | 基准 | V0 | V1 | V4 | V5 | 选型 |
|---|---|---|---|---|---|---|
| 512 | 0.960 | -41.7% | -16.6% | +6.5% | **+33.1%** | **V5** |
| 1024 | 1.340 | -0.1% | +15.7% | **+27.0%** | +18.1% | **V4** |
| 2048 | 3.873 | **+38.8%** | +38.1% | +12.8% | +6.9% | **V0** |

> 数据来源：补 `record_stream` 后的最新运行 `dbuf_v2_scan_recordstream.json`
>
> **噪声观察（重要）**：n=512 档在两次运行间波动显著（V0 从 -25.6% 到 -41.7%，
> V4 从 +20.9% 到 +6.5%），而 n=2048 档稳定（V0 +40.1% → +38.8%）。
> 这印证了**ms 级小负载测量噪声极大**的纪律——小负载结论必须多轮多次复测。
> **选型结论（V5 / V4 / V0）在两次运行中完全一致，故选型本身是稳健的。**

**【完成判据】**
- ① **功能**：四模式 D2H 数据与主机参考逐批一致（**rel_err < 1e-3**，用相对误差——
  主机/设备累加顺序不同，绝对差随 n 增长，绝对阈值 1e-2 会误判）
- ② **性能**：按负载选型后各档重叠率为正（或正确降级为 V5 并说明"小负载不流水线"）
- 判定 `DBUF2_PASS`

**【现状】✅ DBUF2_PASS**（三档扫描，选型与 §5.2 求解结论一致）

**训练侧回补**（P2-⑧ 完整复测，`full_train_d8_20260902.sh`）：
- 改动：`train_qwen_1_5b_npu.py` / `train_qwen_1_5b_flagos.py` 的
  `DataLoader(pin_memory=True)` + `batch.to(dev, non_blocking=True)`
- 复测配置：双卡 hccl，`ASCEND_RT_VISIBLE_DEVICES=2,3 MAX_STEPS=100`
- **结果**：`TORCHRUN_EXIT=0` 全程无崩溃无 traceback；loss **2.64 → 1.37**（s50，
  历史基准 1.9 同量级）；tok/s **642 → 5166** 单调爬升无回落；
  训练后 HBM 完全回落基线（卡2 3161MB / 卡3 2880MB，与训练前一致，**无泄漏**）；
  进程残留 **0**
- 训练映射文档状态：**10/11 → 11/11 全绿**

---

### D9 同步语义（跨卡梯度同步 / 事件语义 / 错误不静默）

**【原理】**
分布式训练依赖跨卡梯度同步（all_reduce）；推理的 TP 依赖每层的 all_reduce/all_gather。
同步语义的正确性包含三层：**集合通信与流绑定**（collective 只在所在流阻塞）、
**事件语义**（record/query 的语义边界）、**错误不静默**（失败必须显式暴露）。

**【实现过程】**
1. **TP_COMM_PASS 4/4**（2026-08-31）：A 线重验 B 线 B2 缺陷（flagcx 异步无同步 → NaN），
   A 线不存在该问题
2. **补验（2026-09-02 O4）**：`test_tp_comm_sync_enhanced.py`，torchrun 2 卡，跑 2 次验证稳定
   - **E** 大张量 64MB all_reduce + 立即设备侧消费：双 rank 正确
   - **F 跨流不阻塞**（**流绑定的排他证据**）：流 S 发大 all_reduce，同时流 T 做独立计算
     → 重叠效率 rank0 54.5%~56.1% / rank1 96.5%~97.7%
   - **G** 4 流并发 all_reduce，结果全对（3072.0）
   - **H** 100 轮（原 20 轮）无 NaN
3. 训练侧：2481 步 DDP 全程稳定；**event 资源泄漏修复**（每步 ~120 个 aclrtEvent 累积 → 析构 + work 完成语义）

**【完成判据】**
- 集合通信无 NaN（H 场景 100 轮）
- **流绑定具排他性**：F 场景证明 collective 只阻塞所在流（原场景 A"立即消费正确"不具排他性——
  内部隐式同步也会通过）
- 事件语义正确（E1/E2/E3）

**【现状】✅**（含 F 场景排他证据与 2 次复测稳定性）

---

### D10 错误码翻译

**【原理】**
厂商（昇腾 ACL）错误码是数字（如 161002、107015），上层无法据此决策。职责要求
**三维翻译**：错误码 → 统一类别（L1-L4）+ 归因位置（location）+ 保留 root cause。
分级直接决定处置策略：

| 分级 | 语义 | 处置 |
|---|---|---|
| L1_RESOURCE | 资源类（OOM 等） | **可重试** |
| L2_PARAM | 参数/契约违反 | **上抛调用方**（重试无意义） |
| L3_EXECUTION | 执行期失败 | **同上下文重放** |
| L4_FATAL | 硬件/致命 | **设备状态恢复（R2-R5）** |

**【实现过程】**
1. **数据源发现**（关键突破）：不需要攒错误示例——**CANN 头文件就是权威错误码全集**
   - `acl/error_codes/rt_error_codes.h`：132 个
   - `aclnn/opdev/op_errno.h`：26 个（**无 `//` 注释**，需从宏名推导语义）
   - 全集 **159 个**
2. **自动分级 + 人工裁决**：规则按语义关键词分级 → 高置信差异人工裁决 → 固化防漂移
3. **覆盖率演进**：0.8% → 15.2% → 30.8% → **64.8%（103/159）**，高置信误判 **24 → 0**
   - **口径说明**：`103/159` 指"命中 CANN 头文件提取的 159 个码全集"的比例（审计口径）；
     `errors.py` 实际映射条目为 **108 条**——多出的 5 个属其他域（如 161xxx 基础段），
     不在 159 全集内。两者均正确，引用时须标明口径。
   - 当前审计快照（`acl_error_map_audit_latest.json`，2026-09-02 重跑）：
     分级不一致 18 个（11.3%），其中高置信建议差异 **3 个**——
     `507018 AICPU_EXCEPTION` / `507035 VECTOR_CORE_EXCEPTION` / `507049 FFTS_PLUS_EXCEPTION`
     （**有意覆盖规则，勿回退**：与 507015 AICORE_EXCEPTION 对齐，同类硬件单元 exception 不应有等级差）
4. **F5 可观测**：新增 `mapped` / `graded_by` / `is_grade_confident`
   （区分"确定分级"与"保守兜底"，防止上层把兜底 L3 当确定结论）
5. **三层验证策略**：
   - **L1** 覆盖审计（159 个码的覆盖率与分级差异）
   - **L2** 构造消息批量验证（不需真实触发）
   - **L3** 真实触发抽样：
     - **107015**（2026-09-01）：A/B 单变量对照 + **可逆性 E4**（subscribe→unsubscribe→launch 复现）
       + block 参数无关 + 跨设备复现 → 根因 = **stream 未 subscribe 即 launch callback**，定级 L2_PARAM
     - **507046**（2026-09-02 P2-⑦）：`TIMEOUT_REALTIME_PASS`
6. **集成到真实推理服务**（P0-②）：`inject_error_translation.py` 挂接 vLLM 全部异常 handler
   （不碰源码），实测 **VLLMValidationError → L2_PARAM**（此前被误兜底为 L3）

**【完成判据】**
- 覆盖率与一致性：64.8% 覆盖，语义组内一致（INVALID 组的 6 个例外为有意分层规则）
- **正确性无 ground truth**——只能保证"可审计 + 关键类真实触发验证通过"
- 翻译真实运行在推理错误路径（集成验证日志可见）

**【现状】✅**（含 107015 + 507046 两个真实触发样本）

---

### D11 设备状态恢复

**【原理】**
四态机 + 五段式恢复：

```
AVAILABLE ──错误──> DEGRADED ──评估失败──> ISOLATED ──重建──> AVAILABLE
                                              │
                                              └──重建失败──> DESTROYED
```

| 段 | 动作 |
|---|---|
| R1 捕获 | 错误归因（经 `translate_error`） |
| R2 评估 | 轻量探针区分 L3（可继续）与 L4（需重建），**避免不必要的高代价重建** |
| R3 隔离 | 损坏上下文从调度池摘除（ISOLATED），停止派发 |
| R4 重建 | 重建后新上下文 AVAILABLE |
| R5 重放 | 在途任务登记 + 重放接口 |

**【实现过程】**
1. **R1-R5 用例**（2026-09-01）：`DEVICE_STATE_RECOVERY_PASS 8/8`
   - 含 L4 完整链路 `captured → isolated → recovered → replay_ready`
   - **关键设计**：`handle_error` 只对 **L4_FATAL** 触发完整 R2-R5；L1-L3 直接重放/上抛
   - **瞬时故障注入**：健康设备下 L4 的 evaluate 探针必过（不重建），
     故用"前 2 次 sync 失败、第 3 次成功"的注入跑完整链路
2. **持久故障证伪**（关键）：S2 场景注入持久故障 → `recovered: False`、状态**保持 ISOLATED**
   → **不会在设备真不可用时假装恢复**
3. **真实重建落地**（2026-09-02 P1-③）：
   - CANN 官方序列（`acl_rt.h`）：`destroyEvent → destroyStream → destroyContext →
     aclrtResetDevice → setDevice → 重建`
   - `recovery.py` 新增 `recover_device(rebuild_mode=probe/real/hybrid)`
   - **多进程本地联调** `MULTIPROC_REBUILD_PASS`（2 次复测稳定）：
     - S1 恢复者真实重建 `recovered=True`
     - **S2 同卡服务进程 30/30 轮零失败**（B 执行 reset 期间 A 完全不受影响 → **隔离性成立**）
     - S3 重建后 pyACL 句柄 + torch_npu 计算均恢复

**【完成判据】**
- 四态可查询、状态转换正确
- **持久故障不假恢复**（`recovered: False` + 保持 ISOLATED）
- 真实重建序列可执行且多进程隔离（MULTIPROC_REBUILD_PASS）
- 回归无破坏（conformance 13/13 + infer 6/6）

**【现状】✅** ⚠️ **real/hybrid 模式生产默认启用前需大模型多卡多进程压力测试调优**
（当前默认 `probe` 保底，进程内安全）

---

## 4. 多流 Stream 子项详解（S-1~S-16）

> 探针：`benchmarks/ascend_regression/probe_stream_semantics_full.py`（判定 `STREAM_SEMANTICS_PASS 8/8`）

### S-1 流内顺序性（FIFO）

**【原理】**同一流上提交的任务必须按提交顺序执行——这是流语义的最基本保证。
若乱序，依赖前序结果的算子会读到未初始化数据。

**【实现】**真正创建流（补强既有用例的"框架层近似"），在流上提交 4 个串行算子：
```
with torch.npu.stream(s):
    x = ones(64,64)     # op1
    y = x * 3           # op2
    z = y + 2           # op3
    r = z.mean()        # op4
```
**【判据】**结果 = `((1*3)+2).mean() = 5.0`
**【现状】✅ 5.000000**

### S-2 显式依赖 / 无隐式同步

**【原理】**不同流之间**不存在隐式同步**——流 A 的任务完成不会自动对流 B 可见。
依赖必须显式建立（event / wait_stream）。这是多流编程最容易出错的地方。

**【实现】**创建两条流 sA/sB，各自独立计算（不建立依赖），验证互不干扰。
**【判据】**流 A 均值 = 7.0、流 B 均值 = 11.0（各自正确）
**【现状】✅ A=7.0000 B=11.0000**

### S-3 跨流可见性（event）

**【原理】**通过 `event.record(streamA)` + `streamB.wait_event(event)` 建立跨流依赖后，
流 A 的计算结果对流 B 可见。

**【实现】**conformance **S3**（训练侧）+ **i3** KV 模拟缓冲跨流可见性（推理侧）
**【判据】**依赖建立后读取结果正确（S3: z 全 3）
**【现状】✅**

### S-4 wait_stream 传递

**【原理】**`streamB.wait_stream(streamA)` 表示"B 等待 A 上**此前所有**任务完成"。
注意其粒度比 event 更粗（见 V2 变体的教训：**过度等待**反而更慢）。

**【实现】**conformance **S4**
**【判据】**`b 全 10 = True`
**【现状】✅**

### S-5 多流并发重叠

**【原理】**多流并行执行，使 H2D / 计算 / D2H 三类任务在时间上重叠，隐藏传输延迟。
**【实现】**推理侧双缓冲 v2 四模式（§3 D8）+ TP 通信 F 场景；训练侧多流流水线
**【判据】**按负载选型后重叠率为正（DBUF2_PASS）
**【现状】✅**

### S-6 集合通信与流绑定

**【原理】**集合通信（all_reduce）必须与**当前流**正确绑定：
它应该只阻塞自己所在的流，**不应阻塞其他流的独立计算**。

**【实现】**`test_tp_comm_sync_enhanced.py` 的 **F 场景**（排他性证据）
**【判据】**流 S 通信与流 T 计算重叠效率 > 5%（实测 54.5%~97.7%）
**【现状】✅** ⚠️ 注意：原场景 A（"all_reduce 后立即消费正确"）**不具排他性**——
若 all_reduce 内部做隐式同步，绑错流也会通过。必须补 F 场景。

### S-7 图捕获（graph capture）流语义

**【原理】**CUDA Graph 语义移植：把一段设备工作"录制"成图，后续可低成本重放，
消除逐算子提交开销。昇腾侧 `torch.npu.graph` 是完整移植（NPUGraph + replay）。

**【实现】**`probe_graph_capture_stream.py`，5 项验证
**【判据】**
- G1 capture 结果与 eager 一致（rel_err < 1e-3）
- G2 两次 replay 结果逐位一致（差 = 0）
- G3 更新输入后 replay 使用新值
- G4 capture 内切换流后结果仍正确
- G5 显式传 stream 参数后结果正确

**【现状】✅ GRAPH_CAPTURE_PASS 5/5**（rel_err 全 0）
⚠️ **踩坑**：**capture 只记录不执行，必须先 replay 才产生结果**——首版探针违反此语义误判 G1/G2。

### S-8 默认流 vs 命名流 + 跨流分配器安全 ★最重要

**【原理】**
torch 的当前流（`current_stream`）即默认流；`torch.npu.Stream()` 创建的是命名流。
**关键的工程陷阱**：PyTorch 缓存分配器按流跟踪内存——若一个 tensor 在流 A 上分配、
在流 B 上使用后于流 A 上释放，分配器**不知道流 B 还在使用它**，可能把这块内存重分配给
流 A 上的新 tensor → **数据竞争**。

**正确做法**：`tensor.record_stream(using_stream)` 显式告知分配器"这块内存还会在该流上使用"。

**【实现】**
1. 探针验证 8a（默认流/命名流各自正确）+ 8b（`record_stream` 后跨流复用正确）
2. **修复真实缺口**：核查发现 `test_double_buffer_pipeline.py` 的 V0/V1/V4 在命名流间传递
   `bufs`/`outs` 但**未调 record_stream**——当前"碰巧安全"仅因 Python 引用长期持有，
   一旦改为批间释放即暴露竞争。已补并注明。

**【判据】**
- 默认流与命名流各自执行正确
- `record_stream` 后跨流复用结果正确（实测 10.0，期望 10）

**【现状】✅**（含实现修复，重跑三档仍 DBUF2_PASS）

### S-9 流错误隔离（分层）★重要语义

**【原理】**错误隔离是**分层的**，不能一概而论：

| 层级 | 影响范围 | 恢复方式 |
|---|---|---|
| **API 调用级**（如 107015） | 仅该次调用失败 | 修正调用前置条件，其他流不受影响 |
| **设备级**（507014 AICORE_TIMEOUT / 507015 AICORE_EXCEPTION 等） | **该设备全部流** | 必须走**设备级恢复**（`aclrtResetDevice`），**流级重试无效** |

这条与 D11 的 L4 分级恢复设计自洽——L4 走设备恢复而非流重试，正是基于这个语义。

**【实现】**流 A 注入真实 107015（`launch_callback` 未 subscribe），验证流 B 仍能正常执行。
**【判据】**流 A `rc=107015`（期望值）且流 B 结果正确（6.0）
**【现状】✅ API 级已实测（rc=107015，流B=6.0）**
⚠️ **设备级为语义推断**——依据 `rt_error_codes.h` 错误码定义（AICORE_TIMEOUT/EXCEPTION 属
芯片级）与 D11 分级设计推断，**未做真实触发**（触发风险高）。已在文档明确标注为推断。

### S-10 流/事件生命周期与配额

**【原理】**流与事件是有限资源。长期运行的服务若每批创建而不销毁，会耗尽配额。
训练侧曾发现真实缺陷：**每步 ~120 个 aclrtEvent 累积**（已修复：析构 + work 完成语义）。

**【实现】**循环 500 次创建流 + 事件 → 使用 → 销毁，之后验证仍能正常创建使用新流。
**【判据】**500 次循环后新流仍可用且结果正确（9.0）
**【现状】✅ 无配额泄漏**

### S-11 跨流内存分配

**【原理】**在流 A 上分配的设备内存，经显式依赖建立后，可被流 B 安全使用。
**【实现】**流 A 分配 `buf` → `event.record` → 流 B `wait_event` → 流 B 使用 `buf`
**【判据】**流 B 使用结果正确（6.0）
**【现状】✅**

### S-12 流优先级

**【原理】**昇腾支持流优先级调度（`aclrtCreateStreamWithConfig(stream, priority, flag)`，
priority 范围 **0~7**）。`aclrtDeviceGetStreamPriorityRange(least, greatest)` 查询范围。

**【实现】**查询优先级范围 + 创建多流验证并发正确性
**【判据】**范围查询成功且多流并发结果正确
**【现状】✅ leastPriority=7 / greatestPriority=0** —— **值越小优先级越高**
（与 `aclrtCreateStreamWithConfig` 注释 "value range:0~7" 一致）
⚠️ 注：**调度效果**（高优先级流是否真的优先）需压力测试验证，非职责验收项。

### S-13 多设备流绑定

**【原理】**流绑定于创建时的当前设备。多卡场景下，各卡的流相互独立、不应互串。
**【实现】**双设备各创建独立流，各自计算并验证结果
**【判据】**各设备结果正确且不互串（实测 [2.0, 3.0]）
**【现状】✅**

### S-14 流同步超时语义

**【原理】**`aclrtSynchronizeStreamWithTimeout(stream, timeout_ms)` 提供**有界等待**——
超时返回错误而非无限阻塞，是长驻服务避免整体hang死的关键能力。

**【实现】**`probe_timeout_realtime.py`（P2-⑦）
**【判据】**真实返回 `ACL_ERROR_RT_STREAM_SYNC_TIMEOUT(507046)` 且翻译为 L3_EXECUTION
**【现状】✅ TIMEOUT_REALTIME_PASS**
⚠️ **踩坑**：任务与同步必须**在同一 stream**——首版同步 pyACL 空流立即返回 0（未触发）。
需使用 `torch.npu.Stream().npu_stream` 获取底层句柄。

### S-15 跨进程流共享（IPC）

**【原理】**流句柄跨进程共享属于分布式 IPC 能力。
**【现状】⬜ 标注不适用**——本方向当前为单机多卡，无跨进程流共享需求；属大规模分布式/上游能力。

### S-16 流数量配额

**【实现】**连续创建 2000 个流探测上限
**【现状】✅ 2000 个流创建成功，无显式配额限制**

---

## 5. 关键纠错历程（3 项错误结论的撤回与重做）

> 这三次纠错是本方向最有价值的产出——**"看起来达标"比"知道没达标"更危险**。

| # | 原错误结论 | 错误性质 | 修正后正确结论 |
|---|---|---|---|
| 1 | **D8/A3**：n≤1024 重叠率为负系"物理限制" | **归因错误 + 不可证伪** | 实为**实现层同步开销**（V4 证明可压降 84%），非硬件不可逾越 |
| 2 | **A5**：TP=1/2/4 greedy 逐字一致 4/4 | **样本量不足** | 16 prompt × 256 token 下**数值不等价**（4/16 全长一致，一致 token 50.1%），**但语义等价** |
| 3 | **A1**：dense 推理 19.5 tok/s | **测量污染** | 同口径重测 **283.0 tok/s**（预热不足污染，差 14.5 倍） |

### 纠错 1 详解：为什么"物理限制"是危险归因

**错误链路**：n=512 单轮测得 V1 = +21.6%（看似达成）→ 未做规模扫描就下结论 →
后又把负值归因于"物理限制"。

**证伪证据**：
- 规模扫描复测：n=512 的 V1 实为 **-41.0%**（首轮 +21.6% 是噪声）
- 严谨重测（每规模独立预热 + rounds=7 + 分段计时）：n=2048 实为 **-15.2%**，
  之前的"+31%"是**跨规模复用小尺寸预热**导致 serial 基线被高估的假象
- **V3 反证早就在表上**（计算×8 却只有 15.5%，低于 V0 的 31.0%），
  直接说明"计算粒度决定论"不成立——但当时只挑了支持假设的行看

**正确结论**：实现层同步开销，可通过 V4（批量提交）压降 84%，**可解决**。

### 纠错 2 详解：TP 分叉是固有特性而非缺陷

**关键对照实验**：

| 对比 | 一致率 | 说明 |
|---|---|---|
| TP=1 vs TP=1（重跑） | **100%** | greedy 与 batch 调度均确定性 → **排除调度变量** |
| TP=1 vs TP=2 | **50.1%** | 分叉确实源自 TP 并行的归约顺序差异 |

**分叉后语义质量**（关键）：
- [8] 博客大纲（前缀 0）：TP=1 "1. Introduction, 2. Understanding LLMs…" /
  TP=2 "Introduction, The Role of Inference in LLMs…" —— **两个都是正确答案**
- [4] fibonacci（前缀 10）：TP=1 返回**列表**版 / TP=2 返回**标量**版 —— 均正确

**结论**：是"**选择分叉**"而非质量退化；**分叉早晚与任务选择空间负相关**
（质数表/数数等确定性任务全长一致，开放式大纲从第 0 个 token 就分叉）。

**对联调压测的价值**：判定缺陷应看「随机性 / NaN / 乱码 / 语义崩坏」，
**不是「与 TP=1 逐字不同」**——否则会把固有特性误判为缺陷。

### 纠错 3 详解：吞吐口径

| 数字 | 口径 | 可信度 |
|---|---|---|
| 19.5 tok/s | P0 的 4 条批处理，121 token / 6.21s | ❌ 预热不足污染 |
| 68.4 tok/s | 4 条**串行** HTTP 请求 | ✅ 但非批量峰值 |
| **283.0 / 285.2 tok/s** | 16 prompt × 256 token **批量** | ✅ **同口径基准** |

批量（284）vs 串行（68.4）差 4.2 倍来自 **batching**；
两模型（1.5B 283.0 / 4B 285.2）吞吐接近 → 大 batch 下瓶颈在**显存带宽**而非算力。

---

## 6. 遗留项与后续计划

| 优先级 | 项 | 说明 |
|---|---|---|
| — | **O3 PR 到 dev-1.0** | 描述已备（`docs/PR_DEV_1_0_20260901.md`），**暂不发起**（等通知） |
| 高 | **D11 real/hybrid 压力测试** | 本地多进程联调已过，**生产默认启用前需大模型多卡多进程压力测试调优**（多卡并发恢复、reset 窗口请求排队、跨容器共享设备竞争） |
| 中 | S-12 流优先级**调度效果** | 范围与并发正确性已验，调度效果需压力测试 |
| 中 | S-9 设备级错误真实触发 | 当前为语义推断；真实触发风险高，建议联调时用压测环境做 |
| 低 | D6 性能项 | 随 D8 选型结论条件性达标，已给出可操作指引 |
| 维护 | CANN 升级复核 | 错误码会变，应重跑 `gen_acl_error_map.py` + `audit_error_map_coverage.py` |

---

## 7. 附录 A：核心资产清单

| 资产 | 路径 | 用途 |
|---|---|---|
| 推理映射文档 | `docs/DEVICE_CONTEXT_INFERENCE_MAPPING_20260831.md` | 推理侧验收依据（A1-A11 + §5.4 stream 核查表） |
| 训练映射文档 | `docs/DEVICE_CONTEXT_TRAINING_MAPPING_20260831.md` | 训练侧验收依据（B1-B11） |
| 错误码映射专题 | `docs/ACL_ERROR_MAP_20260901.md` | D10 数据源/三层验证/裁决原则 |
| 双缓冲 v2 | `benchmarks/ascend_regression/test_double_buffer_pipeline.py` | D8 四模式 + 按负载选型 |
| 双缓冲求解实验 | `benchmarks/ascend_regression/test_dbuf_variants_rigorous.py` | §5.2 规模扫描与分段计时 |
| **Stream 全量核查探针** | `benchmarks/ascend_regression/probe_stream_semantics_full.py` | S-1/S-2/S-8~S-13 共 8 项 |
| graph capture 探针 | `benchmarks/ascend_regression/probe_graph_capture_stream.py` | S-7 |
| timeout 真实触发 | `benchmarks/ascend_regression/conformance/probe_timeout_realtime.py` | S-14 |
| 设备重建探针 | `benchmarks/ascend_regression/probe_device_reset_rebuild.py` | D11 真实重建 |
| 多进程重建联调 | `benchmarks/ascend_regression/conformance/test_rebuild_multiprocess.py` | D11 多进程隔离 |
| 错误翻译注入 | `benchmarks/inference/inject_error_translation.py` | D10 集成 vLLM |
| 设备状态监控 | `benchmarks/inference/device_state_monitor.py` | D11 集成监控 |
| 恢复模块 | `benchmarks/ascend_regression/conformance/recovery.py` | R1-R5 + rebuild_mode |
| 错误码模块 | `benchmarks/ascend_regression/conformance/errors.py` | 103 条映射 + F5 可观测 |
| TP 通信增强 | `benchmarks/inference/test_tp_comm_sync_enhanced.py` | S-6 排他证据 |
| 训练脚本 | `benchmarks/train_qwen_1_5b_npu.py` / `train_qwen_1_5b_flagos.py` | D7/D8 训练侧回补 |

## 8. 附录 B：复现命令

```bash
# SSH 进入 910C
ssh 910C
docker exec -it flagos-infer-910c bash

# ── D8 双缓冲（三档扫描，rounds=5）──
cd /mnt/raid/hliu553/runtime-team/dev/device-context/benchmarks/ascend_regression
python3 test_double_buffer_pipeline.py --scan --rounds 5

# ── S-1/S-2/S-8~S-13 Stream 全量核查 ──
ASCEND_RT_VISIBLE_DEVICES=2,3 python3 probe_stream_semantics_full.py --leak-iters 500

# ── S-7 graph capture ──
python3 probe_graph_capture_stream.py

# ── S-14 timeout 真实触发 ──
cd conformance && ASCEND_RT_VISIBLE_DEVICES=3 python3 probe_timeout_realtime.py

# ── D11 真实重建 + 多进程联调 ──
python3 probe_device_reset_rebuild.py
cd conformance && ASCEND_RT_VISIBLE_DEVICES=2 python3 test_rebuild_multiprocess.py --rounds 30

# ── 回归（conformance 13 例 + infer 6 例）──
cd conformance && ASCEND_RT_VISIBLE_DEVICES=3 python3 runner.py --chip ascend --backend npu
python3 runner.py --chip ascend --backend npu --cases infer_cases

# ── 训练完整复测（双卡 hccl，100 步）──
cd benchmarks
ASCEND_RT_VISIBLE_DEVICES=2,3 BACKEND=hccl MAX_STEPS=100 \
  torchrun --nproc_per_node=2 --master_port=29522 train_qwen_1_5b_npu.py
```

## 9. 附录 C：方法沉淀（测量与归因纪律）

> 这 7 条是本方向用 3 次纠错换来的，适用于所有昇腾性能与语义验证工作。

1. **自身数据的反证优先于假设** —— V3 的反常早就在表上，不能只挑支持假设的行看
2. **测量必须按被测规模预热** —— 跨规模复用小尺寸预热 = 系统性偏差（+31% 假象的根源）
3. **样本量要匹配结论强度** —— 256 token 撑不起"数值等价"这类强结论
4. **因果结论前先排除替代解释** —— 自重复对照 / 可逆性对照（如 TP=1 自重复 100%）
5. **"物理限制"是最危险的归因** —— 它把"我没搞定"包装成"不可能搞定"，且不可证伪
6. **验证 ≠ 集成** —— 集成才暴露真实边界（框架错误类型、异步语义、进程管理）
7. **证伪 ≠ 求解** —— 证伪只是起点，要继续做实验直到拿到"什么是对的"

**补充两条工程纪律**：
- **数据一致性判据用相对误差**（主机/设备累加顺序不同，绝对差随规模增长）
- **比值型指标要检查极端比例下的区分度**（如"节省比例"在两者量级悬殊时天然失效，
  应改用 `实际节省 / 理论最大可节省 min(a,b)`）

---

**文档结束** ｜ 生成时间 2026-09-02 ｜ 对应提交 `75b5ebe` ｜ 三方一致（mac / GitHub / 910C）
