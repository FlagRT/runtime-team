# 调研笔记：vLLM 0.20.2 分层/溢出机制（2.4 任务 #3）

> 日期：2026-08-17 ｜ 来源：容器 venv311 内 vllm 0.20.2 源码（/root/vllm-venv311/lib/python3.11/site-packages/vllm/）
> 用途：子项 B「分层缓存与可控溢出」的可复用挂载点清单

## 1. 模型权重 offload（--cpu-offload-gb）

两个后端（config/offload.py）：
- **UVA 后端**（UVAOffloadConfig，offload.py:19-42）：`--cpu-offload-gb` 把权重放 CPU pinned 内存，靠统一虚拟寻址零拷贝按需取用。要求 CPU-GPU 快速互连（文档明说 fast interconnect），否则 forward 期间取权重会拖垮吞吐。支持 `cpu_offload_params` 按参数名段精确指定哪些权重 offload（offload.py:35-41）。
- **Prefetch 后端**（PrefetchOffloadConfig，offload.py:49-60）：按层分组（offload_group_size/offload_num_in_group），**异步 H2D 预取**隐藏传输延迟——这是分层缓存的核心设计模式：层为粒度、预取提前量、流并行。

抽象层（model_executor/offloader/base.py）：
- `BaseOffloader`（base.py:47-93）：`wrap_modules` / `post_init` / `sync_prev_onload` / `join_after_forward` / `_wait_for_layer` / `_start_prefetch`
- `get_offloader()` / `create_offloader()`（base.py:111-126）— 工厂接入

**对 2.4 的意义**：vLLM 的 offloader 抽象 = 现成的"可控溢出"设计骨架。但注意：
- 它只处理**权重**（低优先级权重放 CPU），不处理 KV cache；
- UVA 后端依赖统一寻址（CUDA 特性），**Ascend/CANN 是否有等价物（aclrtMallocHost + 统一寻址）必须验证**——这是子项 B 调研的关键未知点之一。

## 2. KV cache 预分配与按需释放挂载点

- `v1/core/kv_cache_manager.py:106` KVCacheManager：
  - `allocate_slots`（:265）— 按需分配块
  - `free`（:437）— 释放块
  - **`evict_blocks`（:460）— 块驱逐（prefix cache 淘汰用）→ 分层缓存"按需释放"的天然 hook 点**
  - `cache_blocks`（:534）/ `get_computed_blocks`（:183）— prefix cache 读写
- KV cache 以整块 tensor 从 torch_fl 分配器拿显存（块级逻辑归 vLLM），分块粒度 16 tokens（block_size）。

**对 2.4 的意义**：子项 B 的最现实切入点 = 在 `evict_blocks`/`allocate_slots` 挂"溢出层"（块被驱逐时写 Host 暂存，需要时再取回），不动 vLLM 调度逻辑。这与 vllm-plugin-FL 已有的 `flagcx_connector.py`（KV 跨机传输连接器）是同一类挂载点（KVConnectorBase_V1 接口）。

## 3. 跨实例 KV 传输（--kv-transfer-config）

- vLLM 生态用 KVConnector 体系（distributed/kv_transfer/kv_connector/，base.py 定义 KVConnectorBase_V1 + SupportsHMA）做分布式 KV 共享（Mooncake 等实现）。
- vllm-plugin-FL 已有 `flagcx_connector.py`（KVConnectorBase_V1 子类 + FlagCX 传输）——**本地分层缓存可复用同一接口族**（Host 暂存可视为"远端=本机 CPU/SSD"）。

## 4. 结论与待办

| 发现 | 结论 |
|---|---|
| offloader 抽象（UVA/prefetch 双后端） | 设计骨架可抄；权重溢出是子项 B 的一部分（低优先级权重） |
| UVA 依赖统一寻址 | **待验证**：CANN 是否支持 pinned 内存零拷贝访问（aclrtMallocHost + 寻址语义） |
| evict_blocks / KVConnectorBase_V1 | KV 溢出的挂载点已明确，flagcx_connector 可扩展为本地分层 |
| --cpu-offload-gb 只覆盖权重 | KV cache 溢出无内置（vLLM 设计上 KV 归 KVConnector 管）→ 子项 B 需自研 KV 分层 |

下一步动作：① V3 原型设计基于 evict_blocks hook + flagcx_connector 模式；② 调研 CANN 统一寻址/pinned 内存能力（容器内测 aclrtMallocHost + 零拷贝读）；③ SSD 层评估（V4）留待 Host 层验证后。
