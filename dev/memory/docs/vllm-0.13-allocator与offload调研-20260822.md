# 调研：vllm 0.13.0 allocator / offload 机制（对照 0.20.2 笔记）— 2026-08-22

> 执行：xliu969（Hermes 协助）｜ 性质：源码静态调研（容器内 grep/sed 定位，未改任何代码）
> 环境：容器 flagos-official-moe-recheck（官方发布镜像，昆仑芯 P800 构建），
> vllm 0.13.0 site-packages = /root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/vllm/（下文缩略为 `$V`）
> 对照基线：[vllm-offload-调研笔记-20260817.md](vllm-offload-调研笔记-20260817.md)（0.20.2，来源容器 venv311 已不在运行容器中，0.20.2 侧仅依据笔记、未现场复核）
> 任务来源：[路线A-显存与缓存管理-方案-20260822.md](路线A-显存与缓存管理-方案-20260822.md) §4.3「vllm 0.13 vs 0.20.2 差异：offload/evict_blocks/prefix caching/allocator 接口逐项对照」

---

## 1. 结论摘要（一屏）

- **0.13.0 是 v1 引擎架构**（vllm/v1/engine/core.py），内存摸底 / KV 块表 / prefix caching / KVConnector 体系齐全，与 0.20.2 同族。
- **重大发现：0.13.0 原生内置 KV cache 卸载到 CPU**（`--kv-offloading-size` + `--kv-offloading-backend native|lmcache`），
  实现形态正是 0.20.2 笔记判断的「KV 归 KVConnector 管」——`OffloadingConnector` 是 `KVConnectorBase_V1` 子类，
  底层 `vllm/v1/kv_offload/` 子系统（LRU/ARC 驱逐策略 + CPUBackend + swap_blocks 异步传输）。
  **这直接推翻了 0.20.2 笔记「KV 溢出无内置 → 子项 B 需自研 KV 分层」的结论（对 0.13 而言）**：
  V3 分层缓存的 CPU 溢出层在 0.13 上已有官方实现，优先实测官方路径，不必从零自研。
- 权重 offload（`--cpu-offload-gb`）在 0.13 存在但**无 0.20.2 的 offloader 抽象**（无 model_executor/offloader/ 目录）：
  仅 UVA 单机制 `maybe_offload_to_cpu`（按层模块整体 offload + 全局字节预算），无 prefetch 后端、无按参数名段选择。
- `evict_blocks` / prefix caching / KVCacheManager 挂载点在 0.13 均存在，与 0.20.2 笔记结构一致。
- 平台侧：`vllm/platforms/` 无 kunlunxin.py，昆仑芯构建靠插件 `PlatformFL` 引导；显存相关扩展点 =
  插件 vendor attention backend（KunlunxinAttentionBackend）+ 厂商 torch.cuda caching allocator。
- 两个待实测疑点：① native KV offload 的 `kv_bytes_per_rank → num_cpu_blocks` 换算点未在 0.13 树内定位
  （配置侧占位 0，CPUOffloadingSpec 对 0 直接 raise）→ 实际启用可能需 `--kv-transfer-config` 显式给 num_cpu_blocks；
  ② 权重 offload 依赖 `is_uva_available()`（= pin_memory 可用性），P800 xpytorch 是否满足待实测。

---

## 2. 调研项逐项定位（0.13.0，附 file:line）

### 2.1 内存摸底 / profile_run 路径

与 0.20.2 同族（纯 MoE 报告失败栈 `core._initialize_kv_caches → determine_available_memory → profile_run → _dummy_run` 即此路径）：

| 环节 | 位置 | 说明 |
|---|---|---|
| 入口 | `$V/v1/engine/core.py:109` / `:218` `_initialize_kv_caches` | profile → determine_available_memory → get_kv_cache_configs → `num_gpu_blocks`；`num_cpu_blocks=0`（:253，CPU swap 块） |
| executor | `$V/v1/executor/uniproc_executor.py:178` `determine_available_memory` | 多卡聚合 |
| worker | `$V/v1/worker/gpu_worker.py:298` `determine_available_memory` | `memory_profiling(init_snapshot, weights_memory=...)` 上下文 + `model_runner.profile_run()` |
| 预算计算 | `$V/v1/worker/gpu_worker.py:241` | `requested_memory = init_snapshot.total_memory × cache_config.gpu_memory_utilization`（**基于 total 而非 free**） |
| 可用 KV 内存 | `$V/v1/worker/gpu_worker.py:356` | `available_kv_cache_memory_bytes = requested_memory − non_kv_cache_memory` |
| dummy run | `$V/v1/worker/gpu_model_runner.py:4424` `profile_run` | 最大 batch dummy forward，即 MoE 崩溃点 |

### 2.2 KV cache 分配器

- **无 `PagedAllocator` 类**（grep 全树无命中）。0.13 无独立 KV 专用分配器类名；
  KV cache 以整块 tensor 分配：`$V/v1/worker/gpu/attn_utils.py:69` `_allocate_kv_cache`（:138 调用）。
- CUDA 侧有 `$V/device_allocator/cumem.py:113` `CuMemAllocator`（+ cumem_allocator.abi3.so，CUDA VMM 池，非 KV 专用）。
- `gpu_memory_utilization`：`$V/config/cache.py`（默认 0.9），消费点 gpu_worker.py:241。
- 块管理：`$V/v1/core/kv_cache_manager.py:94` `KVCacheManager` — `allocate_slots` :206 / `free` :326 / **`evict_blocks` :336**；
  `$V/v1/core/block_pool.py:384` `free_blocks` / `:400` `evict_blocks`。
- 0.20.2 对照（笔记）：kv_cache_manager.py:106 KVCacheManager、:265 allocate_slots、:437 free、:460 evict_blocks —— 结构一致，仅行号偏移。

### 2.3 offload 机制

#### (a) 权重 offload：`--cpu-offload-gb` 存在，但无 offloader 抽象

- CLI/配置：`$V/config/cache.py:95` `cpu_offload_gb: float = Field(default=0)`；`$V/engine/arg_utils.py:434` / `:907`（`--cpu-offload-gb`）。
- 实现：`$V/v1/worker/gpu_model_runner.py:290` `set_cpu_offload_max_bytes(int(cpu_offload_gb * 1024**3))`
  → `$V/model_executor/models/utils.py:516` `set_cpu_offload_max_bytes`（全局字节预算 `_CPU_OFFLOAD_MAX_BYTES`）
  → `:522` `maybe_offload_to_cpu(module)`：UVA 零拷贝 pinned 内存 offload（:536 `assert uva_available`）；
  `:606` 在 `make_layers` 逐层构建时包装（**按层模块粒度**，无参数名段选择）。
- `is_uva_available`：`$V/utils/platform_utils.py:55`（= `current_platform.is_pin_memory_available()`）。
- 限制：`$V/v1/worker/gpu_model_runner.py:5035` 断言——非标准 block_size 的 input batch 重初始化路径与
  cpu_offload_gb 不兼容（见 PR vllm-project/vllm#18298）。
- **0.20.2 对照（笔记）**：offloader 双后端 `UVAOffloadConfig`（config/offload.py:19-42，支持 `cpu_offload_params` 按参数名段）
  / `PrefetchOffloadConfig`（offload.py:49-60，按层分组异步 H2D 预取）+ `BaseOffloader` 抽象（model_executor/offloader/base.py:47-93）+
  工厂（base.py:111-126）。**0.13 无 model_executor/offloader/ 目录**——无 prefetch 后端、无参数名段选择、无 offloader 工厂抽象。

#### (b) KV offload：0.13 原生内置（重大发现）⭐

配置与接线：
- `$V/config/cache.py:34` `KVOffloadingBackend = Literal["native", "lmcache"]`；`:151` `kv_offloading_size: float | None`；
  `:157` `kv_offloading_backend`。CLI：`$V/engine/arg_utils.py:570-572`、`:924-925`（`--kv-offloading-size` / `--kv-offloading-backend`）。
- 接线：`$V/config/vllm.py:474-515` `_post_init_kv_transfer_config`——
  - `native` → `kv_transfer_config.kv_connector = "OffloadingConnector"`，extra_config `{"kv_bytes_per_rank": size×GiB/num_kv_ranks, "num_cpu_blocks": 0}`（:493-500）；
  - `lmcache` → `"LMCacheConnectorV1"`（lmcache.local_cpu=True / max_local_cpu_size，:502-506）；
  - 统一 `kv_role = "kv_both"`（:509）。

OffloadingConnector（= KVConnector 形态，同 0.20.2 笔记的 flagcx_connector 接口族）：
- `$V/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:45` `OffloadingConnector(KVConnectorBase_V1)`
  - scheduler 侧：`OffloadingConnectorScheduler`（:140）——block hash 匹配（`_get_block_hashes` :162）、
    store/load 决策（`update_state_after_alloc` :227、`build_connector_meta` :340）、`take_events` :391；
  - worker 侧：`OffloadingConnectorWorker`（:411）——异步 job 化传输（`start_load_kv` :463 / `start_store_kv` :471，`transfer_async`）；
- 注册：`$V/distributed/kv_transfer/kv_connector/factory.py:183-185`（"OffloadingConnector"）。

kv_offload 子系统（`$V/v1/kv_offload/`）：
- `spec.py:19` `OffloadingSpec(ABC)`：`gpu_block_size`（= cache_config.block_size）/ `offloaded_block_size`（extra_config 可配，须整除）；
  **标注 experimental**（spec.py:14-16 warning）。
- `cpu.py:23` `CPUOffloadingSpec`：`num_cpu_blocks` 取自 extra_config（:24-27，**0 值直接 raise**）；
  `eviction_policy` lru|arc（:38-46）→ `LRUOffloadingManager`（lru_manager.py）/ `ARCOffloadingManager`（arc_manager.py）；
  `CPUBackend`（backends/cpu，block_size=offloaded_block_size）；**:77-79「CPU Offloading 目前仅支持 CUDA-alike GPU」**（P800 满足）；
  handlers：`CpuGpuOffloadingHandlers`（worker/cpu_gpu.py:171，gpu_to_cpu + cpu_to_gpu 双向）。
- 传输实现：`worker/cpu_gpu.py:107` `SingleDirectionOffloadingHandler.transfer_async`——
  **`ops.swap_blocks` + CUDA stream/event 串行异步**（v0 CPU swap 同款算子，stream 池 + event 链保证顺序）。
- 管理器接口：`abstract.py:69` `OffloadingManager`（lookup/prepare_load/touch/complete_load/prepare_store/complete_store/take_events）。

**接线疑点（待实测）**：`kv_bytes_per_rank → num_cpu_blocks` 换算点未在 0.13 树内定位
（grep 仅 config/vllm.py:495 写入占位 0；CPUOffloadingSpec 对 0 直接 raise）。
→ 实际启用 native 后端可能需 `--kv-transfer-config` 显式提供 `kv_connector_extra_config.num_cpu_blocks`；
0.13 官方支持度以 P800 实测为准。

#### (c) evict_blocks 挂载点

- `$V/v1/core/kv_cache_manager.py:336` `evict_blocks(block_ids)` → `block_pool.py:400`。
- 调用点：`$V/v1/core/sched/scheduler.py:1803`（KV load 失败处理：`_update_requests_with_invalid_blocks(..., evict_blocks=True)`，
  配合 `recompute_kv_load_failures` 策略）——语义为 prefix cache / KV load 失效驱逐，与 0.20.2 笔记挂载点同类。
- KV offload **不走** evict_blocks：走 KVConnector 事件流（scheduler 侧 take_events → worker 异步传输）。

### 2.4 prefix caching

- 默认开启：`$V/config/cache.py:76` `enable_prefix_caching: bool = True`；`:78` `prefix_caching_hash_algo: "sha256"`。
- 实现：block hash 哈希表（`$V/v1/core/block_pool.py:23-25` get_block_hash / make_block_hash_with_group_id / maybe_convert_block_hash；
  `:36-37` cached_block_hash_to_block；`:93` 命中检查；`:132` 驱逐）。
- 0.20.2 对照（笔记）：cache_blocks/get_computed_blocks —— 一致（0.13 为 hash 表 + 组 id）。

### 2.5 平台相关（kunlunxin 构建）

- `$V/platforms/` 仅 cpu / cuda / rocm / tpu / xpu（**无 kunlunxin.py**）。
- platform 解析：懒加载 + 支持 out-of-tree 插件（`$V/platforms/__init__.py:234-270`，`__getattr__` current_platform →
  `resolve_current_platform_cls_qualname` → `resolve_obj_by_qualname`）。
- 昆仑芯构建由插件引导：`/env/xvllm-plugin-FL/vllm_fl/__init__.py:38` `register()` 返回 `"vllm_fl.platform.PlatformFL"`；
  `vllm_fl/platform.py:41` PlatformFL（`device_name` = device_info.device_type，`is_cuda_alike` :54）。
  无插件时 → UnspecifiedPlatform → 纯 MoE 报告 #3「Device string must not be empty」。
- 显存相关扩展点：
  - vendor attention backend：`/env/xvllm-plugin-FL/vllm_fl/dispatch/backends/vendor/kunlunxin/impl/attention.py:477`
    `KunlunxinAttentionBackend`（KV cache shape `(2, num_blocks, num_kv_heads, block_size, head_size)` 连续张量）、
    `:669` `KunlunxinAttentionBackendImpl`（xtorch_ops）；
  - 0.13 v1 的 attention backend 按模型类型分目录：`$V/v1/attention/backends/`（cpu_attn/flash_attn/flashinfer/
    flex_attention/gdn_attn/linear_attn/mamba1_attn/mamba2_attn）；
  - 显存池 = 厂商 torch.cuda caching allocator（xpytorch CUDA 兼容）+ vLLM 层管理（路线A方案 §2 既有结论；
    本调研补充：0.13 代码树内无 torch_fl/FLAGOS_USE_CACHING_ALLOCATOR 相关物）。

---

## 3. 与 0.20.2 差异对照表（能力 ｜ 0.20.2 ｜ 0.13.0 ｜ file:line ｜ 影响）

| 能力 | 0.20.2（笔记） | 0.13.0（本调研） | file:line | 影响 |
|---|---|---|---|---|
| 权重 offload（--cpu-offload-gb） | offloader 双后端：UVA（config/offload.py:19-42，支持 cpu_offload_params 按参数名段）+ Prefetch（:49-60，按层分组异步 H2D 预取）；BaseOffloader 抽象 + 工厂（model_executor/offloader/base.py:47-126） | **仅 UVA 单机制** `maybe_offload_to_cpu`（按层模块整体 + 全局字节预算），无 prefetch、无按参选择；非标准 block_size 路径禁用（PR #18298） | utils.py:516/:522/:606; gpu_model_runner.py:290/:5035 | 0.13 权重溢出只有 UVA 一种形态、粒度粗；异步预取模式（0.20.2 分层缓存核心设计参考）在 0.13 需自研 |
| KV offload | 笔记：**「KV 溢出无内置，归 KVConnector 管」→ 子项 B 需自研 KV 分层** | **原生内置**：--kv-offloading-size + --kv-offloading-backend native\|lmcache；OffloadingConnector（KVConnectorBase_V1）+ vllm/v1/kv_offload 子系统（LRU/ARC 驱逐、CPUBackend、swap_blocks 流式异步） | config/cache.py:151-160; config/vllm.py:477-515; offloading_connector.py:45; v1/kv_offload/cpu.py:23 | **0.13 直接提供 CPU 溢出底座**，V3 分层缓存不必从零自研；0.20.2 是否保留 kv_offload 待补查（venv311 容器不在运行） |
| evict_blocks 挂载点 | kv_cache_manager.py:460（prefix cache 淘汰） | kv_cache_manager.py:336（→ block_pool.py:400）；调用点 scheduler.py:1803（KV load 失败驱逐） | kv_cache_manager.py:336; scheduler.py:1803 | 语义一致（prefix cache 驱逐 hook），可挂自定义分层逻辑 |
| prefix caching | 有（cache_blocks/get_computed_blocks） | 默认开（cache.py:76），block hash 哈希表（block_pool.py:23-37） | cache.py:76; block_pool.py:23-37 | 一致 |
| 内存摸底 | 笔记未细述（0.20.2 同有 profile_run） | core.py:218 → uniproc_executor.py:178 → gpu_worker.py:298（requested=total×util :241；available=requested−non_kv :356）→ gpu_model_runner.py:4424 | gpu_worker.py:241/:298/:356 | 0.13 的 profile_run 即纯 MoE 报告崩溃路径 |
| KV 分配器 | KVCacheManager（:106） | 无 PagedAllocator；KVCacheManager（kv_cache_manager.py:94）+ attn_utils.py:69 _allocate_kv_cache + CuMemAllocator（device_allocator/cumem.py:113，非 KV 专用） | kv_cache_manager.py:94; attn_utils.py:69; cumem.py:113 | 分配器结构一致（vLLM 块表 + 整块 tensor） |
| 平台扩展 | —（笔记未涉及） | 无 kunlunxin.py；插件 PlatformFL 引导（vllm_fl/__init__.py:38）；vendor attention backend 为显存接口扩展点 | platforms/__init__.py:234-270; vllm_fl/.../attention.py:477 | P800 平台扩展点 = 插件 platform + vendor attention backend |

---

## 4. P800 A 线显存管理可用挂载点清单（结论性建议）

| 挂载点 | 0.13 位置 | 用途 | 优先级 |
|---|---|---|---|
| **原生 KV offload（首选）** | `--kv-offloading-backend native`（OffloadingConnector + kv_offload 子系统） | V3 分层缓存 CPU 溢出层：LRU/ARC 驱逐 + CPUBackend + swap_blocks 异步传输；与 0.20.2 笔记 flagcx_connector 模式同构（KVConnectorBase_V1），无需自研 | ⭐ 最高：先实测容量与吞吐代价 |
| evict_blocks | kv_cache_manager.py:336 / scheduler.py:1803 | prefix cache / KV load 失效驱逐 hook，可挂自定义分层（官方未挂钩的溢出场景） | 中（自研备选） |
| 权重溢出 | maybe_offload_to_cpu（utils.py:522）+ --cpu-offload-gb | 低优先级权重放 CPU（UVA），验证 is_pin_memory_available | 低 |
| KVConnector 扩展 | kv_connector/factory.py 注册表 + KVConnectorBase_V1 | 若官方 native 后端不够用，注册自定义 connector（如 flagcx 本地分层） | 中（备选） |

**结论性建议（V3 分层缓存原型落点）**：
1. **若原型目标平台为 P800（vllm 0.13）**：优先实测官方 `--kv-offloading-backend native`（先解决
   num_cpu_blocks 接线疑点：实测报错则经 `--kv-transfer-config` 显式提供；不满足再走 evict_blocks 自研）。
   符合「不预实现」原则——0.13 已内置官方 CPU 溢出路径，先拿官方数据，避免自研造轮子。
2. **若目标为昇腾（vllm 0.20.2 新线）**：沿用 0.20.2 笔记的 evict_blocks + flagcx_connector 模式；
   **0.20.2 是否同样内置 kv_offload 待补查**（笔记撰写时未覆盖，建议在昇腾侧 vllm 0.20.2 源码补一次
   `ls vllm/v1/kv_offload`，若存在则两线均可复用官方路径）。
3. **P800 待实测清单（2026-09-01 全部实测关闭 ✅）**：
   ① native KV offload 启用路径 —— `--kv-offloading-size` 写入的 num_cpu_blocks=0 会被
   CPUOffloadingSpec 直接 raise（cpu.py:24-29），kv_bytes_per_rank 全树无消费点（config/vllm.py:495-500）；
   **须构造 KVTransferConfig 显式传 kv_connector_extra_config.num_cpu_blocks**（探针
   probes/routeA_s4_kv_host_offload.py / routeA_s4_kv_offload_xfer.py，已跑通）；
   ② `is_pin_memory_available()` 在 xpytorch = **True**（CPU 张量 pinned）；
   ③ swap_blocks 与 kunlunxin vendor attention backend（KV shape (2,N,H,B,S)）**兼容**：
   gpu→cpu store 75 块、重复 prompt cpu→gpu load 命中（run2 仅算 1 新块）均实测通过，
   生成质量无退化，吞吐代价 ~2.4%（72.5 vs 74.3 tok/s，910 CPU 块 ≈ 2GiB）。

---

## 4b. 910C(vllm 0.20.2)移植结论(2026-09-03 实测)

§4 建议 2「昇腾 0.20.2 是否内置 kv_offload」已现场核实:**内置存在**(`vllm/v1/kv_offload/` 齐全,
`OffloadingConnector` 在 factory 注册),但 **native CPU 卸载在昇腾栈不可用**,硬阻塞两处:

1. **平台门**:`v1/kv_offload/cpu/spec.py:84` `get_handlers()` —— `if not current_platform.is_cuda_alike(): raise Exception("CPU Offloading is currently only supported on CUDA-alike GPUs")`。
   `PlatformFL`(device_name=npu)`is_cuda_alike()=False`;P800 的 `PlatformFL`(xpytorch,USE_CUDA=ON)为 True —— **这就是 P800 放行、昇腾拦下的根本差异**。
2. **传输算子**:`vllm._C` 是 CUDA 构建,昇腾镜像无 `libcudart.so.13` → `torch.ops._C_cache_ops.swap_blocks_batch` 不存在(`AttributeError`);`cpu_gpu.py:315` 的 handler 必失败。

**API 变更**(0.13 → 0.20.2):`kv_connector_extra_config` 的键从 `num_cpu_blocks`(块数)改为
`cpu_bytes_to_use`(字节),块数由 `CPUOffloadingSpec` 内部按 `cpu_bytes_to_use // kv_bytes_per_offloaded_block` 自算。
0.13 那个 `num_cpu_blocks=0` 接线缺陷在 0.20.2 已不存在(换了字段)。

详见《[routeA-S4-KV卸载Host-910C尝试-20260903](routeA-S4-KV卸载Host-910C尝试-20260903.md)》。下一步:昇腾需 vllm-plugin-FL 侧补 CPU-offload handlers(ACL memcpy + stream/event)并放行平台门,或先敲定昇腾锁 0.13(0.13 同有 is_cuda_alike 门,未必放行)还是 0.20.2。

---

## 5. 注意事项与未获取项

- ~~「0.20.2 是否含 v1/kv_offload」为待补查项~~ → **2026-09-03 已核实**:含,但 native CPU 卸载被 `is_cuda_alike()` 平台门 + `vllm._C` 缺 libcudart 双重阻塞(见 §4b)。
- kv_offload 子系统标注 experimental（spec.py:14-16），API 可能变动。
- 「kv_bytes_per_rank → num_cpu_blocks 换算点」未在 0.13 树内定位（grep kv_bytes/bytes_per_rank/num_cpu_blocks 全树），
  已如实记录为接线疑点，未编造换算逻辑。
- 本调研仅源码静态定位，未在 P800 上运行任何 offload 配置（GPU 为同事占用，且按 A 线纪律测试须在官方镜像内另行执行）。
