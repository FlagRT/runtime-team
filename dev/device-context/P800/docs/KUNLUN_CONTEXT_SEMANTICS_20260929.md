# 昆仑芯 P800 设备上下文语义判定与 C 项补足（2026-09-29）

> 起因：工作包 C（设备上下文生命周期）在 P800 上**未声明**，原因一行记为
> 「XPytorch 兼容层未暴露上下文原语，与 `recovery_real` 同因」。
> 本次据厂商资料 + 机器实测重新判定 —— **原表述不准确，已更正**（见 §5）。
> 结论：**C 项能补，但语义只能是「只读观测」，不是「生命周期管理」**，且已落地并真机验证。

---

## 1 结论摘要

| 问题 | 答案 |
|---|---|
| P800 有没有上下文的 API？ | **有，而且是完整的 CUDA Driver 风格 `cuCtx*`**（21 个，见 §3.1） |
| 这些 API 是真实现还是壳？ | **真实现**（创建返回真句柄、`push/pop` 语义成立、销毁生效，见 §4.1） |
| 那能不能做「多上下文生命周期」？ | ❌ **不能** —— **primary context 一经激活即占位**，此时显式创建被拒（`cuCtxCreate_v2` → `rc=2`，见 §4.1/§4.5） |
| 能不能由本层建一个上下文给 torch 用？ | ❌ **不能** —— 抢先建会让 torch 起不来（`invalid device ordinal`） |
| 那 C 项能补什么？ | ✅ **只读观测**：`context_query()` —— 读得到"此刻在哪个上下文上、归谁管" |
| 观测有副作用吗？ | **无** —— 真机实测：查询前后同一次计算均为 `512.0` |

**落地**：新增能力键 `context_query`（与 `context_lifecycle` **分开声明**），P800 真机通过
（`managed_by="external"` / `readonly_safe=true`）；不声明 `context_lifecycle`（如实）。

---

## 2 资料来源

用户提供、位于测试机 `/data1/document/` 的两份厂商文档（**受控资料，未入库**）：

| 文件 | 规格 | 有用程度 |
|---|---|---|
| `昆仑芯P800(HCCPK1)测试指导文档.docx` | 3.68 MB · 4851 行纯文本 | 部署/测试指导（OS 镜像、XRE 驱动安装、xpu-smi、XCCL 环境、各模型部署命令）。**无设备上下文 API** |
| `昆仑芯XTDK用户手册V3.5_20260415_昆仑芯.pdf` | 54 页 · 72069 字符 · 文档时间 2025-11-13 | **XPU C/C++ 编程手册**（编程模型、语法、调试、优化）。**`context`/「上下文」零命中** |

### 2.1 手册里查到的、与本判定相关的三条硬事实

1. **Host 侧 runtime 来自 XRE，不在本文档范围内**（原文）：
   > 运行时头文件 `xpu/runtime.h` 是昆仑芯 **XRE** 的一部分 … 本小节中提到的 Runtime 接口相关的
   > 更多用法细节请参考《**昆仑芯 P 系列 XRE 用户手册**》

   ⇒ 这份 XTDK 手册讲的是 **kernel 编程**；device/context 属 host runtime，**本次资料不含 XRE 用户手册**。
2. **「当前工作设备」是线程级状态**（6.2.2.2.1 多线程设备管理，原文）：
   > 在昆仑芯编程模型中，**当前工作设备是线程级的状态** … 新创建的线程默认没有指定工作设备，
   > 若开发者未通过 `xpu_set_device` 接口函数明确指定，该线程将默认使用设备 device0

3. **「默认执行流」是进程级状态**（6.2.2.1.1，原文）：
   > 默认执行流是**进程级的状态**，即在同一进程中，每个设备**仅有一个**默认执行流

   （同节还给出资源上限：**最大 stream 个数 2^8，最大 event 个数 2^64**。）

> ⚠️ **不能据此断定"没有上下文"** —— 这两条只说明**设备绑定与执行流的归属层级**，
> 不等于没有 host 侧上下文抽象。真正的判定必须落到**机器上的头文件与库**（§3）。

---

## 3 机器实测：原语在哪一层

### 3.1 XRE 原生层**没有**上下文 API

`xcudart/include/xpu/runtime.h`（XRE 官方头文件，490 行）的全部 `xpu_*` 入口：

```
xpu_create_cl_func xpu_create_sd_func xpu_current_device xpu_device_count xpu_device_get_attr
xpu_device_list xpu_dim3 xpu_event_create xpu_event_destroy xpu_event_query xpu_event_record
xpu_event_wait xpu_free xpu_get_ccix_peer_id xpu_get_driver_commit xpu_get_driver_version
xpu_get_runtime_commit xpu_get_runtime_version xpu_host_alloc xpu_host_free
xpu_host_pointer_get_attrs xpu_host_register xpu_host_unregister xpu_ipc_close_memhandle
xpu_ipc_get_memhandle xpu_ipc_open_memhandle xpu_kernel_debug xpu_kernel_debug_reset
xpu_last_kernel_exec_time xpu_launch_argument_set xpu_launch_async xpu_launch_config
xpu_launch_config_v2 xpu_malloc xpu_memcpy xpu_memcpy_async xpu_memcpy_peer xpu_mmap
xpu_pointer_get_attrs xpu_profiler_start xpu_profiler_stop xpu_set_device xpu_stream_create
xpu_stream_destroy xpu_stream_wait_event xpu_wait
```

**`grep -i "context\|ctx"` ⇒ 0 命中。** 库侧一致：`libxpurt.so` 的导出符号中 context **0 命中**。

### 3.2 ⭐ 上下文 API 在 **CUDA Driver 兼容层**里，且**就是 XPytorch 在用的那个库**

```
xcudart/lib/libcuda.so      -> libxpucuda.so
xcudart/lib/libcuda.so.1    -> libxpucuda.so
xcudart/lib/libxpucuda.so.1 -> libxpucuda.so.515.58.kunlun     (1.92 MB)
```

`libxpucuda.so.515.58.kunlun`：**415 个 `cu*` 导出符号**（`cuInit`/`cuDeviceGet`/…），
**0 个 `cuda*` runtime 符号** ⇒ 纯 CUDA **Driver** API 兼容层。其中 Ctx 类 **21 个**（去 `_v2` 别名）：

```
cuCtxAttach cuCtxCreate cuCtxDestroy cuCtxDetach cuCtxDisablePeerAccess cuCtxEnablePeerAccess
cuCtxGetApiVersion cuCtxGetCacheConfig cuCtxGetCurrent cuCtxGetDevice cuCtxGetFlags cuCtxGetLimit
cuCtxGetSharedMemConfig cuCtxGetStreamPriorityRange cuCtxPopCurrent cuCtxPushCurrent
cuCtxSetCacheConfig cuCtxSetCurrent cuCtxSetLimit cuCtxSetSharedMemConfig cuCtxSynchronize
```

**关键**：`/proc/self/maps` 证实 XPytorch 进程实际加载的就是这两个库：

```
.../xcudart/lib/libxpurt.so.12.9.1.kunlun
.../xcudart/lib/libcudart.so.12.9.1.kunlun
.../xcudart/lib/libxpucuda.so.515.58.kunlun      ← libcuda.so.1 的真身
.../torch_xmlir/libXMLIRRuntime.so  .../torch_xmlir/_XMLIRC.cpython-310-...so
```

⇒ `cuCtx*` **不是"另一套世界"**，它与 XPytorch 共用同一份驱动状态。

> ⚠️ **踩过的坑（务必避免）**：`triton/backends/xpu/xpu3/so/` 下**也有一份同名** `libxpucuda.so`。
> 先从那处 `dlopen` 会触发 `Libraries loaded from different directories!` 版本错配，
> 随后 torch 直接报 `CUDA_ERROR_NOT_INITIALIZED` —— **污染源是探测方式，不是栈的问题**。
> **纪律：同栈的库一律走 soname（`libcuda.so.1`），不要拿任意副本路径。**

---

## 4 四组判别实验（真机，逐场景独立子进程）

> 卡：`CUDA_VISIBLE_DEVICES=4`（**卡 1 为已知故障卡，禁用**）。
> 全部为**只读或进程内可回收**操作；未触碰任何他人容器/进程。

### 4.1 实验 A：干净进程（不 import torch）—— 能否建、能否建第二个

| 步骤 | 结果 |
|---|---|
| `cuInit(0)` | `rc=0`（CUDA_SUCCESS） |
| `cuCtxGetCurrent`（未触碰设备） | **`0x0`** ⇒ cuInit **不隐式建**上下文 |
| `cuDeviceGetCount` | 1 |
| **`cuCtxCreate_v2` #1** | **`rc=0`，句柄 `0x58e7218dd930`**；创建后 current 即该句柄 |
| **`cuCtxCreate_v2` #2** | ❌ **`rc=2`**（失败）⇒ 该设备上**已有生效上下文**（§4.5 进一步定位为 primary context 占位） |
| `cuCtxDestroy_v2` | `rc=0`；销毁后 current 回 `0x0` |
| **再次 destroy 同一句柄** | **`rc=201`**（`CUDA_ERROR_INVALID_CONTEXT`）⇒ **厂商如实报错** |
| `cuCtxSetCurrent`（已销毁句柄） | ⚠️ **`rc=0`（静默成功）** |

> 最后一条是一处**厂商静默**：与 910C「上下文销毁后用其流当场静默」同族。
> 本轮**未在本层提供 `context_set` 口**（P800 不声明 `context_lifecycle`），故不构成缺口；
> 但**登记在案** —— 将来若本层要暴露切换语义，必须自己查句柄有效性。

### 4.1b ⭐ 进一步定位：不是"只允许一个"，而是 **primary context 占位**（第五轮补测）

符号扫描发现本栈有**完整的 primary context 专用入口**：
`cuDevicePrimaryCtxGetState` / `Retain` / `Release` / `Reset` / `SetFlags`
（与 910C 的 `acl.rt.get_primary_ctx_state` 对应）⇒ 可取观测状态，把 `rc=2` 的成因钉死：

| 阶段 | `cuDevicePrimaryCtxGetState(0)` |
|---|---|
| **触碰设备之前** | `rc=3`（`CUDA_ERROR_NOT_INITIALIZED`），输出参数**未被写入**（保持哨兵 `0xDEAD`/`-1`） |
| **触碰设备之后** | `rc=0`，**`active=1`（primary context 已激活）**，`flags=0` |
| 此时 `cuCtxCreate_v2` | **`rc=2`** |

⇒ **准确表述是**：**该设备上 primary context 已激活 ⇒ 显式创建被拒**，
而非笼统的"平台只允许一个上下文"；且"是否占位"**有可观测判据**（`active` 字段）。

**续测（多卡）进一步精化**：

| 观察（`CUDA_VISIBLE_DEVICES=4,6`） | 结果 |
|---|---|
| `cuda:0` 与 `cuda:1` 的 current context | **互不相同**（`0x…80df0` vs `0x…87fa0`）⇒ **每设备各一个** |
| 切回 `cuda:0` 后 current | **与之前完全相同**（`dev0_ctx_stable=true`）⇒ 切换稳定 |
| `cuDevicePrimaryCtxGetState(0)` / `(1)` | dev0 **`active=1`**；dev1 **`active=0`**（尚未触碰）⇒ **按需激活** |
| `cuCtxCreate_v2(…, dev=1)` | **`rc=2`** ⇒ 任一设备上 primary 已激活即拒 |

⇒ **最终表述**：**每个设备各有自己的 primary context；某设备上 primary 已激活时，该设备上的显式创建被拒（`rc=2`）**。
**不是「全局只允许一个上下文」** —— 那是把**设备级**约束误读成**进程级**
（多设备**并行**使用是允许的，只是每设备一个 primary）。

⭐ **与 910C 同构**：`acl.rt.get_primary_ctx_state(0)` 实测返回 **`(1, 0, 0)`** ——
`state=1` 即 *primary 已存在*；设备 1 返回 `(0, 0, 0)`、越界设备返回 `(0, 0, 107001)`。
**两家栈在同一概念上给出同一语义**（一个是 `active`，一个是 `state`），这使"上层不必写分支"
的主张在**机制层**也站得住。

### 4.2 实验 B：先建上下文，再让 torch 上来 —— 能不能"接管"

| 步骤 | 结果 |
|---|---|
| 建上下文（torch 之前） | `rc=0`，`current` = 我们的句柄 |
| `import torch` + `.cuda()` + 矩阵乘 | ❌ **`AcceleratorError: CUDA error: invalid device ordinal`** |
| `current`（torch 起失败后） | 仍是我们的句柄 ⇒ **状态共享**（不是两套世界） |
| 销毁我们的上下文 | `rc=0` |
| **再跑 torch 计算** | ✅ **`512.0`（恢复正常）** |

⇒ **结论：本层绝不能"抢先在 torch 之前建上下文"** —— 那会让框架起不来；
而**销毁它反而让框架恢复**，说明框架需要**自己**去建（primary context 语义）。

### 4.3 实验 C：正常顺序（torch 先）—— 还有没有建第二个的余地

| 步骤 | 结果 |
|---|---|
| torch 计算 | `512.0` |
| `cuCtxGetCurrent` | **`0x5d0527b29c40`（非 0）** ⇒ **XPytorch 自己在用上下文** |
| **torch 活跃时 `cuCtxCreate_v2`** | ❌ **`rc=2`** |

### 4.4 实验 D：对 **torch 自己的**上下文做只读 / 销毁

| 操作 | 结果 |
|---|---|
| `cuCtxGetCurrent` | `0x590c1044e820` `rc=0` |
| `cuCtxGetDevice` | `rc=0`，`dev=0` |
| `cuCtxGetFlags` | `rc=0`，**`flags=8`** |
| `cuCtxSynchronize` | `rc=0` |
| **只读之后 torch 计算** | ✅ **`512.0`（只读安全）** |
| **`cuCtxDestroy_v2`（torch 的上下文）** | **`rc=201`（`INVALID_CONTEXT`）⇒ 厂商主动拒绝** |
| 再销毁一次 | `rc=201` |

⇒ **危险操作在厂商侧已兜住**；本层无需（也无法）代劳。

---

## 5 ⭐ 认知修正：原表述错了

工作包 C 首轮登记的理由是：

> ~~「XPytorch 兼容层**未暴露**上下文原语，与 `recovery_real` 同因」~~

**实测表明这不准确**：驱动层原语**齐全**，且**正是 XPytorch 在用的那份**，XPytorch **自己在用**上下文。

**准确的原因**是三条平台约束：

1. 平台**只允许一个**上下文（§4.1）；
2. 该上下文由 **XPytorch/XRE 自建**（§4.3）；
3. **本层抢先去建会破坏框架**（§4.2）⇒ 本层**不应也不能**做生命周期管理。

> 教训（已入台账第 17 条）：**「能力缺失」的根因表述本身可能是错的。**
> 写"该栈未暴露 X"之前，必须实测到**原语层**（头文件 + 库符号 + 真调一次），
> 否则会把"平台约束"误记成"厂商没做"，进而误导后续接入者与对外沟通。

---

## 6 本层落地：`context_query`（只读观测）

### 6.1 为什么不硬塞进 `context_lifecycle`

`context_lifecycle` 的语义是**创建/切换/销毁/计数**。P800 上这四件事：
创建（平台不允许第二个）、切换（无第二者可切）、销毁（不该碰框架的）、计数（恒为 0）
⇒ **全都不成立**。把 P800 也标成"支持 `context_lifecycle`"就是把不成立的事实说成成立。

### 6.2 新增 `context_query`：把差异写进字段，而不是留给上层分支

| 键 | 语义 |
|---|---|
| `queryable` | 本后端是否具备该查询能力 |
| `present` | 此刻是否存在生效上下文 |
| `ordinal` | 上下文绑定的设备序号（取不到 ⇒ `None`） |
| `flags` | 上下文的创建标志（取不到 ⇒ `None`） |
| `managed_by` | **`"unified"`（本层创建）/ `"external"`（厂商或框架自建）/ `None`** |
| `reason` | 不可查询 / 不存在 / 取不到字段的**具体原因**（不吞掉） |

未声明该能力的后端**也返回同一 6 键**（`queryable=False` + 具体原因）——
这样上层"换芯片不改行为"：拿到的一定是**同一形态**，"没有上下文"与"查不了"**不会混淆**。

### 6.3 实现要点

- **只用 soname `libcuda.so.1`** 惰性加载（复用 XPytorch 已加载的同一份，不引入副本）；
- **绝不调用 `cuCtxCreate_v2` / `cuCtxDestroy_v2`**（§4.2 的硬证据）；
- 只调 `cuCtxGetCurrent` / `cuCtxGetDevice` / `cuCtxGetFlags`（§4.4 证明只读安全）；
- **不把厂商指针放进任何公开返回**（`ctx` 只在本层内部用于 `managed_by` 比对）。

### 6.4 真机证据（C4 组）

```json
{"queryable": true, "present": true, "ordinal": 0, "flags": 8,
 "managed_by": "external", "reason": ""}
```
- `compute_before = compute_after = 512.0` ⇒ **`readonly_safe = true`（查询无副作用）**
- verdict：**`C4_context_query_readonly = true`**，`PASS = true`

`managed_by = "external"` 是**如实值** —— 本层一个上下文都没造，就不许自称 `unified`；
判据同时守住取值域（`∈ {unified, external}`）。

---

## 7 判据与验证

- 离线自检新增 **6 条**判据（4 条通用 + 2 条「有上下文」分支）；
  判据数 **ascend 64→68 · kunlun 65→71 · cambricon 55→59**。
- ⭐ **非空转验证暴露过一个真实缺口**：离线无设备 ⇒ 只走 `if not raw` 分支，
  于是"丢掉 `managed_by` 键"、"声明了却给 `queryable=False`"两处注入**都不会被抓到**
  ⇒ 补上**可控桩**（构造"有上下文"情形）后，两处均能 FAIL。
  这正是项目纪律「**能造可控假原语就别 SKIP**」的又一实例。
- 三处注入（`present=0` / 缺 `managed_by` / `queryable` 反向）**全部被抓**，还原后 71/0。

## 8 回归（r4 · 共享层改动后两台同口径）

| 判定项 | 910C（r4） | P800（r4） |
|---|---|---|
| 离线契约自检 | **68 / 0 / 1 跳过** | **71 / 0 / 1 跳过** |
| 跨后端对称性 `--all` | 5 / 0 | 5 / 0 |
| 组件冒烟 | **52 / 0** | **46 / 0** |
| conformance 13 + 推理 6 | 13/13 + 6/6 | 13/13 + 6/6 |
| 职责响应审计（39 sub-part） | **39 / 0 / 0** | **36 / 0 / 3** |
| 错误注入→恢复闭环 | 5 / 0 / 0 | 5 / 0 / 0 |
| 工作包 A 功能等价性 | 6 / 6 | 6 / 6 |
| 多流语义 / 配额 | 8/8 · 3/3 | 8/8 · 3/3 |
| 结论 | ✅ 无回归 | ✅ 无回归 |

证据：`../910C/probes/*_ascend_20260929_r4.*` · `probes/*_kunlun_20260929_r4.*` ·
`probes/probe_bc_contract_kunlun_20260929_r2.json`。

## 9 边界项：**逐条做实**（第五轮，2026-09-29 同日补）

初版把下列各条写成"边界（如实，不外推）"。复查后判定：**其中多数当场就能查**
（头文件就在机器上），不该长期挂账。逐条结果：

| 原边界项 | 处置 | 结果 |
|---|---|---|
| `flags=8` 含义未查证 | ✅ **已查证** | 本栈 `cuda.h` 的 `CUctx_flags` 枚举：**`CU_CTX_MAP_HOST = 0x08`**（*Support mapped pinned allocations*）。**运行时复核一致**（`cuCtxGetFlags`→`8` → 解码为 `CU_CTX_MAP_HOST`）。另记本栈**特有** `CU_CTX_NAKED_PRIMARY = 0x20` |
| `cuCtxGet/SetLimit` 未做 | ✅ **已做（读写）** | `CU_LIMIT_STACK_SIZE`=0 · `CU_LIMIT_PRINTF_FIFO_SIZE`=0 · `CU_LIMIT_MALLOC_HEAP_SIZE`=**8388608**（均 `rc=0`）；`CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH` 与 `CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT` **`rc=215`（`CUDA_ERROR_UNSUPPORTED_LIMIT`）**，且**输出参数保持哨兵值未被写入** ⇒ **不支持就如实报、不静默填假值**。只读后 torch 计算仍 `512.0` |
| `xpu_ipc_*` 共享上下文未做 | ✅ **已核符号 + 已真调（跨进程）** | **符号**：context 层 `cuIpcGet/Open/CloseMemHandle`、`cuIpcGet/OpenEventHandle` 五个俱全；XRE 层 `xpu_ipc_get/open/close_memhandle`。**真调（两进程）**：A 侧 `cuMemAlloc_v2` rc=0、`cuMemcpyHtoD_v2` rc=0、**`cuIpcGetMemHandle` rc=0**（64 字节非零 handle）；**B 侧 `cuIpcOpenMemHandle` = `rc=1`（`CUDA_ERROR_INVALID_VALUE`）**。**三种竞争解释逐一排除**：① handle 传递（长度/前 8 字节一致、非全零）；② 上下文未绑定（B 侧 `cuDevicePrimaryCtxRetain` / `cuCtxSetCurrent` 均 rc=0 且 current 非 0）；③ flags 取值（`1` 与 `0` **都**返回 rc=1）。⇒ **该栈 IPC：导出可用、导入未通**。**注意**：它们共享的是**内存/事件句柄，不是上下文** ⇒ 契约无此项 ⇒ **不构成本层缺口**（如实登记厂商能力现状） |
| 多卡下的上下文行为 | ✅ **已做（同日续）** | 转机：dev4 当次**完全空闲**（0 MiB / util 0%）、dev6 util 0% ⇒ 用 `CUDA_VISIBLE_DEVICES=4,6` 完成（**只做 8 字节级分配、不做大分配**）。结果见 §4.1b：**每设备一个 primary context、切换稳定、任一设备上 primary 激活即拒显式创建**。⚠️ 其余卡（dev2/3/5 有 44%/100%/89% **活跃计算**）**未触碰** —— 现场见 §9 末 |
| 单上下文约束未与文档口径交叉核对 | ⏸ **仍不可做** | 《XRE 用户手册》**不在**本次资料内（用户提供的两份中，XTDK 手册明确指向它）。**已在 §2.1 如实登记**，不臆造 |
| 厂商静默：`cuCtxSetCurrent(已销毁句柄)` 返回 0 | ✅ **已登记为判据依据** | 写进接入手册 skill：将来若本层暴露"切换"语义，**必须自己查句柄有效性**（厂商不拦）。本轮 P800 不暴露 `context_set`，故不构成缺口 |

**新增纪律（用卡）**：上一轮记录的"用卡 4"**当次已失效**（dev4 由 0 MiB 变为 16.8 GB 他人占用）
⇒ **用卡前必须当次重探占用并记录**，不得沿用历史记录。

**收尾时的现场（如实）**：普查显示 P800 有 **~10 个容器 Up、全部挂 8 张卡**，其中
`VLLM::EngineCore`（已跑 5 天）、`qwen3-vl-reranker` / `qwen3-vl-embedding-2b` 评测（99% CPU）、
多人 vLLM 服务等**活跃作业**；dev2/3/5 的 util 为 44%/100%/89%。
⇒ **未停任何容器**（`docker stop` 会中断这些作业），改用**当次空闲**的 dev4(+dev6) 完成全部探测。
对照：**910C 全机 8 个 NPU 均 `No running processes found`**，且我方容器只挂 3 张卡、
挂载集不相交 ⇒ 按 09-29 判明的名额规则**本就独占这 3 张**，**同样无需停任何容器**。
⇒ **新纪律：先看「是否需要停」，而不是先停再用** —— 空闲容器不占名额时，停用纯属多余的风险。

---

## 11 910C 侧的多卡与语义细节（同日续，与 P800 对照）

| 项 | 910C（pyACL） | P800 |
|---|---|---|
| 多设备上下文 | **三设备各自独立**（`0x…7e000` / `0x…7e230` / `0x…963f0`，互不相同）；销毁 dev0 后 dev1/dev2 仍非零 ⇒ **无交叉影响** | 每设备一个 primary，互不相同，切换稳定 |
| `get_context(dev)` 的 **`dev` 参数** | ⚠️ **被忽略**（同一当前设备下 `get(0)/get(1)/get(2)` 返回**同一值**）⇒ **必须"先 `set_device` 再查"** | 无此参数（`cuCtxGetDevice` 读回） |
| `managed_by` 能否判出 `unified` | ✅ **能**（单设备紧邻场景：建后 `unified`、销毁后 `present=False`） | 不适用（不声明 `context_lifecycle`） |
| IPC 入口 | `ipc_mem_get_export_key` / `ipc_mem_import_by_key` / `ipc_mem_set_import_pid` / `mem_export_to_shareable_handle` 等 **11 个存在**（`acl.rt`） | 导出 rc=0 / 导入 rc=1（见 §9） |

⚠️ **如实登记的局限（本层）**：`managed_by` 的判定依赖"厂商上下文对象比对"。
实测在**多设备反复切换**的时序下，`create_context` 的返回值与 `get_context` 的返回值**可能不一致**
（单设备紧邻调用时一致）⇒ 该情形下 `managed_by` 可能误报 `external`。
**契约把 `managed_by` 定义为「尽力而为的归属提示」**（不是判据）；多设备大规模混用前需补测。

---

## 10 910C 对齐（第五轮同日完成）

`context_query` 是**跨后端同一形态**的接口，910C 早已具备原语（只是本轮之前未接线）。
实测素材与落地：

| 项 | 910C（pyACL） | P800（CUDA driver 兼容层） |
|---|---|---|
| 只读入口 | `acl.rt.get_context(dev)` → **`(ctx, rc)` 元组**，3 次调用稳定一致 | `cuCtxGetCurrent` → 真句柄 |
| primary 状态 | `acl.rt.get_primary_ctx_state(dev)` → **三元组 `(1, 0, 0)`** | `cuDevicePrimaryCtxGetState` → `active=1` |
| 销毁后查询 | **`(0, 107002)`**（`ACL_ERROR_RT_CONTEXT_NULL`）**如实报错** | 未测（本层不销毁） |
| `flags` | **无此概念 ⇒ 如实 `None`** | `8` → `CU_CTX_MAP_HOST` |
| 真机 C4 结果 | `present=True` · `ordinal=0` · `flags=None` · **`managed_by="external"`** · `readonly_safe=true` | `present=True` · `ordinal=0` · `flags=8` · **`managed_by="external"`** · `readonly_safe=true` |

⇒ **两家在同一 6 键形态下给出同一 `managed_by`**（都是 `external` —— 上下文由框架自建，本层未造），
这正是本层"**让分歧显式且可消费**"的落点：**差异在字段里（`flags=None` vs `8`），不在接口形状里**。

**同时修掉一处工具缺陷**：探针 C4 原先按 `device_type == "cuda"` 才拼设备串
⇒ 在 ascend（`device_type="npu"`）上会退化成 **CPU 计算**，根本触碰不到设备，
`present` 永远为 `False`（**假阴性**）。已改为**按后端 `device_type` 拼**（`npu`/`cuda`/`mlu`），
真机取证显示 `at_device = npu:0`。

**还抓到一处 stub 不完整**：「声明即承诺」判据当场报
`AttributeError: 'types.SimpleNamespace' object has no attribute 'get_context'`
⇒ **离线假 pyACL 缺了三个只读入口**。**已补齐 stub（而非在实现里容错掩盖）**——
否则"stub 的不完整"会被读成"实现的缺陷"。
