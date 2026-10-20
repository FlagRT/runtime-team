# PPU（平头哥 真武 ZW810E）后端接入报告（2026-10-10 · 第 4 家实例）

> 依据《新芯片接入手册》（`prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`）执行；
> 原型版本 **v0.3.0**。**铁律遵守情况**：只迁**规范 / 方法 / 框架代码**，
> 其余实例的**实现与结论一律未迁**（能力声明逐项来自本机实测）。

## 0. 一句话结论

**接入的核心三步已完成并全绿**：backend 落地 → **离线自检 112 通过 / 0 失败 / 1 跳过（stub 边界）** →
**真机 smoke 46/0 · conformance 13/13 · 推理 6/6（`CONFORMANCE_PASS`）**。
手册「验收清单」13 项中**已完成 7 项**；**同日续报**：模型资产已就位，
**训练腿判据 6/6 · 推理腿判据 13/13 亦已取得**（退出期段错误待收口，见 §6）。

## 1. 厂商栈判别 = **路径 B（复用 `torch.cuda`）** —— 实测判定

| 判据 | 读数 |
|---|---|
| `torch.cuda.is_available()` / `device_count()` | **True / 16** |
| `get_device_name(0)` | **`PPU-ZW810E`** |
| 设备上真跑 matmul | ✅ 成功 |
| **`torch.xpu.device_count()`** | **0** ⇒ **不是** XPU 路线（`torch.xpu` 只是 stock torch 带的 Intel XPU） |
| 元数据一致 | `torch.version.cuda == "13.0"`、`CUDA_SDK_VER=cuda-13.0`、`CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK` |

⇒ `name = "ppu"`（厂商标识）· `device_type = "cuda"`（设备串前缀）。
**与 `kunlun` 同名不同厂** ⇒ 厂商标识只能靠设备名/驱动特征（见后端 `info()["vendor_discriminator"]`）。

## 2. 环境与栈（实测）

| 项 | 读数 |
|---|---|
| 宿主 | `ai-server` · Alibaba Cloud Linux 3 · 192 线程 / 2015 GB |
| 加速卡 | **16 × `PPU-ZW810E` · 96 GB/卡（98304 MiB）· 400 W**；节点 `/dev/alixpu_ppu0..15` |
| 驱动 / SDK | KMD **`1.5.1-1d747a`** · **SDK `2.1.0-a5f865`** · HGGC **13.0** · PCCL **2.1.0** |
| 容器基座 | `flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-base`（**本机 4 个「工作负载」容器在用，与之对齐**） |
| 容器内栈 | py 3.12.3 · torch **2.10.0** · **`flagtree 0.6.1+ppu3.6`** · triton 3.6.0 · FlagGems 5.4.0rc2 · vLLM 0.19.0 |

## 3. ⭐ 能力声明 —— 逐项实测（本报告最重要的一张表）

| 能力键 | 声明 | 实测依据（证据文件见 §5） |
|---|---|---|
| `device` / `multidevice` | ✅ | 16 卡可见、设备名、张量运算成功 |
| `memory` / `memory_alloc_stat` | ✅ | `mem_get_info(0)=(98022,98304)` MiB；`memory_allocated` 可用 |
| `memory_alloc` | ✅ | `caching_allocator_alloc/delete` 真调用：512 → **+4 MiB** → 512 |
| `record_stream` | ✅ | `Tensor.record_stream` 真调用 + sync 无异常 |
| `stream` / `event` | ✅ | 均可用；⚠️ **E3 缺口**（见 §4）由 `PpuEventAdapter` 修正 |
| `bounded_sync` | ✅ | 主机侧轮询实现，真有界（`wait_host(200ms)` 实测 201 ms 返回 False） |
| `recovery_probe` | ✅ | 设备上真算 2×2 零张量 |
| `device_state` / `device_state_control` | ✅ | 共享状态机（芯片无关） |
| **`context_lifecycle`** | ✅ | 建 A/B 两个上下文 **rc=0**、`cuCtxSetCurrent` rc=0、`cuCtxDestroy_v2` rc=0，**且 torch 全程可用**（P800 相反：平台只允许一个上下文） |
| `context_query` | ✅ | `cuCtxGetCurrent/GetDevice/GetFlags` rc=0；调完 torch 仍可算 |
| **`stream_priority` / `_control` / `_readback`** | ✅✅✅ | 区间 **`(0,-3)` = 4 档非单点**；C API 建流（flag=1）**请求 == 回读**（-3/-1/0 逐值相等）；`ExternalStream` 包装后真跑 matmul=512；`destroy_rc=0` |
| `graph_capture` | ✅ | `torch.cuda.graph` 捕获 + replay + 数值对照 **PASS** |
| `error_map` | ❌ **不声明** | **已确认不具备**：错误只给错误**名**（`hggcErrorInvalidValue`）**无数值码** |
| `recovery_real` | ❌ **不声明** | **已确认不可用**：`cuDevicePrimaryCtxReset_v2` rc=0 但**之后本进程设备操作全部 `hggcErrorInvalidValue`** |

## 4. 三条实测发现（都会影响下游用卡/用流的方式）

1. ⭐ **`NCCL_SOCKET_IFNAME` 写死 `eth0`，而本机没有 `eth0`（真网卡 `bond0`）** ⇒
   不覆盖则 `init_process_group('nccl')` **能建组**、**真实 `all_reduce` 失败**
   （`Bootstrap : no socket interface found`）。**同机他人容器也是 `eth0`**。
   ⇒ 我们启动容器时显式 `-e NCCL_SOCKET_IFNAME=bond0`；**2 卡 all_reduce 已通过**。
2. **集合通信后端名 = `nccl`**（不是 `pccl`）—— `pccl`/`hccl`/`ucc` 均 `Unknown backend type`，
   `xccl` 未编译；而**底层库确实叫 PCCL**（`libtorch_cuda.so` 内含 `pccl` 字样）
   ⇒ **库名 ≠ 后端名**，拼字符串勿混。
3. **未 record 的 `Event.query()` 原生返回 `True`**（误报"已完成"，E3 缺口）
   ⇒ 必须由适配层用 `recorded` 标志修正（本后端已做，conformance `e3_query_unrecorded` 实测 PASS）。

## 5. 验证明细（全部为本次运行产出）

| 项目 | 结果 | 证据 |
|---|---|---|
| **离线契约自检**（无设备） | **112 通过 / 0 失败 / 1 跳过**（跳过=stub 能力边界） | `PPU/probes/onboard_20261010_out/offline_check_ppu_20261010.log` |
| **跨后端对称性自检**（4 家） | **7 通过 / 0 失败**；四家逐个：ascend 90/0/1 · kunlun 108/0/1 · cambricon 97/0/0 · **ppu 112/0/1** | `…/offline_check_all_symmetry_20261010.log` |
| **真机 smoke** | **46 通过 / 0 失败** | `…/smoke_ppu_20261010.log` |
| **conformance 13 例** | **13/13 · `CONFORMANCE_PASS`** | `…/conformance_13_ppu_20261010.log` |
| **推理 6 例** | **6/6 · `CONFORMANCE_PASS`** | `…/conformance_infer6_ppu_20261010.log` |
| 能力探测（4 轮） | 全项取到（0 项 UNAVAILABLE） | `PPU/probes/probe_ppu_*.log` |

### 5.1 本轮从既有工具/代码里挖出的 3 处缺陷（4 家一起跑才暴露）

1. **`conformance/cases.py` 的 `t3_topology_path` 硬编码 `torch_npu` / `npu-smi`** ——
   该用例是**芯片无关**的，第 4 家不同厂商跑同一套用例时文案失真
   （违反手册 §4.4.4「不得硬编码厂商专有文案」）。**已修**为「统一面未暴露拓扑查询」。
2. **离线自检的假驱动库缺 CUDA 上下文系列符号** ⇒ 声明了 `context_lifecycle` 的后端在离线
   拿不到原语、被判「声明了却没实现」（**误报**）。**已补** `cuCtxCreate_v2/cuCtxSetCurrent/
   cuCtxDestroy_v2/cuCtxGetCurrent/cuCtxGetDevice/cuCtxGetFlags`（纪律：能造可控假原语就别 SKIP）。
3. **后端驱动句柄属性名未对齐自检工具约定**：工具按 `_ctx_driver` 注入可控假驱动；
   本后端初版命名为 `_driver` ⇒ 工具认不出、优先级与上下文判据**退化为 SKIP**。已改名对齐。

## 6. 未完成项（按时序，非缺口掩盖）

> **2026-10-10 续报**：模型资产已就位（`Qwen/Qwen3-Embedding-0.6B`，sha256 与 P800 逐字一致），
> **训练腿判据 `TRAIN_LEG_PASS 6/6`、推理腿判据 `INFER_LEG_PASS 13/13` 均已取得**
> ⇒ 下表第 2 项**已完成**，第 3 项**只剩服务化外壳**。
> ⚠️ 两条腿都在判据通过后于**解释器退出阶段段错误**（归属未定，已排除 16 组对照）。
> 详见 **`PPU_MODEL_AND_TWO_LEGS_20261010.md`** 与证据 `../probes/model_and_legs_20261010_out/`。

| # | 项 | 阻塞 / 前置 | 状态（2026-10-10 续） |
|---|---|---|---|
| 1 | 多流 Stream 16 项基线 | 无需新前置，可直接跑 | ⚪ 未跑 |
| 2 | **训练腿 6/6（2 卡）** | 需**模型资产** | ✅ **已取得**（6/6；退出期段错误见上） |
| 3 | **推理腿服务化** | 需模型 + `serve_standard.sh`；本栈 `current_platform=NvmlCudaPlatform`（**非** `UnspecifiedPlatform`）⇒ 社区 vLLM 路径**可能**可用，待实测 | 🟡 **模型侧已通过（13/13）**；仅缺 `ppu` 分支 + vLLM 起服实测 |
| 4 | 错误闭环四类注入 | 需先确认注入面（本栈无数值错误码） | ⚪ 未跑 |
| 5 | 职责响应审计（78 项口径） | 需两条腿就绪后跑 | 🟡 **前置已满足** |
| 6 | 多卡多进程故障恢复压测（A2） | 本后端**未声明 `recovery_real`** ⇒ 该压测**如实跳过**（脚本已有此分支） | ➖ 如实跳过 |
| 7 | **退出期段错误收口**（本轮新增） | 容器内无 gdb/eu-stack ⇒ 需调试器镜像或厂商协助 | ⚠️ 未收口 |

## 7. 现场与纪律

- 容器 `hliu553-dc-dev`：挂 **ppu0 + ppu1**、`--network host`、`--shm-size=32g`、**非 privileged**；
  启动脚本 `hliu553-dc-dev.run.sh`（在机器 `/bmcp_lvm_fs/hliu553/`）。
- 该机为**共享机**（同机 12+ 个他人容器在跑，卡 4/5/12 有负载）⇒ 只用自己挂的卡、**未碰任何他人容器**。
- ⚠️ **一条方法学教训**：调厂商 C API 前**必须先触碰设备** —— 直接 `ctypes` 调
  `cuCtxGetStreamPriorityRange` 得 `rc=3`（未初始化），会导致后续建流全失败
  （本手册坑 18 的同一族，本轮实测复现一次）。
