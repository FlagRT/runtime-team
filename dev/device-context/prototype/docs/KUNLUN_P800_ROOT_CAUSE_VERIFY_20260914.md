# 昆仑芯 P800 集合通信挂死 · 结论核对与责任层判定（2026-09-14）

> **本文目的**：对 2026-09-14 得出的挂死结论做**独立复核**——① 判断是否准确；
> ② 可复现性如何；③ 该提交**算子层**还是**编译层**。
>
> **一句话结论**：
> 判断**基本准确但有 1 处重要更正**（"三要素"实为"两要素 + 多个暴露点"，其中"不做同步就通过"是**假阴性**）；
> **复现率 16/18 ≈ 89%**（本轮基线 4/4 = 100%）；
> **既不是算子层、也不是编译层的问题，应提交厂商运行时/驱动层（XPytorch / XRE 的 `libxpucuda.so`）**，
> 并抄送 FlagCX 插件（其 `syncStream` 放大了触发面）。

---

## 1. 结论准确性核对

### 1.1 ✅ 核对通过的部分

| 结论 | 核对方式 | 结果 |
|---|---|---|
| 挂死点在厂商 `libcuda.so` 内 | gdb 原生栈（三处一致） | ✅ 成立；该库实为 `libxpucuda.so.515.58.kunlun` |
| 两个 rank 主线程栈一致、其它线程空闲 | gdb `thread apply all bt` | ✅ 成立（非锁竞争） |
| 与数据量、张量形状、reduce op、用卡对无关 | V2 换卡、V6 换 op、A/B 探针（2.38 GB 单次通过） | ✅ 成立 |
| 910C 同款训练腿稳定（非 FlagCX 整体不可用） | 910C 实测 50 步 / 2117 tok/s | ✅ 成立 |
| `BKCL_GC_SIGNAL_MASK=1` 无效 | V4 仍挂死 | ✅ 成立 |
| 挂死点在**我方五域之外** | 探针全程裸 `torch.distributed`，未经过 `RuntimeBackend` | ✅ 成立 |

### 1.2 ❌ 必须更正的部分：**"不做设备同步就能通过"是假阴性**

**原判断**：挂死需要「KL3=1 + 集合通信 + **设备同步**」三条件同时满足，
依据是旧探针 `V5_nosync`（KL3=1、循环内不做同步）报"✅ 120/120 通过"。

**复核发现该依据不成立**——旧探针**不自证**：
它只是"没等"就退出，**从未确认那 120 次集合通信是否真的完成**。

**新探针（`dc_probe_verify.py`）加了结尾真值校验**（各 rank 初值全 1，
经 SUM all_reduce 后应恰为 `2^120`，float32 可精确表示），结果：

| 组 | 条件 | 结果 | 真值校验 |
|---|---|---|---|
| A×4 | KL3=1 + 每 10 次同步 | ❌ **4/4 挂死**（rep=100 / 20 / 0 / 0） | — |
| B×2 | **不设 KL3** + 每 10 次同步 | ✅ 通过 | **`1.329228e+36 = 2^120`，rel_err = 0.000e+00** |
| **C×2** | **KL3=1 + 循环内完全不做同步** | ❌ **2/2 挂死**（均停在 `rep=100`） | — |
| D1 | 不设 KL3 + 不作同步 | ✅ 通过 | **`2^120` 精确匹配** |

**⇒ 更正后的模型**

| 项 | 更正前（错） | 更正后（对） |
|---|---|---|
| 必要条件 | KL3=1 **+ 集合通信 + 设备同步** | **KL3=1 + 设备侧集合通信**（两条） |
| 显式同步的角色 | 必要条件 | **只是暴露点**，不是必要条件 |
| "不做同步就能过" | 规避手段 | ❌ **假阴性**（进程没等而已） |

**机理**：从 H2 的原生栈可见，`dist.all_reduce` 会进入
`c10d::flagcxBackend::allreduce` → **`flagcxBackend::syncStream`** → `CUDAEvent::record`
→ `cudaEventRecordWithFlags`——**flagcx 插件在每次 all_reduce 内部就自带一次设备事件记录**，
所以**即使上层不显式同步，挂死照样发生**（C 组实测挂在第 101–120 次 `all_reduce` 内部）。

### 1.3 三个暴露点（均非"必要条件"，而是同一底层缺陷的不同暴露位置）

| # | 暴露点 | 原生栈 | 观测 |
|---|---|---|---|
| **P1** | **`dist.all_reduce` 内部**（根本暴露点，每次调用都有） | `flagcxBackend::allreduce` → `syncStream` → `CUDAEvent::record` → `cudaEventRecordWithFlags` → 厂商 libcuda 自旋（含 `sched_yield`） | C1/C2（本轮）、V6 |
| **P2** | 上层显式 `torch.cuda.synchronize()` | `THCPModule_cudaSynchronize` → `c10::cuda::device_synchronize()` → `cudaDeviceSynchronize` → 厂商 libcuda 自旋 | A1–A4、V1、V4 |
| **P3** | 通信域**首次初始化**（复现率低） | rank1：`flagcxCommInitRank` → `xcclAdaptorCommInitRank`(`xccl_adaptor.cc:87`) → `bkcl::init_rank` → `bkcl::net_socket_all_gather` 的 `recv()`；rank0：`bkcl::kl3::init_device_param` → `xpu_free` | V2（18 次中仅 1 次） |

**精确偏移（一次实测）**：`libxpucuda.so.515.58.kunlun` 基址 `0x744bb1400000`，
自旋帧 `0x744bb1494080` ⇒ **偏移 `+0x94080`**（该库符号已剥离）。

---

## 2. 可复现性评估

### 2.1 复现率

| 条件组 | 运行次数 | 挂死次数 | 复现率 |
|---|---|---|---|
| **KL3=1 + 设备集合通信** | 18 | **16** | **≈ 89%** |
| └ 其中本轮基线（KL3=1 + 每 10 次同步） | 4 | **4** | **100%** |
| **KL3 关（不设 / =0）+ 设备集合通信** | 8 | **0** | **0%**（其中 4 次另有真值校验，全部精确正确） |

> 统计口径说明：早期一次复核脚本因 `"✅ 完成(未挂死)"` 中的 **"未挂死"含"挂死"** 二字被误判为挂死，
> 已修正为按 ✅/❌ 标记判定；上表为修正后数据。

### 2.2 可复现性判定：**概率性可复现，条件明确，已脚本化**

| 维度 | 评估 |
|---|---|
| **复现率** | **高**（≈89%；特定机器状态下可达 4/4） |
| **复现条件** | 明确且最小化：`XPU_EVENT_KL3_ENABLE=1` + 2 进程 + flagcx 设备集合通信 |
| **最小复现耗时** | **约 10 秒内**挂死（120 次 1024×1024 all_reduce） |
| **是否确定性** | ❌ **非确定性**——挂死步数游走（rep 0 / 20 / 30 / 40 / 70 / 100 均出现） |
| **脚本化程度** | ✅ 探针与命令均已入库：`probes/kunlun/dc_probe_verify.py` 等；复现命令见进度报告 §2.5.7 |
| **机器前提** | 需**空闲卡**（他租户占用会引入噪声；但已验证换到 1,2 号卡同样挂死，非卡特定） |
| **未控变量（诚实标注）** | 共享机他租户负载未受控；`P3`（初始化死锁）18 次仅出现 1 次，**复现率低**，单独计入 |

---

## 3. 责任层判定：提交算子层还是编译层？

### 3.1 先给结论

> **两者都不是。** 应提交 **厂商运行时/驱动层（XPytorch / XRE，`libxpucuda.so`）**；
> 并**抄送 FlagCX**（其 c10d 插件的 `syncStream` 在每次 all_reduce 内部记录设备事件，放大了触发面）。
> **算子层（FlagGems）与编译层（FlagTree/triton）均有硬证据排除。**

### 3.2 排除算子层（FlagGems）——三条硬证据

| 证据 | 实测 |
|---|---|
| 环境变量 | `USE_FLAGGEMS=[<未设>]`、`GEMS_VENDOR=[<未设>]` |
| 运行时导入 | `flag_gems` 未导入（`"flag_gems" in sys.modules` → **False**，base 与目标环境均是） |
| 探针源码 | `dc_probe_rep.py` / `dc_probe_verify.py` 中 `flag_gems` 出现 **0 次** |
| 自动启用钩子 | `site-packages/*.pth` 中含 `flag_gems` 的：**无** |

⇒ 挂死发生在**完全没有算子库参与**的路径上（裸 `dist.all_reduce`），
**不可能是算子实现问题**。

### 3.3 排除编译层（FlagTree / triton）——三条硬证据

| 证据 | 实测 |
|---|---|
| 编译缓存未动 | `/root/.triton` mtime = **2026-08-12 21:33**（镜像构建时，非本次） |
| 近 2 小时无编译产物 | `find /root -mmin -120` 匹配 `*triton*` / `*.xpubin*` → **空** |
| 日志无编译痕迹 | 全部探针日志 `grep -iE 'triton\|jit\|compile'` → **空** |

**补充论据**：
- 挂死路径上的内核是 **BKCL 预编译内核**（`libbkcl.so` 内置），**不经过 triton/FlagTree 编译**；
- FlagTree 官方文档 `third_party/xpu/docs/triton-3.6-validation.md` 里
  `XPU_EVENT_KL3_ENABLE=1` 出现在 **"FlagGems test method"** 一节，
  且该验证用的是**单物理 XPU + 单进程 `pytest`**——**多进程集合通信场景从未被覆盖**，
  所以这个变量在编译/算子侧"看起来没问题"是合理的。

### 3.4 指向厂商运行时/驱动层（本层）——两条硬证据

**证据 A：谁真正读这个变量**

`strings` 扫描厂商库，`XPU_EVENT_KL3_ENABLE` 的出现次数：

| 库 | 该字符串出现次数 |
|---|---|
| **`libxpucuda.so`（→ `libcuda.so.1`）** | **1** |
| `libcudart.so.12` | 0 |
| `libxpurt.so.2`（运行时） | 0 |
| `libxpuml.so*`（管理库） | 0 |
| `libbkcl.so`（集合通信） | 0 |

且它在库内的**邻居全是 `CUDA_*` 运行时旋钮**：

```
CUDA_DEVICE_MAX_CONNECTIONS / CUDA_TSG_CHANNEL_COUNT / CUDA_ENABLE_P2P_NO_UVA
CUDA_ENABLE_PENDING_LIST / XPU_EVENT_KL3_ENABLE / CUDA_GRAPH_OPTIMIZE_STREAM
CUDA_CTA_PREEMPTION / CUDA_ENABLE_ABI_TRAPHANDLER / CUDA_CNP_LAUNCH_QUEUE
```

⇒ **它是厂商 CUDA 兼容运行时自己的功能开关**（KL3 为设备代号/驱动 HAL 层），
**FlagGems 与 FlagTree 只是在环境脚本里"照抄设置"它**，并非读取方。

**证据 B：自旋发生地在厂商库内**

三处暴露点的自旋帧**都在 `libxpucuda.so` 内**（偏移 `+0x94080` 等），
`cudaDeviceSynchronize` 与 `cudaEventRecordWithFlags` 均为**厂商实现**的同步原语
（`libcudart.so.12` 只是转发层）。

### 3.5 提交建议（三份，主次分明）

| 优先级 | 提交对象 | 内容要点 |
|---|---|---|
| **① 主提交** | **昆仑芯 XPytorch / XRE**（`libxpucuda.so`） | 现象 + 最小复现 + 三处原生栈 + 偏移 `+0x94080` + 触发条件（`XPU_EVENT_KL3_ENABLE=1` 且存在设备集合通信）。**核心诉求：`cudaDeviceSynchronize` / `cudaEventRecordWithFlags` 在 KL3 事件开启时不应永久自旋；请说明 `XPU_EVENT_KL3_ENABLE` 的语义与正确用法。** |
| **② 抄送** | **FlagCX** | `c10d::flagcxBackend::syncStream` 在**每次 all_reduce 内部**记录 `CUDAEvent`——请确认该 event record 是否必需；可否改为复用/惰性 record，以缩小触发面。（这是唯一在栈里出现、且我方可以直接对话的一环） |
| **③ 知会** | **FlagGems / FlagTree** | 请复核 `flag_gems/backends.yaml` 的 `XPU_EVENT_KL3_ENABLE: "1"` 在 **xpu3.6** 上是否仍然必要（该条目对应的是 xpu3.0 时代组合 `flagtree==0.5.1+xpu3.0`）；并建议其回归补**多进程集合通信**场景——现有验证为单进程单卡，天然覆盖不到本缺陷。 |

### 3.6 本方向（设备上下文/多流）的处置

| 项 | 处置 |
|---|---|
| 是否修 | ❌ **不修**——不在五域内，且证据显示属厂商层 |
| 是否改上游环境口径 | ❌ **不擅自改**——`XPU_EVENT_KL3_ENABLE=1` 是 FlagGems kunlunxin 官方推荐变量，关闭可能掩盖厂商 KL3 事件上报，须上游确认 |
| 是否影响训练腿验收 | ⚠️ 会。若上游不修，训练腿只能以「**标注缺口的证据 + 归属判定 + 最小复现**」形式验收，**不伪造通过** |
| 已产出 | 探针与三轮原始日志（`probes/kunlun/`）、进度报告 §2.5、本文 |

---

## 4. 证据文件索引

| 文件 | 内容 |
|---|---|
| `probes/kunlun/dc_probe_verify.py` | **带真值校验**的验证探针（本次核对的主要工具） |
| `probes/kunlun/verify_battery.sh` + `D_verify_truthvalue_20260914.log` | 本轮 A/B/C/D 四组核对原始日志 |
| `probes/kunlun/dc_probe_rep.py`、`probe_battery{1,2,3}.sh` | 单变量探针与三轮探针组 |
| `probes/kunlun/A_round1_battery_20260914.log` 等三份 | 原始证据（含 gdb 原生栈） |
| `prototype/docs/PROGRESS_REPORT_20260914.md` §2.5 / §2.5.7 | 根因定位与复现命令 |

**复现命令（最小）**

```bash
# 挂死（约 10 秒内，预期进不去 VERIFY）
CUDA_VISIBLE_DEVICES=6,7 FLAGCX_ADAPTOR=klx XPU_EVENT_KL3_ENABLE=1 \
  DC_MODEL=... python3 -m torch.distributed.run --standalone --nproc_per_node=2 dc_probe_verify.py

# 对照（预期 VERIFY 打印 rel_err=0.000e+00）
#   仅去掉 XPU_EVENT_KL3_ENABLE 即可
```
