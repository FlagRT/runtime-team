# 流优先级统一 API 落地（(A) 方案）—— `create_stream(priority=)` + 回读校验 · 三实例如实矩阵

> 2026-09-30 · 运行时层（device-context 原型）· 分支 `kistich/device-context`
> 前置：`WORKPACKAGE_D_STREAM_PRIORITY_QUOTA_20260930.md`（B2 v2 根因核查）§5「改进项（待裁定）」
> 相关台账：**第 24 / 25 / 26 条**（本文件 §7）；契约：**§1.3 / §1.9**

## 0 结论先行

用户裁定走 **(A) 补能力**（`create_stream(priority=None)` 走真原语 + **回读校验**）。落地后的**如实矩阵**：

| 实例 | 范围可读 | **能设置** | 能回读 | 真实原因（**不是"没做"**） |
|---|---|---|---|---|
| **MLU590** | `(0, -3)` | ✅ | ✅ | 三家**唯一**能完整落地；上游不拦截非法值（本层已兜住） |
| **910C** | `(7, 0)` | ❌ | ✅ | pyACL **接受并保留**该参数，但产出的是**裸 ACL 句柄**；torch_npu **无任何入口**把它包回 torch 流（6 条证据见 §4.3） |
| **P800** | `(0, 0)` | ❌ | ✅ | **优先级空间退化为单点**（三个 CUDA 入口都在，但设备只报一个档位） |

**端到端判定**：`STREAM_PRIORITY_API_PASS` —— **910C 8/8 · P800 7/7**；破坏面回归 **910C r9 / P800 r7 全绿**。

**一句话**：(A) 在接口层**已完整落地**（四道约束 + 判据 + 5 处非空转验证），
但**"能设置"只在 MLU590 成立**；另两家的障碍**不在本层**，已如实降级为"只读"并备好上游诉求。
⭐ 本轮的真正产出是**把"看起来设了"这种假象变成不可能**（见 §7 第 24 条）。

---

## 1 为什么必须做（B2 v2 的结论直接指向这里）

v2 核查（官方文档 + C API 回读）确认了两层事实：

1. **ACL 层接受**：pyACL `aclrtCreateStreamWithConfig(priority, flag)` 传 0/3/7，用
   `aclrtStreamGetPriority` 回读 **== 传入值**（真机证据 `910C/probes/prio_readback_910c_npu_20260930.log`）。
2. **插件层丢弃**：`torch.npu.Stream(priority=7)` 回读 **恒 0**（同一条流？句柄有效、`rc=0`，值就是 0）。

⇒ 统一 API 若照抄厂商 torch 的构造签名，就会产出**最贵的假象**：上层以为"高优先级流建好了"，
据此做调度决策，而设备侧**根本没有这回事**。**这不是"少个功能"，是"多一个谎"。**

## 2 落了什么（只增不改）

| 接口 / 键 | 语义 |
|---|---|
| `create_stream(priority=None)` | `None` = 厂商默认（**与历史逐位一致**）；`int` = 指定优先级 |
| `stream_priority_readback(stream)` | 回读某条流的**实际**优先级（`int`）；**读不出返回 `None`（不猜、不补零）** |
| 能力键 `stream_priority` | **范围可读**（既有键，语义不变） |
| 能力键 `stream_priority_control` | **能设置** ← 新 |
| 能力键 `stream_priority_readback` | **能回读** ← 新 |

**为什么拆成三个键**：同 `device_state` / `device_state_control`、`context_query` / `context_lifecycle`
的既有拆法 —— **能读 ≠ 能改，能改 ≠ 能校验**。一个笼统的 `stream_priority` 无法让下游从元信息判断
"能不能真的设"，只能靠踩坑。

### 2.1 四道约束（实现放基类唯一实现 = 共享资产单一实现）

```
① 未声明 stream_priority_control  ⇒  NotImplementedError   （显式拒绝，绝不静默返回一条无优先级的流）
② 非 int / bool / 越出取值范围     ⇒  ValueError
③ 创建后**强制回读校验**，回读 ≠ 请求  ⇒  RuntimeError      （"参数被厂商静默丢弃"的唯一防线）
④ priority=None                   ⇒  行为与历史逐位一致
```

`priority=None` 之外的**所有**校验都在 `backends/base.py::create_stream` 里 ——
三家后端只实现私有 `_create_stream_raw(priority)`（同 `supports()` / `set_device_state` 的先例：
**共享资产单一实现**，避免三份校验漂移）。

### 2.2 三家实现

| 实例 | `_create_stream_raw(priority)` | 回读原语 |
|---|---|---|
| 910C | `priority` 非 None ⇒ **显式抛 `NotImplementedError`**（带 4 条理由 + 上游诉求）；默认路径 `torch.npu.Stream()` **不变** | ctypes 调**真实** `libascendcl.so` 的 `aclrtStreamGetPriority`（pyACL 未暴露该入口） |
| P800 | 同上（理由不同：空间退化） | ctypes 调 `libcuda.so.1` 的 `cuStreamGetPriority` |
| MLU590 | `torch.mlu.Stream(priority=int(p))` | `Stream.priority`（本家的"原语"就是 torch 侧属性，无独立句柄式 C API） |

⚠️ 三处**实现细节**（本层纪律，不是可选优化）：
- 取库**必须走 `/proc/self/maps`**（ascend）：直接 `CDLL("libascendcl.so")` 会抓到工具链的
  **stub 库**，调用返回 `rc=100039`("stub library cannot be used for execution") ——
  **症状是"调用失败"，极易被误读成"接口不可用"**。
- 回读原语单独成方法（`_stream_priority_read_raw`）：**可注入/可替换**，
  否则离线（无设备、无真库）这条链路压根走不到（"能造可控假原语就别 SKIP"）。
- 回读**不得补零**：原语不可得时返回 `None`；补零会把"读不出"伪装成"优先级就是 0"（已加判据守）。

## 3 判据（离线自检新增 `[8b]` 段，不需卡）

| 判据 | 内容 |
|---|---|
| 形状 | `stream_priority_range()` 必须是 **2 元组**或 `None`（**不得透传厂商三元组**） |
| 耦合 | 声明 `stream_priority_control` ⇒ **必须同时声明** `stream_priority_readback` |
| 设置 | 声明了 control ⇒ 范围内端点**必须能设置且回读一致**；越界/非 int/bool ⇒ `ValueError` |
| 拒绝 | 未声明 control ⇒ `create_stream(priority=…)` 必须抛 **契约级 `NotImplementedError`**（返回流即 FAIL） |
| 回读 | 声明了 readback ⇒ 原语可用时必须返回 `int`；**原语不可得时必须返回 `None`（不得补零）** |

**stub 按真机形态补真**（这一条本身是 §7 第 25 条的教训）：
- 假 pyACL 的 range 由 `(0, -1)`（2 元组）改为 **`(7, 0, 0)`（3 元组，与真机一致）**；
- stub 的 `Stream` 补 `priority` / `npu_stream` / `cuda_stream` / `mlu_stream` / `priority_range()`，
  以及一个 **`drop_priority` 开关**（打开即**精确复现 torch_npu 的静默丢弃**）。

**5 处注入的非空转验证（5/5 当场 FAIL）**：

| 注入 | 目标判据被判 FAIL 的证据 |
|---|---|
| `drop_priority=True`（复现 torch_npu 行为） | `[FAIL] … create_stream(priority=-3) 成功且回读一致  RuntimeError: 流优先级未生效：请求 -3，设备回读 0` |
| 去掉能力门禁 | `[FAIL] … 必须显式拒绝  NotImplementedError: ascend：**无法**在保持… 的前提下设置优先级` |
| `ascend` 改回透传三元组 | `[FAIL] … 形状合规  得到 (7, 0, 0)` |
| 声明 control 却去掉 readback 声明 | `[FAIL] … 必须同时声明  control=True readback=False` |
| 去掉越界校验 | `[FAIL] 越界 priority=-4 ⇒ ValueError  未报错（静默放行非法值）` |

## 4 真机证据

### 4.1 新增探针（跨芯片复用）

`probes/probe_stream_priority_api.py --backend <b>`，7–8 项判定 + 结构化 JSON：

| 判定 | 910C | P800 |
|---|---|---|
| D1 默认路径不回归（建流 + 上下文内计算 16.0） | ✅ | ✅ |
| D2 范围形状 2 元组 | ✅ `(7, 0)` | ✅ `(0, 0)` |
| D2b 回读返回 int | ✅ `0` | ✅ `0` |
| D4 未声明 control ⇒ 显式拒绝（含能力键名） | ✅ ×2 值 | ✅ ×2 值 |
| **D5 厂商路径的参数保留性（观测项）** | ⚠️ **静默丢弃**（请求 7 ⇒ 回读 0） | 请求 0 ⇒ 回读 0（保留；但空间是单点） |
| **D6 受阻点定位（仅 ascend）** | ⭐ pyACL 建流(7) 回读 **7**（ACL **保留**）；`Stream(stream_ptr=handle)` 拿到的却是**另一条**句柄 ⇒ 回读 **0**（句柄被静默忽略、`same_stream=False`） | —— |
| 合计 | **`STREAM_PRIORITY_API_PASS` 8/8** | **`STREAM_PRIORITY_API_PASS` 7/7** |

**D5 与 D6 是本轮最有价值的两个观测项**：它们把"设了没生效"的**位置**钉死了 ——
不在 ACL 层，而在"**把裸 ACL 流包回 torch**"这一步。

### 4.2 910C 第 9 轮回归（r9，破坏面全覆盖）

改动落点 = `create_stream`（**所有流创建都走新路径**）⇒ `create_stream()` 的**每个消费方**都被覆盖：

| 项 | 结果 |
|---|---|
| 离线自检 / 对称性自检 | **88 / 0 / 1** · **7 / 0** |
| 冒烟 | **52 / 0** |
| conformance（13 + 推理 6） | `CONFORMANCE_PASS` ×2 |
| 契约不变式 | `CONTRACT_INVARIANTS_PASS` 4/4（I1④ 入口覆盖 **16/17**） |
| 职责响应审计 | **39 / 0 / 0**（`DUTY_RESPONSE_PASS`） |
| 错误闭环 | `ERROR_RECOVERY_LOOP_PASS` **5 / 0 / 0** |
| B/C 契约探针 | `PASS` |
| **流优先级 API（新）** | **`STREAM_PRIORITY_API_PASS` 8/8** |
| 多流语义 / 流配额 | **`STREAM_SEMANTICS_PASS` 8/8** · **`STREAM_QUOTA_PASS` 3/3** |
| demo（`create_stream` 消费方） | ✅ |
| **训练腿**（2 卡 davinci2,3） | **`TRAIN_LEG_PASS` 6/6** · loss 15.4498→11.1479 · 4163.0 tok/s（2 卡合计） |
| **推理腿** | **`INFER_LEG_PASS` 14/14** · dim 1024 · 66.67 句/s · p50 44.46 ms · 区分度 0.6391 |
| **服务化（`serve_standard.sh`）** | **`SERVE_STANDARD_PASS`**（`ready=1 smoke=1`；**就绪 45 s**、冒烟 5 s；服务端 `/v1/embeddings` 200） |
| **服务化腿（`proto_infer_serve.py`，本层消费方）** | **`SERVE_LEG_PASS` 10/10** —— 关键子项 `[1b] 同卡跨流计算: 3.0 ✅`（即 `create_stream()` 在**服务同卡**上仍正确）；维度 1024 / 范数 1.000000 |

> **免跑理由（可复核，只对本轮改动成立）**：`scripts/serve_standard.sh` **不经过本层**
> （`grep -E "runtime|create_stream" scripts/serve_standard.sh` = 0 真实命中，仅 1 处注释里的镜像名）。
> 但它仍被**跑了一遍**（因为它是"标准形态"，成本低），故此项非免跑而是**已跑**。

### 4.3 910C 为什么"能设置"做不到 —— 6 条证据（`known_issues` 已登记 `ASCEND-STREAM-PRIORITY-SET-BLOCKED`）

| # | 证据 | 来源 |
|---|---|---|
| 1 | `torch.npu.Stream(priority=7)` ⇒ `aclrtStreamGetPriority` 回读 **0**（句柄有效、`rc=0`） | `prio_readback_910c_npu_20260930.log` |
| 2 | `torch.npu.Stream.get_priority()` 官方报 `NPU dose not support Stream.get_priority() currently.` | `prio_api_surface_910c_npu_20260930.log` |
| 3 | 无 `torch.npu.ExternalStream` —— 而**同文件里有 `ExternalEvent`**（插件内自己不对称） | 同上 |
| 4 | `torch_npu._C` **无任何 `*External*` 符号**；12 个候选 kwarg 穷举：只有 `stream_ptr` 被**接受且静默忽略**，其余一律 `TypeError: invalid keyword argument` | `prio_api_surface…log` + `probes` 探针 |
| 5 | `libtorch_npu.so` **导出** `c10_npu::getStreamFromExternal`，但**未暴露到 Python** | `nm`/`strings` 实测 |
| 6 | `acl_rt.h` **无 setter**（只有 Create/Get/Range）⇒ 无法"先建后改" | 头文件原文 |

**结论**：pyACL 能建带优先级的流且**可回读**，但那是一条**裸 ACL 句柄** ——
**在保持"该流可被 torch 执行上下文使用"的前提下，本栈没有任何入口把参数送进设备**。
流若不能进 torch 执行上下文，"设了优先级"等于白设。

**上游诉求（具体、可执行）**：torch_npu 补 `ExternalStream`（同文件已有 `ExternalEvent`），
或把 `Stream(stream_ptr=…)` **真正接上** —— 现状是**静默忽略**，比报错更危险。

### 4.4 P800 第 7 轮回归（r7）

| 项 | 结果 |
|---|---|
| 离线自检 / 对称性 | **89 / 0 / 1** · **7 / 0** |
| 冒烟 | **46 / 0** |
| conformance（13 + 6） · 契约不变式 · 职责审计 · 错误闭环 | `PASS` · `PASS` · `PASS`（39 项）· **5 / 0 / 0** |
| B/C 契约探针 · 入口定向验证 | `PASS` · `PASS` |
| **流优先级 API（新）** | **`STREAM_PRIORITY_API_PASS` 7/7** |
| 多流语义 / 配额 | **8/8** · **3/3** |
| demo（`create_stream` 消费方） | ✅（修复后，见 §7 第 26 条） |
| 训练腿（2 卡 dev4+dev0） | **`TRAIN_LEG_PASS` 6/6** · loss 15.4488→11.1481 · 3461.1 tok/s |
| 推理腿 | **`INFER_LEG_PASS` 13/13** · dim 1024 · 52.06 句/s · 区分度 0.6392 |

**P800 侧的关键更正**：`stream_priority_range()` 原返回 `None`（表述为"不支持"），
理由是"调 torch 的 `priority_range()` 会触发 PyTorch INTERNAL ASSERT"。
复核后发现那是 **torch API 路径**的问题，**底层真原语安全可读**（返回 `(0, 0)`）⇒
现改为**如实回报"退化单点"**（纪律 ⑰：能力缺失的根因表述本身可能是错的）。

### 4.5 MLU590 —— 如实标注未复验

MLU590 的能力声明依据 **2026-09-22 的真机记录**（`priority_range()` = `(0,-3)`、
`Stream(priority=0/-1/-3)` 可创建且回读正确），但**本轮（09-30）未在 MLU590 真机复验本层新接线**
（该机网络未启用 ⇒ 无窗口）。⇒ 已登记入未收尾清单：**待窗口复验**（含 §3 的 `[8b]` 段真机跑一遍）。

## 5 能力矩阵（实测，`--all` 自动呈现）

| | 全集 | 声明 | 范围 | control | readback |
|---|---|---|---|---|---|
| ascend | 21 | **19** | `(7, 0)` | ❌ | ✅ |
| kunlun | 21 | **16** | `(0, 0)`（真机） | ❌ | ✅ |
| cambricon | 21 | **13** | `(0, -3)` | ✅ | ✅ |

`--all` 的差异清单会**自动**把这处不对称摆出来（`能力 stream_priority_control 仅 ['cambricon'] 声明`）——
这正是"如实拆分"要表达的语义，不需要额外文档去"解释"。

## 6 过程中附带修掉的两处（都不是计划内）

1. **`ascend.stream_priority_range()` 透传三元组**（§7 第 25 条）：真机 `(7, 0, 0)`，
   而 `cambricon` 返回 2 元组 ⇒ **同一份下游代码在两台机器上读到不同形状**。
2. **`demo_unified.py` 根解析 off-by-one**（§7 第 26 条）：`parents[1]` 应为 `parents[2]`，
   原来靠"调用方 CWD 恰好是原型根"侥幸可用（910C 容器 `PYTHONPATH` 结尾的 `:` 把 CWD 带进了 `sys.path`），
   P800 上直接 `ModuleNotFoundError`。

## 7 台账新增（第 24 / 25 / 26 条）

- **第 24 条 ·「构造函数收下了参数」≠「参数生效」**：能力声明必须落到「**能设置 + 能回读**」，
  且回读是唯一判定。**「声明」的粒度必须与「能验证的粒度」一致。**
- **第 25 条 · 契约的「形状」也是契约 —— 而 stub 的"简化形态"会把它盖住**：
  离线 stub 写 2 元组、真机返 3 元组 ⇒ 判据看不见真缺陷。
  ⭐ 与 §2.6「stub 的不完整会伪装成实现的缺陷」**正好相反**：这次是**stub 太好看**伪装成了实现的正确。
- **第 26 条 · 隐式依赖调用方 CWD = 不可移植**：
  同一句命令在 910C 能跑、在 P800 报 `ModuleNotFoundError`；
  ⭐ 教训：**"在一台机器上跑得通"可能来自与代码无关的环境巧合**。

## 8 边界（如实，不外推）

- **本报告的"能设置"只在 MLU590 成立**，且 MLU590 侧**本轮未真机复验**（§4.5）。
- **接口能力 ≠ 调度效果**：官方文档明确 **Atlas 训练系列上 `priority` 属「预留参数、暂不使用」**，
  且优先级「不抢占已在运行的低优先级任务」「不动态重评估任务队列」
  ⇒ 即便打通入口**也未必**有可观测效果（本方向 09-30 的实测：两种提交顺序下均"谁先提交谁先完成"）。
  上层若据此做调度决策，**应先补性能侧对照实验**（未做）。
- **`priority` 的语义方向**：官方 `(least, greatest)` 中 `greatest` 是**数值最小**者 ⇒ **0 最高**。
  本层按 `min/max` 规整区间，不按原序。
- **未做（列全）**：`priority` 的**调度效果**实验 · MLU590 真机复验 · 上游 `ExternalStream` 的跟进。
- ⚠️ **P800 上一处新观察到的「口径冲突」（已登记）**：厂商栈在 stderr 提示「官方手册要求测试前 `export XPU_EVENT_KL3_ENABLE=1`，否则**部分路径行为未定义**」，而**该变量正是本方向 09-14 实测的挂死触发条件**（KL3 事件 + 设备集合通信 ⇒ `cudaDeviceSynchronize` 永久自旋）。
  ⇒ **手册要求设置的环境变量 = 我们实测的故障触发条件**。本轮 P800 各项（含新探针）**均在未设该变量的状态下取数**；与既有「错误闭环两设置逐字节一致」的记录相容，但「部分路径行为未定义」这句属厂商声明，如实标注为**上游口径冲突**。
- **证据边界**：所有真机结论只在**该实例 + 该 torch/驱动版本**成立（910C: torch 2.11.0+cu130 / CANN 9.0.0；
  P800: torch 2.9.0+cu129 / XPU-RT 5.0.21）。

## 9 证据清单（`910C/probes/`、`P800/probes/`）

| 文件 | 内容 |
|---|---|
| `910C/probes/prio_api_910c.json`（+ r9 `r9_prio_api_ascend.*`） | 910C 流优先级 API 探针 8/8（含 D5 静默丢弃、D6 受阻点定位） |
| `P800/probes/r7_prio_api_kunlun.*` | P800 7/7（含 `(0,0)` 退化单点） |
| `910C/probes/r9_regress_910c_npu_20260930.log` + `r9_regress_910c_npu_20260930_out/` | 910C 第 9 轮回归全部 15 项日志/JSON |
| `P800/probes/r7_regress_kunlun_20260930.log` + `r7_regress_kunlun_20260930_out/` | P800 第 7 轮回归全部项 |
| `*_prio_readback_*_20260930.log`、`*_prio_api_surface_*_20260930.log` | 根因证据（回读对照 / 接口面与插件源码） |
| `910C/probes/run_regress_ascend_r9.sh`、`P800/probes/run_p800_r7.sh`、`…/run_910c_serve_r9.sh` | 一键复跑脚本（与证据同目录入库，防断链） |
| `prototype/probes/probe_stream_priority_api.py` | 可复用探针（跨芯片，`--backend` 参数化） |

⚠️ 复跑前置（**都是踩过的坑**）：容器 `PYTHONDONTWRITEBYTECODE=1`；训练腿必须给 `DC_ROOT` 且在
`runtime/proto/` 下起；**`DEV` 是容器内索引而不是宿主 smi 索引**（本容器只挂 davinci7 ⇒ `DEV=0`，
填 7 会得到 `aclInit 107001 Invalid device ID`）；P800 侧宿主 `/data2/hliu553` = 容器 `/workspace`
（宿主脚本重定向用宿主路径、给容器程序的 `--out` 用容器路径）。

---

**一句话总结**：这轮把"**看起来设了优先级**"这种假象从接口上消灭了 ——
能设的（MLU590）必须能回读校验，不能设的（910C / P800）**如实拒绝并说清原因**；
而"能设 ≠ 有效果"这一层，留给文档与实测结论，不靠接口暗示。
