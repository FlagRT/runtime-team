# 公开入口补充：设备状态驱动 + 错误编排（2026-09-29 · 只增不改）

> 定位：**接口面补充记录**（读者＝本方向与下游对接人），与工作包 B / C 同类。
> 上游结论来源：`../../910C/docs/ASCEND_910C_A2_RECOVER_STRESS_20260929.md` §7
> —— A2 压测实测发现「某卡 L4 故障 → 设备级恢复」这条链**在公开面上不可触发**。
> 边界：判定与实测只在 **910C 当前档位/环境**成立；离线侧三家同跑；**未触及 P800 / MLU590 真机**。

---

## 0 结论先行

| 判定 | 结果 |
|---|---|
| **补了什么** | **2 个只增公开入口**：`runtime.set_device_state()`（驱动四态）· `runtime.handle_error()`（R1–R5 编排）；**1 个只增能力键** `device_state_control` |
| 为什么要补 | ① `recover_device(mode="real")` **只在 `ISOLATED` 时**才真重建，而公开面**没有置隔离的入口** ⇒ 该链不可触发；② 上层只能拿到「分级」与「重建」两个零件，**中间（评估 → 隔离 → 重放）只能自己拼** |
| **端到端真机证据** | ✅ `ENTRY_VERIFY_PASS` —— 一次 `handle_error(<L4 消息>, mode="real")` 走完 R1–R5：`steps = ['captured', 'evaluated: isolated', 'recovered: True', 'replay_ready']`，设备回 `available`（**全程公开 API**） |
| **守住了没有** | 新增 **4 条离线判据**（不需卡）+ **I1④ 自动覆盖**（「声明 ⇒ 有公共入口」由 13/13 升到 **14/14**）；**4 条判据全部做了非空转验证**（注入缺陷后各自真的 FAIL） |
| **破坏面回归** | ✅ 910C 第 7 轮 **10 项全绿**（离线 **83/0/1** · 对称性 5/0 · 冒烟 52/0 · conformance 13+6 · 契约不变式 4/4 · 职责审计 39/0/0 · 错误闭环 5/0/0 · B/C 探针 7/7 · 入口定向验证 · A2 压测复跑） |
| 只增性 | **未改任何既有签名**；`handle_error` 的既有内部实现只做了**等价改写**（见 §3.3） |

---

## 1 为什么补（实测来由，不是设想）

**① 链不可触发**（A2 压测发现，见 A2 报告 §7）：`recover_device(mode="real")` 的准入条件是
`query_device_state(ordinal) == ISOLATED`，否则直接返回"无需重建"。而 `mark_device_state`
此前**只在 `runtime/conformance/` 内部**（未导出）⇒ 只走公开 API 时：

| 场景 | 修前行为 |
|---|---|
| 健康设备调 `recover_device(0, mode="real")` | `recovered=True` + `detail="…无需重建，探活可用"` —— **`real` 从未真正执行** |
| 想演练"某卡故障 → 设备级恢复" | **无公开入口**，只能 `sys.path` 进 conformance 摸内部模块（A2 压测最初就是这么做的） |

**② 编排缺口**：公开面有 `translate_error`（分级）与 `recover_device`（重建），但
**R2 评估 / R3 隔离 / R5 重放**没有入口 ⇒ 每个上层都得自己拼一遍 R1–R5 —— 而"三套口径必然漂移"是本项目反复吃过的亏。

---

## 2 补了什么（逐项语义）

### 2.1 `runtime.set_device_state(ordinal, state, reason="") -> str`

- **作用**：驱动本层四态状态机，返回新状态 **token 字符串**；与只读的 `device_state()` 成对。
- **取值域**：`DEVICE_STATE_TOKENS`（`available` / `degraded` / `isolated` / `destroyed`），也接受共享枚举成员；
  **非法取值 ⇒ `ValueError`**（不静默）；同态转换不产生事件（幂等）。
- ⚠️ **它是本层的隔离账本，不等于厂商设备的真实状态**：把健康设备标成 `isolated` 只影响
  **本层**的调度 / 恢复判定（演练与混沌注入靠它），**不会**让硬件出错；真实故障的隔离仍应由
  R2 评估（探针失败）驱动。这一句同时写进了 docstring 与契约。

### 2.2 `runtime.handle_error(exc, ordinal=None, location="", mode="probe") -> FlagosError`

- **作用**：五段式编排（R1 分级 → R2 评估 → R3 隔离 → R4 重建 → R5 重放），返回**统一类型**错误对象；
  `recovery_decision` 记录流程事件供监控消费。
- `mode` 与 `recover_device` 同语义，**默认 `"probe"`**（进程内安全）；传 `"real"` 才真重建。
- ⚠️ **如实标注的已知边界**：R5 的**在途登记**入口（`mark_inflight` / `finish_inflight`）与状态机
  **事件订阅**（`last_transition`）**未公开** ⇒ 现阶段 `replay_tasks` 恒为空列表（见 §6）。

### 2.3 能力键 `device_state_control`（只增）

- 三家 `_CAPABILITY_KEYS`（全集）与 `_capabilities`（声明集）**同时**补上 —— 否则下游 `info()` 看不到、
  I1④ 也守不到它。
- **芯片无关**：走共享状态机（进程内账本，不依赖厂商原语）⇒ **三家一致声明**。
- **与 `device_state` 分开声明**，理由同 `context_query` / `context_lifecycle`：**能查 ≠ 能改**。

### 2.4 实现落在**基类唯一实现**（为什么）

四态状态机是芯片无关的共享资产 ⇒ 三家实现必然逐字相同。项目已有先例：`supports()` 也从
"三家各写一份"**收敛到基类唯一实现**（写成三份必然漂移）。故 `set_device_state` / `handle_error`
只在 `RuntimeBackend` 实现一次，三家继承。

---

## 3 怎么守住（判据 + 非空转）

### 3.1 新增 4 条离线判据（**不需卡**）

| 判据 | 守什么 |
|---|---|
| `公开面已导出 set_device_state / handle_error` | 入口确实在 `__all__` 且可调用（防"实现了但没导出"） |
| `set_device_state('isolated') ⇒ device_state() 读到 isolated` | **驱动真的生效**，且判据侧与后端看的是**同一份状态机** —— 这是"公开面可触发"的基础 |
| `set_device_state 非法取值 ⇒ ValueError（不静默）` | 取值域判据（覆盖"只查字段名不查取值域"的老坑） |
| `handle_error 的分级继承本后端码表归属` | ⭐ **未声明 `error_map` 者不得因他厂码判成 L4**（否则**误触发设备级重建**，与台账第 16 条同族）。该条用 `ordinal=None` 调用 ⇒ **不碰设备**，离线可判 |

另：`contract_invariants.CAPABILITY_ENTRYPOINTS` 加 `"device_state_control": ("set_device_state",)`
⇒ **I1④「声明为支持 ⇒ 公共入口可调用」的覆盖面由 13/13 升到 14/14**（新键自动被守）。

### 3.2 非空转验证（4 条判据**各自**都被注入缺陷证明能 FAIL）

| 判据 | 注入方式 | 结果 |
|---|---|---|
| 公开面导出 | 从 `__all__` 移除两者 | ✅ FAIL（`缺/不可调用 ['set_device_state','handle_error']`） |
| 驱动真的生效 | 驱动后立刻改回 `available` | ✅ FAIL（`set 返回='isolated' 读到='available'`） |
| 非法取值报错 | 非法值静默兜底成 `available` | ✅ FAIL（`未报错`） |
| 码表归属继承 | 丢掉 `translate_fn=self.translate_error` | ✅ FAIL（**kunlun** 上 `category=L4_FATAL` 而 `supports(error_map)=False`） |

> 最后一条最值得看：丢掉后端译码器后，**未声明 `error_map` 的 kunlun 会把昇腾码 `507015` 判成 L4**
> —— 正是"会误触发设备级重建"的那条路，判据真的抓住了；还原后 kunlun 正确判 `L3_EXECUTION`。

### 3.3 一处**等价改写**（不是行为变更，但必须说明）

共享编排器 `conformance/recovery.handle_error` 原按**枚举相等**判断级别
（`fe.category != ErrorCategory.L4_FATAL`）。新增的 `translate_fn` 允许传入后端自己的译码器，
它返回的是**统一错误对象**（枚举类型不同）⇒ 原写法会**恒不相等**、把设备级恢复**整段静默跳过**。
故改为**按 `category` 名字**判定（`_cat != "L4_FATAL"`）—— 对本文枚举与统一枚举**都成立**，
是枚举无关的等价改写。同时新增 `rebuild_mode` 关键字（默认 `REBUILD_PROBE` ⇒ **默认行为不变**）。

---

## 4 端到端真机证据（910C）

```
PASS  D1_健康设备real_不得声称已重建   context_recreated=False detail='…无需重建，探活可用'
PASS  D2_公开入口置隔离后real_必须声明真重建
      set 返回='isolated' recovered=True context_recreated=True detail='…真实重建成功（aclrtResetDevice 序列已执行）'
PASS  D2_重建后回到 available（R4）
PASS  D3_隔离后probe_不得声称已重建    context_recreated=False
PASS  D4_handle_error 端到端走到 real 重建并重放就绪
      steps=['captured', 'evaluated: isolated', 'recovered: True', 'replay_ready']
PASS  D4_端到端后设备回 available
ENTRY_VERIFY_PASS
```

⇒ D4 是**一句话证据**：上层捕获异常后**一次 `handle_error(..., mode="real")`** 即完成
分级 → 评估 → 隔离 → 真重建 → 重放就绪，**全程公开 API、不碰内部模块**。

**A2 压测同步改用公开入口并复跑**：3 rank（一卡一进程）× 30 轮、30 次真重建
**零失败**（10.5 s），S0–S3 全 PASS —— 即"演练"这件事现在也是公开 API 能做的。

---

## 5 破坏面回归（910C 第 7 轮 · 10 项全绿）

| # | 判定项 | 结果 |
|---|---|---|
| 1 | 离线契约自检（ascend） | **83 / 0 / 1 跳过**（78 → 83：+4 新判据 +1 条） |
| 2 | 跨后端对称性 `--all` | **5 / 0**（新键三家一致声明，未产生新的不对称行） |
| 3 | 组件冒烟 | **52 / 0** |
| 4 | conformance 13 例 | **13 / 13 `CONFORMANCE_PASS`** |
| 5 | conformance 推理 6 例 | **6 / 6 `CONFORMANCE_PASS`** |
| 6 | 契约不变式 I1–I4 | **4 / 4 `CONTRACT_INVARIANTS_PASS`**（I1④ 覆盖 14/14） |
| 7 | 职责响应审计（39 sub-part） | **39 OK / 0 FAIL / 0 SKIP** |
| 8 | 错误注入→恢复闭环 | **闭环 5 / 跳过 0 / 失败 0** |
| 9 | 工作包 B/C 真机探针 | **7 / 7 PASS** |
| 10 | 入口定向验证 + A2 压测复跑 | **`ENTRY_VERIFY_PASS`** · **`MULTIPROC_REAL_RECOVER_PASS`** |

**离线侧三家同跑**（本机，无设备）：ascend **83/0/1** · kunlun **85/0/1** · cambricon **73/0/0**。

**免跑项（当轮可验证）**：训练腿 / 推理腿 `grep` 0 命中 `recover_device|device_state`；
服务化仅有注释与 `MONITOR=1` 分支（默认 0）⇒ 三条路径都不在本轮破坏面内。

---

## 6 边界（**未**补的，如实登记，不擅自扩张）

| 项 | 状态 | 说明 |
|---|---|---|
| R5「**在途登记**」入口（`mark_inflight` / `finish_inflight`） | **未公开** | ⇒ `handle_error` 的 `replay_ready` 现阶段恒配一个**空** `replay_tasks`。上层若要真正消费重放集合，还需一个登记入口 —— **是否公开另议** |
| 状态机**事件订阅**（`subscribe_device_state` / `last_transition`） | **未公开** | 监控方向若需要"变化时回调"而非轮询，需要它 —— 同样另议 |
| **真实硬件 L4 故障**触发 | 未做 | 健康卡无法制造 AICORE 异常；本轮的隔离是**构造**的（已在 docstring 与契约写明"本层账本 ≠ 硬件状态"） |
| `P800 / MLU590` 真机 | 未跑 | 新入口是共享资产、逻辑与芯片无关，但**结论不得由 910C 外推** ⇒ 两家各自窗口复跑（`--backend` 已参数化） |
| 四态机的**状态持久化 / 跨进程共享** | 不做 | 状态机是**进程内**账本；多进程各自一份（这也是 A2 压测里"每 rank 独立"的前提） |

---

## 7 一键复跑

```bash
cd dev/device-context/prototype

# 离线（不需卡）：三家 + 对称性
for b in ascend kunlun cambricon; do python3 scripts/backend_offline_check.py --backend $b; done
python3 scripts/backend_offline_check.py --all

# 真机（容器内）：入口定向验证 + 压测
DC_ROOT=<prototype> python3 probes/recover_entry_verify.py --backend ascend --dev 0
DC_ROOT=<prototype> python3 probes/recover_multiproc_stress.py --backend ascend --devices 0,1,2 --rounds 30

# 回归（宿主脚本 → 容器）
bash <宿主>/run_regress_ascend_r7.sh
```

---

## 8 证据清单（`../../910C/probes/`）

| 文件 | 内容 |
|---|---|
| ⭐ `entry_verify_910c_npu_20260929.{json,log}` | **入口定向验证**（D1–D4，含 `handle_error` 端到端 steps） |
| ⭐ `a2_recover_multiproc_910c_npu_20260929_pubentry.{json,log}` | **压测改用公开入口后复跑**（3 rank × 30 轮） |
| `r7_regress_910c_npu_20260929.log` + `r7_regress_910c_npu_20260929_out/` | 第 7 轮破坏面回归 10 项原始输出 |
| `offline_3backends_910c_npu_20260929_r7.log` | 离线三家同跑（83 / 85 / 73，全 0 失败） |

> 命名规范与「当前结论 = 哪一份」见 `VERIFICATION_MANIFEST_20260920.md` §2、§5。
