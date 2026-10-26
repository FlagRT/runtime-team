# P800 共享层复跑读数（2026-10-26）

目的：验证四处**共享层**改动在本实例上零回归 ——
① `create_event(enable_timing=…)` 新入口；② `_stream_ctx` 改强持有（退出期段错误第五载体）；
③ 职责审计 C13 的方向/量纲/可用性三态；④ 推理腿模型路径自动探测（不再写死 910C 路径）。

环境：容器 `hliu553-device-context-p800` · 卡 `0`（当次普查：卡1 = 0 MiB 但为**已知故障卡**，按纪律避开；
卡0/4/5 分别 1648/1862/2222 MiB）· conda `python310_torch29_cuda`（torch 2.9.0+cu129，8 卡可见）。
代码：宿主 `/data2/hliu553/dc_rerun_20261026/prototype` = 本仓 `dev/device-context/prototype` 当前分支。

| 段 | 读数 | 结论 |
|---|---|---|
| 离线契约自检 | **114 通过 / 0 失败 / 1 跳过** | PASS（项数 114 与 PPU 的 118 不同属**能力门控**差异，不是回归） |
| smoke | **46 通过 / 0 失败** | PASS |
| 契约不变式 I1–I4 | **4/4 通过**，**`rc=0`，无 `Fatal Python`** | PASS（PPU 侧第五载体段错误**未在本实例出现**） |
| 职责响应审计 78 项 | **OK 68 / FAIL 0 / SKIP 10** ⇒ `DUTY_RESPONSE_PASS` | PASS（逐项与 10-08 D2 轮归档**完全一致**） |
| 推理腿（**不传 `DC_MODEL`**） | **13/13**，50.28 句/s，p50 59.39 ms，区分度 **0.6392** | PASS；`model_source=按实例布局探测` ⇒ 解析到 `/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…` |

## C13 在本实例的新读数（重要）

| | 10-08 归档 | 2026-10-26 |
|---|---|---|
| 状态 | SKIP | **SKIP**（不变） |
| 原因 | `ValueError: Both events must be created with argument 'enable_timing=True'`（**我们没建计时事件**） | `elapsed_time=0.0 恒近 0（跨设备工作 + 10 ms 睡眠）⇒ 平台接受 enable_timing 但计时不可用（如实不具备）` |

**归因（裸 torch 复核，不经本层）**：`e1.elapsed_time(e2)=0.0`、`e2.elapsed_time(e1)=0.0`
⇒ **本栈接受 `enable_timing=True` 但计时恒 0**，属平台能力缺口（PPU 同法实测 10.089 ms，能计时）。
⇒ 已登记为 P800 已知缺口；判据侧改为「如实不具备(SKIP)」而不是 FAIL，并**不得**被读成"通过"。

## 文件

`offline.log` · `smoke.log` · `invariants.log` · `duty2.log`/`duty2.json`（现行）·
`duty.log`（三态细化前那一轮，C13 判 FAIL，留作对照）· `infer_leg.log`/`infer_leg.json`
