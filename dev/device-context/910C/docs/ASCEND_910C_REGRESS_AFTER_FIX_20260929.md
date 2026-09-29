# 910C 第 1 家实例 · 缺陷修复后全套回归（2026-09-29）

> 定位：**实测记录**。回答一个问题 —— 2026-09-29 的两处层内修复
> （① 错误码表归属 `vendor_codes`；② `recover_device()["state"]` 取值域 `state_token`）
> **是否在 910C 上引入回归**。
> 边界：结论只在**当前档位/环境**成立 —— 宿主 `npu1-27`、镜像
> `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64`、
> `torch 2.11.0+cu130` / `torch_npu 2.11.0`、解释器 `venvs/venv-infer-a`（仅 torch_npu，无 torch_fl）、
> 精简容器挂 `davinci1,2,3`（`ASCEND_RT_VISIBLE_DEVICES` 未设，`device_count=3`）。
> 服务化与两条腿**本轮未复跑**（理由见 §5）。

---

## 1 结论先行

> **10 项判定全部通过，0 缺陷，0 回归** —— 本轮**无需修复**；
> 修复对 910C 的影响**符合设计预期**：声明了 `error_map` 的后端（ascend）分级行为**不变**
> （`l4_by_code` 仍为 `L4_FATAL / device_recovery`），而 `state` 取值域已归一为 `available`。

| # | 判定项 | 本轮（09-29） | 历史对照 | 判定 |
|---|---|---|---|---|
| 1 | 离线契约自检（无设备） | **38 / 0 / 1 跳过** | 09-22 为 **35 / 0**（判据数 35→37→38） | ✅ 无回归 |
| 2 | 跨后端对称性自检 `--all` | **5 / 0** | 09-22 **5 / 0** | ✅ 无回归 |
| 3 | 组件冒烟自检 `smoke_runtime` | **52 / 0** | 09-22 **52 / 0** | ✅ 无回归 |
| 4 | conformance 基线 13 例 | **13 / 13** | 09-22 **13 / 13** | ✅ 无回归 |
| 5 | conformance 推理 6 例 | **6 / 6** | 09-22 **6 / 6** | ✅ 无回归 |
| 6 | 职责响应审计（39 sub-part） | **39 / 0 / 0** | 09-28 **39 / 0 / 0** | ✅ 无回归 |
| 7 | 错误注入→恢复闭环 | **5 / 0 / 0** | 09-22 **5 / 0 / 0** | ✅ 无回归 |
| 8 | 分歧代价实验 · 功能等价性（M5） | **6 / 6 一致** | 本轮新增口径 | ✅ |
| 9 | 多流语义基线（S-1…S-13 子集） | **8 / 8** | 09-22 **8 / 8** | ✅ 无回归 |
| 10 | 多流配额 S-16 | **3 / 3** | 09-22 **3 / 3** | ✅ 无回归 |

> 第 1 项与第 2 项为**无设备**判据（`backend_offline_check.py` 以 `sys.meta_path` 阻断真实厂商运行时），
> 可在本机跑；其余 8 项为**真机**判据，在精简容器内跑。
> 两处口径不同：离线自检的判据数在 09-29 由 **37 增至 38**（新增「`state` 须为四态规范 token」；
> 另一条新增判据「外来码表不得影响本后端分类」**只对未声明 `error_map` 的后端生效**，
> ascend 声明了码表 ⇒ **不参与**，这也正是 38（ascend）/ 43（kunlun、cambricon）的差异来源）。

---

## 2 两处修复在 910C 上的定向验证

### 2.1 `state` 取值域（修复 ②）—— ✅ 已归一

| 判据 | 位置 | 实测值 |
|---|---|---|
| `recover_device()["state"]` 须为四态规范 token | 离线自检 | **`state='available'`** ✅（修复前为 `'DeviceState.AVAILABLE'`） |
| `recover_device()` 契约五键齐全 | 离线自检 | 5 键齐全 ✅ |
| `r_recovery` 用例（真机） | conformance 13 | **`评估=available`** · 隔离=True · 重建=True · 在途重放集合=True · 事件 `['isolated','available']` ✅ |

### 2.2 错误码表归属（修复 ①）—— ✅ 行为符合设计，ascend 不受影响

| 注入 | 类别 | disposition | 动作 | 期望核对 |
|---|---|---|---|---|
| `baseline`（无注入对照） | — | — | — | ✅ |
| `shape_mismatch` | `L2_PARAM` | `raise` | `raise_to_caller`（参数类，重试无意义） | ✅ |
| `oom` | `L1_RESOURCE` | `retry` | `retry_ok` | ✅ |
| `stream_timeout` | `L3_EXECUTION` | `replay` | `recover_probe=ok + replay_ok` | ✅ |
| **`l4_by_code`（本厂商码表触发）** | **`L4_FATAL`** | **`device_recovery`** | `recover_probe=ok` | ✅ |

**要点**：`l4_by_code` 一条在 910C 上**仍为 `L4_FATAL / device_recovery`**，这是**正确**的 ——
ascend **声明了 `error_map`**（109 条 ACL 码），注入的是**本厂商**码表内的码 ⇒ 走 `code_map`；
修复 ① 的 `vendor_codes=False` 只用于**未声明码表**的后端（kunlun / cambricon），
在那两家该条由 `L4_FATAL` 变为 **`L3_EXECUTION`**（见 P800 回归）。
⇒ **同一条判据在三家给出不同类别，是"码表归属"的正确表现，不是不一致。**

---

## 3 分歧代价实验（工作包 A）在 910C 的取数

`prototype/probes/exp_divergence_cost.py --backend ascend`（逐场景独立进程）

**M1–M4 静态度量**（与后端无关，与 P800 一致）

| 路径 | 分支数 M1 | 私有知识点 M2 | 上层代码行 M3 | 厂商 API 种类 M4 |
|---|---|---|---|---|
| ① 统一层 `runtime` | **0** | **0** | **77** | **0** |
| ② 直调厂商原生（允许按厂商分支） | **11** | **15** | **163** | **17** |

**M5 功能等价性：6 / 6 一致**

| 场景 | 期望 | 路径 ① | 路径 ② | 触发方式 |
|---|---|---|---|---|
| `S0_device_enum` | 值一致性 | 值一致 | 值一致 | real |
| `S1_param_out_of_range` | `L2_PARAM` + raise | `L2_PARAM` | `L2_PARAM` | real |
| `S2_bounded_sync_timeout` | `L3_EXECUTION` + replay | `L3_EXECUTION` | `L3_EXECUTION` | best-effort |
| `S3_out_of_memory` | `L1_RESOURCE` + retry | `L1_RESOURCE` | `L1_RESOURCE` | real |
| `S5_unknown_error` | `L3_EXECUTION` + replay | `L3_EXECUTION` | `L3_EXECUTION` | real |
| `S6_coded_error_ownership` | 按后端定：声明 `error_map` 者 `L4` | **`L4_FATAL`** | **`L4_FATAL`** | synthetic |

> 与 P800 对照：P800 **修复前** S6 是 **DIFF**（统一层 `L4_FATAL` vs 原生 `L3_EXECUTION`，
> 即本层缺陷），修复后 6/6。910C 因声明了码表，**修复前就应为 6/6**，本轮实测确认。

---

## 4 环境与"跑的是当前版本"核验（避免跑到旧副本）

| 核验项 | 实测 |
|---|---|
| 后端目录 | `runtime/backends/` = `{ascend, kunlun, cambricon}` —— **无 `flagos`**（路线 B 已退出的断链状态） |
| 注册表 | `_KNOWN_BACKENDS = ("ascend", "kunlun", "cambricon")` |
| 修复 ① 在位 | `runtime/conformance/errors.py` 含 `vendor_codes`（3 处） |
| 修复 ② 在位 | `runtime/backends/base.py` 含 `state_token` |
| 同步方式 | `rsync -az --delete prototype/ → /mnt/raid/hliu553/dc_regress_20260929/prototype/` |
| 容器 | 新建 `dc-lean-910c-20260929`，**只挂 `davinci1,2,3` + 3 个管理设备**（见 `ASCEND_HOST_NAMESLOT_RULE_20260929.md`） |
| 设备 | `acl.init rc=0` · `get_device_count=(3, 0)` · `torch.npu.device_count()=3` |

> ⚠️ 宿主既有目录 `dc_full/prototype` 仍是**旧版**（含 `flagos` 后端、无 `state_token`）
> ⇒ 本次建了带日期的干净目录，**不复用旧目录**，避免"命令跑通了但跑的是旧原型"。

---

## 5 本轮**未跑**的项（如实登记，不补零、不外推）

| 项 | 状态 | 理由 |
|---|---|---|
| 服务化（`serve_standard.sh`） | **未在本轮复跑** | 09-28 已按新脚本（v1.2，含 `SMOKE_TIMEOUT`）复跑 ⇒ `SERVE_STANDARD_PASS`（就绪 **35 s**、维度 1024、范数 1.000000、冒烟 0 s）；本轮两处修复**不触及服务路径**，且服务化需与训练容器**串行持卡** |
| 训练腿 / 推理腿前向 | **未在本轮复跑** | 同上：09-22 已取（训练腿 **6/6**、**4402.3 tok/s**；推理腿前向 **13/13**）；本轮修复不触及前向路径 |
| 多卡 TP / 关闭 `--enforce-eager` 的服务化形态 | **未跑** | 已知挂账（09-28 登记），非本轮范围 |
| `probe_device` 真值路径（离线自检第 1 跳过项） | **SKIP** | stub 无真实算子可跑；真机路径已由 conformance `r_recovery` 覆盖 —— **属能力边界，不是缺口** |
| `recover_device(mode="real")` 多卡多进程压测 | **未跑** | 已知挂账，与工作包 C 合并推进（`DEVICE_CONTEXT_GAP_CLOSURE_PLAN_20260929.md` §3） |

---

## 6 一键复跑

```bash
# 0) 起精简容器（只挂空闲卡；不要挂全 16 张 —— 见 ASCEND_HOST_NAMESLOT_RULE_20260929.md）
ssh 910C 'docker run -d --name dc-lean-910c-20260929 --network host --shm-size 64g \
  --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /mnt/raid/hliu553:/mnt/raid/hliu553 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64 sleep infinity'

# 1) 同步当前原型（不要复用 dc_full 等旧目录）
rsync -az --delete prototype/ 910C:/mnt/raid/hliu553/dc_regress_20260929/prototype/

# 2) 跑全套（脚本在宿主 dc_regress_20260929/ 下，逐项落 out/*.log 与 *.json）
ssh 910C 'docker exec -d dc-lean-910c-20260929 bash -lc \
  "bash /mnt/raid/hliu553/dc_regress_20260929/run_regress_ascend.sh > /mnt/raid/hliu553/dc_regress_20260929/run_all.log 2>&1"'

# 3) 收结果
ssh 910C 'cat /mnt/raid/hliu553/dc_regress_20260929/run_all.log'
```

> 备注：`run_regress_ascend.sh` 覆盖 8 项；冒烟自检（第 3 项）为单独一步：
> `python runtime/smoke_runtime.py --backend ascend`（实测 **52/0**）。

---

## 7 证据清单（`../probes/`）

| 文件 | 内容 |
|---|---|
| `regress_offline_ascend_20260929.log` | 离线契约自检 **38/0/1** |
| `regress_conf13_ascend_20260929.{json,log}` | conformance 基线 **13/13** |
| `regress_confinfer6_ascend_20260929.{json,log}` | conformance 推理 **6/6** |
| `regress_duty_ascend_20260929.{json,log}` | 职责响应审计 **39 OK / 0 FAIL / 0 SKIP** |
| `regress_errorloop_ascend_20260929.{json,log}` | 错误闭环 **5 / 0 / 0** |
| `regress_smoke_ascend_20260929.log` | 组件冒烟自检 **52/0** |
| `exp_divergence_cost_ascend_20260929.{json,log}` | 工作包 A 实验（M1–M5），等价性 **6/6** |
| `regress_stream_semantics_ascend_20260929.log` + `stream_semantics_full_result_ascend_20260929.json` | 多流语义 **8/8** |
| `regress_stream_quota_ascend_20260929.log` + `stream_quota_result_ascend_20260929.json` | 流配额 S-16 **3/3** |
| `nameslot_rule_matrix_ascend_20260929.log` | 宿主名额规则判别实验原始记录（5 数据点） |
