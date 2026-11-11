# MLU590（第 3 家实例）· 修复后全套回归 + 一处层内缺陷的发现与修复（2026-09-29）

> 定位：**实测记录**。回答两件事 ——
> ① 09-29 两处层内修复（`vendor_codes` 码表归属 / `state_token` 取值域）在第 3 家实例是否引入回归；
> ② 按同一套判据复验时**新发现的缺陷**是什么、怎么修的、修完是否闭合。
> 边界：结论只在**当前档位/环境**成立 —— 主机 `tza-0a06-ai01-em9`（Mlu-1）、容器 `dc-mlu590-hliu553`、
> `torch 2.7.1+cpu` / `torch_mlu 1.29.2+torch2.7.1`、neuware **4.4.3** 档、解释器 `/flagos/bin/python3`、
> 设备 `MLU_VISIBLE_DEVICES=0`（8 卡容器内取 1 张）。
> 原始留档：`../probes/regress_*_cambricon_20260929.*`、`exp_divergence_cost_cambricon_20260929.{json,log}`、
> 修复前：`exp_divergence_cost_cambricon_PRE_FIX_20260929.json`

---

## 1 结论先行

| 项 | 结果 |
|---|---|
| 首次复验（修复前原型） | **发现 1 处真实层内缺陷** —— 设备序号越界在寒武纪栈下被判成 **`L3_EXECUTION`（`replay`）**，契约期望 **`L2_PARAM`（`raise`）**；工作包 A 等价性 **5/6**、且本层**期望未达标** |
| 修法 | 共享消息规则表的 **L2 规则扩「措辞等价类」**（`invalid argument` / `invalid value` / `illegal …`），位于 `runtime/conformance/errors.py` |
| 修复后复跑 | **10 项判定全部通过**：离线自检 **45/0/0** · 对称性 **5/0** · 冒烟 **46/0** · conformance **13/13 + 6/6** · 职责审计 **36/0/3** · 错误闭环 **5/0/0** · 等价性 **6/6** · 多流语义 **8/8** · 配额 **3/3** |
| 新增判据 | 离线自检 **+2 条**（判据数 43 → **45**），并做**非空转验证**（回退规则 ⇒ 恰好该条 FAIL：44/1） |
| 两处既有修复在本实例的表现 | `state` 归一为 `'available'`（审计 E3 / 冒烟 `recover_device` 双处取证）✓；`l4_by_code` 由 `L4_FATAL/device_recovery` 变为 **`L3_EXECUTION/replay`** ✓（寒武纪**未声明** `error_map` ⇒ 不该用外来昇腾码表，这正是 09-29 修复的**预期效果**） |

---

## 2 新发现的缺陷：参数类错误的「措辞等价类」漏网

### 2.1 现象（工作包 A 实验 · S1 场景）

场景固定为「`set_device(设备数 + 100)` ⇒ 设备序号越界 ⇒ **参数类**，上抛不重试」，
期望值**先写死**（`exp_divergence_cost.py` 文件头 `_EXPECT` 表，禁止事后解释）。

| 路径 | 类别 | 处置 | graded_by |
|---|---|---|---|
| ① 统一层 `runtime` | **`L3_EXECUTION`** ❌ | `replay` | **`default`（兜底）** |
| ② 直接调厂商原生 | **`L2_PARAM`** ✅ | `raise` | `message_hint` |

两家厂商栈抛出的原文（真机抓取，非构造）：

| 实例 | 越界时厂商抛出的原文 | 统一层分级 |
|---|---|---|
| P800（kunlun） | `CUDA error: invalid device ordinal` | `L2_PARAM` ✅（命中 `invalid device`） |
| **MLU590（cambricon）** | **`CNRT error: invalid argument.`** | **`L3_EXECUTION`** ❌（**无任何规则命中**） |
| 910C（ascend） | 带数字码，走 `code_map` | `L2_PARAM` ✅ |

### 2.2 根因

`conformance/errors.py::_MESSAGE_HINTS` 的 L2 规则原本是：

```python
(re.compile(r"(invalid (device|ordinal|data|op|param))", re.I), ErrorCategory.L2_PARAM)
```

这条规则**是从个别厂商的文案反推出来的**（昇腾/昆仑芯的 `invalid device` 写法），
于是同一故障在别家栈下的**等价说法**会漏网：CNRT 说的是 `invalid argument`，
既不在上面的名词表里，也不含任何 L3 关键词 ⇒ 落到最后一条 **兜底 `L3_EXECUTION`**。

### 2.3 危害（与 09-29 已修的 `category` 缺陷同族）

`L3_EXECUTION` 的契约处置是 **`replay`（重放）**。设备序号越界是一个**永久性**参数错误，
无论重放多少次都会以同样方式失败 ⇒ **下游会对着一个不可能成功的操作反复重放**。
这正是本层最该消除的那类问题：**分级错了，下游动作就反**（与 09-29「外来码表把参数类错误升级成
`L4_FATAL`、误触发设备级重建」是同一危害家族的两个方向）。

### 2.4 修法（治本）

把 L2 规则从「按**名词**枚举」改为「按**等价类**覆盖」：

```python
(re.compile(r"(invalid (device|ordinal|data|op|param|arg(?:ument)?s?|value|index)"
            r"|illegal (?:arg(?:ument)?s?|value|param|device))", re.I), ErrorCategory.L2_PARAM)
```

- `invalid argument` / `invalid value` 是 **EINVAL 类调用方错误**的通用措辞 ⇒ 必须覆盖；
- **刻意不做** `invalid \w+` 宽匹配 —— 那会把 `invalid context` 一类可能的 L4 场景吞进 L2；
- 规则只按语义扩展，**不改判定标准**（期望值表仍未动）。

---

## 3 修复后复跑（同实例同判据）

| # | 判定项 | 修复前 | 修复后 | 历史对照（09-28） |
|---|---|---|---|---|
| 1 | 离线契约自检（无设备） | 43 / 0 / 0 | **45 / 0 / 0** | 39 / 0 / 0（判据数 39→43→45） |
| 2 | 跨后端对称性 `--all` | 5 / 0 | **5 / 0** | 5 / 0 |
| 3 | 组件冒烟自检 | 46 / 0 | **46 / 0** | 46 / 0 |
| 4 | conformance 基线 13 例 | 13 / 13 | **13 / 13** | 13 / 13 |
| 5 | conformance 推理 6 例 | 6 / 6 | **6 / 6** | 6 / 6 |
| 6 | 职责响应审计（39 sub-part） | 36 / 0 / 3 | **36 / 0 / 3** | 36 / 0 / 3 |
| 7 | 错误注入→恢复闭环 | 5 / 0 / 0 | **5 / 0 / 0** | 5 / 0 / 0 |
| 8 | 工作包 A 功能等价性（M5） | **5 / 6**（S1 DIFF） | **6 / 6 一致** | 本轮新增口径 |
| 9 | 多流语义（S-1…S-13 子集） | 8 / 8 | **8 / 8** | 8 / 8 |
| 10 | 多流配额 S-16 | 3 / 3 | **3 / 3** | 3 / 3 |

### 3.1 两处既有修复在本实例的定向验证

| 修复 | 判据 | 实测 |
|---|---|---|
| ② `state` 取值域 | 离线自检「`recover_device()["state"]` 须为四态规范 token」 | **`state='available'`** ✅（此前为 `'DeviceState.AVAILABLE'`） |
| ② `state` 取值域 | 冒烟「`recover_device` 返回 dict」 | `{'ordinal':0,'mode':'probe','recovered':True,'state':'available',…}` ✅ |
| ① 码表归属 | 离线自检「外来码表不得影响本后端分类」 | `L3_EXECUTION` / `graded_by=default` ✅（**不再是 L4**） |
| ① 码表归属 | 错误闭环 `l4_by_code`（码表内他厂码·诚实降级负向测试） | **`L3_EXECUTION / replay`** ✅（修复前为 `L4_FATAL / device_recovery`） |
| ① 码表归属 | 工作包 A `S6_coded_error_ownership` | 两条路径均 `L3_EXECUTION` ✅（期望=按后端定） |

> 第 3 家的 `l4_by_code` 与 910C 不同（910C 仍 `L4_FATAL`），**这是正确表现**：
> 910C **声明了** `error_map` ⇒ 注入的是本厂商码，走 `code_map`；寒武纪**未声明** ⇒ 只能兜底 L3。

### 3.2 职责审计的 3 项 SKIP（如实登记，非缺口）

| 项 | 原因 |
|---|---|
| `C13 event.elapsed_time(end)` | 能力未声明（契约外**可选**能力） |
| `D5 L1–L4 处置映射符合处置约定表` | 本后端**无数字错误码**（未声明 `error_map`）⇒ 无从构造码样例；已由错误闭环与 conformance F1 覆盖 |
| `F1 错误隔离分层` | 同上：芯片级 L4 需数字码触发，本后端不具备 |

---

## 4 新增判据与非空转验证

**判据（`scripts/backend_offline_check.py`，对三家后端均生效）**：

```
[PASS] 参数类文案等价类 ⇒ L2_PARAM：'CUDA error: invalid device ordinal'（P800 真机原文 · S1 设备序号越界）
[PASS] 参数类文案等价类 ⇒ L2_PARAM：'CNRT error: invalid argument.'（MLU590 真机原文 · S1 设备序号越界）
```

- 输入用**两家真机抓到的原文**（不发明文案）；**只碰文本、不碰设备** ⇒ 属于"无设备先自查"能拦住的场合
  （该缺陷本可以在上机前被拦住 —— 与 2026-09-22「码表内码串入」那条判据同一教训）。
- **非空转验证**：把 L2 规则回退为旧版后，**恰好该条 FAIL**（`44 通过 / 1 失败`），
  另一条（P800 措辞）仍 PASS ⇒ 判据确有判别力。

**判据数变化**：`ascend 38 → 40` · `kunlun 43 → 45` · `cambricon 43 → 45`（各 +2）。

---

## 5 「跑的是当前版本」核验

| 核验项 | 实测 |
|---|---|
| 同步方式 | `rsync -az --delete prototype/ → /srv/hliu553/dc_regress_20260929/prototype/`（容器内 `/work/…`） |
| 后端目录 | `{ascend, kunlun, cambricon}` —— **无 `flagos`** |
| 修复 ① 在位 | `runtime/conformance/errors.py` 含 `vendor_codes`（3 处） |
| 修复 ② 在位 | `runtime/backends/base.py` 含 `state_token` |
| 新修复在位 | `errors.py` 含 `arg(?:ument)?s?` 等价类 |
| ⚠️ 未复用旧目录 | 既有 `/work/prototype` 是**修复前旧副本**（`state_token`/`vendor_codes` 命中数均为 **0**）⇒ 本次建带日期的干净目录，**不复用** |

---

## 6 本轮**未跑**的项（如实登记，不补零）

| 项 | 状态 | 理由 |
|---|---|---|
| 训练腿 / 推理腿前向 / 服务化 | **未在本轮复跑** | 09-28 已取（训练腿 **6/6 / 3015.3 tok/s**、`dist=cncl`；推理腿前向 **13/13**；服务化 `SERVE_STANDARD_PASS`，就绪 150 s）—— 见 `CAMBRICON_MLU_INFER_LEG_VERIFY_20260928.md`；本轮三处修复均**不触及**前向与服务路径 |
| 多卡 TP / 关闭 `--enforce-eager` 的服务化形态 | **未跑** | 已知挂账（09-28 登记） |
| `S13` 多设备流绑定 | **SKIP** | 本轮取 1 张卡（`MLU_VISIBLE_DEVICES=0`）⇒ 探针如实报「可见设备数=1<2，跳过」 |
| `recover_device(mode="real")` 多卡多进程压测 | **未跑** | 已知挂账，与工作包 C 合并推进 |

---

## 7 一键复跑

```bash
# 0) 同步当前原型（不要复用 /work/prototype 等旧副本）
rsync -az --delete prototype/ Mlu-1:/srv/hliu553/dc_regress_20260929/prototype/
# 1) 全套 10 项（脚本在宿主 /srv/hliu553/dc_regress_20260929/ 下）
ssh Mlu-1 'docker exec -d dc-mlu590-hliu553 bash -lc \
  "bash /work/dc_regress_20260929/run_regress_cambricon.sh > /work/dc_regress_20260929/run_all.log 2>&1"'
ssh Mlu-1 'cat /srv/hliu553/dc_regress_20260929/run_all.log'
```

---

## 8 证据清单（`../probes/`）

| 文件 | 内容 |
|---|---|
| `regress_offline_cambricon_20260929.log` | 离线契约自检 **45/0/0**（含 2 条新判据） |
| `regress_symmetry_all_20260929.log` | 跨后端对称性 **5/0** |
| `regress_smoke_cambricon_20260929.log` | 组件冒烟 **46/0** |
| `regress_conf13_cambricon_20260929.{json,log}` | conformance 基线 **13/13** |
| `regress_confinfer6_cambricon_20260929.{json,log}` | conformance 推理 **6/6** |
| `regress_duty_cambricon_20260929.{json,log}` | 职责响应审计 **36 OK / 0 FAIL / 3 SKIP** |
| `regress_errorloop_cambricon_20260929.{json,log}` | 错误闭环 **5 / 0 / 0** |
| `regress_stream_semantics_cambricon_20260929.log` + `stream_semantics_full_result_cambricon_20260929.json` | 多流语义 **8/8** |
| `regress_stream_quota_cambricon_20260929.log` + `stream_quota_result_cambricon_20260929.json` | 流配额 S-16 **3/3** |
| `exp_divergence_cost_cambricon_20260929.{json,log}` | 工作包 A 实验，**修复后等价性 6/6** |
| ⭐ `exp_divergence_cost_cambricon_PRE_FIX_20260929.json` | **修复前**同一实验（**5/6，S1 DIFF + 期望未达标**）—— 缺陷原始证据，原样留档 |
