# P800（第 2 家实例）· 修复后全套回归 + 一处「卡级环境问题」的判别（2026-09-29）

> 定位：**实测记录**。回答两件事 ——
> ① 09-29 三处层内修复（`vendor_codes` 码表归属 / `state_token` 取值域 / **L2 文案等价类**）
> 在 P800 是否引入回归；
> ② 首轮回归出现「冒烟挂死」，到底是**本层回归**还是**环境（卡）问题**。
> 边界：结论只在**当前档位/环境**成立 —— 主机 `VM-0-2-ubuntu`、容器 `hliu553-device-context-p800`、
> conda `python310_torch29_cuda`（torch 2.9 / XPU-RT 5.0.21 / xpu3.6 档）、
> `CUDA_VISIBLE_DEVICES=4`（**卡 4**，已核空闲且功能正常）、`XPU_EVENT_KL3_ENABLE=1`。
> 原始留档：`../probes/regress_*_kunlun_20260929_r2.*`、`exp_divergence_cost_kunlun_20260929_r2.*`、
> ⚠️ `../probes/DIAG_kunlun_card1_event_hang_20260929.log`

---

## 1 结论先行

| 项 | 结果 |
|---|---|
| 回归结论 | **10 项判定全部通过，无回归**：离线自检 **45/0/1** · 对称性 **5/0** · 冒烟 **46/0** · conformance **13/13 + 6/6** · 职责审计 **36/0/3** · 错误闭环 **5/0/0** · 等价性 **6/6** · 多流语义 **8/8** · 配额 **3/3** |
| 首轮「冒烟挂死」的判定 | **卡 1 的环境问题，与本层代码无关** —— 详见 §3（同卡用**修复前**原型复现同一行挂死；换卡 4 全绿） |
| 三处修复在本实例的表现 | `state='available'` ✓；`l4_by_code` 为 **`L3_EXECUTION/replay`** ✓（09-29 已修）；**S1 参数越界 = `L2_PARAM`** ✓（本轮新修，P800 侧本就正确，未变） |

---

## 2 修复后回归结果

| # | 判定项 | 本轮（09-29 r2） | 历史对照 |
|---|---|---|---|
| 1 | 离线契约自检（无设备） | **45 / 0 / 1 跳过** | 09-29 r1 **43 / 0 / 1**（判据数 43→45） |
| 2 | 跨后端对称性 `--all` | **5 / 0** | 09-29 r1 5 / 0 |
| 3 | 组件冒烟自检 | **46 / 0** | 09-22 **46 / 0** |
| 4 | conformance 基线 13 例 | **13 / 13** | 09-29 r1 13 / 13 |
| 5 | conformance 推理 6 例 | **6 / 6** | 09-29 r1 6 / 6 |
| 6 | 职责响应审计（39 sub-part） | **36 / 0 / 3** | 09-28 36 / 0 / 3 |
| 7 | 错误注入→恢复闭环 | **5 / 0 / 0** | 09-29 r1 5 / 0 / 0 |
| 8 | 工作包 A 功能等价性（M5） | **6 / 6 一致** | 09-29 r1 6 / 6 |
| 9 | 多流语义（S-1…S-13 子集） | **8 / 8** | 09-22 8 / 8 |
| 10 | 多流配额 S-16 | **3 / 3** | 09-22 3 / 3 |

> 命名口径：**无后缀** = 09-29 第 1 轮（`vendor_codes` / `state_token` 修复后）；
> **`_r2`** = 09-29 第 2 轮（**L2 文案等价类**修复后，即本报告）。两轮都保留，**不覆盖**。

**定向验证（三处修复）**

| 修复 | 判据 | 实测 |
|---|---|---|
| ① 码表归属 | 离线自检「外来码表不得影响本后端分类」 | `L3_EXECUTION` ✅ |
| ② `state` 取值域 | 离线自检「`state` 须为四态规范 token」 | **`state='available'`** ✅ |
| ③ L2 文案等价类 | 离线自检「参数类文案等价类」× 2 | 两条均 **PASS**（`L2_PARAM` / `message_hint`）✅ |
| ③ 行为侧 | 工作包 A `S1_param_out_of_range` | 路径①=路径②=**`L2_PARAM`** ✅ |

> 注：`S1` 在 P800 上**本来就正确**（昆仑芯文案 `CUDA error: invalid device ordinal` 命中既有规则），
> 本轮新判据对 P800 是**防回归**性质；真正暴露缺陷的是 MLU590（见
> `../../MLU590/docs/CAMBRICON_MLU_REGRESS_AFTER_FIX_20260929.md`）。

---

## 3 首轮「冒烟挂死」的判别（结论：卡 1 的环境问题）

### 3.1 现象

同一条命令 `python3 runtime/smoke_runtime.py --backend kunlun`：

| 设备 | 结果 |
|---|---|
| `CUDA_VISIBLE_DEVICES=1` | `[FAIL] Event.record + wait_host 有界返回` → 卡在 `probe_device`，**90 s 无返回**（faulthandler 强制 dump 栈） |
| `CUDA_VISIBLE_DEVICES=4` | **全绿 `46 通过 / 0 失败`**（`Event.record + wait_host`、`probe_device(0)`、`recover_device` 均 PASS） |

### 3.2 单变量对照（排除"本轮修复引入"）

用**修复前**的原型副本 `/workspace/prototype`（已核：不含本轮新增的 L2 等价类规则，`grep` 命中数 = **0**）
在**同一张卡 1** 上跑同一条命令 ⇒ **同一行挂死**：

```
  [FAIL] Event.record + wait_host 有界返回
Timeout (0:01:00)!
  File "/workspace/prototype/runtime/backends/kunlun/backend.py", line 245 in probe_device
```

两个版本的**唯一变量**是原型版本，结果相同 ⇒ **挂死与 09-29 的修复无关**。
另：把 `XPU_EVENT_KL3_ENABLE` 关掉重跑**同样挂死** ⇒ **也不是 KL3**。

### 3.3 处置与新增纪律

1. **避开卡 1**，改用卡 4（`xpu-smi` 显示空闲，且**先用最小冒烟验证该卡功能正常**）；
2. **未**把该现象记成层内缺陷，**未**改动任何判据（不"改判据变绿"，也不"改判据变红"）；
3. ⭐ **方法学（本轮新增）**：**`xpu-smi` 显示空闲 ≠ 该卡功能正常**。
   本卡显示 `0 MiB / 0%` 却无法完成 `Event.record + wait_host`。
   ⇒ **选卡后先跑一个最小 Event / 同步探针，再正式取数**；
   否则会把环境问题记成代码缺陷（与 2026-09-22「把卡被他人占用误判为通信库缺陷」同源，属同一纪律的加强）。

---

## 4 本轮**未跑**的项（如实登记，不补零）

| 项 | 状态 | 理由 |
|---|---|---|
| 训练腿 / 推理腿 / 服务化 | **未在本轮复跑** | 历史结论见 `KUNLUN_P800_STAGE34_VERIFY_20260920.md` 与本目录 README；本轮三处修复**不触及**前向与服务路径 |
| 图捕获（契约内 4/4） | **未在本轮复跑** | 非本轮改动面；历史数据保留 |
| `S13` 多设备流绑定 | 见探针输出 | 单卡口径下按探针自身判据处理 |

---

## 5 一键复跑

```bash
# 0) 同步当前原型（不要复用旧副本）
rsync -az --delete prototype/ P800:/data2/hliu553/dc_regress_20260929/prototype/
# 1) 全套 10 项（脚本在宿主 /data2/hliu553/dc_regress_20260929/ 下）
ssh P800 'docker exec -d hliu553-device-context-p800 bash -lc \
  "bash /workspace/dc_regress_20260929/p800_seq.sh > /workspace/dc_regress_20260929/p800_seq.log 2>&1"'
ssh P800 'cat /data2/hliu553/dc_regress_20260929/p800_seq.log'
```

> ⚠️ 复跑前**务必换卡**（脚本里已写死 `CUDA_VISIBLE_DEVICES=4`）；卡 1 当前不可用于 Event/同步类测试。

---

## 6 证据清单（`../probes/`）

| 文件 | 内容 |
|---|---|
| `regress_offline_kunlun_20260929_r2.log` | 离线契约自检 **45/0/1**（含 2 条新判据） |
| `regress_symmetry_all_20260929_r2.log` | 跨后端对称性 **5/0** |
| `regress_smoke_kunlun_20260929_r2.log` | 组件冒烟 **46/0** |
| `regress_conf13_kunlun_20260929_r2.{json,log}` | conformance 基线 **13/13** |
| `regress_confinfer6_kunlun_20260929_r2.{json,log}` | conformance 推理 **6/6** |
| `regress_duty_kunlun_20260929_r2.{json,log}` | 职责响应审计 **36 OK / 0 FAIL / 3 SKIP** |
| `regress_errorloop_kunlun_20260929_r2.{json,log}` | 错误闭环 **5 / 0 / 0** |
| `regress_stream_semantics_kunlun_20260929_r2.log` + `stream_semantics_full_result_kunlun_20260929_r2.json` | 多流语义 **8/8** |
| `regress_stream_quota_kunlun_20260929_r2.log` + `stream_quota_result_kunlun_20260929_r2.json` | 流配额 S-16 **3/3** |
| `exp_divergence_cost_kunlun_20260929_r2.{json,log}` | 工作包 A 实验，等价性 **6/6** |
| ⭐ `DIAG_kunlun_card1_event_hang_20260929.log` | **卡 1 挂死判别记录**（A/B/C 三组原始关键行 + 调用栈） |
