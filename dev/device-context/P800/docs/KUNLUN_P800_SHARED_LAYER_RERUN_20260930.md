# P800 · 共享层改动后的全套复跑（2026-09-30 · r6）

> 定位：**复跑记录**（不是"修复"——P800 的 B/C 接口本身早已通过，见 §1）。
> 目的：共享层在 P800 最后一次全套覆盖之后又被改过 3 轮，按「**按破坏面覆盖**」纪律必须复跑。
>
> 设备：P800（`VM-0-2-ubuntu`，XPU-RT **5.0.21.47** / Driver 5.0.21）·
> 容器 `hliu553-device-context-p800` · conda `python310_torch29_cuda`
> （py3.10 / **torch 2.9.0+cu129** / transformers 4.57.1 / **vllm 0.13.0**）· **单卡 dev4**。
>
> **边界**：结论只在本档位/环境成立；不外推到其他实例或多机。

---

## 1 为什么要复跑（覆盖缺口是可核对的）

| 实例 | 最后一次"全套"覆盖 | 之后未被覆盖的共享层改动 |
|---|---|---|
| 910C | 本轮 r8（9 项全绿） | —— |
| **P800（本轮之前）** | r5（含契约不变式） | **`7225889`**（`serve_standard.sh`）· **`752551b`**（6 个共享层文件：A2 三处缺陷修复）· **`2647370`**（10 个：补公开入口）· **`57cb9dd`**（4 个：`info()` 收口 + 2 条判据）· 本轮（B2 探针） |

⇒ 这些改动**从未在 P800 上跑过**，本次一次性覆盖。

---

## 2 选卡（避开已知故障卡）

| smi 索引 | UUID | 显存 / util | 处置 |
|---|---|---|---|
| 0 | `ed733bb8…` | 1970 MiB / 0% | 备用（训练腿第 2 卡） |
| **1** | **`b3509946…`** | **0 MiB / 0%** | ⛔ **已知故障卡，禁用**（与好卡同貌 ⇒ **看占用挑不出它**，靠 UUID 识别） |
| 2 · 3 · 5 · 6 | — | 20–24 GB / 96–100% | 他人作业，不动 |
| **4** | `b7942319…` | **292 MiB / 0%** | ✅ **本轮主卡** |
| 7 | `4922595e…` | 75.6 GB | 他人占用 |

**选卡后先跑最小探针**（纪律）：dev4 上 `x@tril` 得 **89440**（正确）、event 立即就绪、显存 0.45/96 GiB ⇒ 功能正常。

---

## 3 判定汇总（11 项）

| # | 判定项 | 结果 |
|---|---|---|
| 1 | 离线契约自检 | **85 / 0 / 1 跳过** |
| 2 | 跨后端对称性 `--all` | **7 / 0**（含 2026-09-29 新增的 2 条 `info()` 判据） |
| 3 | 组件冒烟（真机） | **46 / 0** |
| 4 | conformance 13 + 推理 6 | **13/13 + 6/6**（`CONFORMANCE_PASS`） |
| 5 | 契约不变式 I1–I4 | **4/4**（`CONTRACT_INVARIANTS_PASS`） |
| 6 | 职责响应审计（39 sub-part） | **36 OK / 0 FAIL / 3 SKIP**（`DUTY_RESPONSE_PASS`） |
| 7 | 错误注入 → 恢复闭环 | **5 闭环 / 0 跳过 / 0 失败** |
| 8 | B/C 契约探针 | **PASS**（`"PASS": true`） |
| 9 | **公开入口定向验证**（新） | **`ENTRY_VERIFY_PASS`**（修判据后，见 §4） |
| 10 | A2 多卡多进程 `real` 压测 | **`SKIP_UNSUPPORTED`** —— P800 未声明 `recovery_real` ⇒ **如实跳过、不计失败**（设计如此） |
| 11 | **两条腿 + 服务化** | 训练腿 **`TRAIN_LEG_PASS 6/6`** · 推理腿 **`INFER_LEG_PASS 13/13`** · 服务化 **`SERVE_STANDARD_PASS`** |

**两条腿的数值（与历史可比）**：

| 项 | 本轮 | 历史对照 |
|---|---|---|
| 训练腿（2 卡 dev4,0 · 50 步） | `TRAIN_LEG_PASS 6/6` · loss **15.4488 → 11.1481** · **3450.4 tok/s**（两卡合计）· `dist=cpu:gloo,cuda:flagcx` | 09-14 记录同款 loss、3482 tok/s ⇒ **无回归** |
| 推理腿（单卡 dev4） | `INFER_LEG_PASS 13/13` · dim 1024 · 49.24 句/s · p50 59.42 ms · 区分度 **0.6392** | 与 910C 的 14/14 差 1 项：**如实跳过 `vendor_code_map`**（昆仑芯无厂商码表）——**这是如实，不是漏测** |
| 服务化（embedding · dev4 · TP=1 · EAGER=1） | **`SERVE_STANDARD_PASS`**（ready=1 smoke=1）· 停机后释放复查等待 **0 s** ⇒ 新 `RELEASE_WAIT` 逻辑生效、无假信号 | 首次在 P800 上跑**含 `RELEASE_WAIT` 的版本** |

---

## 4 ⭐ 本轮抓到并修掉的 1 处判据缺陷（跨实例口径不一致）

**现象**：`probes/recover_entry_verify.py` 在 P800 上首跑 **`ENTRY_VERIFY_FAIL`**：

```
FAIL  D2_公开入口置隔离后real_必须声明真重建  context_recreated=False
FAIL  D2_重建后回到 available（R4）            post='isolated'
```

**根因**：该探针是我在 **910C（声明了 `recovery_real`）** 上写的，D2 直接假设"real 必然真重建"。
而 P800 **未声明 `recovery_real`** ⇒ 后端**如实拒绝**（`detail` 明写"昆仑芯无设备级重置/重建原语…
real 模式不支持"）⇒ 状态如实停在 `isolated`。

**同一个情形，两个探针口径不一致**：

| 探针 | 对"未声明 recovery_real"的处置 |
|---|---|
| `recover_multiproc_stress.py`（A2） | ✅ **`SKIP_UNSUPPORTED`**，不计失败（既有设计） |
| `recover_entry_verify.py`（D2） | ❌ **判 FAIL**（缺陷） |

**修法**（只增判据，不改既有口径）：
1. 未声明 `recovery_real` ⇒ D2 主判据 **如实跳过**（与 A2 同口径），并加 `skip()` 机制（`skipped=True`、不计失败、留痕）；
2. 补上**该情形下真正适用的两条**判据 —— 否则"跳过即不管"：
   - `D2_未声明 real ⇒ 必须显式拒绝且不得声称已重建`（`context_recreated is False`）；
   - **`D2_账本可显式复原到 available（未声明 real 时的唯一退路）`** —— 实测 `set_device_state(dev,"available")`
     返回 `'available'` 且读回一致 ⇒ **证明了"置隔离后仍有退路"，不会把设备永久卡在隔离态**。

**修后双向复验（探针是共享资产，按破坏面两边都跑）**：

| 实例 | 结果 | 覆盖到的分支 |
|---|---|---|
| **P800**（未声明 real） | **`ENTRY_VERIFY_PASS`** | 跳过 1 条 + 拒绝/复原 2 条（`D2_账本可显式复原` PASS） |
| **910C**（声明 real） | **`ENTRY_VERIFY_PASS`** | 真重建分支完整：D2 `context_recreated=True` + R4 回 `available` + D4 端到端 `steps=['captured','evaluated: isolated','recovered: True','replay_ready']` |

⭐ 这是**跨实例复跑的直接价值**：一个只在 910C 上写、只在 910C 上跑过的判据，**在第二家上立刻暴露口径不一致**。

---

## 5 操作要点（下次少走弯路）

| 坑 | 事实 |
|---|---|
| 宿主/容器路径必须分开 | P800 上 `/data2/hliu553` = 容器内 `/workspace`；脚本在**宿主**执行时重定向要用**宿主路径**，给容器程序的 `--out` 要用**容器路径**（首次跑全 rc=1 就是这个原因） |
| 训练腿必须给 `DC_ROOT` | 脚本用 `DC_ROOT` 把原型根加进 `sys.path`；不给 ⇒ `ModuleNotFoundError: No module named 'runtime'` |
| 训练腿要在 `runtime/proto/` 下起 | 直接 `python3 -m torch.distributed.run … proto_train_leg.py` 必须 `cd` 到脚本目录，否则 `can't open file` |
| 单卡探针不必给 `DEV` | 训练/推理靠 `CUDA_VISIBLE_DEVICES`；服务化才用 `DEV=<n>` |
| `dev1` 永远避开 | 故障卡 `b3509946`（`0 MiB / 0%` 与好卡同貌） |

---

## 6 一键复跑

```bash
# 主回归（宿主上执行）
bash /data2/hliu553/dc_regress_20260929/run_p800_r6.sh

# 两条腿 + 服务化（宿主上执行，已含正确的 cd / DC_ROOT）
bash /data2/hliu553/dc_regress_20260929/run_p800_legs.sh
```

证据（`probes/`，21 份）：`r6_{offline,symmetry_all,smoke,conf13,confinfer6,coninvariants,duty,errorloop,bc_probe,entry_verify,stress}_kunlun_20260930.{log,json}` ·
`legs_{train,infer}_kunlun_20260930.log` · `serve_standard_kunlun_20260930.log` ·
`run_p800_{r6,legs}.log`

---

## 7 未做 / 边界（不补零）

| 项 | 状态 |
|---|---|
| P800 的 B/C 能力缺口 | **无新增**：B1/B3/B4 通过；C 因**平台约束**（每设备一个 primary context 且由框架自建）如实为**只读** `context_query`（本轮未重跑该探针的 C4 组，r2 轮已通过） |
| MLU590 | ⬜ 本轮**未涉及**（B/C 首轮取数未做，需其自身窗口） |
| 多卡 `real` 恢复压测 | ⬜ P800 **不适用**（未声明 `recovery_real`）⇒ 如实跳过 |
| 服务化其他形态（TP=2 / 非 eager） | ⬜ 本轮只跑 embedding 基线形态（与 910C 的 4 形态覆盖不同口径，如实标注） |
