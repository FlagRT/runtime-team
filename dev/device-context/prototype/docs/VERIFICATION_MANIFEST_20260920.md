# 两实例验证复核清单（Verification Manifest）

> 日期：2026-09-20 ｜ 维护：Kistich（hliu553）｜ 性质：**外部复核入口**（第三家接入者 / 验收方 / 后续维护者按此逐行复现）
> 用途：把"我们声明过什么"与"证据在哪、怎么复跑、当前是否齐备"一次说清。
> **本文件只回答可复核性，不重复结论**；结论性数字见各实例 README 与验证报告。

---

## 1. 复核最小可执行清单（10 条）

在**容器内**、目标实例的 `prototype/` 目录下执行；`$B` = 后端名（`ascend` | `kunlun` | `cambricon`）。

| # | 声明 | 命令 | 通过判据 |
|---|---|---|---|
| 1 | 组件自检 | `python3 runtime/smoke_runtime.py --backend $B` | 全部通过、0 失败 |
| 2 | 一致性判据 13 例 | `cd runtime/conformance && python3 runner.py --backend $B` | `CONFORMANCE_PASS 13/13` |
| 3 | 一致性判据 推理 6 例 | `python3 runner.py --backend $B --cases infer_cases` | `CONFORMANCE_PASS 6/6` |
| 4 | 执行语义基线 | `python3 probes/probe_stream_semantics_full.py --leak-iters 1000` | `STREAM_SEMANTICS_PASS 8/8` |
| 5 | 推理腿 · 前向 | `DC_BACKEND=$B DC_MODEL=<模型路径> DC_OUT_DIR=<输出目录> python3 runtime/proto/proto_infer_leg.py` | `INFER_LEG_PASS`，并记录维度 / 语义区分度 / p50 |
| 6 | 推理腿 · 服务化 | `DC_BACKEND=$B MODEL=<模型路径> STOP_AFTER=1 bash scripts/serve_standard.sh` | `SERVE_STANDARD_PASS (ready=1 smoke=1)` |
| 7 | 训练腿 · 2 卡 | `DC_BACKEND=$B torchrun --standalone --nproc_per_node=2 runtime/proto/proto_train_leg.py` | `TRAIN_LEG_PASS 6/6`，并记录 loss 与 tok/s |
| 8 | 错误注入 → 恢复闭环 | `python3 runtime/proto/proto_error_recovery_loop.py --backend $B` | `ERROR_RECOVERY_LOOP_PASS`，记录 闭环/跳过/失败 |
| 9 | 契约不变式（**待补判据**） | `python3 runtime/conformance/runner.py --backend $B --cases contract_invariants` | I1–I4 全绿 —— ⬜ **尚未实现，见 §3 G8** |
| 10 | ⭐ **逐芯片职责验收（发布判定）** | 见 `PROTOTYPE_ACCEPTANCE_3CHIP_20260922.md` §6（10 条命令；或直接跑上表 1–8） | 10 项判定全绿（含多流 3 探针与两条腿）。**09-22 傍晚实测：910C 10/10 · P800 10/10 ⇒ 可发布**；MLU590 主机不可达，未参与 |

> 统一服务启动见 `docs/SERVICE_STARTUP_STANDARD_20260920.md`；接入流程见 `docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`。

---

## 2. 两实例证据索引（当前结论 = 哪一份）

### 2.1 第一实例（昇腾，已完成）

| 声明 | 当前结论证据 | 自证信息 | 状态 |
|---|---|---|---|
| 组件自检 | `prototype/runtime/` 侧历史结果（37 项含跳过） | — | ✅ 有，但**判据项数早于现版本** |
| 一致性判据 13 例（本家后端） | `prototype/runtime/conformance/conformance_runtime_ascend.json` | 含 `backend` | ✅ |
| 一致性判据 13 例（第三后端线） | `prototype/runtime/conformance/conf_proto_flagos_13.json` | 含 `backend` | ✅ |
| 一致性判据 推理 6 例 | `prototype/runtime/conformance/conformance_runtime_infer.json` | 含 `backend` | ✅ |
| **推理腿 · 前向** | 历史：`prototype/runtime/proto/proto_infer_leg_result.json`（旧格式）→ **当前结论：`910C/probes/recheck_infer_leg_ascend_20260922.json`** | 含 `backend=ascend` / `env{python 3.11.15, device_type=npu, device_count=1}` / p50 / p90；checks **14 项** | ✅（2026-09-22 对称复跑补齐） |
| **训练腿 · 2 卡** | 历史：`prototype/runtime/proto/train_leg_result_rank{0,1}.json`（旧格式）→ **当前结论：`910C/probes/recheck_train_leg_ascend_20260922_rank{0,1}.json`** | 含 `backend=flagos` / `dist_backend=flagos`（与 P800 侧格式对称：均有 backend+dist、均无 env） | ✅（2026-09-22 对称复跑补齐） |
| 推理腿 · 服务化 | `prototype/runtime/proto/proto_infer_serve_result.json` + 实例侧服务日志 | 含 `backend=ascend` | ✅ |
| 错误闭环 | `prototype/runtime/proto/error_recovery_loop_ascend.json` | 含 `backend` + `timestamp` | ✅ |
| 执行语义基线 | `910C/distributed_training/ascend_regression/stream_semantics_full_result.json` | `STREAM_SEMANTICS_PASS` | ✅ |
| 服务启动标准 | `910C/probes/L_serve_standard_910c_20260920.log` | 含 backend / 参数 / 时间 | ✅ |

> ⚠️ **历史快照并存**：同类结果在第一实例目录下存在多份（`ascend_regression/results/`、`results_aline_20260826/`、`results_inference_20260831/`，如 `double_buffer_result.json` 出现 3 份）。
> **本清单以本节"当前结论"列为准**，其余视为历史归档。⇒ 见 **G3**。

### 2.2 第二实例（昆仑芯，已完成）

| 声明 | 当前结论证据 | 自证信息 | 状态 |
|---|---|---|---|
| 一致性判据 13 例 | `prototype/runtime/conformance/conformance_runtime_kunlun.json`；镜像变体对照 `P800/probes/I_base_conformance_13_20260920.json` | 含 `backend` | ✅ |
| 一致性判据 推理 6 例 | `prototype/runtime/conformance/conformance_runtime_kunlun_infer.json`；`P800/probes/I_base_conformance_infer6_20260920.json` | 含 `backend` | ✅ |
| 组件自检 | `P800/probes/I_base_smoke_20260920.log` | 含 env | ✅ |
| 推理腿 · 前向 | `P800/probes/F_infer_leg_result_20260920.json`（+ 镜像变体 `I_base_infer_leg_*`） | 含 `backend` / `env`（python、device_type、device_count、capabilities）/ `skipped`；checks **13 + 1 跳过** | ✅ |
| 推理腿 · 服务化 | `P800/probes/F2_serve_result_20260920.json` 等 4 份（含单变量对照） | 含 `backend` | ✅ |
| 训练腿 · 2 卡 | `P800/probes/E_train_leg_result_rank{0,1}.json`（+ 镜像变体 `I_base_train_leg_*`，+ 交替复测 `I_ref_train_leg_*`） | 含 `backend` / `env` | ✅ |
| 错误闭环（两设置对照） | `P800/probes/error_recovery_loop_kunlun_KL3{off,on}.json` | 含 `backend` | ✅ |
| 执行语义基线 | `P800/probes/K_stream_semantics_full_result_p800_20260920.json` | 含 `backend` / 设备 API | ✅ |
| 镜像等价性对照 | `P800/probes/I_base_*`（共 17 份）+ `I_base_kl3_ab_*` | 含镜像标识 | ✅ |
| 服务启动标准 | `P800/probes/L_serve_standard_p800_v2_20260920.log` | 含 backend / 参数 / 时间 | ✅ |

---

## 3. 缺口清单（G）

| # | 缺口 | 影响 | 修补动作 | 状态 |
|---|---|---|---|---|
| **G1** | 第一实例**两腿结果 JSON 无自证信息**（无 `backend` / `env` / 时间戳） | "跨厂商可比"的证据链在证据层**无法自证** | 用后端无关化脚本（V2）复跑两腿 | ✅ **2026-09-22 已关闭**：推理腿 `recheck_infer_leg_ascend_20260922.json`（14/14，含 backend+env+p50/p90）；训练腿 `recheck_train_leg_ascend_20260922_rank{0,1}.json`（6/6，backend=flagos，loss 15.4497→11.1515 与历史逐位吻合） |
| **G2** | 判据项数不对称（第一实例 10 项 vs 第二实例 13+1 项） | "逐项对照"实为**新判据对旧结果** | 同上（复跑即对齐） | ✅ **已关闭**：两侧判据项数对齐（推理腿 14 键 vs 14 键） |
| **G3** | 第一实例同类证据存在**多份历史快照** | 复核者可能取错版本 | 本清单 §2.1 已标注"当前结论"；旧快照保留但不再是引用目标 | 🟡 已缓解（文档层） |
| **G4** | 命名代次不一致（裸名 vs 带镜像+日期） | 无法从文件名判断实验条件 | 本清单规定"证据命名规范"，**不改历史文件名** | 🟡 已缓解 |
| **G5** | 缺"一命令复核"入口 | 第三方需自行拼命令 | 本文件 §1 提供 9 条命令；后续可做一键脚本 | 🟡 已缓解 |
| **G6** | 环境层未闭环：① 第一实例训练镜像**未发布 registry**；② 第二实例镜像**未归档未入锁**；③ 两实例 **LR 实际取值未记录**；④ 第一实例训练腿依赖版本未记录 | 结论的证据基础（环境）不可完全复现 | ①② 已在状态文件登记诉求交由总组裁定；③④ 属低成本补记 | 🟠 部分待办 |
| **G7** | 第一实例**宿主工作副本陈旧**（停在目录重组前，缺 `910C/`、缺 `prototype/scripts/`） | 复核者照该副本操作**路径全部不对** | 下次窗口顺带对齐；本次复跑走独立同步目录规避 | 🟠 待办 |
| **G8** | **契约不变式判据（I1–I4）未实现** | "诚实声明/禁止伪造/失效受管/降级可观测"目前靠人工检查 | 新增 `contract_invariants` 判据集（工作量小、不需卡） | ⬜ 待做 |

---

## 4. 对称复跑记录（2026-09-22 执行完毕）

**两实例均已在当前 HEAD 代码上完成全量复核，全部通过：**

| 项 | 第一实例（ascend） | 第二实例（kunlun） |
|---|---|---|
| smoke | **51 通过 / 0 失败**（修复后） | **42 通过 / 0 失败**（修复后回归） |
| conformance 13 例 | `CONFORMANCE_PASS 13/13` | `CONFORMANCE_PASS 13/13` |
| conformance 推理 6 例 | `CONFORMANCE_PASS 6/6` | `CONFORMANCE_PASS 6/6` |
| 执行语义基线（后端无关 V2 探针） | `STREAM_SEMANTICS_PASS 8/8`（**ascend 首跑**） | `STREAM_SEMANTICS_PASS 8/8` |
| 推理腿前向（V2，含自证字段） | `INFER_LEG_PASS 14/14` · 79.39 句/s · p50 37.73 ms · 区分度 0.6380 | （2026-09-20 已有 13/13+1 skip） |
| 训练腿 2 卡（V2，含 backend 字段） | `TRAIN_LEG_PASS 6/6` · **loss 15.4497→11.1515（与历史逐位吻合）** · 2212.9 tok/s | （2026-09-14 已有 6/6） |
| 错误闭环 | `ERROR_RECOVERY_LOOP_PASS`（5/0/0） | `ERROR_RECOVERY_LOOP_PASS`（5/0/0） |

证据：`910C/probes/recheck_*_20260922.json`（7 份）、`P800/probes/recheck_*_20260922.json`（4 份）。

**⭐ 复跑的价值：挖出并修复第 5 个"跨后端不对称"缺陷（2026-09-22）**
- 现象：ascend 上 smoke 48/2 两项失败——①`translate_error 回填后端名`（`fe.backend` 为 None）；
  ②"声明 error_map → 分级来源为 code_map"（对无码消息被误判）
- 根因（两个独立问题）：
  ① **后端不对称**：`kunlun.translate_error` 在 9-20 修复时加了 `backend=self.name` 回填，
     而 **ascend 没回填**（当时路线 B 后端同款；`FlagosError.backend` 是文档化字段，直调后端方法时为 None）；
  ② **判据不公平**：smoke 给声明了 `error_map` 的后端注入**不含厂商错误码**的消息，却断言必须走 `code_map`
- 修复：① ascend 补回填（与 kunlun 对齐；路线 B 后端同期一并补，该后端现已删除）；② 判据改为两条诚实断言——
  "无码消息不得伪称 code_map" + "含厂商码样例必走 code_map"（样例由各后端自带
  `SAMPLE_CODED_ERROR`，无样例则如实 SKIP 正向检查）
- 修复后：ascend smoke **51/0**（样例码 507015 → `graded_by=code_map`、L4_FATAL；无码消息 → `message_hint`）；
  P800 smoke **42/0** 回归通过；两实例 conformance 13+6 均回归通过
- **教训重演验证**：这再次证明"只在一家芯片上跑过的判据/实现不可信"——smoke 第 [6] 节（后端无关自检）
  是 P800 阶段新增的，新增后从未在 ascend 上跑过，一跑就暴露问题

---

## 5. 证据命名规范（从本批起适用）

```
<层标识>_<项目>_<后端>_<条件>_<YYYYMMDD>.<ext>
例：I_base_conformance_13_20260920.json    ← I = 镜像对照批；base = 镜像变体；conformance_13 = 项目
    L_serve_standard_910c_20260920.log     ← L = 服务启动标准批
```

- **必须**能在文件名中看出：**批次 / 条件（镜像或变量）/ 日期**；
- 结果 JSON **必须**含 `backend` 与 `env`（python 版本、设备类型、设备数、能力清单）字段；
- 新证据一律落在实例目录的 `probes/` 下，并在 `probes/.gitignore` 保留 `!*.log` 例外（否则日志被通用忽略规则吞掉、看似归档实则未入库）。

---

## 6. 复核者须知（两条历史教训）

1. **"已归档"必须实测核验**：写文件前用 `git ls-files <path>` 确认真的入库——
   曾出现"提交信息写已归档、实际因 `.gitignore` 规则从未入库"的情况。
2. **原始日志是证据本体，不是副产物**：`probes/` 下的 `.log` 需纳入版本库；
   通用忽略规则会静默吞掉它们，必须在该目录内显式开例外。
