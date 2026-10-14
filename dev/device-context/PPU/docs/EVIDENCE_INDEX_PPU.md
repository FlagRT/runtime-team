# PPU（平头哥）· 证据索引

> 日期：2026-10-14 ｜ 对应执行单 **T9**（`prototype/docs/TODO_NEXT_WEEK_20261012.md`）
> **定位：这是复核入口，不是结论本身。** 结论请读 `docs/` 下的报告；本文件只回答"证据在哪、怎么读、能证到什么"。

---

## 0. 读法纪律（先读这三条）

1. **每份判据的 verdict 行优先于任何叙述**：`*_PASS` / `*_FAIL` / `NOT_APPLICABLE` / `SKIP_*` 四类**语义不同**，不得混写（如实跳过 ≠ 失败 ≠ 通过）。
2. ⚠️ **判据通过 ≠ 进程干净退出**：本实例多个探针在**解释器退出期段错误**（`rc=139`，见 §4），
   **判据与退出码要一起看**。
3. ⚠️ **`*_out/` 目录内文件可能由容器内 root 写入**（宿主侧 `rm` 会 `Permission denied`）⇒ 清理须在**容器内以 root** 做。

---

## 1. 按任务（T 号）索引

| 任务 | 证据目录/文件 | 判据行 | 备注 |
|---|---|---|---|
| T1（**MLU590**，非本实例） | `../../MLU590/probes/t1_duty_cambricon_20261014.json` · `…log` · `…/t1_coninvariants_cambricon_20261014.json` | `DUTY_RESPONSE_PASS 62/0/16` · `CONTRACT_INVARIANTS_PASS 4/4` | 10-14；⚠️ 该任务按**破坏面**在**另两台**上做，PPU 侧对应物见「本实例判据」各节 |
| T2（**MLU590**，非本实例） | `../../MLU590/probes/t2_preflight_env_cambricon_20261014.log` | `rc=0` | 环境普查（只读） |
| T3 | `../probes/t3_serve_20261014_out/form/serve_standard.log` | `SERVE_STANDARD_PASS (ready=1 smoke=1)` | ⚠️ 判据在**该文件内**（脚本内部 `exec` 重定向，外层重定向得到空文件） |
| T3 | `../probes/t3_serve_20261014_out/consumer/proto_infer_serve_result.json` | `SERVE_LEG_PASS 10/10` | 本层消费方 |
| T4 | `../probes/t456_20261014_out/t4_errorloop_ppu.{json,log}` | `ERROR_RECOVERY_LOOP_PASS` 闭环 5/跳过 0/失败 0 | `rc=139`（退出期段错误，判据已过） |
| T5 | `../probes/t456_20261014_out/t5_stream_semantics_ppu.log` | `STREAM_SEMANTICS_PASS 8/8` | 单卡（S-13 如实跳过） |
| T5 | `../probes/t5_20261014_out/t5_2card_stream_semantics.log` | `STREAM_SEMANTICS_PASS 8/8`（含 S-13 `[2.0, 3.0]`） | **两卡可见**才覆盖 S-13 |
| T5 | `../probes/t5_20261014_out/t5_stream_quota.log` | `STREAM_QUOTA_PASS 3/3`（2000 流 / 0.01 s） | S-16 |
| T5 | `../probes/t5_20261014_out/t5_graph_capture_v2.log` | `GRAPH_CAPTURE_PASS`（契约内 4/4 + 1 观察项） | S-7 |
| T6 | `../probes/t456_20261014_out/t6_sched_effect_ppu_fixed.{json,log}` | `STREAM_PRIORITY_SCHED_EFFECT_PASS` · **`effect_observed=False`** | ⚠️ 修法后；`rc=139` |
| T6（修前） | `../probes/t456_20261014_out/t6_sched_effect_ppu.{json,log}` | `…FAIL`（6 个场景 `RuntimeError`） | **保留作缺陷现场**，勿当"当前状态"读 |
| **T11** | `../probes/t11_20261014_out/` | **本层侧已收口**：修后 `rc=0`（三载体）· 平台侧最小复现仍 `rc=139` | 原生现场 `t4_native*.log` · `t6_native.log`；矩阵 `log/` · `rc_matrix.log`；注入对照 `patch_matrix.log`；修后 `after_matrix.log`；全文 `docs/PPU_T11_EXIT_SEGV_ROOTCAUSE_20261014.md` |
| — | 见 §2 | 缺陷复现与修前/修后对照 | |
| T8（**MLU590**，非本实例） | `../../MLU590/probes/t8_a2_skip_cambricon_20261014.log` | `A2_BOUNDARY: SKIP_UNSUPPORTED`（`rc=3`，未做任何设备操作） | 能力如实不具备；⚠️ PPU 侧的 A2 留档在 `../probes/parity_fill_20261010_out/` |

## 2. 缺陷证据（平台侧弱引用悬垂 ⇒ 本层误拦；2026-10-14）

目录：`../probes/repro_release_recreate_20261014/`

| 文件 | 用途 |
|---|---|
| `probe_release_recreate_loop.py` | **稳定复现器**（建→用→释放 循环 N 轮） |
| `probe_who_holds.py` | 命中瞬间**现场 dump**（两表状态、id 复用、使用点结果） |
| `probe_weakref_2x2.py` | **判据实验**（唯一变量 = 是否 `gc.collect()`） |
| `probe_weakref_verdict.py` | 早期判定（A/B/C 三场景；⚠️ 结果落在"少数不跟随路径"，**勿单独引用**） |
| `probe_release_recreate.py` | 最早的单发实验（同上，仅供参考） |
| `stage_evidence2.sh` | **修前/修后对照的固化脚本**（备份→禁用→跑→恢复→跑） |
| `verify_fix.sh` | 修法自证 |
| `before/` · `after/` | 对照日志（loop / 2×2 / who_holds 三份） |
| `base_fixed_backup.py` | 修好版 `base.py` 备份（对照期用） |

判读：`before/` 与 `after/` 必须**成对**读（同一机器、同一时段、同一进程路径，仅切换一行代码）。

## 3. 接入期证据（2026-10-10，仍在用）

| 目录 | 内容 |
|---|---|
| `../probes/std_steps_20261010_out/` | 离线自检 · smoke · conformance 13+6 · 多流 8 项 · 图捕获 · 配额 · B/C 契约 · 职责审计 72/0/6 |
| `../probes/onboard_20261010_out/` | 能力探测 4 轮（栈普查 / 形态判定 / 后端名 / 流优先级） |
| `../probes/model_and_legs_20261010_out/` | 模型完整性（步 0）+ 两条腿（训练 50 步 / 推理 13/13） |
| `../probes/parity_fill_20261010_out/` | 契约不变式 4/4 · A2 如实跳过留档 |
| `../probes/l1_recheck_20261010/` | 10-10 L1 根因复核（`_owned_streams` 类型复核）——⚠️ 其"悬垂假说被否证"结论**已被 10-14 推翻**，见 `docs/PPU_T3_T6_AND_WEAKREF_20261014.md` §4.2③ |

## 4. 已知退出期段错误（`rc=139`）覆盖表

| 探针 | rc | 判据 |
|---|---|---|
| `proto_train_leg.py`（训练腿） | 139 | 6/6 通过 |
| `proto_infer_leg.py`（推理腿） | 139 | 13/13 通过 |
| `proto_error_recovery_loop.py`（T4） | 139 | 5/0/0 通过 |
| `probe_stream_priority_sched_effect.py`（T6） | 139 | PASS |
| `probe_stream_semantics_full.py`（T5） | **0** | 8/8 通过 |
| `probe_stream_quota.py` / `probe_graph_capture_stream_v2.py` | 0 | 3/3 · 4/4 |

⇒ 与 `runtime.create_stream()`（优先级路径）相关的一组探针会崩；不建优先级流的探针干净退出。

**⭐ 2026-10-14 已定位并收口（本层侧）**：原生现场 = `Py_Exit → Py_FinalizeEx → … → **_PyWeakref_ClearRef**`，
`si_code=1`、**`si_addr=0xab`**（垃圾指针）；根因 = **平台侧弱引用书签失效**（**不含本层**的最小复现
「30 条原生流各挂一个 `weakref.ref` 后丢弃」同样 `rc=139` 2/2）；本层是**暴露面**（为每条流建弱引用）。
修法 = ① 无本层上下文时不登记 `_stream_ctx` ② `_released_streams` 改强持有 ③ 登记表加条数上限
⇒ 修后三个载体（本层最小复现 / T4 / T6）真机 **全部 `rc=0`**，判据读数逐字不变。
⚠️ **平台侧缺陷未修**（纯 torch 最小复现修后**仍崩**）⇒ 转厂商。
全文与全部对照：**`docs/PPU_T11_EXIT_SEGV_ROOTCAUSE_20261014.md`**；证据 `probes/t11_20261014_out/`。

## 5. 命名约定（T10 现状，如实登记）

- **符合 manifest §5**（`<层标识>_<项目>_<后端>_<条件>_<YYYYMMDD>`）：`t1_duty_cambricon_20261014.json` ·
  `t2_preflight_env_cambricon_20261014.log` · `t8_a2_skip_cambricon_20261014.log` 等**任务级产物**。
- ⚠️ **偏离**：本轮新增的**目录名**用 `<T号>_<用途>_<YYYYMMDD>_out`（如 `t456_20261014_out`）与
  `<用途>_<YYYYMMDD>`（如 `repro_release_recreate_20261014`）—— 便于按任务检索，但**不与 §5 模板逐字一致**。
  **处置**：保留现状并在此登记；后续新证据优先按 §5 命名，目录级可另立约定后再统一。
