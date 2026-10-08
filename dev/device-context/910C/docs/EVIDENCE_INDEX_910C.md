# 归档：910C 证据索引（`probes/` 全量清单）

> ⚠️ **这是归档索引，不是看板。** 910C 看板：`../README.md`；口径与命名规范见
> `../../prototype/docs/VERIFICATION_MANIFEST_20260920.md` §2、§5。
> 本文件由 2026-10-08 的看板精简动作从 `910C/README.md` 原样搬入（**内容未改动**）。
> 判读纪律：**证据只作复核入口，结论一律看引用它的报告**；⚠️ 带 `PRE_FIX` / `NAMESLOT_BLOCKED`
> 字样的日志是**失败现场的如实留档**（不是当前结论）。

---


| 证据 | 内容 |
|---|---|
| `recheck_conformance_13_ascend_20260922.json` | 一致性判据 13 例（含 `backend`） |
| `recheck_conformance_infer6_ascend_20260922.json` | 一致性判据 推理 6 例 |
| `recheck_stream_semantics_ascend_20260922.json` | 执行语义基线 8/8（后端无关 V2 探针**首跑 ascend**） |
| `recheck_error_loop_ascend_20260922.json` | 错误闭环 闭环 5 / 跳过 0 / 失败 0（含 `backend` + 时间戳） |
| `recheck_infer_leg_ascend_20260922.json` | 推理腿前向 **14/14**（含 `backend` + `env` + p50/p90） |
| `recheck_train_leg_ascend_20260922_rank{0,1}.json` | 训练腿 2 卡 **6/6** —— ⚠️ 该次 `backend=flagos`（torch_fl），**属切换前的旧口径证据**（loss 15.4497→11.1515、**2212.9 tok/s**） |
| `train_npu_20260922.log` | ⭐ **切换后**训练腿（**torch_npu + HCCL**）：`TRAIN_LEG_PASS 6/6`、loss **15.4498→11.1479**、**4402.3 tok/s** |
| `unified_verify_20260922.log` | ⭐ **统一口径完整复核**：离线自检 35/0 · smoke **52/0** · conformance **13/13 + 6/6** · 训练腿 **6/6**（3954.0 tok/s） · 错误闭环 **5/0/0** · `--all` 5/0 |
| `ev_matrix_20260922.log` / `ev_matrix2_20260922.log` | torch_fl `Event.query()` 语义缺口实测矩阵（审计台账第 13 条的证据） |
| `L_serve_standard_910c_20260920.log` | 《组内服务启动标准》脚本真机验证日志（`SERVE_STANDARD_PASS ready=1 smoke=1`） |
| ⭐ `accept_*_20260922.{log,json}`（15 份） | **三芯片职责验收全套证据**（09-22 傍晚）：离线自检 · 对称性 · 冒烟 · conformance 13/13 与推理 6/6 · 三个多流探针 · 训练腿（`accept_train_npu_20260922/`，含两 rank JSON）· 推理腿前向 · 服务化 · 错误闭环；另有 `accept_probe_results_910c_20260922/`（探针原始 JSON） |
| ⭐ `accept_serve_ascend_*_20260928.{log}`（3 份） | **服务化按新脚本（v1.2，含 `SMOKE_TIMEOUT`）复跑**：`SERVE_STANDARD_PASS`（就绪 **35 s**、维度 1024、范数 1.000000、**冒烟耗时 0 s**），与 09-22 逐项一致 |
| `accept_serve_ascend_*_20260928_NAMESLOT_BLOCKED.log`（3 份） | **同轮首跑失败证据（原样留档，未「改判据变绿」）**：宿主带卡容器名额被他人占满 ⇒ `acl.init`=500000、`get_device_count`=(0,0)、vLLM `Engine core initialization failed`（root cause 原文 `Failed to obtain the console log level … Different containers share the same device`） |
| ⭐ `duty_audit_ascend_20260928.json` | **职责响应审计（39 项 sub-part × 真机）**：`DUTY_RESPONSE_PASS` **39 OK / 0 FAIL / 0 SKIP**；补做后回归复跑仍 39/0/0（无退化） |
| ⭐ `regress_*_ascend_20260929.{log,json}`（11 份） | **缺陷修复后全套回归（真机）**：离线自检 38/0/1 · conformance 13/13 与 6/6 · 职责审计 39/0/0 · 错误闭环 5/0/0 · 冒烟 52/0 · 多流语义 8/8 · 流配额 3/3 |
| ⭐ `regress_*_ascend_20260929_r2.{log,json}`（15 份）+ `exp_divergence_cost_ascend_20260929_r2.{json,log}` | **第 2 轮（含第三处 L2 文案等价类修复）全套回归（真机）**：离线 **40/0/1** · 对称性 5/0 · 冒烟 52/0 · conformance 13+6 · 职责审计 39/0/0 · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3 |
| ⭐ `probe_bc_contract_ascend_20260929.json` | **工作包 B/C 真机契约探针**（5 组逐组独立子进程）：B1/B3/B4/C1/C2/C3 **6/6 通过**；含 allocate→占用→free 的设备空闲变化、二次释放负向、**厂商原生流「销毁后使用 = 静默成功」的对照取证**（本层则如实拦截） |
| ⭐ `regress_*_ascend_20260929_r5.{log,json}`（22 份）+ `recheck_*_r5b.json` | **第 5 轮全套回归（当前原型 · 11 项全绿）**：离线 **75/0/1** · 对称性 5/0 · 冒烟 52/0 · conformance 13+6 · **契约不变式 4/4** · 职责审计 39/0/0 · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3 |
| ⭐ `legs_train_910c_npu_20260929.log` + `train_leg_910c_npu_20260929_rank{0,1}.json` + `legs_infer_910c_npu_20260929.log` + `infer_leg_910c_npu_20260929.json`（5 份） | **A1 两条腿复跑（当前原型）**：训练腿 `TRAIN_LEG_PASS 6/6`（loss 15.4498→11.1479、**4513.2 tok/s**）· 推理腿前向 `INFER_LEG_PASS 14/14`（dim 1024、78.84 句/s、p50 37.59 ms、区分度 0.6391） |
| ⭐ `serve_standard_910c_npu_20260929_r1_tp{1,2}_eager{0,1}.log`（4）+ `serve_vllm_910c_npu_20260929_r1_tp{1,2}_eager{0,1}.log`（4）+ `serve_pool_910c_npu_20260929_r1.log` + `serve_standard_910c_npu_20260929_r1_nonidle_wait5.log`（10 份） | **A1+A4 服务化四形态（当前原型）**：TP=1/2 × EAGER=1/0 **全 `SERVE_STANDARD_PASS`**；TP=2 服务端日志实测 `world_size=2` + `Worker_TP0/TP1` + `backend=hccl`；EAGER=0 实测 `enforce_eager=False` + ACL Graph（PIECEWISE）；`nonidle_wait5` 为**非空转验证**（证明 `⚠️` 释放复查分支能真的触发） |
| ⭐ `a2_recover_multiproc_910c_npu_20260929.{json,log}`（2）+ `a2p1_fix_verified_910c_npu_20260929.{json,log}`（2）+ `a2_p0_public_surface_910c_npu_20260929.{json,log}`（2）+ `r6_regress_910c_npu_20260929.log` + `r6_regress_910c_npu_20260929_out/`（16）+ `offline_3backends_910c_npu_20260929.log` | **A2 全套证据（24 份）**：压测逐轮原始结果（3 rank × 30 轮 digest/快照/返回 dict/状态机转换）· A2-P1 修复双向定向验证 · A2-P0 公开面事实探针 · 第 6 轮破坏面回归 9 项 + 免跑理由验证 · 离线自检三家同跑（78/80/68 全 0 失败） |
| ⭐ `r8_regress_910c_npu_20260929.log` + `r8_regress_910c_npu_20260929_out/`（15 份） | **第 8 轮全套回归（当前原型 · 9 项全绿）**：破坏面 = `info()` 返回值（三家 `backend.py`）⇒ **元信息面**改动。离线 **83/0/1** · 对称性 **7/0**（含新增 2 条 `info()` 判据）· 冒烟 **52/0** · conformance **13/13 + 6/6** · 契约不变式 **4/4**（I1④ 入口存在性覆盖 **14/14**）· 职责审计 **39/0/0** · B/C 探针 **PASS** · 推理腿 **`INFER_LEG_PASS 14/14`**。**免跑项（实测理由）**：服务化（`serve_standard.sh` 仅 1 处注释命中 `device_type`，非真实调用）· 训练腿（`grep -c "\.info()" runtime/proto/proto_train_leg.py` = 0） |
| ⭐ `entry_verify_910c_npu_20260929.{json,log}`（2）+ `a2_recover_multiproc_910c_npu_20260929_pubentry.{json,log}`（2）+ `r7_regress_910c_npu_20260929.log` + `r7_regress_910c_npu_20260929_out/`（17）+ `offline_3backends_910c_npu_20260929_r7.log` | **补公开入口轮证据（23 份）**：入口定向验证（D1–D4 含 `handle_error` 端到端）· 压测改用公开入口后复跑 · 第 7 轮破坏面回归 10 项原始输出 · 离线三家同跑（83/85/73 全 0 失败） |
| ⚠️ `serve_standard_910c_npu_20260929_PRE_FIX_tp{1,2}_eager{0,1}.log`（4）+ `serve_vllm_910c_npu_20260929_PRE_FIX_tp{1,2}_eager{0,1}.log`（4） | **改动前原样留档**（未「改判据变绿」）：TP=2 停机后即时复查读 dev0 **55.06 / 55.07 GiB**（用卡前 61.12）⇒ 会被误读为「未释放」，实为**释放延迟**（约 1 min 后回基线，宿主 `npu-smi info -t proc-mem` 全程 `No process`） |
| ⭐ `exp_divergence_cost_ascend_20260929.{json,log}` | **工作包 A 实验的 910C 取数**：M1–M4（0/0/77/0 vs 11/15/163/17）· 功能等价性 **6/6 一致**（`S6` = `L4_FATAL`，ascend 声明了 `error_map` ⇒ 正确） |
| ⚠️ `nameslot_rule_matrix_ascend_20260929.log` | **宿主名额规则判别实验原始留档**（2×2+1 五个数据点，原样输出）：`davinci7` 成功 / `davinci1`、`davinci2` 失败 ⇒ 旧口径「与挑哪张卡无关」被推翻 |

> 证据命名规范（批次 / 条件 / 日期）与「当前结论 = 哪一份」见 `../prototype/docs/VERIFICATION_MANIFEST_20260920.md` §2、§5。
