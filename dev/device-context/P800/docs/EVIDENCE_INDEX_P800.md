# 归档：P800 证据索引（`probes/` 探针脚本 + 原始证据）

> ⚠️ **这是归档索引，不是看板。** P800 看板：`../README.md`；口径与命名规范见
> `../../prototype/docs/VERIFICATION_MANIFEST_20260920.md` §2、§5。
> 本文件由 2026-10-08 的看板精简动作从 `P800/README.md` 原样搬入（**内容未改动**）。
> 判读纪律：**证据只作复核入口，结论一律看引用它的报告**。
> ⚠️ 本目录 `*.log` 是**原始证据**（随仓库分发），故 `probes/.gitignore` 开了 `!*.log` 例外。

---

## 5. 证据索引（`probes/`）

**探针脚本（可复现）**

| 脚本 | 用途 |
|---|---|
| `dc_probe_p800.py` | 五域探针（设备抽象 / 多流 / 错误 / 分布式），单进程一次跑完 |
| `dc_probe_isolated.py` | **单变量隔离**版：每用例独立进程，避免同进程内错误粘滞污染 |
| `dc_probe_rep.py` | 参数化重复集合通信探针（`MODE`=ar / ar_nosync / ar_max / barrier；`SYNC_EVERY`；`REPS`；`SIZE`） |
| `dc_probe_devonly.py` | 对照组：单进程纯设备计算（无通信） |
| `dc_probe_grad_ar.py` | 梯度通信定点探针：单次大通信（concat）vs 多次小通信（逐参数） |
| `dc_probe_ar_rep.py` | 重复性与确定性判定（同一张量重复 vs 真实梯度按序） |
| `dc_probe_verify.py` | **带真值校验**的验证探针（`2^120` 精确匹配，用于排除假阴性） |
| `smoke_kunlun_20260914.txt` | 组件自检原始输出（**42 通过 / 0 失败**） |

**探针组脚本（自动抓 gdb 原生栈）**

`probe_battery.sh`（第一轮 8 变体）、`probe_battery2.sh`（第二轮重复验证）、`probe_battery3.sh`（第三轮剂量-反应）、`verify_battery.sh`（真值校验四组）

**原始证据日志**

| 日志 | 内容 |
|---|---|
| `A_round1_battery_20260914.log` | 第一轮 8 变体单变量对照 + 挂死现场原生栈 |
| `B_round2_repeat_20260914.log` | 第二轮重复验证（A×3 挂死 / B×3 通过 / C / D / E） |
| `C_round3_dose_20260914.log` | 第三轮剂量-反应（同步间隔 N=1,1,2,3,5,10）→ 证明**无阈值效应** |
| `D_verify_truthvalue_20260914.log` | 真值校验四组（A×4 挂死 / B×2 与 D 精确通过 / C×2 挂死）⇒ 更正「不同步就通过」的假阴性 |
| `E_train_ab.log` | **训练腿 A/B 单变量对照**（不设 → 退出码 0；设=1 → 退出码 124） |
| `E_train_R1_workaround_noKL3.log` | 训练腿规避条件运行日志 |
| `E_train_R2_control_KL3on.log` | 训练腿对照条件运行日志（挂死现场） |
| `E_train_r1_keep.log` | 规避腿复跑（一致性确认） |
| `E_train_leg_result_rank0.json` / `rank1.json` | **训练腿结果 JSON**：`TRAIN_LEG_PASS 6/6`、6 项检查全绿、loss 曲线、perf |
| `F_infer_leg.sh` / `F_infer_leg_20260920.log` / `F_infer_leg_result_20260920.json` | **阶段 3 推理腿**：脚本 + 完整日志 + 结果（13/13，维度 1024 / 区分度 0.6392 / 53.12 句/s / p50 56.17 ms） |
| `F2_vllm_serve.sh` / `F2_vllm_serve_20260920.log` | **阶段 3 补（vLLM 服务化）**：一键脚本（启动 → 就绪 → 验证 → 停机 → 用卡复查）+ 日志 |
| `F2_serve_result_20260920.json` | 服务化结果（`SERVE_LEG_PASS 10/10`：区分度 0.4102 / 30.70 句/s / p50 96.4 ms） |
| `F2_vllm_server_boot_20260920.log` | vLLM 服务启动原始日志（FL 平台插件激活 → 路由注册 → 就绪） |
| `H_kl3_equivalence.sh` | **KL3 缺陷等价性对照脚本**（后台轮询 + `kill -9`；因挂死进程持 GIL 自旋、`timeout` 的 SIGTERM 无法中断） |
| `I_base_*_20260920.*` | **官方 `-base` 镜像全套证据**（17 份）：conformance 13+6（json+log）、smoke 42/0、训练腿两 rank、推理腿前向 13/13、服务化 10/10、**KL3 对照（ab 汇总 + A1–A3 挂死现场 + B1–B2 真值校验）** |
| `I_ref_train_leg_result_rank0_20260920.json` | 同批次**现用镜像**训练腿结果（用于交替复测，证明吞吐差异属共享机噪声） |
| `K_stream_semantics_full_result_p800_20260920.json` | **多流 16 项基线中 8 项探针结果**（`STREAM_SEMANTICS_PASS 8/8`，含 backend=`kunlun` / dev_api=`cuda` / 逐项 detail）——与 910C 侧同名结果逐项对照 |
| `L_serve_standard_p800_20260920.log` | **组内服务启动标准脚本**（`../prototype/scripts/serve_standard.sh`）在 P800 的验证日志：服务就绪 **25 s**、冒烟**维度 1024 / 范数 1.000000**、停机后**无残留进程**且卡 6 释放至 0 MiB、`SERVE_STANDARD_PASS`（脚本 v1.0 首轮） |
| `L_serve_standard_p800_v2_20260920.log` | 同上脚本 **v1.1**（补服务入口自动激活 / 生成形态冒烟 / 容器内卡快照降级后）的复跑日志：`SERVE_STANDARD_PASS (ready=1 smoke=1)`（与 910C 同版本脚本、同日验证） |
| `G_error_loop.sh` / `G_error_loop_20260920.log` | **阶段 4 错误闭环**：两设置对照脚本 + 日志（两组各 5/0/0） |
| `error_recovery_loop_kunlun_KL3off.json` | 阶段 4 结果：**不设** `XPU_EVENT_KL3_ENABLE` |
| `error_recovery_loop_kunlun_KL3on.json` | 阶段 4 结果：**设** `XPU_EVENT_KL3_ENABLE`（与上面除时间戳外**完全一致**） |
| `recheck_conformance_13_kunlun_20260922.json` | **两实例对称复跑（09-22）**：一致性判据 13/13（含 `backend`） |
| `recheck_conformance_infer6_kunlun_20260922.json` | 同上：推理 6 例 6/6 |
| `recheck_stream_semantics_kunlun_20260922.json` | 同上：执行语义基线 8/8 |
| `recheck_error_loop_kunlun_20260922.json` | 同上：错误闭环 闭环 5 / 跳过 0 / 失败 0（含 `backend` + 时间戳） |
| ⭐ `regress_*_kunlun_20260929_r2.{log,json}` + `exp_divergence_cost_kunlun_20260929_r2.{json,log}`（17 份）
| ⭐ `probe_bc_contract_kunlun_20260929.json` | **工作包 B/C 真机契约探针**：B1/B3/B4 **3/3 通过**；C 三组如实 SKIP（未声明 `context_lifecycle`） |
| ⭐ `probe_bc_contract_kunlun_20260929_r2.json` | **B/C 契约探针第二轮（含新增 C4 组）**：B1/B3/B4 + **C4 上下文只读观测** 全部通过；`managed_by="external"`、`compute_before = compute_after = 512.0` |
| `regress_*_kunlun_20260929_r4.*`（15 份） | **第 4 轮全套回归原始证据**（卡 4）：离线 **71/0/1** · 对称性 5/0 · 冒烟 46/0 · conformance 13+6 · 职责审计 36/0/3 · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3 | | **09-29 第 2 轮全套回归原始证据**（卡 4）：离线 **45/0/1** · 对称性 5/0 · 冒烟 46/0 · conformance 13+6 · 职责审计 36/0/3 · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3 |
| ⚠️ `DIAG_kunlun_card1_event_hang_20260929.log` | **卡 1 挂死判别记录**（原样留档）：A 组（卡 1 + 新原型，含 `faulthandler` 调用栈）· B 组（卡 4 + 新原型，**全绿**）· C 组（卡 1 + **修复前**原型，**同一行挂死**）⇒ 判定为**卡级环境问题**，未改动任何判据 |

> ⚠️ **注意**：本目录 `*.log` 为**原始证据**，需随仓库分发，故在此目录放了局部 `.gitignore`（`!*.log`）
> 覆盖根仓库的 `*.log` 通用忽略规则。**此前这批日志因根规则从未入库**，本次整理时一并纳入。
