# PPU v2 镜像 · 全量 13 段验收证据清单（2026-11-03）

> 目的：履行「每发环境/镜像新版，未关闭缺陷逐条三态重表态」的要求 —— 本轮给出 **v2 镜像上的全量复跑**读数，
> 与 2026-10-26 的 v1 基线逐项对照。
> 入口脚本：`PPU/probes/run_acceptance_ppu.sh`（单入口 13 段，`RUNID=v2-20261103`）。
> 被验镜像：`flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64`
> 　　　　　`image id = sha256:1ba7ead15378160ba9aa1dafcfb77686de6b16732590ffab062bab19c60c32b8`
> 　　　　　（= `dev/images/thead-ppu-runtime/v2/image.baseline.yaml` 的 check 项；容器创建于 2026-10-27T02:09Z，
> 　　　　　 证据时间 2026-11-03 ⇒ 满足「证据晚于镜像创建时间」）

## 1 总判

`overall: PASS（13 段全部 rc=0）` —— 见 `console.log` 末尾。

| 段 | rc | verdict（逐字） |
|---|---|---|
| s01 离线契约自检 | 0 | `PASS` |
| s02 跨后端对称性自检 | 0 | `PASS` |
| s03 真机 smoke | 0 | `PASS` |
| s04 conformance 13 例 | 0 | `CONFORMANCE_PASS` |
| s05 conformance 推理 6 例 | 0 | `CONFORMANCE_PASS` |
| s06 契约不变式 I1–I4 | 0 | `CONTRACT_INVARIANTS_PASS` |
| s07 训练腿 2 卡 50 步 | 0 | `TRAIN_LEG_PASS 6/6` |
| s08 推理腿 | 0 | `INFER_LEG_PASS` |
| s09 服务化 + 消费方腿 | 0 | `SERVE_LEG_PASS \| 10/10` |
| s10 错误闭环 + 多流 + 调度效果 | 0 | `ERROR_RECOVERY_LOOP_PASS` · `STREAM_SEMANTICS_PASS: 8/8` · `SCHED_EFFECT_PASS` |
| s11 图捕获 | 0 | `GRAPH_CAPTURE_PASS`（4/4） |
| s12 流配额 | 0 | `STREAM_QUOTA_PASS`（3/3） |
| s13 职责响应审计 | 0 | `DUTY_RESPONSE_PASS`（OK 73 / FAIL 0 / SKIP 5，共 78） |

## 2 与 2026-10-26 基线（v1）逐项对照

| 项 | 10-26 归档 | **11-03 v2** | 判定 |
|---|---|---|---|
| 训练 50 步 | 6/6 · loss 15.4498→**11.1530** | 6/6 · loss 15.4498→**11.1530** | ✅ 逐字相同 |
| 训练吞吐 | 5090 tok/s | 5237.1 / 5218.9 tok/s | 区间内波动（共享机） |
| 推理前向 | 13/13 · 区分度 **0.6391** | 13/13 · 区分度 **0.6391** | ✅ 逐字相同 |
| 推理吞吐 | 138.66 句/s | 140.95 句/s（p50 21.12 ms） | 区间内波动 |
| 服务化消费方腿 | 10/10 · 区分度 **0.4118** | 10/10 · 区分度 **0.4118** | ✅ 逐字相同 |
| 服务化吞吐 | 105.74 句/s | **91.65 句/s** | ⚠️ 差 −13%，见 §4 |
| 错误恢复闭环 | 5 项 · 0 失败 · 0 遗漏 | 5 / 跳过 0 / 失败 0 | ✅ |
| 多流语义 | 8/8 | 8/8 | ✅ |
| 图捕获 | 4/4 | 4/4 | ✅ |
| 流配额 | 3/3 | 3/3 | ✅ |
| 职责审计 | 73 通过 / 0 失败 | OK 73 / FAIL 0 / SKIP 5 | ✅ |

## 3 ⚠️ s09 的「假 FAIL」（本轮核出，已修脚本）

`console.log` 里 s09 打印 `[A] 未通过`，但**实际是 PASS**：`serve_standard.sh` 臂的日志明确写着
`✅ 服务就绪 t=45s` / `[verdict] SERVE_STANDARD_PASS (ready=1 smoke=1)`（见 `s09_t3_form_flat/serve_standard.log`）。

根因＝**判据读错目录**：`run_t3_ppu_serve_20261014.sh` 只把宿主视角变量 `HO` 拼了 `$RUNID`，
容器视角变量 `O` 没拼 ⇒ `DC_OUT_DIR="$O/form"` 写到**扁平旧目录**，而 verdict 的 grep 读 `$HO/<runid>/form/`
⇒ ① 每轮 form 证据都被下一轮覆盖；② verdict 恒读不到 ⇒ 假 FAIL。

- 已修：`O=$O/$RUNID`（`run_t3` / `run_t456` / `run_t5` 三处同族问题一并修）。
- 反证保留：`s09_t3_form_flat/serve_standard.log` 就是那份「被判未通过、实则 PASS」的原始日志。
- ⇒ 10-26 归档里的同一位置同样是空的，说明**这个假 FAIL 一直存在**，此前未被发现。

## 4 服务化吞吐 −13% 的归因（进行中 / 已排除项）

v2 上消费方腿吞吐 91.65 句/s vs 归档 105.74 句/s。**已排除**「v2 镜像本身坏」：同轮区分度 0.4118 逐字相同、
10/10 全 PASS、就绪 40 s。**待定**两项候选：① 本轮起用新默认 `gmu=0.4`（归档是 vLLM 默认 0.9）；
② 本轮起用 `max_num_seqs=128`（归档未传，vLLM 默认 256）。⇒ 见 `PPU/probes/gmu_ab_20261103_out/`
与同批吞吐对照（`tput_ab_*`），结论以实测为准，**不在本节下结论**。

## 5 文件索引

| 路径 | 说明 |
|---|---|
| `console.log` | 单入口全量控制台（逐段 rc + verdict + `overall: PASS`） |
| `acceptance/summary.tsv` | 逐段机读汇总（脚本自己写） |
| `acceptance/s0*.log` / `*.json` | 各段原始日志与结构化结果 |
| `acceptance/s09-serve.log` | 服务化段控制台（含 §3 的假 FAIL 现象） |
| `s09_t3/` | s09 委派脚本按 runid 分目录的证据（本轮为空 = §3 的 bug 现场） |
| `s09_t3_form_flat/serve_standard.log` | **反证**：实际写出 PASS 的那份日志（扁平目录） |
| `s10_t456/` | s10 委派脚本按 runid 分目录的证据 |
