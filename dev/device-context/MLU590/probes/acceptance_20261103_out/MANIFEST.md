# MLU590 单入口全量复跑证据清单（2026-11-03）

> 入口：`MLU590/probes/run_mlu590_acceptance_20261103.sh`（`RUNID=run1103b`）
> 容器 `dc-mlu590-hliu553` ｜ torch 2.7.1+cpu / torch_mlu 1.29.2 ｜ 选卡 `MLU_VISIBLE_DEVICES=3`
> 结论：**`overall: PASS`（9 段逐段 rc=0）**，与 10-27 归档逐项一致。详见 `../../docs/CAMBRICON_MLU_RERUN_20261103.md` §2。

| 段 | rc | verdict |
|---|---|---|
| s01 离线契约自检 | 0 | 103 通过 / 0 失败 / 0 跳过 |
| s02 跨后端对称性 | 0 | 7 通过 / 0 失败 |
| s03 真机 smoke | 0 | 46 通过 / 0 失败 |
| s04 conformance 13 例 | 0 | `CONFORMANCE_PASS` |
| s05 conformance 推理 6 例 | 0 | `CONFORMANCE_PASS` |
| s06 契约不变式 I1–I4 | 0 | `CONTRACT_INVARIANTS_PASS` |
| s11 图捕获 | 0 | `GRAPH_CAPTURE_PASS` 4/4 |
| s12 流配额 | 0 | `STREAM_QUOTA_PASS` 3/3 |
| s13 职责响应审计 | 0 | `DUTY_RESPONSE_PASS`（OK 63 / FAIL 0 / SKIP 15，共 78） |

## 文件
| 路径 | 说明 |
|---|---|
| `console.log` | 控制台全文（逐段 rc + verdict + `overall: PASS`） |
| `run/summary.tsv` | 机读汇总（脚本自写） |
| `run/{offline,symmetry,smoke,conf13,confinfer6,invariants,graphcap,quota,duty}.log` | 各段原始日志 |
| `run/s0*.json` · `run/s13_duty.json` | 结构化结果 |

⚠️ 另有一轮 `RUNID=run1103`（**作废**）：脚本第一版把代码副本路径拼在 run-id 目录下 ⇒ 9 段全部 rc=1 且日志为空。
保留说明是为了让"同一脚本两次结果不同"可解释；本目录只放有效轮。
