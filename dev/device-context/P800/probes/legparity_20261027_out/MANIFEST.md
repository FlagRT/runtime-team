# P800 · `DC_MODEL` 两腿口径复验读数（2026-10-27）

目的：验证「训练腿补 `resolve_model`」这次**共享层改动**在本实例的读数与既有归档一致，
并且两条腿对**同一份** `DC_MODEL` 解析到**同一路径**（改前只有推理腿会解析）。

脚本：`P800/probes/run_p800_legparity_20261027.sh`（薄封装）→
`prototype/scripts/run_legparity_dcmodel.sh`（判据核心，md5 `307197a620c8c611565fe94f0b39fb0b`）

环境：容器 `hliu553-device-context-p800`（10-27 11:39 由 `docker start` 拉起，此前 Exited）·
解释器 `/root/miniconda/envs/python310_torch29_cuda/bin/python3`（torch 2.9.0+cu129）·
单卡段 `CUDA_VISIBLE_DEVICES=0`、训练腿 `0,7`（**卡 1 为已知故障卡，避开**；
11:40 普查：卡0 12.2GiB/19%、卡2/3 已 78GiB/100%、卡7 11.5GiB/0%）·
`DC_DIST_BT=cpu:gloo,cuda:flagcx`

## roundC（官方读数）

| 判据 / 段 | 读数 | 结论 |
|---|---|---|
| P1 离线口径自检（含 AST 同源守卫） | **23 通过 / 0 失败 / 0 跳过** | PASS |
| P2-a 推理腿 × HF 缓存根 | `INFER_LEG_PASS 13/13` · dim 1024 · 区分度 **0.6392** · 27.16 句/s · p50 111.09 ms | PASS |
| P2-b 训练腿 50 步 × HF 缓存根 | `TRAIN_LEG_PASS 6/6` · loss **15.4488→11.1481** · 2995 tok/s（两卡合计） | PASS |
| P4-a 推理腿 × 普通目录 | `13/13` · 0.6392 · 42.84 句/s · p50 69.85 ms | PASS |
| P4-b 训练腿 50 步 × 普通目录 | `6/6` · 15.4488→11.1481 · 2915 tok/s | PASS |
| P2 两腿解析同一 snapshot（= refs/main 真 hash） | 缓存根 → `…/legparity_hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…` | **PASS** |
| P3 未选中诱饵 snapshot（`zzzz…` 空目录） | 未命中 | **PASS** |
| P4 两种传法两腿解析一致 | 普通目录 → `…/legparity_plain/Qwen3-Embedding-0.6B` | **PASS** |
| 两次训练腿 `\|Δloss\|` | **0.000000** | 两条传法喂的是同一份权重 |
| 总判 | **`LEGPARITY_PASS` 9 通过 / 0 失败** | |

**与归档对齐**：区分度 0.6392 = 10-26 复跑归档 0.6392；训练腿 loss 15.4488→11.1481 = 归档同值。

⚠️ **吞吐不可直接比**：本轮单卡段与其它用户作业同卡共跑（卡 0 已有 12.2 GiB 占用），
27.16 与 42.84 句/s 的差来自同机并发，不是版本差异。

## 轮次与文件

| 轮 | 说明 |
|---|---|
| `roundA/` | 首轮；判据末行标签把宿主路径写成「（容器）」（**标签瑕疵，路径本身正确**），已修 |
| `roundB/` | 判据抽取写法用 `.*`，被 torchrun **双 rank 输出交错**骗到（P2 误判 FAIL）；本实例结果仍为 9/0 |
| **`roundC/`** | **官方读数**（判据修正后）；`console_roundC.log` 为控制台全文 |

| 文件 | 内容 |
|---|---|
| `console_roundC.log`（顶层） | **官方轮**控制台全文（每段 rc + verdict + 判据） |
| `roundA/console.log` / `roundB/console.log` | 前两轮控制台全文（保留不覆盖） |
| `roundC/s1_offline.log` | 离线口径自检 23 项逐条 |
| `roundC/s2_infer_hub.log` / `s4_infer_plain.log` | 推理腿两种传法 |
| `roundC/s3_train_hub.log` / `s5_train_plain.log` | 训练腿两种传法（含逐 5 步 loss） |
| `roundC/infer_hub.json` / `infer_plain.json` | 推理腿结果（`model` / `model_raw` / `model_source` 可逐字对账） |
| `roundC/train_leg_result_rank{0,1}.json` | 训练腿两 rank 结果 |

## 结论边界

只覆盖「模型路径口径」这一个改动面。不改 `runtime/__init__.py` / `api/` / `backends/` /
`duty_response_audit.py` ⇒ 离线契约 114 项、smoke、契约不变式、审计 78 项**未重跑**（范围裁定，
不是"已证明无回归"）。整轮回归请用 `run_p800_*_rerun_*` 系列。
