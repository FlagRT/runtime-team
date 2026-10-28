# MLU590 · `DC_MODEL` 两腿口径复验读数（2026-10-27）

目的：验证「训练腿补 `resolve_model`」这次**共享层改动**。⚠️ **本实例就是缺陷的发现地**：
10-27 复跑时同一份 `DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B`
推理腿过、训练腿报 `ValueError: Unrecognized model in …`（当时只能先手工解析 snapshot 绕过）。

脚本：`MLU590/probes/run_mlu590_legparity_20261027.sh`（薄封装）→
`prototype/scripts/run_legparity_dcmodel.sh`（判据核心，md5 `307197a620c8c611565fe94f0b39fb0b`）

环境：容器 `dc-mlu590-hliu553`（自 2026-09-22 Up，长期复用）· 容器内 `/work` == 宿主 `/srv/hliu553` ·
解释器 `/flagos/bin/python3`（py3.10 / torch 2.7.1+cpu / torch_mlu 1.29.2）·
选卡 `MLU_VISIBLE_DEVICES`（单卡 0、训练腿 0,1）· **`DC_DIST_BT=cncl`**（本原型对寒武纪拒绝兜底 gloo）·
11:41 普查：8 张卡**均无他人进程**（`cnmon` 无 running processes）

## roundC（官方读数）

| 判据 / 段 | 读数 | 结论 |
|---|---|---|
| P1 离线口径自检（含 AST 同源守卫） | **23 通过 / 0 失败 / 0 跳过** | PASS |
| P2-a 推理腿 × HF 缓存根 | `INFER_LEG_PASS 13/13` · dim 1024 · 区分度 **0.6391** · 40.67 句/s · p50 73.13 ms | PASS（**改前本项就过 —— 缺陷在训练腿**） |
| P2-b 训练腿 50 步 × HF 缓存根 | `TRAIN_LEG_PASS 6/6` · loss **15.4498→11.1479** · 3000 tok/s（两卡合计） | **改前此项直接 `Unrecognized model`** |
| P4-a 推理腿 × 普通目录 | `13/13` · 0.6391 · 40.24 句/s · p50 71.98 ms | PASS |
| P4-b 训练腿 50 步 × 普通目录 | `6/6` · 15.4498→11.1479 | PASS |
| P2 两腿解析同一 snapshot（= refs/main 真 hash） | 缓存根 → `…/legparity_hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…` | **PASS** |
| P3 未选中诱饵 snapshot（`zzzz…` 空目录） | 未命中 | **PASS** |
| P4 两种传法两腿解析一致 | 普通目录 → `…/legparity_plain/Qwen3-Embedding-0.6B` | **PASS** |
| 两次训练腿 `\|Δloss\|` | **0.000000** | 两条传法喂的是同一份权重 |
| 总判 | **`LEGPARITY_PASS` 9 通过 / 0 失败** | |

**与归档对齐**：区分度 0.6391 = 10-27 复跑归档 0.6391；训练腿 loss 15.4498→11.1479 = 归档同值
⇒ 改共享层**没有动**原有读数。

## 轮次与文件（三轮全部保留，不覆盖）

| 轮 | 说明 |
|---|---|
| `roundA/console.log` | 首轮；判据末行标签把宿主路径写成「（容器）」（**标签瑕疵，路径本身正确**），已修 |
| `roundB/console.log` | **判据自己误判**：抽取写法 `grep -oE '→ 解析 .*'` 被 torchrun **双 rank 输出交错**骗到 ⇒ P2 报 FAIL（8/1）。被测对象无问题；修法＝抽取只认路径字符集 `[A-Za-z0-9._/-]+` |
| **`roundC/` + `console_roundC.log`** | **官方读数**（判据修正后） |

`roundC/` 内文件：`s1_offline.log`（23 项逐条）· `s2_infer_hub.log` / `s4_infer_plain.log`（推理腿两种传法）·
`s3_train_hub.log` / `s5_train_plain.log`（训练腿两种传法，含逐 5 步 loss）·
`infer_hub.json` / `infer_plain.json`（`model` / `model_raw` / `model_source` 可逐字对账）·
`train_leg_result_rank{0,1}.json`

## 结论边界

只覆盖「模型路径口径」这一个改动面。不改 `runtime/__init__.py` / `api/` / `backends/` /
`duty_response_audit.py` ⇒ 离线契约 103 项、smoke、契约不变式、审计 78 项（C13 = 11.381 ms）
**未重跑**（范围裁定，不是"已证明无回归"）。整轮回归请用 `run_mlu590_rerun_20261027.sh`。
