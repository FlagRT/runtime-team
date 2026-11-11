# 910C · 两腿 `DC_MODEL` 口径复验（2026-11-03）

> 补齐 10-27 那次遗留：当日 910C 网络不可达，只有 P800 / MLU590 跑过。判据本体见
> `prototype/scripts/run_legparity_dcmodel.sh`（四实例共用），本文件只记本实例读数。
> 脚本：`probes/run_910c_legparity_20261027.sh`（`C=dc-probe-priv` 覆盖容器名后执行）。

环境：`npu1-27` · 容器 `dc-probe-priv`（**privileged**，只挂 `davinci6/7`）·
`torch 2.10.0+cpu` / `torch_npu 2.10.0` · `DC_BACKEND=ascend` · `DC_DIST_BT=hccl` ·
模型 `/workspace/models/Qwen3-Embedding-0.6B`（宿主 `/tgm_rent_data/…/models` 只读挂载）

## 读数（run-id `20261103_110713`）

| 判据 | 读数 | 与归档对照 |
|---|---|---|
| P1 离线口径自检（含 AST 同源守卫） | **23 / 0 / 0** | 与 P800/MLU 同 |
| P2-a 推理腿 × HF 缓存根 | `INFER_LEG_PASS` **14/14** · 70.31 句/s · 区分度 **0.6380** | 与 10-27 复跑 0.6380 逐字同 |
| P2-b 训练腿 50 步 × HF 缓存根 | `TRAIN_LEG_PASS` **6/6** · loss **15.4498→11.1479** · 4286 tok/s | **与 10-27 归档逐字同** |
| P4-a 推理腿 × 普通目录 | 14/14 · 69.82 句/s · 0.6380 | 同 |
| P4-b 训练腿 50 步 × 普通目录 | 6/6 · 15.4498→11.1479 · 4163 tok/s | 同 |
| P2 两腿解析同一 snapshot | 缓存根 → `…/legparity_hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…` | **PASS** |
| P3 未选中诱饵 snapshot（`zzzz…` 空目录） | 未命中 | **PASS** |
| P4 两种传法两腿解析一致 | 普通目录 → `…/legparity_plain/Qwen3-Embedding-0.6B` | **PASS** |
| 两次训练腿 `\|Δloss\|` | **0.000000** | 两条传法喂同一份权重 |
| **总判** | **`LEGPARITY_PASS` 9 通过 / 0 失败** | |

## ⭐ 本轮的实质意义（对照 10-27 的缺陷）

10-27 在 MLU590 上暴露的缺陷是：**同一份 HF 缓存根，推理腿能过、训练腿直接**
`ValueError: Unrecognized model in …/models--Qwen--Qwen3-Embedding-0.6B`。
本实例今日**直接用缓存根跑训练腿 50 步**（脚本里**刻意不做任何手工 snapshot 解析**）
⇒ `TRAIN_LEG_PASS 6/6`、loss 与归档逐字相同 ⇒ 修复在 910C 上同样成立。

## 四实例口径复验进度

| 实例 | 状态 | 读数 |
|---|---|---|
| P800 | ✅ 10-27 roundC | 9/0 |
| MLU590 | ✅ 10-27 roundC | 9/0 |
| **910C** | ✅ **11-03（本文件）** | **9/0** |
| PPU | ⏳ 待跑（容器/脚本已备：`PPU/probes/run_ppu_legparity_20261027.sh`） | — |

## 文件

`20261103_110713/` 下：`s1_offline.log`（23 项逐条）· `s2_infer_hub.log` / `s4_infer_plain.log` ·
`s3_train_hub.log` / `s5_train_plain.log`（含逐 5 步 loss）· `infer_hub.json` / `infer_plain.json` ·
`train_leg_result_rank{0,1}.json`；`console.log` 为控制台全文。
