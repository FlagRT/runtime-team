# PPU `DC_MODEL` 两腿口径复验证据清单（2026-11-03）

> 判据本体在 `prototype/scripts/run_legparity_dcmodel.sh`（四实例共用一份）；本目录是 PPU 这一轮的产物。
> 入口：`PPU/probes/run_ppu_legparity_20261027.sh`（薄封装，宿主侧执行）。
> 被验栈：`hliu553-dc-dev` 容器 · **v2 镜像**（id `sha256:1ba7ead1…`）· torch 2.10.0 · 2 卡。
> 结论：**`LEGPARITY_PASS` 9 通过 / 0 失败**。

## 1 判据逐项

| 判据 | 读数 | 说明 |
|---|---|---|
| P1 离线口径自检（含 AST 同源守卫） | **23 / 0 / 0** | 两条腿不得各自实现 `resolve_model` |
| P2-a 推理腿 × HF 缓存根 | `INFER_LEG_PASS` **13/13** · 131.02 句/s · p50 22.99 ms · 区分度 **0.6391** | |
| P2-b 训练腿 50 步 × HF 缓存根 | `TRAIN_LEG_PASS` **6/6** · loss **15.4498→11.1530** · 5237.0 tok/s（两卡） | |
| P2 两腿解析到同一 snapshot 且 = `refs/main` | PASS | |
| P3 未选中诱饵 snapshot（字典序更大的空目录） | PASS | 防止实现退化成「排序取最后」 |
| P4-a 推理腿 × 普通模型目录 | `INFER_LEG_PASS` 13/13 · 区分度 0.6391 | 回归 |
| P4-b 训练腿 50 步 × 普通模型目录 | `TRAIN_LEG_PASS` 6/6 · loss 15.4498→11.1530 | 回归 |
| P4 两种传法两腿解析同一路径 | PASS | |
| `\|Δloss\|` 两种传法 | **0.000000** | 逐字相同 ⇒ 喂的是同一份权重 |

## 2 与归档的对照

loss 终点 **11.1530**、推理区分度 **0.6391** 与 2026-10-26 全量重跑归档**逐字相同** ⇒ 共享层改动未动读数。
训练吞吐 5237.0 tok/s vs 归档 5090 tok/s（+2.9%）属共享机波动，不作结论。

## 3 结论边界

只覆盖「模型路径口径」改动面。离线 114/103/92、smoke、契约不变式、审计 78 **不在本轮范围**——
它们由同日的 v2 全量验收覆盖（`../v2_acceptance_20261103_out/`）。

## 4 文件索引

| 路径 | 说明 |
|---|---|
| `console.log` | 控制台全文（逐判据 PASS/FAIL + `VERDICT: LEGPARITY_PASS`） |
| `run/` | 该轮的原始产物：各判据日志、`infer_hub.json`、`train_leg_result_rank*.json` |
| `run/*.rc` | 逐段退出码（判据通过 ≠ 进程干净退出） |
