# MLU590 `DC_MODEL` 两腿口径复验证据清单（2026-11-03）

> 判据本体：`prototype/scripts/run_legparity_dcmodel.sh`（四实例共用一份）
> 入口：`MLU590/probes/run_mlu590_legparity_20261027.sh` ｜ 结论：**`LEGPARITY_PASS` 9/0**

| 判据 | 读数 |
|---|---|
| P1 离线口径自检（含 AST 同源守卫） | 23 / 0 / 0 |
| 推理腿 × HF 缓存根 | `INFER_LEG_PASS` 13/13 · 41.89 句/s · p50 70.54 ms · 区分度 **0.6391** |
| 训练腿 50 步 × HF 缓存根 | `TRAIN_LEG_PASS` 6/6 · loss **15.4498→11.1479** |
| 推理腿 × 普通目录 | 13/13 · 42.54 句/s · 区分度 0.6391 |
| 训练腿 × 普通目录 | 6/6 · loss **15.4498→11.1479** |
| 两腿解析同一 snapshot / 诱饵未命中 / `\|Δloss\|` | PASS / PASS / **0.000000** |

与 10-27 归档**逐字相同**（11.1479 / 0.6391）⇒ 共享层改动未动读数。

| 路径 | 说明 |
|---|---|
| `console.log` | 控制台全文（逐判据 + `VERDICT: LEGPARITY_PASS`） |
| `run/` | 原始产物（各段日志、`infer_*.json`、`train_leg_result_rank*.json`） |
