# 归档：MLU590 证据索引（`probes/`）

> ⚠️ **这是归档索引，不是看板。** MLU590 看板：`../README.md`。
> 本文件由 2026-10-08 的看板精简动作从 `MLU590/README.md` 原样搬入（**内容未改动**）。
> 判读纪律：**证据只作复核入口，结论一律看引用它的报告**；⚠️ 带 `PRE_FIX` 字样的日志是
> **失败现场的如实留档**（不是当前结论）。命名规范见
> `../../prototype/docs/VERIFICATION_MANIFEST_20260920.md` §2、§5。

---

| 证据 | 内容 |
|---|---|
| `probes/accept_*_20260928.*` | **本轮 12 项判定的原始证据**（命名与两实例同规范）：离线自检 · 冒烟 · conformance 13+6 · 多流三项 · 训练腿 · 推理腿前向 · 服务化（含 `PRE_FIX` 失败留档与 vLLM 服务端日志）· 错误闭环 |
| ⭐ `probes/duty_audit_cambricon_20260928.json` | **职责响应审计（39 项 sub-part × 真机）**：`DUTY_RESPONSE_PASS` **36 OK / 0 FAIL / 3 SKIP**（3 项 SKIP 均为如实不具备）；离线自检 41/0/0 · 对称性 5/0 |
| `probes/preflight_env_mlu1_20260922.log` | **Mlu-1 环境普查原始日志**（`preflight_env.sh` 首跑产出） |
| `probes/preflight_env_mlu2_20260922.log` | **Mlu-2 环境普查原始日志** |
| `probes/.gitignore` | `!*.log` 例外（否则根 `.gitignore` 的 `*.log` 会让证据静默不入库） |
| ⭐ `probes/regress_*_cambricon_20260929.*` + `probes/exp_divergence_cost_cambricon_20260929.*`（共 18 份） | **09-29 修复后全套回归原始证据**：离线 **45/0/0** · 对称性 5/0 · 冒烟 46/0 · conformance 13+6 · 职责审计 36/0/3 · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3 |
| ⭐ `probes/exp_divergence_cost_cambricon_PRE_FIX_20260929.json` | **修复前**同一实验的原始证据（**5/6，`S1` DIFF + 期望未达标**）—— 缺陷发现的**第一手证据**，原样留档 |
