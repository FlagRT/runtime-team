# 验收样例 · fm-smoke-ascend —— FlagMemory 池冒烟（昇腾 910C）

> 规范：[`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md) §0（四件套）。

## 用途与判据

- **验收对象**：FlagMemory v0.3.0 池核心语义在昇腾 910C 真卡上的最小功能面。
  判据载体 = FM 仓 `tests/pool/test_pool_ascend_realcard.py`（3 用例：真分配与跨流依赖 /
  复用拒绝旧租约 / 借用方保护；有界预算安全契约，累计分配上限 64MiB）。
- **运行环境**：直接引用现行环境目录
  [`dev/images/ascend-train-comm/v3/env/v2`](../../../images/ascend-train-comm/v3/env/v2/)
  （引用方式见 [`env/README.md`](env/README.md)），`up.sh --strict` 机器核对。
- **单入口**：`TRAIN_DIE_A=<die> TRAIN_DIE_B=<die> bash dev/memory/cases/fm-smoke-ascend/run.sh`
  （选卡纪律：npu-smi 挑零计算进程 die；容器内 0 号设备 = TRAIN_DIE_A）。

## 判据表

| 判定项 | 判据 | 结果 |
|---|---|---|
| C1 环境核对 | `up.sh --strict` 全绿（镜像 id、三挂载 commit、cann-shmem 装后核对） | **通过**（镜像 e13a15d0d02b／FlagCX 62fe615 覆盖生效／dc 984687f／FM 58e60a9＋cann-shmem 1.6.0 装后核对） |
| C2 版本面 | 容器内 `import flagmemory` == 0.3.0 | **通过**（FM_VERSION 0.3.0） |
| C3 功能面 | `test_pool_ascend_realcard.py` 3/3 pass（0 fail / 0 error / 0 skip） | **通过**（3 passed in 9.14s） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | **通过**（退出码 0，`overall: PASS`） |

## 通过态

首次真机执行：2026-11-19，机器 910c-11，die 0/1（执行前后两 die 零计算进程、回落核验干净）。

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)
- 完整日志：执行机 `/tmp/fm-smoke-ascend-<时间戳>/`（不入仓）

<!-- changelog
2026-11-18 首建
2026-11-19 首次真机执行全绿（910c-11 die0/1）
2026-11-19 按审核意见去除教学性说明，保持标准样例形态
-->
