# 验收样例（S5 教学示范首件）· fm-smoke-ascend —— FlagMemory 池冒烟（昇腾 910C）

> **定位**：S5 模板推广的教学示范——按《验收样例上手指引》五步实现的**最小完整样例**，
> 各方向做首件样例时对照此件。规范权威 = [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md) §0。

## 用途与判据

- **验收对象**：FlagMemory（显存池管理器）v0.3.0 池核心语义在昇腾 910C 真卡上的最小功能面。
  判据载体 = FM 仓 `tests/pool/test_pool_ascend_realcard.py`（**3 用例**：真分配与跨流依赖 /
  复用拒绝旧租约 / 借用方保护；自带**有界预算安全契约**，累计分配上限 64MiB）。
- **运行环境**：直接引用现行环境目录
  [`dev/images/ascend-train-comm/v3/env/v2`](../../../images/ascend-train-comm/v3/env/v2/)
  （引用方式与教学注解见 [`env/README.md`](env/README.md)），`up.sh --strict` 机器核对。
- **单入口**：`TRAIN_DIE_A=<die> TRAIN_DIE_B=<die> bash dev/memory/cases/fm-smoke-ascend/run.sh`
  （选卡纪律：npu-smi 挑零计算进程的 die；容器内 0 号设备 = TRAIN_DIE_A）。

## 判据表

| 判定项 | 判据 | 结果 |
|---|---|---|
| C1 环境核对 | `up.sh --strict` 全绿（镜像 id、三挂载 commit、cann-shmem 装后核对） | **通过**（四项 PASS：镜像 e13a15d0d02b／FlagCX 62fe615 覆盖生效／dc 984687f／FM 58e60a9＋cann-shmem 1.6.0 装后核对） |
| C2 版本面 | 容器内 `import flagmemory` == 0.3.0 | **通过**（FM_VERSION 0.3.0） |
| C3 功能面 | `test_pool_ascend_realcard.py` 3/3 pass（0 fail / 0 error / 0 skip） | **通过**（3 passed in 9.14s） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | **通过**（退出码 0，判定行 `overall: PASS`） |

## 通过态

**首次真机执行：2026-11-19，机器 910c-11，die 0/1（执行前后两 die 零计算进程、回落核验干净）。**
三判据全过；完整日志执行机 `/tmp/fm-smoke-ascend-20261119-100355/`（首跑因前置核对 awk 字段
笔误 FAIL 后即修，同分支同判据重跑通过——修正经历本身即教学样例的示范内容之一：失败如实记录、
单入口幂等重跑）。

## 教学对照（做首件样例时看这里）

| 指引要求 | 本样例的落点 |
|---|---|
| 五步之①定判据 | 上面判据表——三条，全部机器可判 |
| 五步之②定环境 | `env/README.md`——引用式，一行 ENV_DIR |
| 五步之③单入口 | `run.sh`——前置核对→自证→strict→用例→summary→down+回落，失败路径也清理 |
| 五步之④跑真的 | 本节通过态 + `evidence/summary.json` |
| 五步之⑤写 REPORT | 本文件（现状-only） |

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)
- 完整日志：执行机 `/tmp/fm-smoke-ascend-<时间戳>/`（不入仓）

---

<!-- changelog
2026-11-18 首建（S5 教学示范首件）
2026-11-19 首次真机执行全绿（910c-11 die0/1，三判据过，判据表与通过态回填）
-->
