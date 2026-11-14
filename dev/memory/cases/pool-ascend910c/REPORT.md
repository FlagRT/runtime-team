# 验收样例一 · pool-ascend910c —— FlagMemory 池真卡（昇腾 910C）

> 功能验收样例（形态：四件套 run.sh / env / scripts / REPORT+evidence，规范见
> [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md)「验收样例四件套」）。
> 与 `dev/issues/` 的跨方向缺陷 case 包不混放：本目录是子方向功能验收样例。

## 用途与现状

- **验收对象**：FlagMemory（显存池管理器）池本体在昇腾 910C 真卡上的功能面——
  分配/复用、生命周期、流序、回收屏障等，载体 = FM 仓
  `tests/pool/test_pool_realcard.py` 全量 **11** 用例（torch.cuda 兼容命名空间套件，
  torch_npu 接管 torch.cuda，按 FM 仓文件头部 npu 先例原样运行）。
- **运行环境**：直接引用现行环境目录
  [`dev/images/ascend-train-comm/v3/env/v2`](../../../images/ascend-train-comm/v3/env/v2/)
  （镜像 ascend-train-comm v3 + FlagCX/flagos-device-context/FlagMemory 三挂载 +
  cann-shmem 1.6.0 层包），`up.sh --strict` 机器核对。
- **通过态**：见下表（本表只记现行状态，历史运行进 evidence/summary.json 与日志指针）。

| 判定项 | 判据 | 结果 |
|---|---|---|
| 环境核对 | `up.sh --strict` 全绿（镜像 id、三挂载 commit、cann-shmem 实装版本） | 未执行（本样例未起容器；env/v2 本体已于 2026-11-15 两次 `--strict` 全绿，见下注） |
| 池真卡套件 | `test_pool_realcard.py` 11/11 pass（0 fail / 0 error / 0 skip） | 未执行（2026-11-15 机器全 16 die 被并存容器映射，非特权容器 open 一律 EBUSY、device_count=0；待窗口重跑） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | 未执行（无 summary.json，不虚构） |

## 判据与口径

- 单入口：`TRAIN_DIE_A=<die> TRAIN_DIE_B=<die> bash dev/memory/cases/pool-ascend910c/run.sh`
  （选卡纪律：npu-smi 挑显存为驱动基线带且零进程的两张 die，每次执行前重新核对；
  并存空壳带卡容器可共存——控制台 B3 判读 + 2026-11-14 共存实证）。
- 运行口径（FM 仓 `tests/README` 与该测试文件头部）：显式指到
  `tests/pool/test_pool_realcard.py`，置 `FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS`，
  `-o addopts= -m realcard`，`FLAGMEMORY_REALCARD_DEVICE=0`（容器内 0 号 = TRAIN_DIE_A）。
- 自判定：退出码 0 / 判定行 `overall: PASS` = 11/11 通过；非 0 = 有失败或环境不满足。
- 外部依赖（基线外）：pytest（容器内按需 pip 安装，默认 PyPI）；两张空闲 die；
  机器 910c-11。

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)（用例名→pass/fail、
  计数、镜像 id、三挂载 commit 锚、cann-shmem 实装版本；由 run.sh 生成）。
- 完整日志：执行机器 910c-11 的 `/tmp/pool-ascend910c-<时间戳>/`（run.log、pytest.log、
  junit.xml、npu-smi 前后对照；完整日志不进仓）。
- 环境版本事实：`dev/images/ascend-train-comm/v3/env/v2/env.baseline.yaml`。

<!-- changelog: 2026-11-15 新建（S4-A 验收样例一，模板首件）。
     2026-11-15 真机窗口尝试：run.sh 未跑成——机器全 16 die 被并存带卡容器映射（npu-eval：2-15；
     flagos-op-ascend-daizijian：0-1），非特权容器 open die 一律 EBUSY、torch_npu device_count=0
     （die 0/1 与 10/11 各试一次；privileged 对照可枚举 16 卡）。诊断与 up.sh --strict 两次全绿
     记录：910c-11 /tmp/s4-env2-20261115/。待空闲窗口后重跑本样例。 -->
