# 验收样例一 · pool-ascend910c —— FlagMemory 池真卡（昇腾 910C）

> 功能验收样例（形态：四件套 run.sh / env / scripts / REPORT+evidence，规范见
> [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md)「验收样例四件套」）。
> 与 `dev/issues/` 的跨方向缺陷 case 包不混放：本目录是子方向功能验收样例。

## 用途与现状

- **验收对象**：FlagMemory（显存池管理器）池本体在昇腾 910C 真卡上的功能面——
  分配/复用、生命周期、流序、回收屏障等，载体 = FM 仓
  `tests/pool/test_pool_realcard.py` 全量 **11** 用例（torch.cuda 兼容命名空间套件，
  torch_npu 接管 torch.cuda，按 FM 仓文件头部 npu 先例原样运行；接管经
  `torch_npu.contrib.transfer_to_npu` 预导入——torch 2.10 上裸 `import torch_npu`
  不接管 torch.cuda，夹具取不到卡会全量 skip）。
- **载体口径**：FM 仓昇腾侧另有正式真卡入口 `tests/pool/test_pool_ascend_realcard.py`
  （独立适配器，3 个有界生命周期用例，单芯片受控、累计额度 ≤1 GiB、显式 enable 门控）与
  `tests/pool/test_pool_device_context_ascend_realcard.py`（统一桥接入口，同 3 用例形态）；
  二者与本样例载体**不同构且互不包含**——正式入口服务单芯片受控验证实例（映射 / 4096 字节
  健康 / 三生命周期），本样例在现行训练腿运行环境内全量跑 11 用例（判据一/二、自管路、水位、
  整理回收、全生命周期、粒度、屏障恢复、三类误用拦截）。`dev/memory/STATUS.md` 矩阵昇腾池格
  的「已验收」锚 = `ascend-pool-lifecycle-3de6d5a` 等证据（正式入口产物）；矩阵中链接
  `test_pool_realcard.py` 的是寒武纪 MLU 格。
- **运行环境**：直接引用现行环境目录
  [`dev/images/ascend-train-comm/v3/env/v2`](../../../images/ascend-train-comm/v3/env/v2/)
  （镜像 ascend-train-comm v3 + FlagCX/flagos-device-context/FlagMemory 三挂载 +
  cann-shmem 1.6.0 层包），`up.sh --strict` 机器核对。
- **通过态**：见下表（本表只记现行状态，历史运行进 evidence/summary.json 与日志指针）。

| 判定项 | 判据 | 结果 |
|---|---|---|
| 环境核对 | `up.sh --strict` 全绿（镜像 id、三挂载 commit、cann-shmem 实装版本） | 通过（2026-11-15 真机 die 10/11，镜像 id / 三挂载 commit / cann-shmem 1.6.0 实装全 PASS） |
| 池真卡套件 | `test_pool_realcard.py` 11/11 pass（0 fail / 0 error / 0 skip） | 通过（2026-11-15 真机 11/11 pass，0 fail / 0 error / 0 skip；daizijian 空壳带卡容器共存不挡） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | 通过（`evidence/summary.json` `result: PASS`；run.sh 退出码 0，判定行 `overall: PASS`） |

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
- 完整日志：执行机器 910c-11 的 `/tmp/pool-ascend910c-<时间戳>/`（现行一次：
  `/tmp/pool-ascend910c-20261115-132716/`，run.log、pytest.log、junit.xml、npu-smi 前后对照）（run.log、pytest.log、
  junit.xml、npu-smi 前后对照；完整日志不进仓）。
- 环境版本事实：`dev/images/ascend-train-comm/v3/env/v2/env.baseline.yaml`。

<!-- changelog: 2026-11-15 新建（S4-A 验收样例一，模板首件）。
     2026-11-15 真机窗口尝试：run.sh 未跑成——机器全 16 die 被并存带卡容器映射（npu-eval：2-15；
     flagos-op-ascend-daizijian：0-1），非特权容器 open die 一律 EBUSY、torch_npu device_count=0
     （die 0/1 与 10/11 各试一次；privileged 对照可枚举 16 卡）。诊断与 up.sh --strict 两次全绿
     记录：910c-11 /tmp/s4-env2-20261115/。待空闲窗口后重跑本样例。
     2026-11-15 真机窗口补验完成：占用者已释放，die 10/11（daizijian 空壳容器共存）；
     --strict 全绿、11/11 pass、summary PASS、down 后 die 回落基线、容器删净。同笔修
     容器侧 runner 第三处潜伏缺陷：torch 2.10 裸 import torch_npu 不接管 torch.cuda
     （夹具 is_available()=False → 全量 skip），改为预导入 transfer_to_npu 后起 pytest。
     2026-11-15 载体口径核实（收尾件）：比对 FM dev-1.0@58e60a9 的 tests/README 备注
     （test_pool_realcard.py=旧 CUDA 草稿、禁作昇腾结果）与昇腾正式入口
     （test_pool_ascend_realcard.py / test_pool_device_context_ascend_realcard.py，各 3 个
     有界生命周期用例）——与 11 用例不同构亦非超集（不同物），裁定维持本样例载体不变、不切换。
     事实链更正一处：STATUS.md 矩阵昇腾池格锚的是 ascend-pool-lifecycle 证据（正式入口产物），
     链接 test_pool_realcard.py 的是寒武纪 MLU 格（建议交 FM 线与设备方向复核该链接口径）。
     FM tests/README 过时备注未改（FM 文档由 FM 线自修）。正文新增「载体口径」条（现状-only）。 -->
