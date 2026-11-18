# 验收样例三 · tiering-async-ppu —— FlagMemory PPU 分层存储异步轨 E2E

> 功能验收样例（四件套 run.sh / env / scripts / REPORT+evidence，规范见
> [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md)「验收样例四件套」）。

## 用途与现状

- **验收对象**：FlagMemory 分层存储管理器「异步在途轨」端到端（`evict_async` /
  `promote_async` 发起即返回、`poll_transfers` 确认后落账/还块/交块），载体 = FM 仓
  `tests/tiering/test_ppu_tier_manager_realcard.py` 全量 **3** 用例（T-8 负例 /
  异步轨全链 / destroy 排空）。
- **运行环境**：本目录自建 [`env/`](env/)——宿主 = **flagrelease vllm0.24 定档镜像**
  （`harbor.baai.ac.cn/flagrelease-public/flagrelease_ppu_vllm024_gems_540:1012`，
  image id `sha256:6cd05dcf…`，digest `sha256:399937f1…`）+ FlagMemory@58e60a9
  （v0.3.0）pip-editable 挂载。镜像权威登记处 = FM 仓
  `flagmemory/tiering/offload/ppu/RUNBOOK.md` §0（定档 2026-11-14），**暂不入
  stack.lock**（用户裁定 2026-11-15，入锁另行立项）；case 内锚 = 
  [`env/image.anchor.yaml`](env/image.anchor.yaml)（手写登记桩），版本事实 =
  [`env/env.baseline.yaml`](env/env.baseline.yaml)（手写登记式，check 核对镜像 id +
  FM commit）。
- **已登记边界（环境性，非 FM 代码缺陷）**：本样例在 thead-ppu-runtime v2 镜像
  （`dev/stack.lock.thead-ppu.v1.yaml` 现行锁定的训练/推理同底座，vllm 0.19 线）上
  **不可用**——测试夹具懒加载 `ppu_copy_backend` 连带 import connector，其
  `from vllm.utils.torch_utils import PIN_MEMORY` 在 vllm 0.19 不存在（2026-11-15
  本样例容器内实测；FM RUNBOOK 2026-11-14 登记同款）。PPU 侧本样例锚定 0.24 定档
  镜像执行。
- **通过态**：**真机实测 PASS（2026-11-15 16:36，PTG_0 / PPU0，3/3 全绿：T-8 负例 /
  异步轨全链 / destroy 排空，run.sh 退出码 0，`overall: PASS`）**。

| 判定项 | 判据 | 结果 |
|---|---|---|
| 环境核对 | `up.sh ./env --strict` 全绿（镜像 id、FlagMemory commit） | **PASS**（镜像 id 逐字一致；FM 58e60a9 = v0.3.0） |
| 异步轨 E2E 套件 | `test_ppu_tier_manager_realcard.py` 3/3 pass | **PASS**（3/3，`3 passed in 11.19s`） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | **PASS**（`result: PASS`，counts 3/3，退出码 0） |

## 判据与口径

- 单入口：`PPU_CARD=0 bash dev/memory/cases/tiering-async-ppu/run.sh`（ppu-smi 挑空闲卡；
  PPU 无带卡容器独占，实际用卡由 CUDA_VISIBLE_DEVICES 授予）。
- 运行口径（FM 仓该文件头部 + `tests/README`）：显式指到本测试文件，置
  `FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS`，`-o addopts= -m realcard`；
  PPU 复用 torch.cuda 命名空间（PPU SDK v2.1.0）。
- SDK 三件套（RUNBOOK §1，0.24 镜像 SDK 环境不固化在镜像 ENV）：compose 覆盖镜像
  ENTRYPOINT（原 `bash /opt/t-head/entrypoint.sh` 为服务化引导）、environment 固化
  `UMD_PLATFORM_TYPE=1` + `HGGC_DRIVER_CANDIDATE=UMD`；exec 会话由
  `scripts/run_tiering_ppu.sh` source `/usr/local/PPU_SDK/envsetup.sh`（无参 →
  MODE=CUDA，与 FM 文件头一致）。**坑（已内置规避）**：envsetup.sh 内部引用 `$1`，
  须在 `set -u` 生效前 source，否则 unbound variable 即退。
- 自判定：退出码 0 / 判定行 `overall: PASS` = 3/3 通过。
- 外部依赖（基线外）：pytest 镜像内自带（7.2.0，无须安装）；一张空闲 PPU 卡；
  机器 PTG_0；宿主 python ≥3.10（PTG_0 默认 python3=3.6，run.sh 自动建 shim 前置
  python3.11）。
- 容器装配事实：pip-editable 安装前须 `git config --global --add safe.directory
  /workspace/FlagMemory`（宿主/容器 uid 不同，git 拒绝读取挂载仓）；安装完成落
  `flagmemory-build.done` 标记，run.sh 等标记出现再进测试（消除装配竞态）。

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)（2026-11-15 16:36
  实测生成：PASS 3/3，镜像 id 与 FM commit 锚在内）。
- 完整日志：PTG_0 `/tmp/s4-sample3b-20261115/`（run.log / pytest.log / junit.xml /
  ppu-smi-before.txt；run.log 含改锚重跑全记录）。
- 环境版本事实：[`env/env.baseline.yaml`](env/env.baseline.yaml) +
  [`env/image.anchor.yaml`](env/image.anchor.yaml)。

<!-- changelog: 2026-11-15 新建（S4-A 验收样例三骨架；真机挂起待 PTG_0 VPN 窗口）。 -->
<!-- changelog: 2026-11-15 真机补完：修 run.sh 仓根定位少一层与宿主 python≥3.10 前置、compose 增 safe.directory 与装配就绪标记；实测 0/3（3 setup ERROR，vllm 0.19 无 PIN_MEMORY，FM RUNBOOK 已登记边界），如实记 FAIL。 -->
<!-- changelog: 2026-11-15 改锚重跑全绿：样例锚 thead-ppu-runtime v2 → flagrelease vllm0.24 定档镜像（FM RUNBOOK §0 定档，暂不入锁——用户裁定）；env/ 重写（compose 换镜像+SDK 三件套+entrypoint 覆盖、baseline 手写登记式、新增 image.anchor.yaml 登记桩）、run_tiering_ppu.sh 增 envsetup source（含 set -u 前置规避）；实测 3/3 PASS（16:36，PTG_0/PPU0）。 -->
