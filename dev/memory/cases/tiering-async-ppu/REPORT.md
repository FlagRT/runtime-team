# 验收样例三 · tiering-async-ppu —— FlagMemory PPU 分层存储异步轨 E2E

> 功能验收样例（四件套 run.sh / env / scripts / REPORT+evidence，规范见
> [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md)「验收样例四件套」）。

## 用途与现状

- **验收对象**：FlagMemory 分层存储管理器「异步在途轨」端到端（`evict_async` /
  `promote_async` 发起即返回、`poll_transfers` 确认后落账/还块/交块），载体 = FM 仓
  `tests/tiering/test_ppu_tier_manager_realcard.py` 全量 **3** 用例（T-8 负例 /
  异步轨全链 / destroy 排空）。
- **运行环境**：本目录自建 [`env/`](env/)（thead-ppu-runtime v2 镜像，id `1ba7ead1…` +
  FlagMemory@58e60a9 pip-editable 挂载；装配骨架沿用 thead-ppu-runtime/v2/env/v1 实测，
  含两条 PPU 启动坑规避）。
- **通过态**：**真机实测 FAIL（2026-11-15，PTG_0 / PPU0，0/3，3 用例全部 setup ERROR，
  run.sh 退出码 1）**。失败点（环境性，非 FM 代码缺陷）：测试夹具 `PpuTierBackend.bind`
  懒加载 `flagmemory.tiering.offload.ppu.ppu_copy_backend.DmaCopyBackend`，该包 `__init__`
  连带 import connector，其 `from vllm.utils.torch_utils import PIN_MEMORY` 在本环境
  vllm 0.19 线不存在（0.24 符号）——即 FM 仓 PPU RUNBOOK 已登记的「connector 在
  thead-ppu-runtime v2 镜像内环境性不可用」边界（RUNBOOK 2026-11-14 容器内实测同款）。
  历史 11-10 3/3 全绿基线的执行载体是 flagrelease vllm0.24 定档镜像（FM RUNBOOK 登记），
  不是本样例锁定的 thead-ppu-runtime v2。

| 判定项 | 判据 | 结果 |
|---|---|---|
| 环境核对 | `up.sh ./env --strict` 全绿（镜像 id、FlagMemory commit） | **PASS**（镜像 id 逐字一致；FM 58e60a9 = v0.3.0） |
| 异步轨 E2E 套件 | `test_ppu_tier_manager_realcard.py` 3/3 pass | **FAIL**（0/3，3 setup ERROR，import 链断点如上） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | **FAIL**（`result: FAIL`，counts errors=3，退出码 1） |

## 判据与口径

- 单入口：`PPU_CARD=0 bash dev/memory/cases/tiering-async-ppu/run.sh`（ppu-smi 挑空闲卡；
  PPU 无带卡容器独占，实际用卡由 CUDA_VISIBLE_DEVICES 授予）。
- 运行口径（FM 仓该文件头部 + `tests/README`）：显式指到本测试文件，置
  `FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS`，`-o addopts= -m realcard`；
  PPU 复用 torch.cuda 命名空间（PPU SDK v2.1.0）。
- 自判定：退出码 0 / 判定行 `overall: PASS` = 3/3 通过。
- 外部依赖（基线外）：pytest（容器内自带，无须安装）；一张空闲 PPU 卡；机器 PTG_0；
  宿主 python ≥3.10（PTG_0 默认 python3=3.6，run.sh 自动建 shim 前置 python3.11）。
- 真机执行事实（2026-11-15）：v2 镜像 ENV 固化的 SDK 变量下 `import torch` 与设备枚举
  正常（`is_available()=True` / `device_count()=1`），FM 文件头一次性容器的 envsetup /
  `UMD_PLATFORM_TYPE` 口径在本镜像非必需——本样例失败与此无关，失败点见上。
- 容器装配事实：pip-editable 安装前须 `git config --global --add safe.directory
  /workspace/FlagMemory`（宿主/容器 uid 不同，git 拒绝读取挂载仓）；安装完成落
  `flagmemory-build.done` 标记，run.sh 等标记出现再进测试（消除装配竞态）。

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)（2026-11-15 13:31
  实测生成：FAIL，3 errors，镜像 id 与 FM commit 锚在内）。
- 完整日志：PTG_0 `/tmp/tiering-async-ppu-20261115-133131/`（run.log / pytest.log /
  junit.xml / ppu-smi-before.txt）；归档副本 `/tmp/s4-sample3-20261115/`。
- 环境版本事实：[`env/env.baseline.yaml`](env/env.baseline.yaml)。

<!-- changelog: 2026-11-15 新建（S4-A 验收样例三骨架；真机挂起待 PTG_0 VPN 窗口）。 -->
<!-- changelog: 2026-11-15 真机补完：修 run.sh 仓根定位少一层与宿主 python≥3.10 前置、compose 增 safe.directory 与装配就绪标记；实测 0/3（3 setup ERROR，vllm 0.19 无 PIN_MEMORY，FM RUNBOOK 已登记边界），如实记 FAIL。 -->
