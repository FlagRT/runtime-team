# 验收样例三 · tiering-async-ppu —— FlagMemory PPU 分层存储异步轨 E2E（骨架）

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
- **通过态**：**真机验证待 PTG_0 VPN 窗口**——目标机器（PTG_0 / ai-server）当前经 VPN
  不可达，本样例文件齐备、run.sh 可执行，但**尚未在真机执行，无结果可引**（不虚构）。

| 判定项 | 判据 | 结果 |
|---|---|---|
| 环境核对 | `up.sh ./env --strict` 全绿（镜像 id、FlagMemory commit） | 未执行（待 PTG_0 窗口） |
| 异步轨 E2E 套件 | `test_ppu_tier_manager_realcard.py` 3/3 pass | 未执行（待 PTG_0 窗口） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | 未执行（待 PTG_0 窗口） |

## 判据与口径

- 单入口：`PPU_CARD=0 bash dev/memory/cases/tiering-async-ppu/run.sh`（ppu-smi 挑空闲卡；
  PPU 无带卡容器独占，实际用卡由 CUDA_VISIBLE_DEVICES 授予）。
- 运行口径（FM 仓该文件头部 + `tests/README`）：显式指到本测试文件，置
  `FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS`，`-o addopts= -m realcard`；
  PPU 复用 torch.cuda 命名空间（PPU SDK v2.1.0）。
- 自判定：退出码 0 / 判定行 `overall: PASS` = 3/3 通过。
- 外部依赖（基线外）：pytest（容器内按需 pip 安装，默认 PyPI）；一张空闲 PPU 卡；
  机器 PTG_0（经 VPN 窗口可达）。
- 首次真机窗口注意：SDK 环境变量按镜像 ENV 固化假定直接运行；若 `import torch` 或设备
  枚举异常，按 FM 仓该文件头部的 envsetup/UMD 变量口径复核后更新本样例（如实记录）。

## 证据指针

- 机器可判结果：`evidence/summary.json`（由 run.sh 生成；当前无——未执行）。
- 完整日志（执行后）：PTG_0 的 `/tmp/tiering-async-ppu-<时间戳>/`。
- 环境版本事实：[`env/env.baseline.yaml`](env/env.baseline.yaml)。

<!-- changelog: 2026-11-15 新建（S4-A 验收样例三骨架；真机挂起待 PTG_0 VPN 窗口）。 -->
