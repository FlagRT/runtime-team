# 验收样例五 · connector-ppu —— FlagMemory ppu KV connector 字节级搬运真卡（平头哥 PPU）

> 功能验收样例（形态：四件套 run.sh / env（借用 `../tiering-async-ppu/env`）/ scripts / REPORT+evidence，
> 规范见 [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md)「验收样例四件套」）。
> 与 `dev/issues/` 的跨方向缺陷 case 包不混放：本目录是子方向功能验收样例。
> 方法与 [`connector-ascend`](../connector-ascend/REPORT.md) 同构（同一轮工作先做的昇腾那份）。

## 用途与现状

- **验收对象**：FlagMemory（显存池管理器）ppu KV→Host 卸载 connector
  （`flagmemory.tiering.offload.ppu` / `DmaCopyBackend`）的**真实搬运行为**在平头哥 PPU 真卡
  上是否字节级正确。判据同昇腾那份：③D2H 字节级逐字一致 / ④H2D 换入后新旧块号分叉在场 /
  ⑥destroy 双形态（带在途任务排空 / 空闲幂等）。
- **载体与方法**：`scripts/ppu_offload_byte_probe.py` 直接驱动
  `flagmemory/tiering/offload/ppu/ppu_copy_backend.py` 的 `DmaCopyBackend`（PPU 走标准
  `torch.cuda` 命名空间，PlatformFL 提供 CUDA 兼容层）——不经 vLLM、不受调度策略影响，方法与
  `connector-ascend` 完全一致，只是设备命名空间从 `torch.npu` 换成 `torch.cuda`。
- **运行环境**：借用 [`dev/memory/cases/tiering-async-ppu/env`](../tiering-async-ppu/env/)
  （flagrelease vllm0.24 定档镜像 + FlagMemory pip-editable 挂载），不新建第二份装配。
  **本次运行未加 `--strict`**：该 env 的 `env.baseline.yaml` 把 FlagMemory commit 钉在
  `58e60a9`（2026-11-15 建该 env 时的点），本次容器内实际装配的是现行 `dev-1.0@3b7cdb1`
  （PTG_0 工作副本本轮会话已同步到最新）——这是 FlagMemory 往前走了、不是环境漂移，镜像 id
  仍逐字一致（已在步骤 1 单独自证）。若要用 `--strict` 的完整形态验收，需要先把该 env 的
  FlagMemory 锚点刷新到当前 commit（跨 case 共享文件，是否刷新待单独决定，不在本样例改动）。

## 通过态（2026-11-24 PTG_0 真机执行，PPU 卡 0，FlagMemory@3b7cdb1）

| 判据 | 结果 |
|---|---|
| D2H 字节级逐字一致 | PASS |
| H2D 换入新旧块号分叉在场 | PASS |
| destroy：带在途任务排空 | PASS |
| destroy：空闲幂等 | PASS |

证据：[`evidence/summary.json`](evidence/summary.json)（机器可判）。完整执行日志未入库
（`/tmp/connector-ppu-<时间戳>/`，临时目录）。

## 附带清理

本次执行前发现 `tiering-async-ppu` 留下的容器 `flagrt-fm-tiering-ppu-ptg0` 已空闲运行 5 天
（2026-11-19 创建，仅 `sleep infinity`，未占用计算），经用户确认后拆除，为本样例让出环境名。

## 适用边界

- 只验证 `offload/ppu` 这一个连接器的搬运机制本身，不验证 vLLM/调度器何时决定触发搬运。
- 四家芯片（昇腾/昆仑芯/寒武纪/平头哥）中，本轮已完成昇腾、平头哥两家；昆仑芯、寒武纪是否
  铺开待定。
