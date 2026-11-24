# 验收样例七 · connector-xpu —— FlagMemory xpu KV connector 字节级搬运真卡（昆仑芯 P800）

> 功能验收样例（形态：四件套 run.sh / env（直接引用现行环境目录）/ scripts / REPORT+evidence，
> 规范见 [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md)「验收样例四件套」）。
> 方法与 [`connector-ascend`](../connector-ascend/REPORT.md)、[`connector-ppu`](../connector-ppu/REPORT.md)、
> [`connector-mlu`](../connector-mlu/REPORT.md) 同构（同一轮工作的第四家，也是最后一家）。

## 用途与现状

- **验收对象**：FlagMemory（显存池管理器）xpu KV→Host 卸载 connector
  （`flagmemory.tiering.offload.xpu` / `XpuCopyBackend`）的**真实搬运行为**在昆仑芯 P800 真卡
  上是否字节级正确。判据同前三家：③D2H 字节级逐字一致 / ④H2D 换入后新旧块号分叉在场 /
  ⑥destroy 双形态（带在途任务排空 / 空闲幂等）。
- **载体与方法**：`scripts/xpu_offload_byte_probe.py` 直接驱动 `XpuCopyBackend`（昆仑芯
  `torch.cuda` 命名空间，`torch_xmlir` 符号重写）。
- **运行环境**：直接引用现行环境目录
  [`dev/images/kunlun-infer-vllm/v1/env/v1`](../../../images/kunlun-infer-vllm/v1/env/v1/)。
  同 `connector-ppu` 的情况：该 env 的 `env.baseline.yaml` 把 FlagMemory commit 钉在
  `58e60a9`（connector-kunlunxin 建样例时的点），本次实际装配的是现行 `dev-1.0@3b7cdb1`，
  未用 `--strict`（镜像 id 已单独自证一致）。

## 通过态与一个未解释的失败（2026-11-24）

| 机器/卡 | 结果 |
|---|---|
| P800-C，卡 0 | **FAIL**——H2D 换入步骤驱动报错 `kl3ChannelCheckErrors ... status=299` /
  `CUDA error: wait for noc idle timeout` |
| P800-D，卡 5 | **PASS**（五项判据全过） |

**P800-C 卡 0 的失败需要如实说明，不下结论**：报错签名与已登记的 **B21**（P800-A 机器级
D2H 拷贝挂死）几乎一致，但 B21 此前的结论是"只在旧机 P800-A 复现，新机 P800-C/D 测过四次
全部正常"（T-058）——这次在 P800-C 上出现，与那个结论不一致。报错发生前后，P800-C 上恰好
有其他用户新起了一批容器（`xbg-*-local-on-{2,3,4,5,6,7}`，疑似评测负载，占用了多张卡），
不能排除是资源争抢触发；但也不能排除是真实复现。**两个变量（机器 + 卡号）同时换了才拿到
P800-D 卡 5 的 PASS，不能确定是"换机器"还是"换卡"解决了问题**——这个问题本身没有被诊断
清楚，只是如实记录下来，已登记为新发现（见 flag_RT 控制台 anchors B28）。

## 证据

- P800-D 卡 5 PASS：[`evidence/summary.json`](evidence/summary.json)（机器可判）。
- P800-C 卡 0 FAIL 的完整日志未入库（`/tmp/connector-xpu-20261124-172617/run.log`，临时目录，
  机器重启或清理后会丢失；控制台会话记录里摘录了关键报错行）。

## 适用边界

- 只验证 `offload/xpu` 这一个连接器的搬运机制本身，不验证 vLLM/调度器何时决定触发搬运。
- 四家芯片（昇腾/平头哥/寒武纪/昆仑芯）本轮全部完成字节级验收样例；昆仑芯这一份额外带出了
  一个未解释的真机异常（见上），不能算作"昆仑芯这条路径已完全验证干净"。
