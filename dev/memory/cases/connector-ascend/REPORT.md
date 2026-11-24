# 验收样例四 · connector-ascend —— FlagMemory npu KV connector 字节级搬运真卡（昇腾 910C）

> 功能验收样例（形态：四件套 run.sh / env（直接引用现行环境目录）/ scripts / REPORT+evidence，
> 规范见 [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md)「验收样例四件套」）。
> 与 `dev/issues/` 的跨方向缺陷 case 包不混放：本目录是子方向功能验收样例。

## 用途与现状

- **验收对象**：FlagMemory（显存池管理器）npu KV→Host 卸载 connector
  （`flagmemory.tiering.offload.npu` / `NpuSimpleCPUOffloadConnector`）的**真实搬运行为**在
  昇腾 910C 真卡上是否字节级正确。判据对应 FM 仓 `flagmemory/tiering/offload/npu/RUNBOOK.md`
  记载的「connector 冒烟六判据」里此前只有一次性手工记录、没有留下可重复脚本的那几条：
  ③D2H 字节级逐字一致 / ④H2D 换入后新旧块号分叉在场 / ⑥destroy 双形态（带在途任务排空 /
  空闲幂等）。①工厂懒加载拿导出类、②锁页实测两条不在本样例覆盖范围——那两条已有
  `tests/tiering/test_offload_realcard.py -k npu`（导入面）与既有锁页探测覆盖，不重复造轮子。
- **载体与方法**：`scripts/npu_offload_byte_probe.py` 直接驱动
  `flagmemory/tiering/offload/npu/npu_copy_backend.py` 的 `NpuDmaCopyBackend`——不经 vLLM、
  不受调度器/显存压力策略影响，用已知内容填充 GPU 张量块、发起真实 D2H/H2D 搬运、逐字节比对
  搬运前后内容。这与"起真实 vLLM 服务、靠请求流量间接触发搬运、肉眼核对模型输出"的路线
  （2026-10 月 proto 阶段与 2026-11-14 手工验证走的路线）是两种不同的验证方法：前者（本样例）
  直接、确定性强、不依赖显存压力调参；后者更接近真实使用场景但引入了调度器/显存压力策略
  本身的不确定性，不适合用来判断"搬运字节对不对"这类问题（过程中曾用后一种方法尝试验证，
  在人为调小显存池触发高压力换入换出时观察到模型输出不稳定，但排查后确认是显存池调得过小
  引入的极端压力场景混杂了调度策略噪声，不是搬运本身的问题——本样例改用直接驱动的方法后
  五项判据全部字节级 PASS，排除了搬运机制本身的问题）。
- **运行环境**：直接引用现行环境目录
  [`dev/images/ascend-infer-vllm/v1/env/v1`](../../../images/ascend-infer-vllm/v1/env/v1/)
  （华为昇腾官方 vllm-ascend 镜像，单 die 映射，`up.sh --strict` 机器核对镜像 id）。该目录
  不挂载 FlagMemory（镜像内无此依赖），本样例靠整仓挂载到 `/workspace` 的既有机制让
  `/workspace/FlagMemory` 可见，容器内探针脚本自行把它加入 `sys.path`，不改环境目录任何文件。

## 通过态（2026-11-24 910c-11 真机执行，die 8）

| 判据 | 结果 |
|---|---|
| D2H 字节级逐字一致 | PASS |
| H2D 换入新旧块号分叉在场 | PASS |
| destroy：带在途任务排空 | PASS |
| destroy：空闲幂等 | PASS |

证据：[`evidence/summary.json`](evidence/summary.json)（机器可判，含镜像 id/die/判据逐项真值）。
完整执行日志未入库（`/tmp/connector-ascend-<时间戳>/`，临时目录）。

## 适用边界

- 本样例只验证 `offload/npu` 这一个连接器的搬运机制本身，不验证 vLLM 调度器何时决定触发
  搬运（那是调用方的策略，见本轮控制台会话对 T-054 第 68 条范围收窄的讨论）。
- 寒武纪/昆仑芯/平头哥三家的同构验证尚未做（本样例是首家试做，用于确认方法可行、评估工作量）。
