# 验收样例六 · connector-mlu —— FlagMemory mlu KV connector 字节级搬运真卡（寒武纪 MLU590）

> 功能验收样例（形态：四件套 run.sh / env（直接引用现行环境目录）/ scripts / REPORT+evidence，
> 规范见 [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md)「验收样例四件套」）。
> 方法与 [`connector-ascend`](../connector-ascend/REPORT.md)、[`connector-ppu`](../connector-ppu/REPORT.md)
> 同构（同一轮工作的第三家）。

## 用途与现状

- **验收对象**：FlagMemory（显存池管理器）mlu KV→Host 卸载 connector
  （`flagmemory.tiering.offload.mlu` / `MluCopyBackend`）的**真实搬运行为**在寒武纪 MLU590
  真卡上是否字节级正确。判据同前两家：③D2H 字节级逐字一致 / ④H2D 换入后新旧块号分叉在场 /
  ⑥destroy 双形态（带在途任务排空 / 空闲幂等）。
- **载体与方法**：`scripts/mlu_offload_byte_probe.py` 直接驱动 `MluCopyBackend`（寒武纪
  `torch_mlu` 原生命名空间 `torch.mlu.*`，不像昆仑芯那样是符号重写），不经 vLLM。
- **运行环境**：直接引用现行环境目录
  [`dev/images/cambricon-infer-vllm/v1/env/v1`](../../../images/cambricon-infer-vllm/v1/env/v1/)。
- **已知环境缺陷绕过（B24，登记处 flag_RT 控制台 anchors）**：HWJ_1 宿主 `python3.8.10`
  （本机确认无 3.9 以上版本可做 shim），`dev/lib/baseline.py` 的 PEP604 写法在该机直接报错，
  `up.sh` 不可用（strict/非 strict 皆然，env_packages 校验步骤即崩）。按既有先例（2026-11-22
  寒武纪 env/v1 真机验证同款绕过，见该轮会话记录）：**镜像 id 已手工核对一致（步骤 1）后，
  直接 `docker compose up -d` 起容器，不经 `up.sh`/`baseline.py`，未改动任何脚本或配置文件**。
- **一条可忽略的 stderr 噪音**：容器内 `import torch` 链路触发 vLLM 的 `vllm_fl` 插件自动加载，
  因镜像内缺 `triton` 包而报 `Failed to load plugin fl`——这是镜像自带、与本探针无关的已有
  现象（探针本身不走 vLLM/vllm_fl 路径），不影响判据结果，如实记录不算发现。

## 通过态（2026-11-24 HWJ_1 真机执行，MLU 卡 0）

| 判据 | 结果 |
|---|---|
| D2H 字节级逐字一致 | PASS |
| H2D 换入新旧块号分叉在场 | PASS |
| destroy：带在途任务排空 | PASS |
| destroy：空闲幂等 | PASS |

证据：[`evidence/summary.json`](evidence/summary.json)（机器可判）。完整执行日志未入库
（`/tmp/connector-mlu-<时间戳>/`，临时目录）。

## 适用边界

- 只验证 `offload/mlu` 这一个连接器的搬运机制本身，不验证 vLLM/调度器何时决定触发搬运。
- 四家芯片中，本轮已完成昇腾、平头哥、寒武纪三家；昆仑芯是否铺开待定（P800-C/D 随时可做，
  不需要 VPN）。
