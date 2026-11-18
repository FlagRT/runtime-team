# 验收样例二 · connector-kunlunxin —— FlagMemory xpu KV connector 真卡（昆仑芯 P800）

> 功能验收样例（形态：四件套 run.sh / env / scripts / REPORT+evidence，规范见
> [`dev/images/CASE-TEMPLATE.md`](../../../images/CASE-TEMPLATE.md)「验收样例四件套」）。
> 与 `dev/issues/` 的跨方向缺陷 case 包不混放：本目录是子方向功能验收样例。

## 用途与现状

- **验收对象**：FlagMemory（显存池管理器）xpu KV→Host 卸载 connector
  （`flagmemory.tiering.offload.xpu` / `XpuSimpleCPUOffloadConnector`，vLLM 0.20.2
  `SimpleCPUOffloadConnector` 的昆仑芯适配）在昆仑芯 P800 真卡上的功能面。判据与命令
  以 FM 仓 `flagmemory/tiering/offload/xpu/RUNBOOK.md` 现行口径为准，载体 = 判据三件
  （`tests/tiering/` 三个 realcard 文件，共 **5** 用例）：
  栈可导入取到导出类（×1）· 工厂懒加载/锁页实测/D2H·H2D 逐字/换入分叉源行（×1）·
  T8 负例 + 异步轨全链（确认前不落账、审计三等）+ destroy 双形态（×3）。
- **运行环境**：直接引用现行环境目录
  [`dev/images/kunlun-infer-vllm/v1/env/v1`](../../../images/kunlun-infer-vllm/v1/env/v1/)
  （tele4 镜像 + 单卡授予/xpuctrl + FlagMemory@v0.3.0 只读挂载 PYTHONPATH 生效 +
  network none），`up.sh --strict` 机器核对。
- **通过态**（2026-11-15 P800-A 首次真机执行，卡 7）：环境与导入面全绿，
  判据三件 **2/5**，未达通过态——阻塞点为**机器级 D2H 拷贝路径问题**（详见下节），
  非样例/环境/FlagMemory 代码缺陷。

| 判定项 | 判据 | 结果 |
|---|---|---|
| 环境核对 | `up.sh --strict` 全绿（镜像 id、FlagMemory 挂载 commit） | **通过**（起容器前后两轮 check 全 PASS，`evidence/summary.json` image/env 锚） |
| FlagMemory 版本 | 容器内 `import flagmemory` == 0.3.0（PYTHONPATH 生效口径） | **通过**（FM_VERSION 0.3.0） |
| 判据三件 | 三文件 5/5 pass（0 fail / 0 error / 0 skip） | **未达**（2/5：栈可导入 ×1 过、T8 负例过；backend 与异步轨 fail 于 `wait for noc idle timeout`，destroy setup error 同因） |
| 机器可判 | `evidence/summary.json` `result: PASS`，run.sh 退出码 0 | **未达**（`result: FAIL`，退出码 1；如实记录） |

## 失败点定位（2026-11-15，如实记录）

- **现象**：backend 与 tier-manager 两件里一切走 D2H/H2D 拷贝的用例（懒加载闭环、
  异步轨全链、destroy 的 bind fixture）死于 `torch.AcceleratorError: CUDA error: wait
  for noc idle timeout`（synchronize 处）；不走拷贝的用例（栈可导入、T8 负例）通过。
- **与 FlagMemory 无关的证据**：同日同镜像跑**纯 torch** D2H 探针（`torch.arange` 设备
  张量 → `pin_memory` 主机张量 `copy_` + `synchronize`，不经 FM 代码）在**卡 0 与卡 7**
  均挂死不归（`/tmp/s4-kunlun-infer-20261115/pure_copy_probe.py`，两设备形态各试）。
- **对照**：RUNBOOK 记载 2026-11-09 本机（P800-A）同镜像判据三件验证通过；当日与今日
  差异未定因（观察项：宿主驱动 5.0.21.47 vs 镜像 tag 栈代际 v5.0.21.43，不作因果断言；
  机上另有他人常驻负载与「卡 1/卡 2 分配挂死前科」）。判据结论 = **待机器拷贝路径
  恢复后重跑本样例**（单入口幂等，直接重跑 run.sh 即可）；不在本样例内修 FM 产品代码。
- 选卡记录：当日可用空闲卡 0/5/7（避开卡 1/卡 2 与有占用卡 3/4/6），执行取卡 7；
  收尾卡 7 显存回落 0MiB、样例容器删净（run.log 留痕）。

## 判据与口径

- 单入口：`XPU_CARD=<n> bash dev/memory/cases/connector-kunlunxin/run.sh`
  （选卡纪律：xpu-smi 挑零负载/零显存卡，避开 `AVOID_CARDS`（默认「1 2」，分配挂死前科）
  与有残留占用的卡；每次执行前重新核对；映射全 8 卡的空壳容器可共存，以 xpu-smi 实测为准）。
- 运行口径（xpu RUNBOOK 与 `test_xpu_tier_manager_realcard.py` 文件头的已验证命令）：
  `python -m pytest -m realcard -p no:cacheprovider`（conda env 绝对路径解释器），
  三件分开跑、失败不短路；FlagMemory 经 PYTHONPATH 生效；**容器不挂仓根**（仓根挂
  /workspace 会遮蔽镜像自带 vllm-plugin-FL 源树，触发 `Vendor 'kunlunxin' not found`，
  2026-11-15 二分实测，详见 env/v1 compose 头注与 README）。
- 自判定：退出码 0 / 判定行 `overall: PASS` = 5/5 通过；非 0 = 有失败或环境不满足。
- 外部依赖（基线外）：一张空闲 P800 卡；机器 P800-A（镜像 conda env 自带 pytest，
  零 pip 依赖）。

## 证据指针

- 机器可判结果：[`evidence/summary.json`](evidence/summary.json)（用例名→pass/fail、计数、
  镜像 id、FM commit 锚、容器内实装版本；2026-11-15 执行，result: FAIL 2/5）。
- 完整日志：执行机器 P800-A 的 `/tmp/connector-kunlunxin-20261115-073326/`（run.log、
  三件 pytest.log 与 junit.xml、xpu-smi 前后对照）；探查与纯 torch 拷贝探针留痕
  `/tmp/s4-kunlun-infer-20261115/`（完整日志不进仓）。
- 环境版本事实：`dev/images/kunlun-infer-vllm/v1/env/v1/env.baseline.yaml`。

<!-- changelog: 2026-11-15 新建（S4-B 验收样例二，四件套同构样例一）。
     2026-11-15 首次真机执行（P800-A 卡 7）：--strict 与 FM 0.3.0 导入全绿；判据三件 2/5，
     阻塞点为机器级 D2H 拷贝路径问题（纯 torch 探针卡 0/卡 7 均挂死；2026-11-09 本机曾全过）。
     待机器恢复后重跑。 -->
