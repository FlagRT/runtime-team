# PPU Operation 实测结果

**两入口各完成 308 个适用组合：299 项正确性通过、9 项失败；正确性通过项中仍有 9 个缓存路由缺口。** 代码随 [PR #6](https://github.com/FlagRT/FlagPerf/pull/6) 合入 dev-1.0。本文采用 10-27 更正后的归因边界，复核日期：2026-11-05。

| 测试条件 | 取值 |
|---|---|
| 矩阵 / 对照实验 | 2026-10-23 / 2026-10-26；矩阵源码 `00202323` |
| 镜像 | `flagperf-ppu-operation:654e2a3a`，ID `b3246f3fa617…` |
| 协议 | daily，原生 / FlagGems 双路径，矩阵 profiling off；代表组合另做 full 采集 |
| 环境 | 共享宿主；local 首轮中断后由 28 个独立续跑补齐，原中断记录保留 |

## 实测结果

| 入口 | 适用 | correctness passed | routing passed / partial | 任务 partial / failed | 类型不适用 |
|---|---:|---:|---|---|---:|
| Docker `c8cccff934e7` | 308 | 299 | 290 / 9 | 299 / 9 | 420 |
| local `eee0accf05f0` + 续跑 | 308 | 299 | 290 / 9 | 299 / 9 | 420 |

`partial` 表示部分条件或证据未满足。共享宿主不能取得独占性能资格；另有 9 项缓存路由缺口，不能把全部 partial 只归因于共享环境。

| 失败组 | 数量 | 已确认事实与边界 |
|---|---:|---|
| mm/addmm/bmm FP32 | 5 | 公共容差下可复现超差；不能根据误差签名识别内部格式，重复设置已为 False 的开关不能证明开关无效 |
| layer_norm FP16/BF16 原生 | 2 | 低精度输入 + FP32 参数在所测 PPU 路径报 dtype mismatch；CPU FP16 对照可执行，不声称已完成 CPU BF16/CUDA 对照 |
| FlagGems mm/linear FP32 | 2 | 原矩阵 300 秒超时；最小 mm 到 1500 秒仍无完成标记，linear 实际 301.2 秒完成、未验正确性；不能仅凭超时断言永久死锁或编译器缺陷 |

九项全部保持 failed，仍由后端/算子与框架维护者联合复核。撤回“九项根因全部闭合、全部与框架无关”的早期表述，不转换 dtype、不放宽容差掩盖失败。

| 补充能力 | 证据与限制 |
|---|---|
| PPU-TRACE-001 | cross_entropy_loss、mv、prod 的三种浮点组合共 9 个缓存路由缺口 |
| Asight | 7 个组合各采 5 个 kernel、205 条 metric；含 5 个原生 section，非全矩阵计数器验收 |
| 官方 P2P | 16 卡 240 有向对、4 类矩阵 data matched；共享环境工具实测不等于 Base 全设备对独占资格 |
| 16-rank AllReduce | 共享功能 smoke；20 ITERS 窗口不足 15 秒，不构成五次独占资格 |
| FP8 | 当前 runtime 路径未提供，不能据此认定硬件无 FP8 |
| runtime | candidate / `validated:false`，16-target 监控等待补 |

来源：10-27 `matrix-audit.json`、最终阶段验收报告、组会运行分析与公开文档；原始张量、日志和计数器文件不随本文提供，来源指纹见[索引](README.md)。[公开验收及归因更正](https://github.com/FlagRT/FlagPerf/blob/41ab2dce5a94906919905437ebc8dec97a3d36c5/docs/thead/validation.md)、[运行机制](https://github.com/FlagRT/FlagPerf/blob/41ab2dce5a94906919905437ebc8dec97a3d36c5/docs/thead/operation.md)。
