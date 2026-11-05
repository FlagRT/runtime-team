# PPU Toolkit 实测结果

**已登记的 12 个 Case 测量、监控和诊断均通过；不包含失败的 INT8 官方工具路径，不构成性能资格。** 复核日期：2026-11-05；本文为人工摘要，未重新运行硬件负载。

| 项目 | 测试条件 |
|---|---|
| 运行 ID | `20261020T140938Z-aea8308d` |
| 镜像 | `flagtree-ppu-3.6-dev:202608`，本地 image ID `be1606e81fb2…` |
| 设备 | 单机两张 PPU；原始记录含独立 BDF/UUID 绑定 |
| 测量实现 | 官方 sailbandwidth / p2pBandwidthLatencyTest 与独立 acblas 子进程 |
| 规模 | 8192³，传输 512 MiB，P2P 40,000,000 bytes；repeat=5，warmup=5 |

## 实测结果

| 检查 | 结果 |
|---|---|
| 覆盖 | manifest 中 12/12 Case 的测量、监控、诊断与最终状态均为 passed |
| 清理与后检 | passed，租约已释放 |
| 性能资格 | `not-qualified`，candidate / `validated:false` |
| 容器退出码 | 原记录为 **2**，不改写为 0；runner 先将监控未收尾状态记 partial，宿主完成监控合并后按 manifest 判断最终分层状态 |

| 指标 | 本轮观测值 | 解释 |
|---|---|---|
| FP32 / FP64 | 34.504 / 5.515 TFLOPS | acblas 原生 GEMM |
| FP16 / BF16 / TF32 | 135.064 / 134.686 / 67.555 TFLOPS | 独立精度路径，不互作精度替代 |
| H2D / D2H | 含 CE、SM 两通道结果 | 按模式分列解释，不用混合中位数当作单通道性能 |
| D2D | 约 1050.630 GB/s | 本轮官方搬运工具口径，不等同 Base 的 PyTorch 带宽口径 |
| 容量 | Total 98,304 MiB | 静态容量查询，不是性能负载 |
| P2P | 所测单向约 46.5、双向约 88–90 GB/s | 限本轮设备对，不能外推全机或跨机 |

## 保留的问题

| 项目 | 边界 |
|---|---|
| INT8 | acblasGemmEx 输出错误、acblasLtMatmul 崩溃，排除在这 12 项之外；Base INT8 通过不能替代它 |
| 诊断含义 | 正确性回读、工具校验、契约覆盖与健康观测，不是 DMI 式阈值认证 |
| 历史 harness 缺陷 | P2P stdout 交错与 BF16 小缓冲 layout 越界已修复，旧误判撤回；不能继续归咎厂商 |
| 独占资格 | Toolkit 通过不关闭 Base 16-rank 独占资格、16-target 监控等待验项 |

来源：同 run 的 `summary.json`、`toolkit-evidence/manifest.json`、封存 `evidence_runner.py` 与组会《PPU-Toolkit-实验报告》；原始文件未随本文复制，来源指纹见[索引](README.md)。[公开验收记录](https://github.com/FlagRT/FlagPerf/blob/41ab2dce5a94906919905437ebc8dec97a3d36c5/docs/thead/validation.md)、[INT8 问题](https://github.com/FlagRT/FlagPerf/blob/41ab2dce5a94906919905437ebc8dec97a3d36c5/docs/thead/acblas-issues.md)。
