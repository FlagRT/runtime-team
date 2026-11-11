# PPU Inference v2 实测结果

**Qwen3-Embedding 的已测单卡、TP2 和采集分析报告链路可用；性能资格与 OneRec 设备端生成验证未完成。** 本文依据 11-03/04 实测与固定提交文档整理，复核日期：2026-11-05，没有新跑 PPU 硬件。

| 项目 | 测试身份 |
|---|---|
| 适配源码 | `ff1f834f`，现并入整合提交 `315a2c32`；PR #12 待评审，独立 PR #11 已关闭未合并 |
| 模型 / 设备 | Qwen3-Embedding-0.6B；两台同型 16 卡 PPU 主机，实测单卡与 TP2 |
| 默认镜像 | `flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64`，ID `1ba7ead15378…` |
| coverage 镜像 | `flagperf/thead-ppu-runtime-gems:1.0.0`，ID `087a234b7abe…`；只在首台部署 |
| 两机驱动 | `1.5.1-1d747a` / `1.5.3-5239a4`，在当轮契约范围内 |

## 实测结果

| 范围 | 实测结果 | 限制 |
|---|---|---|
| 单卡分析链路 | record → import → inspect → report 完成 | 不外推其他模型 |
| 预设覆盖 | perf、perf-lite、accuracy、modules、perf+memory、coverage 已测 | coverage 使用单独镜像，不表示两机所有预设重复验收 |
| 精度 A/A | 同栈重复 oracle first_inequal=null | 不是独立精度参考，也不证明单卡/TP2 相同 |
| modules | 首台封存采集 5,293,952 源记录 | 数据规模，不是模型吞吐或全部算子数量 |
| TP2 | nccl 后端；两台各双 rank 返回 0，导入 complete；通信分类 comm=5 | 分类计数不等于全部通信都已唯一归因 |
| delivery | publish → verify → restore 跨容器通过，19 files / 168,472 bytes | 恢复保持依赖映射；不表示所有任意包均可迁移 |
| 参数面 | BF16 / FP16 / FP32、batch 8、eager / sdpa 的已测配置输出有限且重复确定 | 不是参数笛卡尔积全量验收 |
| 软件回归 | 11-04 两机各 17 passed | 该计数属于历史 PPU 分支，不与整合套件叠加 |

## 保留的问题

| 问题 | 当前处理 |
|---|---|
| 性能资格 | 时钟一致性未完成验收，performance_disposition 为 `not_qualified`，数值仅作观测 |
| OneRec | 权重未到位，PPU generate 端到端未验证；不能据 P800 结果推定 PPU 通过 |
| FlagGems | coverage 注册 6 个算子：mm/add/mul/softmax/rms_norm/silu；layer_norm 未注册 |
| 新模型入口 | 内置 generate 为 OneRec 专用；其他生成模型需自定义目标 + attach，并独立验证 |
| 整合后范围 | 本次整合复核在 P800 上进行，不代替 PPU 在整合提交上的新增硬件验收 |

来源：11-04 组会实测摘要、健康检查记录与固定提交适配指南。本文未附 PPU 原始 capture/日志；对本轮未直接复读原件的结果，仅按已有文档交叉核对，不声称已重新逐文件验证。[固定提交指南](https://github.com/FlagRT/FlagPerf/blob/315a2c32829fa080ebfe0ff39416562d6c8da03e/docs/thead/inference-v2.md)、[PR #12](https://github.com/FlagRT/FlagPerf/pull/12)；来源指纹见[索引](README.md)。
