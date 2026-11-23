# P800 平台进度与待办（最近更新：2026-11-23）

## 当前状态：⚠️ 机器驱动故障中，实测暂停

- 2026-11-18 起 P800-A 三张配额卡（2/3/7）对新计算上下文全部挂起
  （bge-reranker 加载崩溃毒化驱动，与 11-15 同模式），**待管理员修复**
- 管理员修复后按下方"恢复后清单"继续

## 已完成（FlagScheduler × P800）

| 项 | 结果 | 证据 |
| --- | --- | --- |
| 功能单元测试 | 251 过 / 4 跳过 / 0 败（官方脚本口径） | verify-records/ |
| 32 请求映射正确性验收 | 三到达顺序全过（32/32 恰好一次回填） | data/flagscheduler-32req-summary.json |
| kunlun 设备画像登记 | verified（预算 32/32768/4096） | FlagScheduler `sched_engine/devices/registry.py` |
| n64 / n192 组批基准 | A 87.6 req/s；C 分桶 p50 -83% | data/ |
| **vllm-kunlun 0.11 双缺陷定案** | ①缓存命中扭曲 ②跨 prefill 步读无效 KV；生成路径同样中招 | investigation/INVESTIGATION_20261117.md |
| 厂商缺陷报告 | 两个最小复现 + 触发边界 + 双路径证据 | investigation/KUNLUN_BUG_REPORT_20261118.md |

## 恢复后清单（按序，约 15 分钟）

1. **embed sanity**（30s）：新基座（flagos + plugin-FL dev-1.0 修复版）上
   `EMBED_OK dim=1024`——顺带复验插件修复在本机生效
2. **FlagScheduler 新基座验收**：单测 + 32 请求验收（vllm 0.20.2 与 910C
   同版本线，跨平台可比性改善）
3. **缺陷一/二在新栈重检**：vllm-plugin-FL 是另一套实现，缓存命中/跨步
   行为必须重新检验——结论可能改写 P800 数值口径
4. **score()/classify()**：Qwen3-Reranker-0.6B（本地已有，标准键名）；
   **避开 bge-reranker**（其加载会让整机再次卡死）
5. 每步结果回写本文件与 data/（同步纪律见 ../BOOTSTRAP.md）

## 悬置事项

- FlagScheduler 主仓已按双仓分工瘦身（reports/results 迁入本分支）——
  详见 FlagScheduler README 变更说明
- vllm-plugin-FL 修复未随镜像重打包：等 `kunlun-infer-vllm` 升版后
  核对打包 commit ≥ `039fb46`（issue 文档的分版表态要求）
