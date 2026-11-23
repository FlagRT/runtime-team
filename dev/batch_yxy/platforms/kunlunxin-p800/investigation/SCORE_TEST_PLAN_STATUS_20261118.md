# score()/classify() 入口测试——计划与现状（2026-11-18）

> 背景：embed 与 generate 两条入口的缺陷已定案（见
> [INVESTIGATION_20261117.md](INVESTIGATION_20261117.md)）。本文件记录
> 第三条入口 score()（及同族的 classify()）的测试计划、已尝试结果与
> 阻断原因。**测试以 FlagScheduler 为主体**（用户指定的口径）。

## 1. 测试目标与顺序

1. score() 平台可用性确认（能否加载、能否出分）
2. **FlagScheduler 功能验收**（主体）：reranker 模型画像登记 → 32 请求
   三到达顺序调度验收（恰好一次回填）→ legacy/scheduled 对照
3. 顺带验证缺陷一/二在 score 路径的表现（CLS 池化模型是否免疫缺陷一）

## 2. 本地权重盘点（已完成）

| 模型 | 架构 | 状态 |
| --- | --- | --- |
| `bge-reranker-base`（/data1/.../p800-models/，2.1GB） | `XLMRobertaForSequenceClassification`（单标签→score，CLS 池化） | 权重完整；**加载即崩（见 §3）** |
| `Qwen3-Reranker-0.6B`（1.2GB） | `Qwen3ForCausalLM`，**标准键名**（带 model. 前缀，无需重映射） | 首选待测 |
| `Qwen3-Reranker-8B`（16GB） | 同上 | 备选 |
| classify 专用（多标签分类头） | — | **本地无**；score/classify 共用 pooling 流水线，缺陷发生在池化之前的 KV 寻址，score 结论可迁移 |

FlagScheduler 侧：`sched_engine/models/registry.py` 已声明
`Qwen3-Reranker`（repo 指向 0.6B）与 `BGE-Reranker` 两个画像（declared）；
`VllmEmbedFrameworkAdapter` 的执行协议是负载无关的
（`executor.embed(payloads) -> list`），score 执行器可直接复用，框架层零改动。

## 3. 已尝试结果（2026-11-18，全部阻断在平台层）

| 尝试 | 结果 |
| --- | --- |
| `bge-reranker-base` + `task="score"` 加载 | ❌ **glibc 堆损坏崩溃**（`corrupted double-linked list`，引擎初始化阶段）。两次独立复现。**疑似平台缺陷三**：XLMRoberta/score 路径在 vllm-kunlun 0.11.0 上加载即崩 |
| 崩溃的副作用 | ❌ **驱动再度全局卡死**（与 2026-11-15 故障同模式：崩溃进程毒化驱动，所有新建计算上下文挂起；已运行进程不受影响）。2/7 号卡实测 60 秒矩阵乘挂起，重启容器无效 |
| `Qwen3-Reranker-0.6B` + `task="score"` 加载 | ⏸ 引擎初始化挂起（发生在堆损坏崩溃**之后**，无法区分是 score 路径问题还是当时栈已被毒化——**需在健康机器上重测**） |
| FlagScheduler 功能验收 | ⏸ 被 embed 路径都无法加载阻断 |

## 4. 恢复后的执行清单（脚本与流程已备好）

1. embed sanity（30 秒确认栈健康，`~/fgw/ctools/` 现成）
2. `Qwen3-Reranker-0.6B` 试 `task="score"`（**避开 bge——其加载会让整机再次卡死**）
   - 若能出分：跑缓存命中重复调用 + 跨步两缺陷小实验（~3 分钟）
3. FlagScheduler 主体验收（按 KUNLUN 设备同样的登记流程）：
   - reranker 画像翻 verified（verified_on=`vllm-score+kunlun-p800`，预算与
     embed 口径一致）+ 同步 `test_model_registry` 断言
   - `examples/sched_engine_p800_score.py`：ScoreExecutor（token_count 同源
     tokenizer、execute 走 `llm.score`）+ 复用 `VllmEmbedFrameworkAdapter`
     + KUNLUN 设备画像，32 对 (query, doc) 三顺序验收 + legacy 对照
   - **全程安全包络**：`enable_prefix_caching=False` + 单次调用 ≤16384 tok
     （策略 max_batch=8、max_total_tokens=8192，单步内执行）
4. 结果归档：本文件更新 + 缺陷证据 JSON 补充 score 章节

## 5. 环境与卡位约定

- 容器：`xianyiyuan-vllm-kunlun`（chenyunlong/vllm-kunlun-runtime:0.11.0，
  引擎配方同 INVESTIGATION §5；p800-models 目录以同路径只读挂载）
- 配额卡：宿主 **2、3、7** 号（空闲时可用；XPU_VISIBLE_DEVICES 为容器内
  相对编号，主容器 0=物理2号卡）
- 2026-11-18 故障期间：3 号卡他人任务正常（未受影响），我方无任何残留进程
