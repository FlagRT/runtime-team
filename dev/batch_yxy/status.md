# FlagScheduler（调度方向）进展

更新日期：2026-11-04

## 已完成进展

**FlagScheduler 已独立成仓（`FlagRT/FlagScheduler`，v0.2.0）：节点级统一推理调度引擎完成六层框架落地，246 项测试全绿，在线调度的终态语义（超时/取消/降级基础）与观测闭环可用；此前在 NVIDIA A6000 完成动态组批真机 A/B/C 验收（长度分桶合批吞吐较直连 +20%）。**

| 成果 | 当前进展 |
|---|---|
| 动态组批与长度分桶 | ✅ `BatchCoordinator` 决策核心：固定桶/相邻桶合并/padding 硬约束/三维预算/优先级；910C 历史线开发，GPU3 真机验收（数值等价 max diff 2.83e-3） |
| 真机性能对照（A/B/C） | ✅ 192 条混合长度负载：统一入口 batch=32 吞吐 561 req/s（较直连 **+20%**），自定义 scheduler 无可测开销；决策层微基准 <0.3% 非瓶颈。证据 `artifacts/nvidia_gpu3_perf_20261025/` |
| 调度引擎四层架构 | ✅ 15 类搜广推模型契约 + 6 家国产芯片设备画像注册表（910C 已验证、昆仑芯/寒武纪/平头哥/壁仞/燧原待回归） |
| FlagScheduler 节点运行时 | ✅ CLI/发现/版本锁定启动计划/scheduler catalog/常驻队列/回环 HTTP（171 项测试基线） |
| PyTorch 推理后端 | ✅ `torch-predict`：进程内引擎（CPU/CUDA 一等，device 显式不回退、fp16 仅 cuda、批内单次 H2D）；DeepFM/DIN/MMoE 三契约（nnz/sequence/sample 三种计量族）；CPU e2e 20/20，GPU 链路待空闲卡补证据 |
| v0.2.0 六层调度框架 | ✅ 四终态机（completed/failed/timed_out/cancelled）+ 分段耗时；**slo_ms 排队阶段强制**（超期未派发收割为 timed_out，不占设备）；**全链路取消**（未派发移除即 cancelled / 已派发 cancel-pending、执行完丢弃结果，无设备抢占）；aging 防饥饿（可选）；`/metrics` Prometheus 导出 + 结构化事件；planner/execution/observe/profiles 四子包 + 模型族策略模板 CLI。**246 passed / 2 skipped** |

## 仓布局与报备状态

- **代码与版本锚**：https://github.com/FlagRT/FlagScheduler（独立仓，`main` 唯一协作线，v0.2.0，版本锚 commit `1e157db`）。报备五项已齐：仓址/目录名 `FlagScheduler`/协作线 `main`/可见性 Private（待定稿）/名单（含 rag_ljy）；**请总组收录 `dev/repos.yaml` 并给袁先乙、刘继阳 write 权限**。
- 本目录 `dev/batch_yxy/` 保留历史看板与昇腾分支记录；FlagScheduler 新开发全部在独立仓。
- 本地两处 docs 改动（2026-10-25 GPU3 真机与性能对照记录）内容已进独立仓，本仓未提交，是否入档由总组口径定。

## 与 dev/scheduling 的两线关系（报备对齐点）

- **接口边界**：FlagScheduler 提供并维护稳定接口——Planner/Backend 协议（含 v0.2.0 可选 cancel/sweep_expired 钩子）、`schema_version` 配置、CLI 与回环 HTTP `/v1` API、`/metrics` 指标。调度方向共建的通用策略不在公共仓内联评审，以版本锚引用 FlagScheduler 接入。
- **验证平台**：现役 NVIDIA 4×A6000（Conda `vllm` 0.21.0 / `vllm_add` torch 2.13+cu130）；910C 为历史验证平台，真机回归待有空闲卡。

## 主平台口径（对应课题「≥5 家国产芯片」验收锚点）

调度核心（dynamic_batching/sched_engine 四层架构）与芯片解耦，设备画像层已登记 6 家国产芯片契约；当前在 NVIDIA + vLLM 上完成功能与性能基线，国产芯片验收路径：通过设备画像 + vllm-ascend 环境复用同一套测试与合批策略，按画像逐卡回归（910C 联调待办已在独立仓挂账）。

## 下一阶段

1. GPU 链路补证据：torch-predict 的 cuda/fp16 e2e + GPU 测试用例（GPU3 空闲后，命令已备好）。
2. 多实例并发执行（InstanceRegistry 缝已留，N=2 双卡 A/B）与 CostVector 多维预算装箱。
3. rerank 执行适配器（vllm-score，族级覆盖 Qwen3-Reranker/BGE-Reranker）+ 其余 7 个 CTR zoo 条目。
4. 离线批处理调度（批量向量生成/索引构建/候选重排/大数据协同，验收方向 C）与 TUI。
