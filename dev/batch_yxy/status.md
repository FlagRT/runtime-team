# FlagScheduler（调度方向）进展

更新日期：2026-11-04

## 已完成进展

**FlagScheduler 已独立成仓（`FlagRT/FlagScheduler`，v0.2.0）：节点级统一推理调度引擎完成六层框架落地，246 项测试全绿；在线调度的完整终态语义（completed/failed/timed_out/cancelled）、全链路取消、排队超期控制与 Prometheus 观测闭环已可用；动态组批在 NVIDIA A6000 真机验收中取得吞吐 +20%。**

| 成果 | 说明 |
|---|---|
| 动态组批与长度分桶 | `BatchCoordinator` 决策核心：固定桶/相邻桶合并/padding 硬约束/三维预算/优先级；GPU3 真机验收，原生与自定义 scheduler 数值等价（max diff 2.83e-3） |
| 真机性能对照（A/B/C） | 192 条混合长度负载：统一入口 batch=32 吞吐 561 req/s（较直连 **+20%**），自定义 scheduler 无可测开销；决策层微基准 <0.3%。证据 `artifacts/nvidia_gpu3_perf_20261025/` |
| 调度引擎四层架构 | 15 类搜广推模型契约 + 6 家国产芯片设备画像注册表（昇腾 910C 已验证，昆仑芯/寒武纪/平头哥/壁仞/燧原契约已登记） |
| FlagScheduler 节点运行时 | CLI/框架发现/版本锁定启动计划/scheduler catalog/常驻队列/回环 HTTP，171 项测试基线 |
| PyTorch 推理后端（torch-predict） | 进程内引擎，CPU/CUDA 一等路径（device 显式声明不回退、fp16 支持、批内单次 H2D）；DeepFM/DIN/MMoE 三契约覆盖 nnz/sequence/sample 三种计量族；CPU e2e 20/20 完成 |
| v0.2.0 六层调度框架 | 四终态机 + 分段耗时（queue_ms/execution_ms）；slo_ms 排队阶段强制（超期请求收割为 timed_out，不占设备）；全链路取消（未派发移除即 cancelled，已派发执行完丢弃结果）；aging 防饥饿；`/metrics` Prometheus 导出 + 结构化事件；planner/execution/observe/profiles 四子包 + 模型族策略模板 CLI（15 契约输出可直接被 parse_config 接受）。**246 passed / 2 skipped** |

## 仓布局与报备状态

- **代码与版本锚**：https://github.com/FlagRT/FlagScheduler（独立仓，`main` 唯一协作线，v0.2.0，版本锚 commit `1e157db`）。
- 报备五项已备齐：仓址 / 目录名 `FlagScheduler` / 协作线 `main` / 可见性 Private / 名单（含 rag_ljy）；请总组收录 `dev/repos.yaml` 并给袁先乙、刘继阳 write 权限。
- 本目录 `dev/batch_yxy/` 保留历史看板与昇腾分支记录；FlagScheduler 开发全部在独立仓进行。

## 与 dev/scheduling 的两线关系

- **接口边界**：FlagScheduler 提供并维护稳定接口——Planner/Backend 协议（含 v0.2.0 cancel/sweep_expired 钩子）、`schema_version` 配置、CLI 与回环 HTTP `/v1` API、`/metrics` 指标。调度方向共建的通用策略以版本锚引用 FlagScheduler 接入，不在公共仓内联评审。
- **验证平台**：NVIDIA 4×A6000（Conda `vllm` 0.21.0 / `vllm_add` torch 2.13+cu130）+ 昇腾 910C（历史验证平台）。

## 主平台口径（对应课题「≥5 家国产芯片」验收锚点）

调度核心与芯片解耦（四层架构 + 设备画像层）；当前在 NVIDIA + vLLM 上完成功能与性能基线，国产芯片按设备画像逐卡回归——同一套测试与合批策略直接复用。
