# FlagScheduler 插件与适配器开发

## Backend：先绑定执行语义，再谈统一调度

每个 binding 的工厂接收 `BindingConfig`，创建一个 Backend 实例。CLI/节点按配置显式装载插件，发现框架时不装载插件。

```python
from flagscheduler.contracts import ExecutionOutput, positive_int

class MyBackend:
    def __init__(self, binding):
        self.options = binding.options
        # 在这里加载当前 binding 的模型/客户端，不在模块导入时加载。

    def count_cost(self, payload):
        # 与执行输入相同的 tokenizer/特征预处理，不使用字符数估算 token。
        # 纯 CPU、无 I/O、无线程间执行状态变更；执行时节点仍可调用它。
        return positive_int(payload["sample_count"], "sample_count")

    def execute(self, inputs):
        # 可调用本地 Python、ONNX、TF/Triton 客户端等。
        # 输入项携带 ID、priority、deadline_mono_ns 和 metadata。
        return [ExecutionOutput(item.request_id, {"score": 0.5}) for item in inputs]

    def close(self):
        # 只释放自己拥有的资源，不杀全局框架进程。
        pass
```

输出必须是 JSON 可序列化的值，禁止 NaN/Inf；多头、分数、向量和字典都可以。即使后端返回顺序变化，也必须保留对应请求 ID。缺项、重复、未知 ID 会令整批失败。输入校验、特征 shape、词表、截断、normalize/后处理、模型输出头命名都应在真实后端内完成。

直接源码插件用配置：

```json
{
  "backend": "python:my_package.backend.MyBackend",
  "options": {"model_path": "/local/weights"}
}
```

插件所在目录加入当前运行环境的 PYTHONPATH；无需修改 flagscheduler 的 CLI。正式包可用标准 entry point：

```toml
[project.entry-points."flagscheduler.backends"]
my-backend = "my_package.backend:MyBackend"
```

随后配置 `backend: "my-backend"`。entry point 重名或不存在显式失败。示例见 `examples/flagscheduler_custom_backend.py`。在当前“只修改 batch_yxy”约束下，开发/测试插件也放本目录，暂不安装到外部环境。

十五类模型注册表是任务描述，不能直接推断执行方式。例如 Qwen reranker 需要联合上下文及 yes/no 评分适配，Jina 需要 task adapter，DIN 需要一致的行为序列和特征 schema。建议每个后端附自己的输入输出样例和与原框架直接调用对照，不共用 Mock 特征计数器作为真实计量。

## Planner：后续策略的插入点

Planner 工厂签名为 `factory(policies=..., clock=...)`，协议见 `flagscheduler.contracts.Planner`：

- `submit(BatchRequest)`：校验、按兼容 lane 排队、固定策略快照。
- `poll_ready(now_mono_ns)`：一次性输出可执行 BatchPlan，不重复派发。
- `next_wakeup_mono_ns()`：下一等待截止，用同一单调时钟。
- `flush()`：停止接纳后排空，仍须满足硬约束。
- `pending_count`：当前未派发请求数。

通过 `flagscheduler.planners` entry point 或 `planner: "python:my_package.planner.MyPlanner"` 注册。策略独立于 CLI、HTTP、Backend，不在 UI 中重写。

建议先组合已有 `BatchCoordinator`，只改候选优先规则或准入，不复制组批算法。SLO 策略可读取 `deadline_mono_ns`，但要先测定运行时、定义过期请求的终态、补全取消/降级契约后再宣称 SLO 感知执行。当前 Planner 协议没有独立 drop/cancel 通道，排队过期拒绝需要先演进契约和 `schema_version`，不要悄悄丢弃请求。

## vLLM 内部 scheduler

`flagscheduler/integrations/vllm_scheduler.py` 提供同步与异步的原生继承入口；当前没有新的内部算法。继承入口的作用是验证替换配置与为后续策略提供明确文件位置。

将专属实现放在新的版本模块中，声明精确 vLLM 版本。使用 `scheduler inspect` 查看安装位置与候选，`scheduler validate` 检查导入/继承/签名；随后在空闲 GPU 上按 KV/cache、preemption、chunked prefill、pooling、同步/异步等适用路径验证。不要只重排 `waiting` 就认为已经兼容完整 SchedulerInterface。

vLLM 官方依据：[`SchedulerConfig.scheduler_cls`](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)、[`vllm serve --scheduler-cls`](https://docs.vllm.ai/en/latest/cli/serve/)。线上文档只作扩展入口依据，具体实现以目标环境源码和接口为准。

## TUI 的接入方式

TUI 首版先实现发现/环境选择、scheduler 计划预览、节点状态、最近批次、请求提交和结果查询。所有操作调用既有 Python 应用 API 或 `/v1/` 接口；不要启动子进程解析终端文字来获取状态。界面显示任务、binding、版本、验证状态，区分只读检查、生成配置和启动进程。

框架 discovery、GPU inventory、插件加载/启动等可能阻塞，放在 UI 工作线程；Planner 更新仍由节点事件循环串行执行。TUI 是后续交付项，本轮不引入 Textual/Rich 等依赖，也不预实现未确定的交互。
