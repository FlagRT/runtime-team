# Device-context 接入现状

更新：2026-10-14。设备初始化由 [device.py](../src/rag_engine/device.py) 统一处理。
当前精排使用此接口；默认embedding由独立vLLM HTTP引擎执行。

## 当前使用方式

```bash
export PYTHONPATH="/workspace/dev/device-context/prototype${PYTHONPATH:+:$PYTHONPATH}"
export DC_BACKEND=ascend
python scripts/checks/check_device.py --device npu:0
```

PYTHONPATH指向共享Python模块所在目录，不会切换Python解释器。
`prepare_torch_device()` 选择backend、验证设备前缀与可见索引，调用
`runtime.use()`、`runtime.device_count()`、`runtime.set_device()`，再返回torch设备。
同一个RAG进程只选择一个runtime backend。

| 选择 | 行为 |
|---|---|
| `DC_BACKEND=ascend` | Ascend runtime，设备前缀npu |
| `DC_BACKEND=kunlun` | Kunlun runtime，XPytorch使用cuda前缀；需要匹配镜像 |
| `--device auto` | 选定backend的逻辑设备0，默认Ascend |
| `--device cpu` | 原生CPU路径 |
| `--device cuda:0`且未设置DC_BACKEND | 原生CUDA路径 |
| `RAG_DTYPE` | bfloat16/float16/float32；加速器默认BF16，CPU默认FP32 |

当前已有Ascend物理NPU7实跑证据，容器内为逻辑npu:0。

## 两个模型的边界

精排：`server.py` 的模型worker调用Transformers，经device-context初始化NPU。
vLLM embedding：`serve_embeddings.py` 独立启动pooling服务，设备由
`--device-index`/`ASCEND_RT_VISIBLE_DEVICES`控制；客户端不加载embedding权重。
该独立服务启动器补充CANN路径并强制 `TRITON_ENABLE_TASKQUEUE=true`。

runtime提供设备、内存、stream/event与错误分类等接口；匹配的torch、驱动、
镜像和模型算子仍由各厂商环境提供。RAG现已接入设备初始化及内存采样接口；
runtime错误恢复、内存感知准入及跨芯片完整验收尚未接入调度器。

## 设备内存探测

[memory_probe.py](../src/rag_engine/memory_probe.py) 通过 `runtime.memory_stats(ordinal)`
读取后端报告的 `total_mb` / `used_mb` / `free_mb`，单位为 MiB。查询API默认每秒
在独立采样线程读取一次，`/health.device_memory` 返回缓存读数、时间、年龄及错误状态；
HTTP健康检查不会同步调用驱动。采样绑定本地精排设备，不使用 `remote-vllm` 字符串
作为设备编号。采样失败会报告 `unavailable` 并清除数据，不影响查询服务。

```bash
python scripts/checks/check_memory.py --device npu:0 --count 5 --interval 1
python scripts/checks/check_memory.py --device npu:0 --count 3 --cross-process-mib 128
python scripts/runtime/serve_queries.py --device npu:0 --memory-probe-interval 1
```

独立检查每次只生成一个小型 `report.md`，位于 `artifacts/checks/memory_probe/<日期时间>-<ID>/`。
可选 `--cross-process-mib 128` 验证独立子进程128MiB分配及释放是否被探测到，
验证结束即释放，不作为常驻负载。
可用 `--no-memory-probe` 关闭API采样。CPU显示 `not_applicable`；未指定DC_BACKEND的
原生CUDA路径显示 `unavailable`，需接入支持该平台的device-context后端。

统一接口目前没有标明底层统计范围及是否使用回退；因此读数标记为
`scope: backend_reported_unverified`。Ascend实现查询失败时可能回退到当前进程的
PyTorch分配量，不能保证覆盖独立的vLLM进程。其他后端缺少字段时显示 `partial`，
不自行推算缺失的可用内存。计算利用率不属于这个接口。

本次仅启用可观察的内存探测；准入上限仍是请求数量，后续需确认统计范围、采样
时效性及请求内存预算后再使用读数控制准入。

2026-10-14物理NPU7实测：独立进程128MiB分配后，已用158→348MiB，退出后回到
158MiB（增量含运行环境开销）。该结果确认本次Ascend后端能观察跨进程占用；
后端仍可能使用兼容回退，因此API继续保留未验证范围标记。
完整测试120项通过，详见 [单文件实测报告](../artifacts/checks/memory_probe/2026-10-14_132823-f0ad80/report.md)。
真实Elasticsearch API运行期间173次采样均正常且未过期；2条串行及同样2条并发查询
全部成功，结果与串行参考一致，确认后台探测可与模型执行同时运行。

当前操作见 [运行指南](operations.md)。
[2026-09-22原始接入研究](history/device-context-study-2026-09-22.md) 保存当时的设计判断。
