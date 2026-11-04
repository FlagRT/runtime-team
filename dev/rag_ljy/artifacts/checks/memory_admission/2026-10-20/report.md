# 内存感知准入原型验证

运行日期：2026-10-20 16:49:42 +0800（北京时间）。
范围：真实FastAPI路由、阶段队列和后台采样器，使用假模型及受控内存读数；未运行NPU压力测试、未重跑双库200次链路。

## 结果

完整硬件无关回归 **169/169通过**，失败0、错误0、跳过0；pytest耗时2.394秒。
新增准入不改变原有检索/精排结果格式，默认不开启，显式开启时同时受请求数量上限约束。

| 场景 | 验证结果 |
|---|---|
| 高水位与回落恢复 | 受控占用900/1000MiB时503拒绝；降到800仍保持暂停；降到低水位以下且足够容纳预算后恢复200。 |
| 同时到达与预算竞争 | 8请求同时进入；设备总量1000MiB、已用300MiB、每请求预算200MiB、安全余量50MiB、高水位85%；接受2个、其余6个预期503，活动预算400MiB。 |
| 后续阶段预算 | 请求阻塞在检索时仍保留完整预算，下一请求不能花费这份余量；已接收请求可继续精排完成。 |
| 请求完成与缓存读数 | 请求完成后转为待释放预算；旧采样或开始于完成前的采样不能释放，只有完成后开始的有效新采样才释放。 |
| 客户端断开 | HTTP等待取消后模型/检索工作继续，预算仍占用至实际处理结束，不因断开提前开放容量。 |
| 请求异常 | 对应请求返回500，模型不重载；预算在完成后新采样确认时释放，后续请求正常。 |
| 探测异常与过期 | pending、unavailable、partial、过期、缺字段、错误单位、无效时间均不接收新请求；有效读数恢复后可接收。 |
| 统计范围 | 未验证范围默认拒绝；只有显式允许后才使用backend_reported_unverified，process范围仍拒绝。 |
| 后台采样接入 | 使用真实DeviceMemoryMonitor与受控探测器，验证pending期间503及有效采样后的200；探测器返回错误类型也可恢复。 |
| 配置与CLI | 开启却缺每请求预算、同时关闭探测、错误高低水位均启动前报错；CLI --help和Python3.11语法检查通过。 |

以上内存数字为测试输入，不是今天在NPU上测得的容量或占用。

## 运行环境与复跑

Python 3.14.7，aarch64；pytest 8.4.2, fastapi 0.151.0, starlette 1.9.0, pydantic 2.14.1, httpx 0.28.1。
本轮在宿主机隔离的/tmp虚拟环境运行，未启动、重建或重启现有容器和模型服务。
语法按Python3.11解析通过；本轮行为回归使用Python3.14，未替代容器内Python3.11与NPU实测。
出现1条Starlette TestClient/httpx弃用提示，无测试失败。

```bash
cd dev/rag_ljy
python -m pytest tests -q
```

需要已安装pyproject.toml中的test依赖。测试不需要模型权重或NPU。

| 测试模块 | 通过数 |
|---|---|
| `tests.test_admission` | 37 |
| `tests.test_benchmark` | 14 |
| `tests.test_concurrent_retrieval` | 7 |
| `tests.test_config` | 3 |
| `tests.test_corpus` | 7 |
| `tests.test_device` | 17 |
| `tests.test_documents` | 2 |
| `tests.test_ingestion` | 8 |
| `tests.test_memory_probe` | 18 |
| `tests.test_multi_request_checker` | 4 |
| `tests.test_retrieval` | 3 |
| `tests.test_scheduler` | 2 |
| `tests.test_server` | 32 |
| `tests.test_store_factory` | 3 |
| `tests.test_timing` | 4 |
| `tests.test_vllm_embedding` | 8 |

## 启用及边界

按照docs/operations.md准备原有数据库、vLLM和API环境后，替换查询API启动命令：

```bash
python scripts/runtime/serve_queries.py --device npu:0 --port 8088 \
  --memory-admission --memory-request-mib 2048 \
  --memory-allow-unverified-scope
```

2048MiB是示例预算，**不是实测峰值**。默认高/低水位85%/75%、安全余量1024MiB、采样1秒、读数最大年龄3秒。
当前device-context没有承诺统一的整卡统计范围，必须先验证适用环境才显式允许未验证范围的读数。

预算是进程内逻辑记账，不是预分配或设备独占；测量占用与预算相加可能重复计算，策略偏保守。
请求完成后的预算等待新采样确认，可能使短突发在下一次采样前先返回503。
只覆盖被探测的设备；主机RAM、其他设备、其他进程突然分配、超出配置预算的峰值、碎片化及OOM后模型恢复尚未覆盖。
后续需在真实长查询与并发负载下校准预算、水位和压力恢复，不能据此宣称任意场景零OOM。

`/health.memory_admission`展示当前状态、活动/待释放预算、计数与最后决策；服务日志event=memory_admission记录请求ID、决策原因与预算，不记录查询原文。

## 源码校验

工作分支：`jliu171/rag_ljy`；测试时HEAD：`ed731aabda138f1c19665421e6f6401174022f6a`（改动尚未提交）。
本次实现与测试文件的SHA256：

```text
c230322e591a0da4df9569407daca96316bf06d90b094390862d405f9b3181ac  src/rag_engine/admission.py
c112cc5e3d06d0e47e9e1f37285b03c53d854a5b132e1701df35772c18a8f112  src/rag_engine/server.py
835f6f2ed2058843649be87377aeb15b8ea6c9df77dd56812801aabf42044959  src/rag_engine/memory_probe.py
b0c51c8327f7e3a75276120ff18f2818a4ce60ff1e59aacd960373fbff232619  scripts/runtime/serve_queries.py
6822b00bc49a7d27d10298d8f7eb3e62efd1a1cf348c35a1d1384b573104edfa  tests/test_admission.py
680f2d22087ea5b13fba60d062afa9944d140ef9b7ff6d187e14ce3bac8c5830  tests/test_server.py
2194e23bfc634a92dde409919c3faaaa73d2ba116f8bedd41c1c6a7b037f74f3  tests/test_memory_probe.py
```
