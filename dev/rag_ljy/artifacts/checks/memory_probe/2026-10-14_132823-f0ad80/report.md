# Device-context 设备内存探测

日期：2026-10-14T13:28:23.732731+08:00（北京时间）；探测结果：passed。

请求设备：`npu:0`；采样 3 次，间隔 0.2s。
调用路径：RAG → device-context runtime.memory_stats → 所选平台后端。

| 时间（UTC） | 状态 | 设备 | 总量（MiB） | 已用（MiB） | 可用（MiB） | 采样耗时（ms） |
|---|---|---|---|---|---|---|
| 2026-10-14T05:28:23.330824+00:00 | ok | npu:0 | 62748 | 158 | 62589 | 5267.768 |
| 2026-10-14T05:28:23.531825+00:00 | ok | npu:0 | 62748 | 158 | 62589 | 0.832 |
| 2026-10-14T05:28:23.732664+00:00 | ok | npu:0 | 62748 | 158 | 62589 | 0.725 |

## 独立进程内存占用验证

子进程通过device-context绑定同一逻辑设备，分配并持有小块内存；父进程通过统一内存接口观测分配与释放。
变化量包含子进程运行环境的额外占用。此次验证不证明所有版本、平台及兼容回退都能报告整卡内存。

```json
{
  "status": "passed",
  "allocation_mib": 128,
  "child_pid": 310,
  "child_exit_code": 0,
  "before": {
    "enabled": true,
    "source": "device-context.runtime.memory_stats",
    "backend": "ascend",
    "device": "npu:0",
    "ordinal": 0,
    "unit": "MiB",
    "scope": "backend_reported_unverified",
    "stats": {
      "total_mb": 62748,
      "used_mb": 158,
      "free_mb": 62589,
      "memory_used_percent": 0.25
    },
    "status": "ok",
    "sampled_at": "2026-10-14T05:28:23.733484+00:00",
    "sample_duration_ms": 0.457
  },
  "held": {
    "enabled": true,
    "source": "device-context.runtime.memory_stats",
    "backend": "ascend",
    "device": "npu:0",
    "ordinal": 0,
    "unit": "MiB",
    "scope": "backend_reported_unverified",
    "stats": {
      "total_mb": 62748,
      "used_mb": 348,
      "free_mb": 62399,
      "memory_used_percent": 0.55
    },
    "status": "ok",
    "sampled_at": "2026-10-14T05:28:28.881732+00:00",
    "sample_duration_ms": 0.933
  },
  "after": {
    "enabled": true,
    "source": "device-context.runtime.memory_stats",
    "backend": "ascend",
    "device": "npu:0",
    "ordinal": 0,
    "unit": "MiB",
    "scope": "backend_reported_unverified",
    "stats": {
      "total_mb": 62748,
      "used_mb": 158,
      "free_mb": 62589,
      "memory_used_percent": 0.25
    },
    "status": "ok",
    "sampled_at": "2026-10-14T05:28:33.001706+00:00",
    "sample_duration_ms": 1.107
  },
  "deltas": {
    "used_increase_mib": 190,
    "free_decrease_mib": 190,
    "used_release_mib": 190,
    "free_recovery_mib": 190
  },
  "note": "Differences include the child runtime/context allocation; scope is validated only for this backend/run, not for fallback paths."
}
```

统计范围标记为 backend_reported_unverified：当前统一接口未提供底层查询/回退路径的元数据。
Ascend 后端查询失败可能回退到当前进程分配量；这些读数尚未用于内存准入控制。
该接口测量内存占用，不提供加速卡计算利用率。

## 原始采样（仅小型指标记录）

```json
[
  {
    "enabled": true,
    "source": "device-context.runtime.memory_stats",
    "backend": "ascend",
    "device": "npu:0",
    "ordinal": 0,
    "unit": "MiB",
    "scope": "backend_reported_unverified",
    "stats": {
      "total_mb": 62748,
      "used_mb": 158,
      "free_mb": 62589,
      "memory_used_percent": 0.25
    },
    "status": "ok",
    "sampled_at": "2026-10-14T05:28:23.330824+00:00",
    "sample_duration_ms": 5267.768
  },
  {
    "enabled": true,
    "source": "device-context.runtime.memory_stats",
    "backend": "ascend",
    "device": "npu:0",
    "ordinal": 0,
    "unit": "MiB",
    "scope": "backend_reported_unverified",
    "stats": {
      "total_mb": 62748,
      "used_mb": 158,
      "free_mb": 62589,
      "memory_used_percent": 0.25
    },
    "status": "ok",
    "sampled_at": "2026-10-14T05:28:23.531825+00:00",
    "sample_duration_ms": 0.832
  },
  {
    "enabled": true,
    "source": "device-context.runtime.memory_stats",
    "backend": "ascend",
    "device": "npu:0",
    "ordinal": 0,
    "unit": "MiB",
    "scope": "backend_reported_unverified",
    "stats": {
      "total_mb": 62748,
      "used_mb": 158,
      "free_mb": 62589,
      "memory_used_percent": 0.25
    },
    "status": "ok",
    "sampled_at": "2026-10-14T05:28:23.732664+00:00",
    "sample_duration_ms": 0.725
  }
]
```

## 实现与回归验证

探测实现：`src/rag_engine/memory_probe.py`；API接入：`src/rag_engine/server.py`。
每秒独立线程采样，`/health.device_memory`读取缓存；默认开启，可用`--no-memory-probe`关闭。

2026-10-14完整回归：`python -m pytest tests -q`，**120 passed in 2.19s**。覆盖指标校验、失败恢复、采样线程隔离、health/请求不被慢采样阻塞，以及既有RAG功能。

复跑本报告的设备/跨进程测试：

```bash
python scripts/checks/check_memory.py --device npu:0 --count 3 --interval 0.2 --cross-process-mib 128
```


## 验证结论

| 验证项 | 结果 |
|---|---|
| runtime内存读取 | 连续3次正常，总量62,748MiB；初始化后的读取约1ms |
| 独立进程占用/释放 | 128MiB分配时已用158→348MiB，进程退出后回到158MiB；变化包含运行环境开销 |
| 真实API后台采样 | 每秒刷新，173次health采样全部正常且未过期 |
| 同时处理查询 | 2条串行及同样2条同时提交全部成功，结果与串行参考一致 |
| 全量回归 | 120项通过 |

当前只启用设备内存探测，准入仍按请求数量限制，未启用内存水位准入。

## 真实API后台采样及查询验证

物理NPU7、逻辑npu:0；Elasticsearch使用宿主机9201，embedding由8089 vLLM服务执行。
通过实际HTTP API进行2条串行及同样2条同时提交；请求期间连续检查内存指标及时间戳刷新。以下仅保存小型指标与结果ID。

```json
{
  "status": "passed",
  "backend": "elasticsearch",
  "url": "http://127.0.0.1:8088",
  "http_queries": 4,
  "serial_successes": 2,
  "concurrent_successes": 2,
  "concurrent_matches_serial": true,
  "health_samples": 173,
  "all_memory_samples_ok_and_fresh": true,
  "memory_timestamp_refreshes": true,
  "peak_inflight": 2,
  "health_before": {
    "status": "ready",
    "device": "remote-vllm",
    "busy": false,
    "inflight": 0,
    "max_inflight": 8,
    "embedding_backend": "vllm",
    "retrieval_backend": "elasticsearch",
    "execution_mode": "concurrent",
    "device_memory": {
      "enabled": true,
      "source": "device-context.runtime.memory_stats",
      "backend": "ascend",
      "device": "npu:0",
      "ordinal": 0,
      "unit": "MiB",
      "scope": "backend_reported_unverified",
      "stats": {
        "total_mb": 62748,
        "used_mb": 10756,
        "free_mb": 51991,
        "memory_used_percent": 17.14
      },
      "status": "ok",
      "sampled_at": "2026-10-14T05:34:51.544596+00:00",
      "sample_duration_ms": 0.927,
      "age_seconds": 0.852,
      "stale": false
    },
    "stage_queues": {
      "embedding": {
        "capacity": 8,
        "queued": 0,
        "running": 0,
        "workers": 2
      },
      "retrieval": {
        "capacity": 8,
        "queued": 0,
        "running": 0,
        "workers": 4
      },
      "rerank": {
        "capacity": 8,
        "queued": 0,
        "running": 0,
        "workers": 1
      }
    }
  },
  "health_after": {
    "status": "ready",
    "device": "remote-vllm",
    "busy": false,
    "inflight": 0,
    "max_inflight": 8,
    "embedding_backend": "vllm",
    "retrieval_backend": "elasticsearch",
    "execution_mode": "concurrent",
    "device_memory": {
      "enabled": true,
      "source": "device-context.runtime.memory_stats",
      "backend": "ascend",
      "device": "npu:0",
      "ordinal": 0,
      "unit": "MiB",
      "scope": "backend_reported_unverified",
      "stats": {
        "total_mb": 62748,
        "used_mb": 13269,
        "free_mb": 49478,
        "memory_used_percent": 21.15
      },
      "status": "ok",
      "sampled_at": "2026-10-14T05:35:00.556516+00:00",
      "sample_duration_ms": 1.151,
      "age_seconds": 0.702,
      "stale": false
    },
    "stage_queues": {
      "embedding": {
        "capacity": 8,
        "queued": 0,
        "running": 0,
        "workers": 2
      },
      "retrieval": {
        "capacity": 8,
        "queued": 0,
        "running": 0,
        "workers": 4
      },
      "rerank": {
        "capacity": 8,
        "queued": 0,
        "running": 0,
        "workers": 1
      }
    }
  },
  "responses": [
    {
      "query_id": "PLAIN-1028",
      "status": 200,
      "request_id": "1205d22fc50347059804f9ddc3f7c744",
      "latency_ms": 6426.617,
      "chunk_ids": [
        "00b9d6d2790efba7d6416c03d640d647e4661303747d2d5b7ba0847bec4b8ee1",
        "2896b1f8700e0856897ede5b594db77285d23de45c38af751878eb0c4a11694b",
        "49cabf73a148fcaca762a4b5a7838d8ba94bcb68be3807bfc05129a189c7a1db",
        "7d87b28f3d234d51650af2d317ef6c27ccb5d813955122d000d912bde3d88a92",
        "c777435c28a65b3226ef70e90e439a2702e0f6fb2503623e335dc11b8f731155"
      ]
    },
    {
      "query_id": "PLAIN-1130",
      "status": 200,
      "request_id": "6f71d2223e1b45cab831de88093f7da3",
      "latency_ms": 491.644,
      "chunk_ids": [
        "8a9cc428dd1bfb62a92e0cf7db5364a5990da72a5ee7d6648c495cb3d6c31eea",
        "9428d16dc095f33c0bace70422622f59f433c1b745589a1c674ee1e38bd1a81a",
        "4e6ebbda3bada5c280c4040116ba38318056085d0c93f9212b0cbdde5c4cc3e1",
        "33117571e36f2140fa61abad41d6f03a342f52a11f594650006ce074384b52ca",
        "d6e0970310806a6882eb1dc09ff295822953003dff3181beef5464644d29037c"
      ]
    },
    {
      "query_id": "PLAIN-1028",
      "status": 200,
      "request_id": "46cc0c3abe584e7688fe5070df40e1f9",
      "latency_ms": 441.235,
      "chunk_ids": [
        "00b9d6d2790efba7d6416c03d640d647e4661303747d2d5b7ba0847bec4b8ee1",
        "2896b1f8700e0856897ede5b594db77285d23de45c38af751878eb0c4a11694b",
        "49cabf73a148fcaca762a4b5a7838d8ba94bcb68be3807bfc05129a189c7a1db",
        "7d87b28f3d234d51650af2d317ef6c27ccb5d813955122d000d912bde3d88a92",
        "c777435c28a65b3226ef70e90e439a2702e0f6fb2503623e335dc11b8f731155"
      ]
    },
    {
      "query_id": "PLAIN-1130",
      "status": 200,
      "request_id": "e3a0906e12c54beba1c29f47574f8a5a",
      "latency_ms": 737.369,
      "chunk_ids": [
        "8a9cc428dd1bfb62a92e0cf7db5364a5990da72a5ee7d6648c495cb3d6c31eea",
        "9428d16dc095f33c0bace70422622f59f433c1b745589a1c674ee1e38bd1a81a",
        "4e6ebbda3bada5c280c4040116ba38318056085d0c93f9212b0cbdde5c4cc3e1",
        "33117571e36f2140fa61abad41d6f03a342f52a11f594650006ce074384b52ca",
        "d6e0970310806a6882eb1dc09ff295822953003dff3181beef5464644d29037c"
      ]
    }
  ],
  "note": "API uses the existing .env nfcorpus index/collection names; the original container carries older default names."
}
```
