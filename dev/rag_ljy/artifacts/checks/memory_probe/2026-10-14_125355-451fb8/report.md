# Device-context 设备内存探测

日期：2026-10-14T12:53:55.202504+08:00（北京时间）；探测结果：passed。

请求设备：`npu:0`；采样 3 次，间隔 0.3s。
调用路径：RAG → device-context runtime.memory_stats → 所选平台后端。

| 时间（UTC） | 状态 | 设备 | 总量（MiB） | 已用（MiB） | 可用（MiB） | 采样耗时（ms） |
|---|---|---|---|---|---|---|
| 2026-10-14T04:53:54.599229+00:00 | ok | npu:0 | 62748 | 158 | 62589 | 5354.35 |
| 2026-10-14T04:53:54.901306+00:00 | ok | npu:0 | 62748 | 158 | 62589 | 0.898 |
| 2026-10-14T04:53:55.202374+00:00 | ok | npu:0 | 62748 | 158 | 62589 | 0.929 |

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
    "sampled_at": "2026-10-14T04:53:54.599229+00:00",
    "sample_duration_ms": 5354.35
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
    "sampled_at": "2026-10-14T04:53:54.901306+00:00",
    "sample_duration_ms": 0.898
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
    "sampled_at": "2026-10-14T04:53:55.202374+00:00",
    "sample_duration_ms": 0.929
  }
]
```
