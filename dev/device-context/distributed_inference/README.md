# distributed_inference · 分布式推理（分支看板）

> 定位：**分布式推理侧的既有工作与资产**（vLLM + TP、错误码映射、服务化状态与恢复、双缓冲）。
> 与统一标准的关系：本目录是**历史与推理侧资产**；统一 API/Backend 标准在 `../prototype/`。
> 上级看板：`../README.md`

---

## 1. 目录

```
distributed_inference/
├── inference/           # 推理侧 conformance、探针、错误码工具、TP 对照、服务化监控
│   └── results_p3/      # P3 阶段结果
├── start_infer_container.sh
└── docs/                # 推理映射与阶段报告
```

---

## 2. 关键资产

| 资产 | 说明 |
|---|---|
| `inference/gen_acl_error_map.py`、`audit_error_map_coverage.py` | 错误码映射生成与覆盖审计（108 条 / 64.8%） |
| `inference/probe_acl_107015.py` | ACL 107015 根因探针（stream callback 契约） |
| `inference/inject_error_translation.py` | 错误翻译注入（挂接 vLLM 异常路径） |
| `inference/device_state_monitor.py` | 设备状态监控 |
| `inference/compare_tp_outputs.py` | TP=1/2/4 输出对照 |
| `inference/probe_serve_*` | 服务化健康与状态恢复探针 |

---

## 3. 验证状态

| 项 | 结果 |
|---|---|
| 推理侧 conformance | 6/6 |
| 推理腿自验证（统一原型） | ✅ 10/10（向量区分度 0.638、66–79 句/s、无 NaN） |
| 错误码映射 | 108 条（CANN 159 码全集命中 64.8%） |
| 跨天长驻 | 28 小时 HBM +0.07% / RSS +0.6%，零泄漏 |
| TP 输出对照 | 数值不等价（50.1% 一致）但语义等价，属自回归固有特性 |

---

### 3.1 代码路径说明（重要）

| 推理 | 路径 | 是否经统一原型 |
|---|---|---|
| 历史 vLLM + TP（Qwen3-4B，TP=1/2/4） | `inference/` 下各探针与对照脚本 | ❌ 直接 import 厂商扩展，不经统一 API |
| **本轮推理腿单卡（前向，Qwen3-Embedding-0.6B）** | `../prototype/runtime/proto/proto_infer_leg.py` | ✅ `runtime.use("ascend")` + `set_device(0)` |\n| **本轮推理腿单卡（vLLM 服务化）** | `../prototype/runtime/proto/proto_infer_serve.py` | ✅ 同上 + HTTP 调 `/v1/embeddings`，10/10 |

**已知缺口（如实标注）**：

1. ~~vLLM 服务化尚未基于统一原型验证~~ → **已于 2026-09-09 补齐**：
   `proto_infer_serve.py` 经统一 API 接入 + HTTP 调服务，10/10 通过（验收标准 2 达成）；
2. 历史 Qwen3-4B 的 TP 验证**尚未用统一原型复跑**。


## 4. 运行注意

- 推理腿锁定镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`，设备后端 `npu（torch_npu）`
- 带卡容器并发上限 3；与训练腿串行
- 推理腿自验证：`python3 ../prototype/runtime/proto/proto_infer_leg.py`

---

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/DEVICE_CONTEXT_INFERENCE_MAPPING_20260831.md` | 推理侧职责映射（D1–D11、A1–A11） |
| `docs/DEVICE_CONTEXT_INFERENCE_PLAN_20260831.md` | 推理侧推进计划 |
| `docs/INFERENCE_P0_P1_RUN_20260831.md` | P0/P1 跑测记录 |
| `docs/INFERENCE_P3_SERVE_STATE_ERROR_20260901.md` | P3 服务状态与错误 |
| `docs/INFERENCE_QWEN3_TP_COMPARE_20260901.md` | Qwen3 TP 对照结论 |
