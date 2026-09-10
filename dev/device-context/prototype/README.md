# prototype · 统一运行时原型（分支看板）

> 定位：**我们制定的统一标准**——统一基座之上的统一 API、统一 Backend、统一验证。
> 主张：**换芯片不改代码**；新增一家芯片/后端 = 实现一个 backend + 跑通 conformance。
> 上级看板：`../README.md` ｜ 基座配置：`../../stack.lock.910c.v1.yaml`

---

## 1. 目录

```
prototype/
├── runtime/                     # 实现代码（可独立分发，自包含）
│   ├── __init__.py              # 用户入口（use / set_device / create_stream / translate_error / recover_device）
│   ├── api/                     # 统一错误对象与 Stream/Event 封装
│   ├── backends/
│   │   ├── base.py              # RuntimeBackend 抽象（接口规范）
│   │   ├── registry.py          # 注册表（register / use / discover）
│   │   ├── ascend/              # 昇腾后端（torch_npu，推理腿）
│   │   └── flagos/              # FlagOS 后端（torch_fl，训练腿）
│   ├── conformance/             # 验收用例 13 例 + 推理 6 例 + runner（含资产模块）
│   ├── proto/                   # 两条腿自验证脚本与结果
│   ├── demos/                   # 设备无关演示
│   └── smoke_runtime.py         # 接入自检
└── docs/                        # 标准说明文档
```

---

## 2. 统一 API 面（承诺）

| 域 | 接口 |
|---|---|
| 后端选择 | `use / available / discover / register` |
| 设备 | `device_count / set_device / memory_stats / probe_device` |
| 流 / 事件 | `create_stream / create_event / current_stream / synchronize`（有界） |
| 错误 | `translate_error / FlagosError / ErrorCategory / DISPOSITION`（L1–L4 → retry/raise/replay/device_recovery） |
| 恢复 | `recover_device / device_state`（probe / real / hybrid） |

完整约定见 `docs/INTERFACE_CONTRACT_DC_20260908.md`。

**两条硬纪律**：跨流传缓冲必须 `record_stream`；异常处理一律按 `disposition` 处置，不按消息字符串判断。

---

## 3. 验证状态（910C 实跑）

| 验证 | 结果 |
|---|---|
| 冒烟自检 | 37/37 |
| conformance（ascend 后端） | 13/13 + 推理 6/6 |
| conformance（flagos 后端） | 13/13 |
| 推理腿自验证 | 10/10（向量区分度 0.638、66–79 句/s） |
| 训练腿 2 卡微调 | 6/6（loss 15.45→11.15、2117 tok/s、通信三类对照） |
| **错误注入 → 恢复闭环**（设备侧） | ✅ 推理腿 **5 闭环 / 0 失败**（含真实超时 → L3 重放）；训练腿 **4 闭环 / 1 跳过**（无有界同步，如实跳过）。详见 `docs/ERROR_RECOVERY_LOOP_20260909.md`（含归因核查：此前两条"发现"已推翻） |

---

### 3.1 基于统一原型的训推复跑（2026-09-09，验收模型 Qwen3-Embedding-0.6B）

两条腿的复跑**都经本原型的统一 API 接入设备**（`runtime.use(...)` + `set_device`）：

| 腿 | 脚本 | 后端 | 结果 |
|---|---|---|---|
| 训练腿 2 卡微调 | `runtime/proto/proto_train_leg.py` | flagos | 6/6：loss 15.45 → 11.15（50 步）、2117 tok/s、通信三类对照全对 |
| 推理腿单卡（前向） | `runtime/proto/proto_infer_leg.py` | ascend | 10/10：向量区分度 0.638、66–79 句/s、无 NaN |\n| **推理腿单卡（服务化）** | `runtime/proto/proto_infer_serve.py` | ascend | **10/10 SERVE_LEG_PASS**：vLLM OpenAI 兼容服务，维度 1024、区分度 0.4123、108 句/s（p50 27.4ms）、超长输入 → L2_PARAM/raise 且业务继续 |
| 错误注入→恢复闭环 | `runtime/proto/proto_error_recovery_loop.py` | 双后端通用 | 推理腿 5 闭环 / 训练腿 4 闭环（1 项因无有界同步如实跳过） |

**与历史资产的关系（易混淆，务必看清）**：本原型跑的是**验收模型的新验证**；
`../distributed_training/`、`../distributed_inference/` 里的历史训推（Qwen2.5-1.5B DDP、
Qwen3-4B vLLM+TP）是**旧代码路径**（直接 import 厂商扩展，不经统一 API），
**尚未用本原型复跑**。

**已知缺口**：推理腿服务化已补齐（vLLM OpenAI 兼容接口）；历史模型（Qwen2.5-1.5B / Qwen3-4B）尚未在统一原型上复跑。


## 4. 运行方式

```bash
python3 runtime/smoke_runtime.py
python3 runtime/conformance/runner.py --backend ascend [--cases infer_cases]
python3 runtime/conformance/runner.py --backend flagos      # 训练腿镜像（AUTOLOAD=0 + 先 import torch_fl）
python3 runtime/proto/proto_infer_leg.py
torchrun --nproc_per_node=2 runtime/proto/proto_train_leg.py
```

注意：带卡容器并发上限 3（超限时 `acl.init()`=500000），两条腿串行跑。

---

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/INTERFACE_CONTRACT_DC_20260908.md` | **接口约定（设备上下文章节）**：API 承诺、Backend 插件规范、版本承诺 |
| `docs/RUNTIME_PROTOTYPE_DESIGN_20260904.md` | 运行时原型整体设计（插件模式、目录、双后端策略、IR、CI） |
| `docs/RUNTIME_DC_STREAM_PLAN_20260907.md` | 设备上下文 + 多流 Stream 职责框架 |
| `docs/RUNTIME_LAYER_MONTHLY_PLAN_20260908.md` | 9 月工作计划（统一基座 / 本月目标 / 逐周任务） |
| `docs/DESIGN_DIST_COMM_20260908.md` | 2 卡分布式微调的通信路线思考（推荐 flagcx） |
| `docs/ASCEND_910C_DC_STREAM_MAPPING_20260902.md` | 训练 ‖ 推理 双侧职责全景 |
| `docs/event_semantics_contract.md` | 统一事件契约 |
| `docs/ACL_ERROR_MAP_20260901.md` | 错误码映射（108 条 / 64.8% 覆盖） |
| `docs/DIAG_TRAIN_IMAGE_NPU_20260908.md` | 训练镜像 NPU 初始化失败排查记录 |
| `docs/DC_STAGE_SUMMARY_20260909.md` | **阶段性总结**：两条腿证据 + 组件自检 + 已知缺口（设备上下文部分） |
| `docs/ERROR_RECOVERY_LOOP_20260909.md` | **错误注入 → 恢复闭环**：验证记录、两个发现、代码修正 |
