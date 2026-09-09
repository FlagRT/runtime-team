# distributed_training · 分布式训练（分支看板）

> 定位：**分布式训练侧的既有工作与资产**（910C 同构训练验证、通信缺陷修复、训练脚本）。
> 与统一标准的关系：本目录是**历史与训练侧资产**；统一 API/Backend 标准在 `../prototype/`。
> 上级看板：`../README.md`

---

## 1. 目录

```
distributed_training/
├── ascend_regression/   # 训练侧 conformance 与探针资产（S/T/E/F/R 系列、双缓冲、恢复语义等）
├── scripts/             # 训练脚本与通信工具（train_qwen_*、flagcx/hccl 相关 patch 与测试）
├── patches/             # 通信缺陷补丁（P2/P6/P7、O3/O4 等）
└── docs/                # 训练映射与报告
```

---

## 2. 关键资产

| 资产 | 说明 |
|---|---|
| `ascend_regression/conformance/` | 训练侧 conformance（含 errors / recovery / device_state / npu_events 原始实现） |
| `ascend_regression/` 各探针 | 双缓冲、事件语义、错误注入、超时等探针与结果 |
| `scripts/train_qwen_1_5b_npu.py` 等 | 910C 双卡 DDP 训练脚本（flagcx / hccl 两种后端） |
| `scripts/setup_910c.sh` | 910C 环境初始化 |
| `patches/` | FlagCX / HCCL 通信缺陷补丁 |

> 注：`prototype/runtime/conformance/` 已**自包含**复制了 errors/recovery/device_state/npu_events，
> 原型分发不依赖本目录；本目录保留的是训练侧的原始资产与历史证据。

---

## 3. 验证状态

| 项 | 结果 |
|---|---|
| 910C 双卡 DDP（Qwen2.5-1.5B） | 2481 步 loss 1.95、4245–5428 tok/s |
| 训练侧 conformance | 13/13 |
| event 泄漏缺陷 | 已修复（每步 ~120 个 aclrtEvent 累积） |
| **训练腿 2 卡微调（统一原型）** | ✅ 见 `../prototype/runtime/proto/proto_train_leg.py`：loss 15.45→11.15、2117 tok/s |

---

## 4. 运行注意

- 训练腿锁定镜像的设备后端是 **flagos（torch_fl）**，禁止 torch_npu 共存；需 `AUTOLOAD=0` 且先 `import torch_fl`
- 容器内需补装 `transformers`
- 带卡容器并发上限 3；与推理腿串行
- 2 卡启动示例：
  `torchrun --nproc_per_node=2 ../prototype/runtime/proto/proto_train_leg.py`

---

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/DEVICE_CONTEXT_TRAINING_MAPPING_20260831.md` | 训练侧职责映射（D1–D11、B1–B11） |
| `docs/FLAGCX_CORE_DEFECT_FIXES_20260826.md` | FlagCX 核心缺陷修复 |
| `docs/4090_training_report.md` | 4090 训练报告（历史） |
| `docs/flagcx_ascend_aline_validation_20260824.md` | FlagCX 昇腾适配验证 |
| `docs/netcc_chunk_race_investigation.md` | 通信 chunk 竞争排查 |
