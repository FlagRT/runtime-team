# 910C（昇腾）· 第一个芯片落地实例（分支看板）

> 定位：**统一运行时原型的第一个落地实例** —— 设备上下文（device-context）与多流 Stream 在**昇腾 910C** 上的完整验证。
> 状态：✅ **已完成**（训练腿与推理腿均闭环，错误注入→恢复闭环通过）
> 上级看板：`../README.md` ｜ **通用规范与原型：`../prototype/`（芯片无关，不在此目录）**

---

## 1. 本目录放什么 / 不放什么

| | 内容 |
|---|---|
| ✅ **放** | 910C **专属**的落地实例资产：分布式训练与推理的既有工作、昇腾专属文档与结论（ACL 错误码表、910C 实测映射、CANN/镜像约束、阶段总结） |
| ❌ **不放** | **统一运行时原型与通用规范** —— 它们在 `../prototype/`（`runtime/api`、13 个抽象、conformance 用例、接口约定、事件语义契约）。这些是**芯片无关**的，新芯片按同一份规范接入 |

> ⚠️ **铁律（新芯片方向务必先读）**：本目录里的**实现与结论不迁移**。
> 910C 是规范的**第一个**落地实例，不是规范的载体。新芯片（如 P800）应
> **按 `../prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` §2 的 Backend 插件接入规范新建实例**，
> 迁移的是**规范与方法**；本目录的 `backends/ascend|flagos` 绑定、**108 条 ACL 错误码表**、
> CANN 约束、镜像基座、并发上限 3、`ASCEND_RT_VISIBLE_DEVICES` **一律不迁移**。

---

## 2. 目录结构

```
910C/
├── distributed_training/            # 分布式训练既有工作
│   ├── ascend_regression/           #   训练侧 conformance、探针、通信补丁验证、原始结果
│   ├── docs/                        #   训练映射、flagcx 缺陷修复、双缓冲/netcc 调查、4090 报告
│   ├── patches/                     #   FlagCX 缺陷补丁（O2/O3/O4/P2/P6/P7/P9）
│   └── scripts/                     #   训练脚本、通信工具、环境搭建
├── distributed_inference/           # 分布式推理既有工作
│   ├── inference/                   #   推理探针、错误码工具、TP 对照、服务化脚本与结果
│   ├── docs/                        #   推理映射、P0–P3 阶段报告、Qwen3 TP 对照
│   └── start_infer_container.sh
├── probes/                          # 910C 侧验证证据（跨后端脚本/标准脚本的原始日志）
└── docs/                            # 910C 专属文档（见 §4）
```

---

## 3. 关键成果（实跑证据）

| 项 | 结果 |
|---|---|
| 统一运行时 API + Backend 注册表 | ✅ 真机 **37/37** |
| 昇腾后端（torch_npu，推理腿） | ✅ conformance **13/13 + 6/6**、推理腿自验证 **10/10** |
| FlagOS 后端（torch_fl，训练腿） | ✅ conformance **13/13**（锁定训练镜像） |
| 训练腿 2 卡分布式微调（Qwen3-Embedding-0.6B） | ✅ loss **15.4497 → 11.15**（50 步）、**2117 tok/s**、通信三类对照全对 |
| 推理腿单卡 · 前向形态 | ✅ 向量区分度 **0.638**、66–79 句/s、无 NaN |
| 推理腿单卡 · **服务化形态** | ✅ vLLM OpenAI 兼容服务 **10/10 SERVE_LEG_PASS**：维度 1024、区分度 0.4123、108 句/s（p50 27.4 ms） |
| **错误注入 → 恢复闭环** | ✅ 推理腿 **5 闭环 / 0 失败**（含真实流同步超时 → L3_EXECUTION → 重放）；训练腿 **4 闭环 / 1 跳过 / 0 失败** |
| **统一启动脚本**（组内服务启动标准 v1.1，09-20） | ✅ **`SERVE_STANDARD_PASS (ready=1 smoke=1)`**：服务就绪 **30 s**；生成冒烟 **8 tokens**（`1+1=` → `'2 is a basic arithmetic fact, but'`）；用卡快照 `free=60.91GiB / total=61.27GiB`（停机前后一致）；宿主侧 8100 端口已释放。证据：`probes/L_serve_standard_910c_20260920.log` |
| 组件打包 | ✅ Git tag `runtime-v0.1.0` + Release note（`../prototype/RELEASE_NOTES_v0.1.0.md`） |

> **两条腿都是基于统一原型跑通的**（经 `runtime.use(...)` 接入设备），
> 脚本在 `../prototype/runtime/proto/`；**不是**把历史训推用原型重跑了一遍 ——
> 本目录 `distributed_training/` 的双卡 DDP（Qwen2.5-1.5B）与 `distributed_inference/` 的 vLLM+TP（Qwen3-4B）
> 属**旧代码路径**（直接 `import torch_npu` + `torch.distributed`，`runtime.use` 出现 0 次）。

---

## 4. 文档索引（本目录）

**设备上下文与多流（核心）**

| 文档 | 内容 |
|---|---|
| `ACL_ERROR_MAP_20260901.md` | 108 条 ACL 错误码映射表（L1–L4 分级依据）—— **不迁移**给新芯片 |
| `ASCEND_910C_DC_STREAM_MAPPING_20260902.md` | 910C 设备上下文 × Stream 双侧全景（职责映射与实测） |
| `DC_STAGE_SUMMARY_20260909.md` | 设备上下文阶段性总结（两条腿证据并入） |
| `ERROR_RECOVERY_LOOP_20260909.md` | 错误注入 → 恢复闭环，含归因核查推翻的两条"发现" |
| `DIAG_TRAIN_IMAGE_NPU_20260908.md` | 训练镜像与 NPU 可见性诊断 |

**8 月早期工作（FlagCX 补丁与准备）**

| 文档 | 内容 |
|---|---|
| `DEVICE_CONTEXT_PLAN_20260827.md` | 设备上下文计划 |
| `PROGRESS_20260822.md` | 8-22 进展快照 |
| `910C-env-issue-report.md` | 环境问题报告 |
| `O3_getlasterror_fix.md` / `O4_socket_seq_guard.md` | FlagCX O3/O4 缺陷修复 |
| `PR_DEV_1_0_20260902.md` | PR #11 合入 dev-1.0 记录（157 文件） |

**分支看板**

- `distributed_training/README.md` ｜ `distributed_inference/README.md`

---

## 5. 环境要点（910C 专属，不迁移）

- **带卡容器并发上限 3**（`dev/stack.lock.910c.v2.yaml` 置顶规则）：超限后 `acl.init()` 返 **500000**，
  表现为 `device_count=0`；出现该现象**先查并发容器数**，不要先怀疑镜像/驱动/代码。
- **训练镜像** `flagrt/ascend-operator-runtime-comm:0.1.3` → 后端 **flagos（torch_fl）**：
  禁止 `torch_npu` 共存；`AUTOLOAD=0` 且先 `import torch_fl` 再 `import torch`。
- **推理镜像** `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` → 后端 **npu（torch_npu）**。
- 选卡变量：`ASCEND_RT_VISIBLE_DEVICES`。
- **容器内没有 `npu-smi`**（实测 `npu-smi: command not found`）——它是**宿主工具**。
  容器内查卡请退回 torch 侧（`torch.npu.mem_get_info`）；要看整机 16 卡全貌在宿主执行 `npu-smi info`。
  统一启动脚本已按此降级（`[torch.npu:0] free=… / total=…`）。
- **起服务统一走《组内服务启动标准》**：`../prototype/scripts/serve_standard.sh`（唯一入口，
  `DC_BACKEND=ascend`）。本目录的 `distributed_inference/inference/start_vllm_serve_910c.sh`
  含 D10/D11 集成（错误翻译包装器 + 设备状态监控），**保留但仅供该集成场景**，下游新需求请走统一脚本。
