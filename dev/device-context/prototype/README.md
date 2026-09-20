# prototype · 统一运行时原型（分支看板）

> 定位：**我们制定的统一标准**——统一基座之上的统一 API、统一 Backend、统一验证。
> 主张：**换芯片不改代码**；新增一家芯片/后端 = 实现一个 backend + 跑通 conformance。
> 上级看板：`../README.md` ｜ 基座配置：`dev/stack.lock.910c.v2.yaml`（总组定稿，位于 `dev-1.0` 分支）

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
│   │   ├── flagos/              # FlagOS 后端（torch_fl，训练腿）
│   │   └── kunlun/              # 昆仑芯后端（torch.cuda / xpytorch，P800）
│   ├── conformance/             # 验收用例 13 例 + 推理 6 例 + runner（含资产模块）
│   ├── proto/                   # 两条腿自验证脚本与结果
│   ├── demos/                   # 设备无关演示
│   └── smoke_runtime.py         # 接入自检
├── probes/                      # 跨后端验证探针（目前：多流 16 项基线探针，后端无关 V2）
└── docs/                        # 标准说明文档
```

> **要接新芯片？** 直接读 **《新芯片接入手册》** `docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`
> ——厂商栈判别（4 条路径）→ 镜像就绪判据 → 13 个抽象方法清单 → conformance → 两条腿 → 错误闭环
> → **可勾选验收清单**；坑与厂商上报模板也在里面。
> 组件版本 `runtime-v0.2.0`（第二实例接入版，见 `RELEASE_NOTES_v0.2.0.md`）。

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

## 3. 验证状态（两实例实跑）

**同一套 conformance 判据、同一份两条腿脚本**，在两个芯片实例上分别跑通：

| 验证 | 910C（第一实例，昇腾） | P800（第二实例，昆仑芯） |
|---|---|---|
| 冒烟自检 | 37/37 | **42 通过 / 0 失败** |
| conformance（设备上下文与多流 13 例） | 13/13（`ascend`）+ 13/13（`flagos`） | **13/13**（`kunlun`） |
| conformance（推理 6 例） | 6/6 | **6/6** |
| 训练腿 2 卡微调 | 6/6（loss 15.45→11.15、2117 tok/s、通信三类对照全对） | **6/6**（loss 15.4488→11.1481、3482 tok/s） |
| 推理腿（单卡前向） | 10/10（区分度 0.638、66–79 句/s） | **13/13**（0.6392、53.12 句/s、p50 56.17 ms） |
| 推理腿（vLLM 服务化） | 10/10（区分度 0.4123、108 句/s、p50 27.4 ms） | **10/10**（0.4102、30.70 句/s、p50 96.4 ms） |
| 错误注入 → 恢复闭环（设备侧） | 推理腿 **5 闭环 / 0 失败**（含真实超时 → L3 重放）；训练腿 **4 闭环 / 1 跳过**（无有界同步，如实跳过）。详见 `../910C/docs/ERROR_RECOVERY_LOOP_20260909.md`（含归因核查：此前两条"发现"已推翻） | 设/不设 `XPU_EVENT_KL3_ENABLE` 两组**各 5 闭环 / 0 失败**，结果**逐字节一致** |
| **官方镜像等价性**（换镜像后结论是否成立） | — | ✅ 全部结论在**官方 `-base` 镜像**上复现（conformance 逐用例一致、推理腿 `detail` **14/14 逐字相同**、KL3 挂死一致重现） |

---

### 3.1 基于统一原型的训推复跑（2026-09-09，验收模型 Qwen3-Embedding-0.6B）

两条腿的复跑**都经本原型的统一 API 接入设备**（`runtime.use(...)` + `set_device`）：

| 腿 | 脚本 | 910C（后端 → 结果） | P800（后端 → 结果） |
|---|---|---|---|
| 训练腿 2 卡微调 | `runtime/proto/proto_train_leg.py` | `flagos` → 6/6：loss 15.45 → 11.15（50 步）、2117 tok/s、通信三类对照全对 | `kunlun` → **6/6**：loss 15.4488 → 11.1481、3482 tok/s |
| 推理腿单卡（前向） | `runtime/proto/proto_infer_leg.py` | `ascend` → 10/10：向量区分度 0.638、66–79 句/s、无 NaN | `kunlun` → **13/13**：区分度 0.6392、53.12 句/s、p50 56.17 ms |
| 推理腿单卡（服务化） | `runtime/proto/proto_infer_serve.py` | `ascend` → **10/10 SERVE_LEG_PASS**：vLLM OpenAI 兼容服务，维度 1024、区分度 0.4123、108 句/s（p50 27.4 ms）、超长输入 → L2_PARAM/raise 且业务继续 | `kunlun` → **10/10**：区分度 0.4102、30.70 句/s、p50 96.4 ms、超长输入 → L2_PARAM/raise |
| 错误注入→恢复闭环 | `runtime/proto/proto_error_recovery_loop.py` | 双后端通用 → 推理腿 5 闭环 / 训练腿 4 闭环（1 项因无有界同步如实跳过） | `kunlun` → 设/不设 `XPU_EVENT_KL3_ENABLE` 两组**各 5 闭环 / 0 失败**且逐字节一致 |

> **换芯片只改一行**：同一条命令，只把 `--backend` / `DC_BACKEND` 从 `flagos`／`ascend` 换成 `kunlun`
> （设备串、选卡变量、通信后端名由后端各自封装，见 §5 两实例配置手册）。

**与历史资产的关系（易混淆，务必看清）**：本原型跑的是**验收模型的新验证**；
`../910C/distributed_training/`、`../910C/distributed_inference/` 里的历史训推（Qwen2.5-1.5B DDP、
Qwen3-4B vLLM+TP）是**旧代码路径**（直接 import 厂商扩展，不经统一 API），
**尚未用本原型复跑**。

**已知缺口**：① 历史模型（Qwen2.5-1.5B / Qwen3-4B）尚未在统一原型上复跑；
② P800 侧阶段 5（接入手册 / 接口约定修订建议 / 原型 release）未做；
③ 厂商缺陷 KL3 事件同步概率性挂死（归属厂商运行时层，**在两个镜像上均复现**）待上报与裁定。


## 4. 运行方式

```bash
# ── 910C ──
python3 runtime/smoke_runtime.py
python3 runtime/conformance/runner.py --backend ascend [--cases infer_cases]
python3 runtime/conformance/runner.py --backend flagos      # 训练腿镜像（AUTOLOAD=0 + 先 import torch_fl）
python3 runtime/proto/proto_infer_leg.py
torchrun --nproc_per_node=2 runtime/proto/proto_train_leg.py

# ── P800（后端改 kunlun；脚本与参数由 DC_* 环境变量驱动）──
export PYTHONPATH=/env/FlagGems/src FLAGCX_ADAPTOR=klx
CUDA_VISIBLE_DEVICES=$DEV python3 runtime/smoke_runtime.py --backend kunlun
CUDA_VISIBLE_DEVICES=$DEV python3 runtime/conformance/runner.py --backend kunlun [--cases infer_cases]
DEV=6 bash ../P800/probes/F_infer_leg.sh          # 推理腿前向（含用卡复查）
DEV=6 bash ../P800/probes/F2_vllm_serve.sh        # 推理腿服务化（含停机与子进程清理）
DEV=6 bash ../P800/probes/G_error_loop.sh         # 错误闭环两设置对照
```

注意（910C）：带卡容器并发上限 3（超限时 `acl.init()`=500000），两条腿串行跑。
注意（P800）：用卡前先 `xpu-smi` 挑空闲卡；**若用官方 `-base` 镜像须先补齐 `triton`**（见 §5 两实例配置手册 §2.1.1）。

---

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/INTERFACE_CONTRACT_DC_20260908.md` | **接口约定（设备上下文章节）**：API 承诺、Backend 插件规范、版本承诺 |
| `docs/RUNTIME_PROTOTYPE_DESIGN_20260904.md` | 运行时原型整体设计（插件模式、目录、双后端策略、IR、CI） |
| `docs/RUNTIME_DC_STREAM_PLAN_20260907.md` | 设备上下文 + 多流 Stream 职责框架 |
| `docs/RUNTIME_LAYER_MONTHLY_PLAN_20260908.md` | 9 月工作计划（统一基座 / 本月目标 / 逐周任务） |
| `docs/DESIGN_DIST_COMM_20260908.md` | 2 卡分布式微调的通信路线思考（推荐 flagcx） |
| `../910C/docs/ASCEND_910C_DC_STREAM_MAPPING_20260902.md` | 训练 ‖ 推理 双侧职责全景 |
| `docs/event_semantics_contract.md` | 统一事件契约 |
| `../910C/docs/ACL_ERROR_MAP_20260901.md` | 错误码映射（108 条 / 64.8% 覆盖）—— **不迁移**给新芯片 |
| `../910C/docs/DIAG_TRAIN_IMAGE_NPU_20260908.md` | 训练镜像 NPU 初始化失败排查记录 |
| `../910C/docs/DC_STAGE_SUMMARY_20260909.md` | **阶段性总结**：两条腿证据 + 组件自检 + 已知缺口（设备上下文部分） |
| `../910C/docs/ERROR_RECOVERY_LOOP_20260909.md` | **错误注入 → 恢复闭环**：验证记录、两个发现、代码修正 |
| `docs/REFERENCE_TWO_INSTANCES_CONFIG_20260920.md` | **两实例配置与依据参考**（910C / P800）：镜像 · 模型 · 训推框架 · 参数 + **依据链** + 复用坑清单 + 可复现命令 + 未覆盖项 |
| `docs/IMAGE_SELECTION_GUIDE_20260920.md` | **镜像选择与确定指南**：需求画像 · 来源优先级 · 入档/入锁两道门槛 · 实操五步 · 判据清单（适用于第三家芯片接入与现有实例镜像补齐） |
| `docs/IMAGE_REQUIREMENT_SPEC_20260920.md` | **镜像需求说明书（提交总组）**：上游官方文档清单（FlagTree per-backend User Manual 等）· 我们镜像与官方推荐的差异 · 硬性/期望/可协商三级需求（逐条带实测依据）· 请总组裁定的三件事 |
| `../P800/docs/KUNLUN_P800_BASE_IMAGE_EQUIVALENCE_20260920.md` | **官方 `-base` 镜像等价性验证报告**：§0 镜像速查（两镜像 tag/digest/大小 · 官方镜像获取与补齐三步 · 容器启动参数对照）· 全部结论复现对照（逐用例 / 逐 `detail`）· **KL3 缺陷与镜像无关** · 入锁建议 |
| `docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md` | **《新芯片接入手册》**（月度计划 11 月交付物）：4 条厂商栈判别路径 · 13 个抽象方法清单 · 8 步接入流程 · **可勾选验收清单** · 9 条跨芯片坑 · 厂商问题上报模板 |
| `docs/INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md` | **接口约定修订建议 6 条**（P800 是现行约定的首次非昇腾检验）：`device_type`/`vendor` 分离 · `device_state` 入契约 · `.native` 逃生舱约束 · **错误对象跨模块类归一** · 有界同步降级契约 · `known_issues()` 入契约 |
| `RELEASE_NOTES_v0.2.0.md` | **组件 v0.2.0 发布说明**（第二实例接入版）：kunlun 后端 · 4 个框架修复 · 脚本后端无关化 · 两实例验证结果 · 已知限制 9 条 |
| `probes/probe_stream_semantics_full.py` | **多流 16 项基线探针（后端无关 V2）**：覆盖 S-1/S-2 补强 + S-8~S-13 共 8 项（真正创建流的验证）；设备 API 前缀由统一运行时给出，同一份脚本跨芯片复用（`DC_BACKEND` / `DC_TAG`） |
| `../P800/docs/KUNLUN_P800_STREAM_BASELINE_16_20260920.md` | **多流 16 项基线逐项比对报告**（P800 第三实例视角）：16 项结论 + 与 910C 对照（仅 S-12 差异）+ S-7 图捕获首测 5/5 + 一处自我纠错 + 证据形态差异说明 |
| `../P800/docs/KUNLUN_P800_BASE_IMAGE_EQUIVALENCE_20260920.md` | **官方 `-base` 镜像等价性验证报告**：全部 P800 结论在官方推荐镜像上复现（逐用例/逐 `detail` 对照）· **KL3 缺陷与镜像无关**（排除"是我们镜像的问题"）· `-base` 开箱缺 `triton` 的完整调用链与补齐命令 · 对镜像入锁的建议 |
