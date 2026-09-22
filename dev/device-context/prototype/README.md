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
├── scripts/                     # **组内服务启动标准脚本**（serve_standard.sh —— 下游起服务唯一入口）
└── docs/                        # 标准说明文档
```

> **要接新芯片？** 直接读 **《新芯片接入手册》** `docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`
> ——厂商栈判别（4 条路径）→ 镜像就绪判据 → 13 个抽象方法清单 → conformance → 两条腿 → 错误闭环
> → **可勾选验收清单**；坑与厂商上报模板也在里面。
> **要起推理服务？** 走 **《组内服务启动标准》** `docs/SERVICE_STARTUP_STANDARD_20260920.md`
> ——唯一入口 `scripts/serve_standard.sh`，跨芯片只改 `DC_BACKEND`，**不要各自维护启动脚本**。
> **要复核我们说过的话？** 走 **《两实例验证复核清单》** `docs/VERIFICATION_MANIFEST_20260920.md`
> ——逐条声明 → 证据文件 → 复跑命令 → 当前缺口，9 条命令即可自行判定 ✅/❌。
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
| 冒烟自检 | **51 通过 / 0 失败**（09-22 对称复跑；此前 48/2 的两项失败已修，见 §3.2） | **42 通过 / 0 失败** |
| conformance（设备上下文与多流 13 例） | 13/13（`ascend`）+ 13/13（`flagos`） | **13/13**（`kunlun`） |
| conformance（推理 6 例） | 6/6 | **6/6** |
| 执行语义基线（多流 16 项中的 8 项探针） | **8/8**（09-22 后端无关 V2 探针**首跑 ascend**） | **8/8**（与 910C 逐项一致） |
| 训练腿 2 卡微调 | **6/6**（09-22：loss 15.4497→11.1515、2212.9 tok/s；与历史 15.4497→11.15 / 2117 tok/s 一致） | **6/6**（loss 15.4488→11.1481、3482 tok/s） |
| 推理腿（单卡前向） | **14/14**（09-22：区分度 0.6380、79.39 句/s、p50 37.73 ms） | **13/13 + 1 如实跳过**（0.6392、53.12 句/s、p50 56.17 ms） |
| 推理腿（vLLM 服务化） | 10/10（区分度 0.4123、108 句/s、p50 27.4 ms） | **10/10**（0.4102、30.70 句/s、p50 96.4 ms） |
| 错误注入 → 恢复闭环（设备侧） | 推理腿 **5 闭环 / 0 失败**（含真实超时 → L3 重放；09-22 复跑同为 5/0/0）；训练腿 **4 闭环 / 1 跳过**（无有界同步，如实跳过）。详见 `../910C/docs/ERROR_RECOVERY_LOOP_20260909.md`（含归因核查：此前两条"发现"已推翻） | 设/不设 `XPU_EVENT_KL3_ENABLE` 两组**各 5 闭环 / 0 失败**，结果**逐字节一致** |
| 官方镜像等价性（换镜像后结论是否成立） | — | ✅ 全部结论在**官方 `-base` 镜像**上复现（conformance 逐用例一致、推理腿 `detail` **14/14 逐字相同**、KL3 挂死一致重现） |

> 表中标注 09-22 的数字来自**两实例对称复跑**（当前代码、同一组命令），证据见
> `../910C/probes/recheck_*_20260922.json` 与 `../P800/probes/recheck_*_20260922.json`；
> 复核入口与缺口状态见 `docs/VERIFICATION_MANIFEST_20260920.md`。

### 3.2 对称复跑（2026-09-22）与修掉的第 5 个跨后端缺陷

在两实例上用**同一组命令**重跑全部判据，暴露出并修复了一个真实缺陷：

- **现象**：ascend 冒烟 **48/2** —— ①`translate_error 回填后端名` 失败（`fe.backend` 为 `None`）；
  ②"声明 `error_map` → 分级来源为 `code_map`"失败（对**不含厂商错误码**的消息误判）
- **根因**：① **后端不对称**——`kunlun.translate_error` 回填了 `backend=self.name`，`ascend`/`flagos` 未回填
  （`FlagosError.backend` 是文档化字段，直调后端方法时为 `None`）；
  ② **判据不公平**——给声明了 `error_map` 的后端注入无厂商码消息，却要求必须走 `code_map`
- **修复**：`ascend`/`flagos` 补回填；判据改为两条诚实断言——"无码消息不得伪称 `code_map`"
  +"含厂商码样例必走 `code_map`"（样例由各后端自带 `SAMPLE_CODED_ERROR`，无样例则如实 SKIP 正向检查）
- **修后**：ascend 冒烟 **51/0**（样例码 507015 → `code_map`/L4_FATAL；无码 → `message_hint`，双向均正确）；
  P800 **42/0** 回归通过；两实例 conformance 13+6 回归全绿
- **教训重演**：冒烟第 [6] 节（后端无关自检）是第二实例阶段**新增**的，新增后从未在第一实例上跑过——
  **对称复跑一跑即暴露**。再次验证"只在一家芯片上验证过的判据/实现不可信"。

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

> **本目录 = 芯片无关**（规范 / 手册 / 标准 / 跨实例参考）；**芯片专属结论在 `../910C/`、`../P800/`**。
> 全量文档的效力分层与一句话说明见**主看板 §6**；本表按同一分层排列。

### 5.1 规范 / 效力文件（下游必须遵守；变更需走流程）

| 文档 | 回答什么 |
|---|---|
| `docs/INTERFACE_CONTRACT_DC_20260908.md` | **我承诺什么接口语义**：统一 API 面 + **Backend 插件接入规范（13 个抽象方法）** + 两条纪律 |
| `docs/event_semantics_contract.md` | **事件语义契约 E1–E4**（昇腾实测驱动的 v2 修订） |
| `docs/SERVICE_STARTUP_STANDARD_20260920.md` | **你们必须怎么起服务**：唯一入口 + 参数表 + 六条硬纪律 + 三个已知行为 |
| `docs/INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md` | **接口约定修订建议 6 条**（**尚未生效，待裁定**）：每条含现状 / 实测依据 / 建议条文 / 兼容性 |

### 5.2 操作手册 / 标准（照做即可）

| 文档 | 回答什么 |
|---|---|
| `docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md` | **新芯片怎么接进来**：8 步流程 + **可勾选验收清单 13 项** + 跨芯片坑 9 条 + 厂商上报模板 |
| `docs/VERIFICATION_MANIFEST_20260920.md` | **怎么复核**：9 条「声明 → 命令 → 判据」+ 证据索引 + 缺口 G1–G8 + 证据命名规范 |
| `scripts/serve_standard.sh` | **服务启动唯一入口**：`DC_BACKEND` 切芯片；verdict = `ready=1 且 smoke=1` |
| `docs/RUNTIME_PROTOTYPE_DESIGN_20260904.md` | **原型怎么设计的**：五域划分、13 个抽象方法的来由、目录结构、验证方式 |
| `docs/RUNTIME_DC_STREAM_PLAN_20260907.md` | **本层职责边界与方法**：设备抽象 / 多流 / 错误翻译 / 状态恢复的划分与上下游分工；**多流 16 项基线出处** |
| `docs/RUNTIME_LAYER_MONTHLY_PLAN_20260908.md` | **月度里程碑与交付物口径** |

### 5.3 参考 / 实测记录（描述"我们当时怎么做"，不含承诺）

| 文档 | 回答什么 |
|---|---|
| `docs/REFERENCE_TWO_INSTANCES_CONFIG_20260920.md` | **两实例配置与依据**：镜像 / 模型 / 训推框架 / 参数逐项对照 + 依据链 + 可比性说明 + 坑清单 + 可复现命令 |
| `docs/IMAGE_SELECTION_GUIDE_20260920.md` | **镜像怎么选**：需求画像（5 条可执行判据）· 来源优先级 · 入档/入锁两道门槛 |
| `docs/IMAGE_REQUIREMENT_SPEC_20260920.md` | **向上游要什么**：官方文档清单 · 我们与官方推荐的差异 · 硬性/期望/可协商三级需求 · 请总组裁定的三件事 |
| `docs/DESIGN_DIST_COMM_20260908.md` | 2 卡分布式微调通信路线思考备忘（**尚未实跑**；与分布式方向的接口约定待回复） |
| `RELEASE_NOTES_v0.2.0.md` | **组件 v0.2.0 发布说明**（第二实例接入版）：纪律 3 条 + 已知限制 9 条 |
| `RELEASE_NOTES_v0.1.0.md` | 组件 v0.1.0 发布说明（初版，单实例） |

### 5.4 跨实例参考（正文在芯片目录）

| 文档 | 回答什么 |
|---|---|
| `../910C/docs/ASCEND_910C_DC_STREAM_MAPPING_20260902.md` | 设备上下文 × Stream 双侧全景（**含 S-1～S-16 编号基线**） |
| `../910C/docs/DC_STAGE_SUMMARY_20260909.md` | 阶段性总结（两条腿证据并入） |
| `../910C/docs/ERROR_RECOVERY_LOOP_20260909.md` | 错误注入 → 恢复闭环验证记录 |
| `../910C/docs/DIAG_TRAIN_IMAGE_NPU_20260908.md` | 训练镜像 NPU 初始化失败排查 |
| `../910C/docs/ACL_ERROR_MAP_20260901.md` | 108 条 ACL 码表与 L1–L4 分级依据 ⚠️ **不迁移** |
| `../P800/docs/KUNLUN_P800_STAGE34_VERIFY_20260920.md` | 阶段 3/4 验证 + **第 4 例框架缺陷**根因与修复 |
| `../P800/docs/KUNLUN_P800_STREAM_BASELINE_16_20260920.md` | 多流 16 项逐项比对（14 / 1 如实声明不支持 / 1 不适用） |
| `../P800/docs/KUNLUN_P800_BASE_IMAGE_EQUIVALENCE_20260920.md` | 官方 `-base` 镜像等价性验证（缺陷与镜像无关） |
| `../P800/docs/PROGRESS_REPORT_20260914.md` | 全量进度报告（待办四分类 + 证据索引） |

### 5.5 探针与原始证据

| 路径 | 内容 |
|---|---|
| `probes/probe_stream_semantics_full.py` | **多流 16 项基线探针（后端无关 V2）**：设备 API 前缀由统一运行时给出，同一份脚本跨芯片复用 |
| `../910C/probes/` ｜ `../P800/probes/` | 两实例原始证据（JSON + 日志），含 **2026-09-22 对称复跑 `recheck_*`**；命名规范见 `docs/VERIFICATION_MANIFEST_20260920.md` §5 |
