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
│   │   ├── ascend/              # 第 1 家 昇腾 910C（torch_npu，训推两腿）
│   │   ├── cambricon/           # 第 3 家 寒武纪 MLU590（torch_mlu）
│   │   └── kunlun/              # 第 2 家 昆仑芯 P800（torch.cuda / XPytorch）
│   ├── conformance/             # 验收用例 13 例 + 推理 6 例 + runner（含资产模块）
│   ├── proto/                   # 两条腿自验证脚本与结果
│   ├── demos/                   # 设备无关演示
│   └── smoke_runtime.py         # 接入自检
├── probes/                      # 跨后端验证探针（目前：多流 16 项基线探针，后端无关 V2）
├── scripts/                     # **标准化脚本**（serve_standard.sh 起服务唯一入口；preflight_env.sh 环境普查）
└── docs/                        # 标准说明文档
```

> **要接新芯片？** 直接读 **《新芯片接入手册》** `docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`
> ——厂商栈判别（4 条路径）→ 镜像就绪判据 → 13 个抽象方法清单 → conformance → 两条腿 → 错误闭环
> → **可勾选验收清单**；坑与厂商上报模板也在里面。
> **要起推理服务？** 走 **《组内服务启动标准》** `docs/SERVICE_STARTUP_STANDARD_20260920.md`
> ——唯一入口 `scripts/serve_standard.sh`，跨芯片只改 `DC_BACKEND`，**不要各自维护启动脚本**。
> **要复核我们说过的话？** 走 **《验证复核清单》** `docs/VERIFICATION_MANIFEST_20260920.md`
> ——逐条声明 → 证据文件 → 复跑命令 → 当前缺口，9 条命令即可自行判定 ✅/❌。
> **要查路线选型与归档？** 走 **《路线 B 退出归档 · 三实例厂商分支核验》**
> `docs/ROUTE_B_ARCHIVED_20260922.md` —— 删了什么 / 保留什么 / 三家为何都走厂商分支 / 残留全量清单 / 复跑清单。
> **要不要发布？** 走 **《统一原型 · 三芯片职责验收与发布结论》**
> `docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260922.md` —— 职责清单 × 验收项 × 三芯片实测结果 + 发布结论（含复跑命令）。
> 组件版本 **`runtime-v0.2.0`（三芯片统一原型版）** —— 见 `RELEASE_NOTES_v0.2.0.md`；
> **版本以 [GitHub Release](https://github.com/FlagRT/runtime-team/releases/tag/runtime-v0.2.0) 为准**（git tag 仅作提交指针）。

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

## 3. 验证状态（三实例实跑）

> **本表只给"过没过"**；每条的过程、判据明细与我方缺陷复盘在专题报告里（见末列）。
> 发布判定：**三实例 12 项判定全部通过**（09-28 收口）⇒ **可发布**；
> 全文 `docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md` **L11–L27**（§一 结论）· **L65–L123**（§三 逐芯片）。
> ⭐ **2026-10-08 增量**：三实例在**同一代码版本**上跑完**扩口径后的 78 项职责审计**
> （910C **73/0/5** · P800 **67/0/11** · MLU590 **62/0/16**，SKIP 均如实不具备），
> 并各配**逐条注入的非空转验证**。

| 验证 | 910C（`ascend`） | P800（`kunlun`） | MLU590（`cambricon`） | 详见 |
|---|---|---|---|---|
| 冒烟自检 | **52/0** | **46/0** | **46/0** | `runtime/smoke_runtime.py` |
| conformance | **13/13 + 6/6** | **13/13 + 6/6** | **13/13 + 6/6** | `runtime/conformance/` |
| 执行语义基线（多流 8 项探针） | **8/8** | **8/8** | **8/8**（另 S-7 图捕获 5/5） | `probes/probe_stream_semantics_full.py` |
| 训练腿 2 卡微调 | **6/6**（15.4498→11.1479 · 3954–4402 tok/s · `hccl`） | **6/6**（3533.5 tok/s） | **6/6**（3015.3 tok/s · `cncl`） | `runtime/proto/proto_train_leg.py` |
| 推理腿（单卡前向） | **14/14**（77.49 句/s · 区分度 0.6391） | **13/13 +1 跳过**（53.28 句/s） | **13/13 +1 跳过**（41.08 句/s） | `runtime/proto/proto_infer_leg.py` |
| 推理腿（vLLM 服务化） | **`SERVE_STANDARD_PASS`**（35 s 就绪） | **`SERVE_STANDARD_PASS`**（25 s） | **`SERVE_STANDARD_PASS`**（150 s；⚠️ 须 **vLLM 应用镜像**容器） | `scripts/serve_standard.sh` · `docs/SERVICE_STARTUP_STANDARD_20260920.md` |
| 错误注入 → 恢复闭环 | 推理腿 **5/0/0** · 训练腿 **5/0/0** | 设/不设 KL3 **各 5/0/0**（逐字节一致） | **5/0/0**（四类注入） | `runtime/proto/proto_error_recovery_loop.py` |
| ⭐ **职责响应审计（78 项 · 10-08）** | **73 / 0 / 5** | **67 / 0 / 11** | **62 / 0 / 16** | `docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md` **L42–L92** |
| 离线契约自检（**无设备**） | **90/0/1** | **108/0/1** | **97/0/0** | `scripts/backend_offline_check.py` |
| 官方镜像等价性 | — | ✅ 全部结论复现 ⇒ **缺陷与镜像无关** | — | `../P800/docs/KUNLUN_P800_BASE_IMAGE_EQUIVALENCE_20260920.md` |
| ⭐ 流优先级「调度效果」 | ⚪ `NOT_APPLICABLE`（未声明 `control`） | ⚪ `NOT_APPLICABLE`（单档 `(0,0)`） | ✅ **分场景**：同时就绪 ⇒ 无实质效果 / 排队争用 ⇒ 有实质效果 | `docs/STREAM_PRIORITY_SCHED_EFFECT_20261008.md` **L21–L54** |

> **换芯片只改一行**：同一条命令，只把 `--backend` / `DC_BACKEND` 在 `ascend`／`kunlun`／`cambricon`
> 之间换（设备串、选卡变量、通信后端名由后端各自封装，见 `docs/REFERENCE_TWO_INSTANCES_CONFIG_20260920.md`）。
>
> **两个易混点（务必看清）**：① 两条腿**都是经统一 API**（`runtime.use(...)`）接入设备的，但
> `../910C/distributed_training|distributed_inference/` 里的历史训推（Qwen2.5-1.5B DDP、Qwen3-4B
> vLLM+TP）是**旧代码路径**，**尚未用本原型复跑**；② **09-22 对称复跑修掉的第 5 个跨后端缺陷**、
> **训推复跑逐腿细节**与**历史缺口登记**（含 KL3 待上报）已移入
> `docs/BOARD_ARCHIVE_PROTOTYPE_VALIDATION.md`。


## 4. 运行方式

```bash
# ── 910C ──
python3 runtime/smoke_runtime.py
python3 runtime/conformance/runner.py --backend ascend [--cases infer_cases]
                                    # 910C（训推统一 torch_npu，2026-09-22 起）
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

注意（910C）：**带卡容器名额（2026-09-29 更正）—— 独占单位是「已 init 容器的挂载设备集」**：某容器一旦 `acl.init` 成功，就把它挂载的全部设备整组独占，其他容器与之有交集即失败（`acl.init()`=500000、`get_device_count()=(0,507899)`）；**空闲（Up 未 init）容器不占名额**。⇒ **不要挂全 16 张**，**只挂要用的空闲卡** 即可与现网并存（不必停他人容器）；两条腿仍**串行**。旧口径「同一时刻只留 1 个 / 只挂无人卡也一样」**均已更正** ⇒ `../910C/docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md`。
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
| `scripts/preflight_env.sh` | **环境普查一键脚本**（接入手册 §1 那 7 项的可执行版）：只读、不装东西，缺项如实标注；输出可直接作为环境报告 |
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
| `RELEASE_NOTES_v0.2.0.md` | **组件 v0.2.0 发布说明**（三芯片统一原型版）：纪律 3 条 + 已知限制 10 条 + 三实例验证矩阵 |
| `RELEASE_NOTES_v0.1.0.md` | 组件 v0.1.0 发布说明（初版，单实例） |

### 5.4 跨实例参考（正文在芯片目录）

| 文档 | 回答什么 |
|---|---|
| `probes/probe_graph_capture_stream_v2.py` | **图捕获探针（后端无关 V2）**：**契约内 4 项判据**（G1/G2/G3/G5）+ **1 项宽容度观察项**（G4「捕获区内切流」= **上游契约外用法**，不计判定；MLU590 容忍 / P800 不容忍）；设备 API 前缀由统一运行时给出、图对象类**动态发现**；**观察项放最后 + 条目失败后清理状态**（失败捕获会污染后续条目）。910C 原版（硬编码 `torch_npu`）**保留不删**作历史归档 |
| `probes/probe_stream_quota.py` | **S-16 流数量配额探针**（后端无关）：连续创建 2000 流 / 首流仍可用 / 释放后重建，三项判据 |
| `scripts/backend_offline_check.py` | **无设备离线契约自检**：**四家内置 stub** + **显式 SKIP 机制** + **真实厂商运行时阻断器**（保证"离线"名副其实）+ **入口兜底**（未预期异常不再整轮崩掉）+ **可控 rc 的假 pyACL**（把 `ascend` 的有界同步从 SKIP 变实测）。`--backend <名>` 单跑：cambricon **39/0/0** · kunlun **39/0/1 跳过** · flagos **32/0/2 跳过** · ascend **35/0/1 跳过**；`--all` 跨后端对称性自检：**5/0** |
| `docs/BACKEND_SYMMETRY_AUDIT_20260922.md` | **跨后端对称性审计台账**（第 6/7/8 条缺陷 + 2 处证据污染 + 防回归判据 + 非空转验证） |
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
