# 路线 B（torch_fl）退出归档 · 三实例厂商分支核验

> 状态：✅ **已执行完毕（2026-09-22）** ｜ 作者：Kistich（hliu553）
> 定位：**处置记录 + 核验报告**。回答两件事：
> ① 路线 B 的资产**哪些删了、哪些保留、为什么**；② **三个芯片实例当前是否都走厂商分支**。
> 规范/效力文件见 `INTERFACE_CONTRACT_DC_20260908.md`；取舍依据见
> `summary/DEVICE_ABSTRACTION_ROUTE_AB_SUMMARY_20260922.md`。

---

## 一、结论（一句话）

**原型（`prototype/`）内已无路线 B 的活跃路径**：`runtime/backends/flagos/` 已整体删除，
注册表 / 离线自检 / 复跑命令 / 后端清单全部清理；**三个芯片实例（910C / P800 / MLU590）
当前一律走厂商官方 torch 插件路线**。历史上落地的路线 B 资产**保留为归档**
（保住路线 A/B 选型结论的原始证据链），并在各入口加统一横幅标注「已冻结，非当前口径」。

---

## 二、处置清单

### 2.1 删除（原型内的活跃路径）

| 对象 | 动作 | 说明 |
|---|---|---|
| `prototype/runtime/backends/flagos/`（`backend.py` + `__init__.py`） | **删除** | 该后端的唯一实现；删除后 `use("flagos")` 会走 `BackendNotFound`（不再静默降级） |
| `runtime/backends/registry.py` `_KNOWN_BACKENDS` | 改为 `("ascend", "kunlun", "cambricon")` | 自动发现不再扫描该家 |
| `scripts/backend_offline_check.py` | 删除 `_stub_flagos` 与其 `_STUBS` 项 | 自检工具的参与后端由四家收敛为三家 |
| `runtime/smoke_runtime.py` | 后端候选列表去该项 | 冒烟自检的挑选顺序与兜底列表 |
| `runtime/proto/proto_train_leg.py` | 删 `if BACKEND == "flagos": import torch_fl` 分支、删 `_DIST_BT_DEFAULT["flagos"]` | 训练腿脚本不再有任何该路径的分支 |
| `probes/probe_*.py`（3 个） | 去该后端的映射项与 `BACKEND in (...)` 条件 | 探针的后端无关化表 |
| 文档内的**复跑命令 / 后端清单 / 口径行** | 改为三家厂商后端 | 含 `prototype/README.md`、`runtime/README.md`、`docs/REFERENCE_TWO_INSTANCES_CONFIG`、`docs/VERIFICATION_MANIFEST`、`docs/NEW_CHIP_ONBOARDING_MANUAL`、`dev/device-context/README.md`、`STATUS.md`、`summary/*` 等 |

### 2.2 保留（归档；不删证据）

| 对象 | 处置 | 理由 |
|---|---|---|
| `910C/distributed_training/ascend_regression/**` | **保留 + 归档横幅** | 这是路线 A/B 选型的**原始证据链**（同机同模型吞吐对照、108 条 ACL 错误码表来源、事件语义缺口复现）。删掉它，"为什么不用路线 B"就只剩结论没有依据 |
| `910C/docs/{DC_STAGE_SUMMARY,ERROR_RECOVERY_LOOP,PROGRESS}_*.md` | 同上 | 当时阶段的完整记录 |
| `P800/docs/{PROGRESS_REPORT,KUNLUN_P800_ADAPT_PLAN}_20260914.md` | 同上 | P800 接入期的跨后端结论来源 |
| `prototype/RELEASE_NOTES_v0.1.0.md`、`docs/BACKEND_SYMMETRY_AUDIT_20260922.md`、`docs/INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md` | 同上 | 已发布版本说明 / 缺陷台账 / 契约修订建议 —— **它们的价值恰恰在于记录了"发现过什么"** |
| `runtime/conformance/conf_proto_flagos_13.json`、`runtime/proto/error_recovery_loop_flagos.json`、`runtime/proto/train_leg_result_rank{0,1}.json` | **保留**（文件名不变，按本文件标注为历史证据） | 是当时的原始结果 JSON；改文件名会破坏既有文档指针，改内容则等于篡改证据 |
| `910C/docs/OFFICIAL_RUNTIME_COUNTERPART_20260922.md` 等的**对照行** | 保留 + 就地划掉/标注「已取消」 | 镜像对照是"官方走 Route A、我们也走 Route A"的论据，删掉反而看不出对齐点 |

### 2.3 未动（不是路线 B 内容，勿误清）

| 对象 | 说明 |
|---|---|
| `flagos-runtime-*` / `flagos-app/*` / `flagos-base` / `flagos-dev` | **FlagOS 官方镜像仓名**，如寒武纪定档的 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0` —— 是**资产名**，不是设备后端名 |
| `resource.flagos.net/repository/flagos-pypi-*/simple` | 官方私有 PyPI 源 |
| `FlagosError` / `DISPOSITION` / `ErrorCategory` | **统一错误对象与分级的契约名**（见接口约定），芯片无关，与设备后端路线无关 |
| `scripts/backend_offline_check.py` 的 `VENDOR_ROOTS` 含 `torch_fl` | **刻意保留**：该清单的语义是"厂商运行时一个都不许碰"（保证自检真的离线），多列一家只增安全性 |
| `docs/RUNTIME_PROTOTYPE_DESIGN_20260904.md`、`docs/event_semantics_contract.md` 的 `torch_fl` 引用 | **设计来源 / v2 修订动因**的历史说明，非落地路径（后者已在正文顶部加了说明） |

---

## 三、三实例厂商分支核验（本次实测）

**核验命令**（在仓库根执行，可自行复现）：

```bash
# ① 注册表里到底注册了哪几家
grep -n "_KNOWN_BACKENDS = " dev/device-context/prototype/runtime/backends/registry.py
ls dev/device-context/prototype/runtime/backends/

# ② 三家后端的厂商标识（device_type = torch 命名空间）
grep -nE '^    name = |^    device_type = ' dev/device-context/prototype/runtime/backends/*/backend.py

# ③ 原型内是否还有该路线残留（应为 0 条命中，除下方 §4 声明的保留点）
grep -rnE 'torch_fl' dev/device-context/prototype/
grep -rnE '`flagos`|--backend flagos|DC_BACKEND=flagos' dev/device-context/prototype/
```

**核验结果**：

| 实例 | 芯片 | 后端名 | 厂商官方 torch 栈 | `device_type` | 集合通信后端 | 训练腿 | 推理腿 |
|---|---|---|---|---|---|---|---|
| 第 1 家 | 华为昇腾 910C | `ascend` | **`torch_npu`** | `npu` | `hccl`（torch_npu 原生） | ✅ 统一 torch_npu（2026-09-22） | ✅ torch_npu |
| 第 2 家 | 昆仑芯 P800 | `kunlun` | **`torch.cuda` 兼容层（XPytorch + torch_xray 符号重写）** | `cuda` | `cpu:gloo,cuda:flagcx` | ✅ | ✅ |
| 第 3 家 | 寒武纪 MLU590 | `cambricon` | **`torch_mlu`** | `mlu` | `cncl` | ✅ | ⏳ 未做 |

**判定**：

- ✅ **三家一致走厂商官方 torch 插件路线**（路线 A），`device_type` 分属三种命名空间 `npu` / `cuda` / `mlu`；
- ✅ **原型内无该路线（路线 B）的活跃路径**：后端目录已删、注册表不含、自检工具不含、训练腿脚本无分支、复跑命令无引用；
- ✅ `use("flagos")` 现在会**如实报错**（`BackendNotFound`），不会静默落回别家 —— 这是"删除"相对"保留但禁用"更可信的地方。

---

## 四、仓库内仍含 `torch_fl` / `flagos` 的位置（全量清单，逐条说明）

以下为**有意保留**的残留，均可解释；除此之外原型内应为零命中（§3 命令 ③ 可验）。

| # | 位置 | 性质 | 为什么保留 |
|---|---|---|---|
| 1 | 归档横幅本体（11 个文件顶部） | 归档标注 | 横幅自身要写明被归档的路线名 |
| 2 | 归档文档正文（910C `ascend_regression/**`、`DC_STAGE_SUMMARY_20260909.md`、`ERROR_RECOVERY_LOOP_20260909.md`、`PROGRESS_20260822.md`；P800 `PROGRESS_REPORT_20260914.md`、`KUNLUN_P800_ADAPT_PLAN_20260914.md`；prototype `RELEASE_NOTES_v0.1.0.md`、`BACKEND_SYMMETRY_AUDIT_20260922.md`、`INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md`） | 历史档案正文 | 原始证据与结论；**每条都已被横幅标注为"非当前口径"** |
| 3 | `prototype/runtime/proto/proto_train_leg.py:28` | 历史说明 | 解释"那个权宜例外为什么曾存在"，紧接着写明**已取消且后端已删除** |
| 4 | `prototype/scripts/backend_offline_check.py:96,98,317` | 离线阻断清单 | 见 §2.3；已就地注释说明 |
| 5 | `prototype/docs/event_semantics_contract.md`、`RUNTIME_PROTOTYPE_DESIGN_20260904.md`、`docs/DESIGN_DIST_COMM_20260908.md` 的引用 | 设计来源 / 修订动因 | 契约与设计的**来源说明**；契约正文顶部已加"本文件芯片无关"的说明 |
| 6 | `prototype/runtime/conformance/conf_proto_flagos_13.json` 等 4 个结果 JSON | 原始证据 | 见 §2.2；文件名与内容一律不改 |
| 7 | 镜像仓名 / pypi 源（`flagos-runtime-*` 等） | 官方资产名 | 见 §2.3 |

> **口径纪律**：凡引用本表中第 2 类内容，**必须同时标注"路线 B 历史（已归档）"**，
> 不得作为当前能力或当前路线引用。

---

## 五、复跑复核（当前原型可复跑清单）

删除该后端后，**全部判据与脚本仍可原样复跑**，且参与后端由"两个设备后端"变为"三家厂商后端"。

| # | 复跑项 | 命令（`prototype/` 目录内） | 本次实测结果（2026-09-22） |
|---|---|---|---|
| 1 | 离线契约自检 · 昇腾 | `python3 scripts/backend_offline_check.py --backend ascend` | **35 通过 / 0 失败 / 1 跳过** |
| 2 | 离线契约自检 · 昆仑芯 | `python3 scripts/backend_offline_check.py --backend kunlun` | **39 通过 / 0 失败 / 1 跳过** |
| 3 | 离线契约自检 · 寒武纪 | `python3 scripts/backend_offline_check.py --backend cambricon` | **39 通过 / 0 失败 / 0 跳过** |
| 4 | 跨后端对称性自检 | `python3 scripts/backend_offline_check.py --all` | **5 通过 / 0 失败**（三家全部可加载、方法齐备、`info().supports` 键集合与 `capabilities` 自洽） |
| 5 | 语法校验 | 全部 30 个 `.py` 编译 | **30 通过 / 0 失败** |
| 6 | conformance 13 例 | `python3 runtime/conformance/runner.py --backend <ascend\|kunlun\|cambricon>` | 三家均 **13/13**（真机已验，见各芯片目录） |
| 7 | conformance 推理 6 例 | `python3 runtime/conformance/runner.py --backend <B> --cases infer_cases` | 三家均 **6/6**（真机已验） |
| 8 | 冒烟自检 | `python3 runtime/smoke_runtime.py [--backend <B>]` | 910C **51/0** · P800 **42/0** · MLU590 **42/0** |
| 9 | 训练腿 2 卡 | `DC_BACKEND=<B> DC_DIST_BT=<bt> torchrun --nproc_per_node=2 runtime/proto/proto_train_leg.py` | 三家均 **6/6**（通信后端 `hccl` / `cpu:gloo,cuda:flagcx` / `cncl`） |
| 10 | 错误注入 → 恢复闭环 | `python3 runtime/proto/proto_error_recovery_loop.py --backend <B>` | 910C **5/0/0** · P800 **5/0/0**（两组）· MLU590 **5/0/0** |
| 11 | 推理服务启动标准 | `DC_BACKEND=<B> bash scripts/serve_standard.sh` | 910C / P800 已验 `SERVE_STANDARD_PASS`；MLU590 待跑 |

**边界（如实标注）**：第 1–5 项为**本次在本机实测**；第 6–11 项为**各芯片真机历史实测**，
本轮**未重跑**（910C 带卡容器并发名额受限、寒武纪两台主机当日 SSH 超时，详见 `../STATUS.md`）。
本文件只做"命令与资产仍然成立"的复核，**不替代真机复跑**。
