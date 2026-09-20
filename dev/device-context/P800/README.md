# P800（昆仑芯）· 第二个芯片接入实例（分支看板）

> 定位：**统一运行时原型的第二个接入实例** —— 按接入规范**新建** `kunlun` backend，
> 而不是「把 910C 的能力适配/移植过来」。
> 迁移的是**规范与方法**；910C 的实现与结论**不迁移**（见 `../910C/README.md` §1 铁律）。
> 状态：🔄 **进行中**（阶段 0–4 已全部完成 ✅；阶段 5 收敛待执行，见 §6）
> 上级看板：`../README.md` ｜ 通用规范与原型：`../prototype/` ｜ 910C 实例：`../910C/`

---

## 0. 结论速览

| 项 | 状态 | 关键数字 |
|---|---|---|
| 阶段 0 · 环境与基线 | ✅ 完成 | 8× P800（96 GB/卡，全空闲）、1.5 TiB 内存、384 线程 |
| 阶段 1 · 单卡接入 | ✅ 完成 | `kunlun` backend 落地；conformance **13/13 + 6/6**；smoke **42/0** |
| 阶段 2 · 训练腿（多卡） | ✅ 完成（**标注条件**） | 两 rank **TRAIN_LEG_PASS 6/6**；loss **15.4488 → 11.1481**；**3482 tok/s** |
| 阶段 3 · 推理腿 | ✅ **完成（PASS 13/13）** | 维度 **1024** ｜ 语义区分度 **0.6392** ｜ **53.12 句/s** ｜ p50 **56.17 ms** |
| 阶段 4 · 错误闭环 | ✅ **完成（PASS，两设置完全一致）** | 闭环 **5 / 跳过 0 / 失败 0**；KL3 设与不设**逐字节一致** ⇒ **关闭该变量不损失诊断能力** |
| 阶段 5 · 收敛 | ⏳ 待执行 | 《新芯片接入手册》+ 接口约定修订建议（4 条）→ release |
| **已知厂商缺陷** | ⚠️ 已定性、已上报 | KL3 事件同步概率性挂死（≈89%），归属**厂商运行时层**；**不影响单进程设备上下文路径**（阶段 4 两设置一致即证据） |
| **框架缺陷（第 4 例）** | ✅ 已发现并修复 | 错误对象**跨模块类不相等** → `disposition` 取 `KeyError`；已修在框架层（详见 `docs/KUNLUN_P800_STAGE34_VERIFY_20260920.md` §3） |

---

## 1. 本目录放什么

| | 内容 |
|---|---|
| ✅ **放** | P800 **这一实例**的验证资产：环境汇总、五域基线实测、接入工作方案、根因核对报告、全量进度报告、探针脚本与**原始证据** |
| ❌ **不放** | `kunlun` backend 代码（属通用原型，在 `../prototype/runtime/backends/kunlun/`）；conformance 用例与结果（在 `../prototype/runtime/conformance/`） |

```
P800/
├── docs/                            # 见 §4
│   ├── KUNLUN_P800_ENV_REPORT_20260914.md
│   ├── KUNLUN_P800_BASELINE_PROBE_20260914.md
│   ├── KUNLUN_P800_ADAPT_PLAN_20260914.md
│   ├── KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md
│   └── PROGRESS_REPORT_20260914.md
└── probes/                          # 探针脚本 + 原始证据（见 §5）
```

---

## 2. 关键认知（写进《新芯片接入手册》，均经实测）

| # | 认知 | 证据 |
|---|---|---|
| **1** | **设备 API 走 `torch.cuda`，`torch.xpu` 不可用** | `torch.xpu.is_available() = False`（AssertionError: Torch not compiled with XPU enabled）；`torch.cuda.device_count() = 8`；编译标志 **`USE_XPU=OFF`**；官方 xpu3.6 单测 `conftest.py` 的 `--device` 默认值即 `'cuda'`。机制为 XPytorch + `torch_xray` 符号重写 |
| **2** | **选卡变量是 `CUDA_VISIBLE_DEVICES`** | 实测 `=2` → `device_count()=1`；`=2,5` → `2`。它才是 910C `ASCEND_RT_VISIBLE_DEVICES` 的对应物，**不是** XPU 侧变量 |
| **3** | **同一 FlagCX，两芯片后端名不同** | 910C 注册为 **`flagos`**；P800 注册为 **`flagcx`**，且**必须显式 `import flagcx`** 才会注册；用法 `init_process_group("cpu:gloo,cuda:flagcx")` + `FLAGCX_ADAPTOR=klx` |
| **4** | **只有 `flagcx` 这一条通信路径可用** | `nccl` 挂死；`xccl` 未编译（`Distributed package doesn't have XCCL built in`）；`kccl` 无响应 |
| **5** | **HF cache 要指向 `snapshots/<hash>`** | 传仓库根目录报 `Unrecognized model ... Should have a model_type key`（根目录只有 `blobs/`、`refs/`、`snapshots/`） |
| **6** | **镜像本机已有，无需联网** | `flagtree-xpu3.6-...-flaggems-main-dev:202608`（38.3 GB）已在本地镜像库 → 官方手册的 59.9 GB `pull` 与 32 GB `load` 全部跳过；FlagGems 源码亦已在容器内 `/env/FlagGems` |

**环境差异（相对 910C 的部署前置，建议入基座说明）**

| 项 | P800 现状 |
|---|---|
| 连接 | 非标准端口 **26008**（`~/.ssh/config` 的 `Host P800` 曾缺 `Port` 行） |
| 权限 | 需加入 `docker` 组 + 需自有可写数据目录（`/data2/hliu553`） |
| 镜像落盘 | `/var/lib/docker` 已 **bind mount 到 `/data1`**（5.8 TB NVMe），充足 |
| 共享程度 | 25 人在线、22 个容器；**用卡前必须 `xpu-smi` 挑「连续且空闲」的卡并记录用卡** |
| 拓扑 | XPU0-3 属 NUMA0、XPU4-7 属 NUMA1；组内 XL 私有链路、跨组 SYS；NIC 与卡 PIX 直连 |

---

## 3. 已知厂商缺陷（⚠️ 需芯片厂商适配，本方向不阻塞）

**现象**：`XPU_EVENT_KL3_ENABLE=1` 且存在设备侧集合通信时，设备事件同步原语**概率性永久自旋**。

| 项 | 内容 |
|---|---|
| **挂死点** | 全部在厂商 `libcuda.so`（实为符号 `libxpucuda.so.515.58.kunlun`）与 flagcx c10d 插件的交界处：① `dist.all_reduce` 内部 `flagcxBackend::syncStream` → `cudaEventRecordWithFlags`；② 上层显式 `torch.cuda.synchronize` → `cudaDeviceSynchronize`；③ 通信域首次初始化（复现率低） |
| **判别条件** | **两要素**，缺一不挂：① `XPU_EVENT_KL3_ENABLE=1`；② 存在设备侧集合通信 |
| **复现率** | **18 次运行 16 次挂死（≈89%）**；本轮基线 4/4 = 100%；挂死步数游走（rep 0/20/30/40/70/100） |
| **无关项** | 数据量、张量形状、reduce op、用哪对卡、同步间隔 **均无关**（逐一单变量排除） |
| **责任层** | **厂商运行时/驱动层** —— 算子层（FlagGems：`flag_gems` 未导入、探针 0 引用）与编译层（FlagTree/triton：`/root/.triton` mtime 仍为镜像构建时、无编译产物变更、走 BKCL 预编译内核）**均已硬证据排除** |
| **给上游的偏移** | `libcuda.so.1` 映射基址 `0x744bb1400000`，自旋帧 `0x744bb1494080` ⇒ **偏移 `+0x94080`** |
| **临时规避** | **不设置** `XPU_EVENT_KL3_ENABLE`（我方训练腿为 transformers + 原生 torch、推理腿为 vLLM/vllm-plugin-FL，**源码中 `flag_gems` 0 处引用 ⇒ 均不依赖 FlagGems**，故该变量**不属于我方验证前置条件**） |
| **规避的代价** | 该变量是 **FlagGems kunlunxin 后端的官方推荐变量**（`tools/env.sh`、`src/flag_gems/backends.yaml`、CI `P800.yml` 三处均设 1），且**厂商文档与镜像里对它的说明为零** ⇒ 关闭是否损失设备异常上报**须上游确认**。本方向**不擅自改锁定镜像口径、不改公共资产** |
| **机器可读声明** | `kunlun` 后端 `info()["known_issues"]`（12 字段结构化，含复现率/责任层/规避/上报对象）；其他子方向接入时**读到后端即可获知** |
| **开跑前告警** | `proto_train_leg.py` 的 `_preflight_env_check()`：设该变量时明确告警（不设时不误报） |

> **统一措辞**：根本原因在厂商 CUDA 兼容运行时 `libxpucuda.so`（KL3 事件机制与设备事件同步原语的交互），
> **需上报芯片厂商适配**；我方已按上述方式规避以不阻塞本方向验证，并如实标注条件。

---

## 4. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/KUNLUN_P800_ENV_REPORT_20260914.md` | **任务 1 环境汇总**：主机/系统、8× P800、内存/CPU、数据盘与权限、网络与拓扑、可复用资产 |
| `docs/KUNLUN_P800_BASELINE_PROBE_20260914.md` | **五域基线实测**：设备抽象 / 多流 Stream-Event / 算子 / 错误 / 分布式，含与 910C 对照与 3 条缺失项 |
| `docs/KUNLUN_P800_ADAPT_PLAN_20260914.md` | **接入工作方案**：定位与交付边界、接入路线（§2.2 后端划分依据）、验证与验收标准（6 条）、阶段计划、风控、对外提交物、**§7.5 接入过程暴露并已修的 3 个框架缺陷** |
| `docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md` | **结论核对与责任层判定**：准确性 / 可复现性 / 该提算子层还是编译层（三问全答） |
| `docs/PROGRESS_REPORT_20260914.md` | **全量进度报告**（910C 回顾 + P800 主体 + 待办总清单按「谁来做」四分类 + 证据索引 + 风险与下一步） |
| `docs/KUNLUN_P800_STAGE34_VERIFY_20260920.md` | **阶段 3/4 验证报告**：推理腿 13/13 与 910C 同构对照、错误闭环两设置对照（逐字节一致）、**§3 第 4 个框架缺陷的根因与修复**、待办 |

**规范与原型（在 `../prototype/`，不属本目录）**

- 接入规范本体：`../prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` §2「Backend 插件接入规范」
- 事件语义契约：`../prototype/docs/event_semantics_contract.md`
- `kunlun` backend 实现：`../prototype/runtime/backends/kunlun/backend.py`
- conformance 结果：`../prototype/runtime/conformance/conformance_runtime_kunlun.json`（13/13）、`..._kunlun_infer.json`（6/6）

---

## 5. 证据索引（`probes/`）

**探针脚本（可复现）**

| 脚本 | 用途 |
|---|---|
| `dc_probe_p800.py` | 五域探针（设备抽象 / 多流 / 错误 / 分布式），单进程一次跑完 |
| `dc_probe_isolated.py` | **单变量隔离**版：每用例独立进程，避免同进程内错误粘滞污染 |
| `dc_probe_rep.py` | 参数化重复集合通信探针（`MODE`=ar / ar_nosync / ar_max / barrier；`SYNC_EVERY`；`REPS`；`SIZE`） |
| `dc_probe_devonly.py` | 对照组：单进程纯设备计算（无通信） |
| `dc_probe_grad_ar.py` | 梯度通信定点探针：单次大通信（concat）vs 多次小通信（逐参数） |
| `dc_probe_ar_rep.py` | 重复性与确定性判定（同一张量重复 vs 真实梯度按序） |
| `dc_probe_verify.py` | **带真值校验**的验证探针（`2^120` 精确匹配，用于排除假阴性） |
| `smoke_kunlun_20260914.txt` | 组件自检原始输出（**42 通过 / 0 失败**） |

**探针组脚本（自动抓 gdb 原生栈）**

`probe_battery.sh`（第一轮 8 变体）、`probe_battery2.sh`（第二轮重复验证）、`probe_battery3.sh`（第三轮剂量-反应）、`verify_battery.sh`（真值校验四组）

**原始证据日志**

| 日志 | 内容 |
|---|---|
| `A_round1_battery_20260914.log` | 第一轮 8 变体单变量对照 + 挂死现场原生栈 |
| `B_round2_repeat_20260914.log` | 第二轮重复验证（A×3 挂死 / B×3 通过 / C / D / E） |
| `C_round3_dose_20260914.log` | 第三轮剂量-反应（同步间隔 N=1,1,2,3,5,10）→ 证明**无阈值效应** |
| `D_verify_truthvalue_20260914.log` | 真值校验四组（A×4 挂死 / B×2 与 D 精确通过 / C×2 挂死）⇒ 更正「不同步就通过」的假阴性 |
| `E_train_ab.log` | **训练腿 A/B 单变量对照**（不设 → 退出码 0；设=1 → 退出码 124） |
| `E_train_R1_workaround_noKL3.log` | 训练腿规避条件运行日志 |
| `E_train_R2_control_KL3on.log` | 训练腿对照条件运行日志（挂死现场） |
| `E_train_r1_keep.log` | 规避腿复跑（一致性确认） |
| `E_train_leg_result_rank0.json` / `rank1.json` | **训练腿结果 JSON**：`TRAIN_LEG_PASS 6/6`、6 项检查全绿、loss 曲线、perf |
| `F_infer_leg.sh` / `F_infer_leg_20260920.log` / `F_infer_leg_result_20260920.json` | **阶段 3 推理腿**：脚本 + 完整日志 + 结果（13/13，维度 1024 / 区分度 0.6392 / 53.12 句/s / p50 56.17 ms） |
| `G_error_loop.sh` / `G_error_loop_20260920.log` | **阶段 4 错误闭环**：两设置对照脚本 + 日志（两组各 5/0/0） |
| `error_recovery_loop_kunlun_KL3off.json` | 阶段 4 结果：**不设** `XPU_EVENT_KL3_ENABLE` |
| `error_recovery_loop_kunlun_KL3on.json` | 阶段 4 结果：**设** `XPU_EVENT_KL3_ENABLE`（与上面除时间戳外**完全一致**） |

> ⚠️ **注意**：本目录 `*.log` 为**原始证据**，需随仓库分发，故在此目录放了局部 `.gitignore`（`!*.log`）
> 覆盖根仓库的 `*.log` 通用忽略规则。**此前这批日志因根规则从未入库**，本次整理时一并纳入。

---

## 6. 下一步

**执行前置（把最新原型与脚本送进容器）**

```bash
tar czf /tmp/dc.tgz -C dev/device-context prototype P800/probes \
    --exclude='__pycache__' --exclude='*.tgz'
scp /tmp/dc.tgz P800:/data2/hliu553/            # 端口 26008
ssh P800 'cd /data2/hliu553 && tar xzf dc.tgz --overwrite && rm dc.tgz'
```

**任务表**

| # | 动作 | 状态 | 结果 / 判据 |
|---|---|---|---|
| 1 | **阶段 3 推理腿（单卡前向）** | ✅ **09-20 完成** | `INFER_LEG_PASS 13/13`（1 项如实跳过）：维度 **1024** ｜ 区分度 **0.6392** ｜ **53.12 句/s** ｜ p50 **56.17 ms** |
| 1b | 阶段 3 推理腿（vLLM 服务化形态，对齐 910C 的 108 句/s 口径） | ⏳ 待补 | `proto_infer_serve.py` 的昆仑芯适配（容器内已有 vLLM 0.13.0） |
| 2 | **阶段 4 错误闭环（两设置对照）** | ✅ **09-20 完成** | 两组均 `ERROR_RECOVERY_LOOP_PASS`（闭环 **5 / 跳过 0 / 失败 0**），**逐字节一致** ⇒ 关闭 KL3 **不损失**诊断能力 |
| 3 | **阶段 5 收敛**：《新芯片接入手册》（§2 六条认知 + §3 已知缺陷 + 验收清单）；接口约定修订建议（4 条）；原型 release | ⏳ 待执行 | — |
| 4 | **上报渠道待确认**：直连昆仑芯支持，还是经总组转达 | ⏳ 待定 | STATUS「阻塞与需要协调」已登记 |

**本次为执行做的准备（2026-09-20）**

- `prototype/runtime/proto/proto_infer_leg.py` → **后端无关化 V2**：路径/模型走 `DC_*` 环境变量、
  设备串取 `runtime.current().device_type`、同步走 `runtime.synchronize()`、真实异常注入替代伪造错误码、
  补 p50/p90 时延；`DC_MODEL` 支持 hub 的 `models--xxx` 目录（自动解析 snapshot）。
- `prototype/runtime/proto/proto_error_recovery_loop.py` → 补 `DC_BACKEND` 环境变量（与另两条腿一致）。
- 新增 `probes/F_infer_leg.sh`、`probes/G_error_loop.sh`（超时兜底 + 挂死快照 + 用卡复查，沿用 battery 体例）。

**环境核对结论（2026-09-20 只读实测）**：容器 `hliu553-device-context-p800` Up 5 天；
conda env `python310_torch29_cuda`（py3.10.18 / torch 2.9.0+cu129 / transformers 4.57.1 /
sentence_transformers 5.7.0 / **vLLM 0.13.0** / flagcx 可导入）；`torch.cuda.device_count() = 8`；
共享缓存模型 `/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…`（挂载 `/hf_cache`）。
