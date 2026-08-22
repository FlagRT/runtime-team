# VERSIONS.md —— 物资起始清单与版本事实源

> 用途：**凭此清单可准确复现开发环境**
> 双线结构：**A 主线（生产交付）+ B legacy（预研支线）**，详见 §1.2。

## 0. 物资起始清单（一屏）

要准备的东西：

| # | 物资 | 数量 | 明细见 |
|---|---|---|---|
| 1 | 组织主干仓（FlagRT 组织下，成员直接 clone） | 7 个（6 子库 + runtime-team） | §1.1 |
| 2 | GitHub 账号（SSH key + 组织成员 write 权限） | 1 个 | §1.1 |
| 3 | 基础镜像 | 6 个（本机已有 5 个，1 个待拉取） | §2 |
| 4 | venv 运行组合（容器内） | 2 套实测 + 1 占位（待验证） | §3 |
| 5 | 运行环境变量 | 14 个（昇腾 5 + 昆仑芯 9，按芯片/路线拆分） | §4 |
| 6 | 容器启动参数（子方向专属） | 见 memory 子方向 `dev/memory/README.md` | 子方向目录 |

## 1. 开源代码仓（6 个；原 5 个 fork 基线于 2026-08-17 锁定）

> 原 5 个仓的基线 commit 与上游一致（见 §1 表格）；FlagPerf 按 FlagRT fork 的 `main` 对齐。成员直接 clone FlagRT 组织仓，无需 fork。
> **注**：PyTorch-Plugin-FL 上游已改名 **Torch-FL**（2026-08-18 确认，旧名 301 重定向）；本地目录/容器路径沿用 PyTorch-Plugin-FL（clone 时指定目录名）。

| 仓（fork 名） | 上游链接 | 基线 | 版本标识 | 角色 | 依赖（源码核实） |
|---|---|---|---|---|---|
| Torch-FL（本地目录 PyTorch-Plugin-FL） | github.com/flagos-ai/Torch-FL | main@caefaae | torch_fl 0.1.0 | **B 线（预研支线）**：torch_fl 设备层验证与上游贡献；**A 线不使用**（上层 6 仓对其零引用） | torch 2.10 固定（setup.py TORCH_PIN `>=2.10,<2.11`）；Python≥3.8；triton/flag_gems 由平台注入（不装 PyPI triton） |
| FlagCX | github.com/flagos-ai/FlagCX | main@0a747f6 | flagcx 0.13.0 | 多卡通信 + KV 传输 | torch（构建期自动检测，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`）；源码构建需 make + git submodule（plugin/torch 为 torch 插件） |
| FlagGems | github.com/flagos-ai/FlagGems | master@c22f8eb | flag_gems 0.0.0（editable） | 多芯片 triton 算子库（昇腾/昆仑芯/真武等） | packaging≥26.0、PyYAML==6.0.1、sqlalchemy==2.0.48、numpy；ascend 组合 torch==2.10.0+cpu（官方 extra 含 torch-npu；**A 线用 torch_npu**，B 线以 torch_fl 替代） |
| vllm-plugin-FL | github.com/flagos-ai/vllm-plugin-FL | main@db9afd6 | vllm-plugin-fl 0.0.0（editable） | 推理插件（KV Cache 挂载点、Platform 层） | 运行时 pyyaml；配套 vllm==0.20.2；Python 3.10~3.13；构建需 torch≥2.7.1 |
| FlagTree | github.com/flagos-ai/FlagTree | 分支 triton_v3.2.x | triton_ascend 3.2.1（import 报 3.2.0，已知差异） | 编译层（triton kernel 编译） | 构建 setuptools/wheel/cmake≥3.18/ninja≥1.11.1；triton_ascend 3.2.1 wheel 无 PyPI 发行，**从 vllm-ascend 镜像拷出**（cp311） |
| FlagPerf | github.com/flagos-ai/FlagPerf | main（FlagRT fork） | FlagPerf | AI 硬件评测与基准测试 | 依赖以 FlagPerf 仓库自身 README/requirements 为准；不自动并入 venv 核心组合 |

### 1.1 协同开发链接（FlagRT 组织主干）

> 协作模式（同仓分支 PR，GitHub Flow）：成员 clone FlagRT 组织仓（无需 fork）→ 本地分支（`<名字>/<功能>`）→ push 组织仓同名分支 → 同仓 PR；`dev-1.0` 为当前公共开发集成分支（宽松，日常合并），`main` 为稳定主线（唯一变更来源：fork sync + dev-1.0 验收 PR）；flagos-ai 上游由组织仓 Sync fork 统一同步，成员无需各自配置

| 仓 | 上游（只读参照） | 组织仓（共同开发主干） | 主干分支 |
|---|---|---|---|
| Torch-FL（本地目录 PyTorch-Plugin-FL） | github.com/flagos-ai/Torch-FL | github.com/FlagRT/Torch-FL | main + dev-1.0 |
| FlagCX | github.com/flagos-ai/FlagCX | github.com/FlagRT/FlagCX | main + dev-1.0 |
| FlagGems | github.com/flagos-ai/FlagGems | github.com/FlagRT/FlagGems | main + dev-1.0 |
| vllm-plugin-FL | github.com/flagos-ai/vllm-plugin-FL | github.com/FlagRT/vllm-plugin-FL | main + dev-1.0 |
| FlagTree | github.com/flagos-ai/FlagTree | github.com/FlagRT/FlagTree | triton_v3.2.x（与上游同步） |
| FlagPerf | github.com/flagos-ai/FlagPerf | github.com/FlagRT/FlagPerf | main（clone 基线；协作分支按 fork 实际配置） |

### 1.2 双线说明（2026-08-22 设备层路线变更）

> 权威依据：`dev/memory/docs/FlagOS设备层路线变更指南.md`。FlagOS 栈四层中，**A/B 只在设备层不同**，上层（FlagGems 算子层、FlagCX 通信层、vllm-plugin-FL 推理插件）完全相同。

| | 路线 A（生产主线） | 路线 B（预研支线） |
|---|---|---|
| 设备层 | 各芯片厂商设备插件：昇腾 **torch_npu**、昆仑芯 **xpytorch**、真武 **PPU SDK** | **torch_fl**（= PyTorch-Plugin-FL = Torch-FL，同一组件） |
| 定位 | **FlagOS 官方发布配置**；生产交付全部走 A，覆盖所有芯片（推理与训练均适用） | 保留验证资产（昇腾 Qwen3-4B TP=1/2/4、16 卡 DDP、FlagCX 补丁、上游 PR 推进）；**不承担交付，不作为任何团队前置依赖** |
| 引用关系 | 上层 6 仓（vllm/sglang/Megatron/TE/verl/FlagScale）的"flagos"指 **FlagGems 算子优先级**（`VLLM_FL_PREFER=flagos`），与 torch_fl 设备同名不同物 | 上层 6 仓对 torch_fl **零引用**；torch_fl 仅是选定 B 路线后的内部依赖（PyTorch 单 PrivateUse1 槽位机制决定），不是全栈共同基础 |
| 当前状态 | 昇腾/昆仑芯均已实测跑通 dense 推理；训练链路只存在于 A | 昇腾 ACCELERATOR=ascend 标注 Beta；昆仑芯/真武无 CI、无端到端验证 |

**测试环境隔离原则**：路线 A 的所有测试一律在**官方发布镜像**内进行；dev 容器仅用于 B 的开发。已有教训：dev 容器的 triton 版本偏差曾导致 FlagGems GEMM 编译崩溃，一度被误判为组件缺陷——版本偏差会污染结论。

## 2. 基础镜像（按芯片/路线拆分；昇腾 3 个 tag 保留）

| 用途 | 地址 | tag | 大小 | 获取方式 |
|---|---|---|---|---|
| 昇腾开发基础镜像（harbor 源） | harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl | manual-20260807-ascend-dev | 11.2GB | 私有 harbor，需 BAAI 账号 |
| 昇腾开发基础镜像（本地重建，推荐） | flagos-dev/pytorch-plugin-fl | manual-20260807-ascend-dev-hostnet | 13.7GB | 2026-08-13 快照重建（同 commit），CANN 9.0.0 + Python 3.12 + 昇腾工具链全套 |
| 昇腾对照/工具镜像 | quay.io/ascend/vllm-ascend | v0.20.2rc1-a3 | 18GB | 公开可直接 pull；triton_ascend wheel 来源、torch_npu 生态参照、上线容器基础 |
| 昆仑芯开发镜像（B 开发/dev 容器） | flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev | 202608 | 107GB（本机已有） | triton **3.6.0**——与官方发布镜像声明的 triton 3.0.0 不一致，**注意版本偏差警示**（见 §1.2 隔离原则） |
| 昆仑芯官方发布镜像（**A 线首选测试环境**） | harbor.baai.ac.cn/flagrelease-public/qwen3.6-35b-a3b-nomtp-kunlunxin-gems_4.2.1rc0-vllm_0.13-plugin_0.1-cx_0.10.0-python_3.10.18-x86_64-driver_515.58 | 2604161518 | 92.8GB（本机已有） | A 线所有测试一律在此镜像内进行 |
| 昆仑芯新线镜像（**未拉取，待获取**） | kunlunxin001 线：gems 5.0.0 + vllm 0.20.2（路线 A 实测文档 S1.4 提到存在） | 待核实 | 未拉取 | 完整地址/tag 待核实 |

> **注**：官方 `install-stack-flagos` 技能的 vendor-mappings.md **无 kunlunxin 条目**（文档滞后，源码实际支持 klx）。

## 3. venv 运行组合（容器内，按芯片/路线拆分）

### 3.1 昇腾 B 线（预研，venv311，2026-08-17 实测验证）

python 3.11.15 + torch 2.10.0+cpu + vllm 0.20.2 + triton_ascend 3.2.1
+ flag_gems + torch_fl（安装 ACCELERATOR=ascend）+ flagcx + vllm-plugin-FL

### 3.2 昆仑芯 P800 A 线（实测可复现，2026-08-21/22 验证）

python 3.10.18 + torch 2.9.0+cu129（xpytorch，CUDA 兼容，USE_CUDA=ON）+ vllm 0.13.0（昆仑芯版，插件强依赖做平台引导）+ vllm-plugin-fl 0.1.0 + flag_gems 4.2.1rc0 @73c5aff1（editable）+ flagcx 0.10.0（klx 适配器）+ flagtree 0.6.1+xpu3.6（dev，triton 3.6.0；官方发布镜像为 triton 3.0.0）+ xtorch_ops 0.1.2640 + torch_xray 2.0.4 + xmlir 1.0.0.1 + torch_plugin 0.1.0（仅 runtime 初始化，无 torch_fl 显存池）

### 3.3 昇腾 A 线（占位，待验证）

torch_npu + vllm 0.20.2 + triton_ascend 3.2.1 + flag_gems + flagcx + vllm-plugin-FL

> 组合待另一台 910c 环境实测后补齐（具体版本号待核实）。

> `clone_all.sh` 会将 FlagPerf 源码收拢到 `/workspace/FlagPerf`；FlagPerf 的运行环境和依赖不自动安装到上述 venv 组合，使用时按其仓库说明单独准备。

## 4. 运行必需环境变量（按芯片拆分）

> 均为容器内路径约定：`/workspace` = `compose.base.yml` 相对路径 `../` 自动挂载的公共仓根。

### 4.1 昇腾（现有 5 个保留）

- `GEMS_VENDOR=ascend`
- `TRITON_ENABLE_TASKQUEUE=false`
- `FLAGCX_PATH=/workspace/FlagCX/plugin/torch`
- `DO_NOT_TRACK=1`
- `HCCL_NPU_SOCKET_PORT_RANGE=16666,16676`（多进程 HCCL 必需，compose 已注入）

### 4.2 昆仑芯 P800 A 线（实测口径）

- `VLLM_PLUGINS=fl`
- `VLLM_FL_PLATFORM=kunlunxin`
- `VLLM_FL_PREFER=flagos|vendor`
- `USE_FLAGGEMS=1`
- `GEMS_VENDOR=kunlunxin`
- `KLX_USE_AUTOTUNE=0`
- `CUDA_VISIBLE_DEVICES`（选卡）
- `DO_NOT_TRACK=1`
- `FLAGCX_PATH=/workspace/FlagCX/plugin/torch`

> **注**：vllm 0.13 昆仑芯构建必须靠 vllm-plugin-FL 做平台引导——不带插件会报 `Device string must not be empty`。

## 5. 变更记录

- 2026-08-22：**设备层路线变更（B → A 主线，B 降为预研支线）**，依据 `dev/memory/docs/FlagOS设备层路线变更指南.md`；VERSIONS.md 双线化（A 主线 + B legacy），§2/§3/§4 按芯片/路线拆分，新增昆仑芯 P800 A 线实测组合
- 2026-08-20：`clone_all.sh` 纳入 FlagPerf（FlagRT/FlagPerf，main）；同步更新仓库清单、目录约定、模板和 `.gitignore`
- 2026-08-18：上游 PyTorch-Plugin-FL 改名 **Torch-FL**（旧名 301 重定向）；组织仓/文档链接改用 Torch-FL，本地目录与容器路径沿用 PyTorch-Plugin-FL
- 2026-08-18：**WORKSPACE_HOST 机制删除**——`dev/compose.base.yml` 挂载改用相对路径 `../` 自动解析公共仓根（compose 相对路径按首个 -f 文件所在目录解析，已实测），`.env` 无需再填任何宿主路径
- 2026-08-18：**include 改为 -f 多文件合并**——compose v2 实测 include 不允许同名 service 覆盖（报 conflicts）；子方向启动命令改为 `docker compose -f ../compose.base.yml -f docker-compose.yml up -d`（后文件覆盖前文件）
