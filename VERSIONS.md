# VERSIONS.md —— 物资起始清单与版本事实源

> 用途：**新成员起步 + 组级版本事实源**——只记仓库、设备层构成与公共环境变量；各芯片可复现的镜像 / venv 组合以对应子方向 README 为准。
> 设备层 = 各芯片厂商设备插件（昇腾 `torch_npu`、昆仑芯厂商 CUDA 兼容 torch）；其上 FlagGems（算子）/ FlagCX（通信）/ vllm-plugin-FL（推理插件）/ vLLM。

## 0. 物资起始清单（一屏）

要准备的东西：

| # | 物资 | 数量 | 明细见 |
|---|---|---|---|
| 1 | 组织主干仓（FlagRT 组织下，成员直接 clone） | 7 个（6 子库 + runtime-team） | §1.1 |
| 2 | GitHub 账号（SSH key + 组织成员 write 权限） | 1 个 | §1.1 |
| 3 | 基础镜像 | 见 §2 | §2 |
| 4 | 运行组合（容器内，按芯片） | 见 §3 + 子方向 README | §3 |
| 5 | 公共环境变量（昇腾 910C） | `compose.base.yml` 注入 | §4 |
| 6 | 容器启动参数（子方向专属） | 见对应的 `dev/<子方向>/README.md` | 子方向目录 |

## 1. 开源代码仓（6 个；原 5 个 fork 基线于 2026-08-17 锁定）

> 原 5 个仓的基线 commit 与上游一致（见 §1 表格）；FlagPerf 按 FlagRT fork 的 `main` 对齐。成员直接 clone FlagRT 组织仓，无需 fork。
> **注**：PyTorch-Plugin-FL 上游已改名 **Torch-FL**（2026-08-18 确认，旧名 301 重定向）；本地目录/容器路径沿用 PyTorch-Plugin-FL（clone 时指定目录名）。

| 仓（fork 名） | 上游链接 | 基线 | 版本标识 | 角色 | 依赖（源码核实） |
|---|---|---|---|---|---|
| Torch-FL（本地目录 PyTorch-Plugin-FL） | github.com/flagos-ai/Torch-FL | main@caefaae | torch_fl 0.1.0 | 设备接入层参照实现（`csrc/runtime/`：设备句柄 / 显存池 allocator / Stream·Event 抽象） | torch 2.10 固定（setup.py TORCH_PIN `>=2.10,<2.11`）；Python≥3.8 |
| FlagCX | github.com/flagos-ai/FlagCX | main@0a747f6 | flagcx 0.13.0 | 多卡通信 + KV 传输 | torch（构建期自动检测，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`）；源码构建需 make + git submodule（plugin/torch 为 torch 插件） |
| FlagGems | github.com/flagos-ai/FlagGems | master@c22f8eb | flag_gems 0.0.0（editable） | 多芯片 triton 算子库 | packaging≥26.0、PyYAML==6.0.1、sqlalchemy==2.0.48、numpy；昇腾组合 torch==2.10.0+cpu + torch_npu（厂商设备插件） |
| vllm-plugin-FL | github.com/flagos-ai/vllm-plugin-FL | main@db9afd6 | vllm-plugin-fl 0.0.0（editable） | 推理插件（KV Cache 挂载点、Platform 层） | 运行时 pyyaml；配套 vllm==0.20.2；Python 3.10~3.13；构建需 torch≥2.7.1 |
| FlagTree | github.com/flagos-ai/FlagTree | 分支 triton_v3.2.x | triton_ascend 3.2.1（import 报 3.2.0，已知差异） | 编译层（triton kernel 编译） | 构建 setuptools/wheel/cmake≥3.18/ninja≥1.11.1；triton_ascend 3.2.1 wheel 无 PyPI 发行，**从 vllm-ascend 镜像拷出**（cp311） |
| FlagPerf | github.com/flagos-ai/FlagPerf | main（FlagRT fork） | FlagPerf | AI 硬件评测与基准测试 | 依赖以 FlagPerf 仓库自身 README/requirements 为准；不自动并入 §3 运行组合；性能评测的宿主入口、运行环境和验证边界见 `dev/performance/README.md` |

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

## 2. 基础镜像

> 昇腾 910C 用途分两类：**开发 / 构建**（在容器内装 torch_npu + 编译 FlagCX / FlagGems 等）与**推理结论性测试**（直接用华为官方昇腾 vLLM 镜像自带栈）。昆仑芯 P800 一律在厂商官方发布镜像内做，tag 见 `dev/memory/README.md`。

| 用途 | 地址 | tag | 大小 | 获取方式 |
|---|---|---|---|---|
| 昇腾 910C 开发 / 构建（harbor 源） | harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl | manual-20260807-ascend-dev | 11.2GB | 私有 harbor，需 BAAI 账号 |
| 昇腾 910C 开发 / 构建（本地重建，`compose.base.yml` 默认） | flagos-dev/pytorch-plugin-fl | manual-20260807-ascend-dev-hostnet | 11.2GB | 2026-08-13 快照重建（同 commit）；CANN 9.0.0 + Python 3.12 + 昇腾工具链全套（通用昇腾 dev 底座，A 线在容器内装 torch_npu） |
| 昇腾 910C 推理结论性测试 | quay.io/ascend/vllm-ascend | v0.20.2rc1-a3 | 18GB | 华为官方，公开可直接 pull（本机现有 `nightly-main-a3`，tag 以实拉为准）；自带 torch_npu + vllm 0.20.2 + vllm_ascend；triton_ascend wheel 来源 |

> **本镜像 = 昇腾工具链底座**（非设备层交付物）：预装 torch_fl 0.1.0，但 A 线（厂商 torch_npu）不使用它——A 线在容器内自建 torch_npu venv，结论性测试改用官方镜像（见各子方向 README）。
> 镜像名 `pytorch-plugin-fl` 与 `compose.base.yml` 的 `/data_lib/PyTorch-Plugin-FL` 挂载均为 B 线（torch_fl，已冻结归档）血统残留，保留无害，勿据此误判底座绑定 B 线。
> harbor 源与本地重建副本为**同一镜像**（IMAGE ID 一致，均 11.2GB），换用 harbor 地址不改变任何栈行为。

## 3. 运行组合（容器内按芯片组装；权威 pin 在子方向 README）

**昇腾 910C**
- 推理结论性测试：`quay.io/ascend/vllm-ascend` 镜像自带 python 3.11 + torch 2.10.0+cpu + **torch_npu 2.10.0** + vllm 0.20.2 + vllm_ascend（device-context P0–P3 在此验证）
- 训练 / device-context / conformance / 通信：容器内装 **torch_npu 2.10.0** + FlagCX（`plugin/torch`）+ FlagGems（± vllm-plugin-FL）；具体镜像与 venv 组合以 `dev/device-context/README.md`、`dev/communication/README.md` 为准

**昆仑芯 P800**（厂商官方发布镜像内）
- python 3.10 + 厂商 CUDA 兼容 torch 2.9 + vllm 0.13 + **vllm-plugin-FL** + FlagGems + FlagCX + triton 3.0.0；口径见 `dev/memory/README.md`

> `clone_all.sh` 会将 FlagPerf 源码收拢到 `/workspace/FlagPerf`；FlagPerf 的运行环境和依赖不自动安装到上述任一组合，使用时按其仓库说明单独准备。

## 4. 公共环境变量（昇腾 910C，由 `dev/compose.base.yml` 注入）

> 均为容器内路径约定：`/workspace` = `compose.base.yml` 相对路径 `../` 自动挂载的公共仓根。其他芯片的专属变量（如昆仑芯 `GEMS_VENDOR=kunlunxin` / `VLLM_PLUGINS=fl`）见对应子方向 compose。

- `GEMS_VENDOR=ascend`
- `TRITON_ENABLE_TASKQUEUE=false`
- `FLAGCX_PATH=/workspace/FlagCX/plugin/torch`
- `DO_NOT_TRACK=1`
- `HCCL_NPU_SOCKET_PORT_RANGE=16666,16676`（多进程 HCCL 必需，compose 已注入）

## 5. 变更记录

- 2026-09-03：§2 基础镜像加注——明确该镜像为「昇腾工具链底座」而非设备层交付物；`pytorch-plugin-fl` 名与 `/data_lib` 挂载为 B 线（已冻结）血统残留、保留无害；harbor 源与本地副本同一 IMAGE ID；大小订正 13.7GB→11.2GB。同步 `compose.base.yml` 顶部与 `/data_lib` 挂载注释
- 2026-09-03：§2/§3 按芯片口径整理——设备层以各芯片厂商设备插件为准（昇腾 torch_npu、昆仑芯厂商 torch），其上 FlagGems / FlagCX / vllm-plugin-FL / vLLM；各芯片镜像 / venv / 环境变量的权威 pin 归对应子方向 README
- 2026-09-02：新增 performance 性能评测协调入口；FlagPerf 正式代码仍由独立仓库维护，宿主入口、运行环境与验证边界见 `dev/performance/README.md`
- 2026-08-27：**公共配置对齐（组级）**——`DO_NOT_TRACK=1` 补注入 `compose.base.yml`（原"待补注入"）；子方向 compose 必须声明独立 `name:`（默认 project 取首个 -f 文件目录名 `dev`，会跨方向重建 `runtime-dev`）；`!override` 覆盖设备约束与昇腾 vLLM 通用坑写入根 README
- 2026-08-20：`clone_all.sh` 纳入 FlagPerf（FlagRT/FlagPerf，main）；同步更新仓库清单、目录约定、模板和 `.gitignore`
- 2026-08-18：上游 PyTorch-Plugin-FL 改名 **Torch-FL**（旧名 301 重定向）；组织仓/文档链接改用 Torch-FL，本地目录与容器路径沿用 PyTorch-Plugin-FL
- 2026-08-18：**WORKSPACE_HOST 机制删除**——`dev/compose.base.yml` 挂载改用相对路径 `../` 自动解析公共仓根（compose 相对路径按首个 -f 文件所在目录解析，已实测），`.env` 无需再填任何宿主路径
- 2026-08-18：**include 改为 -f 多文件合并**——compose v2 实测 include 不允许同名 service 覆盖（报 conflicts）；子方向启动命令改为 `docker compose -f ../compose.base.yml -f docker-compose.yml up -d`（后文件覆盖前文件）
