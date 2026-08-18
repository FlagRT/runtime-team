# VERSIONS.md —— 物资起始清单与版本事实源

> 用途：**凭此清单可准确复现开发环境**

## 0. 物资起始清单（一屏）

要准备的东西：

| # | 物资 | 数量 | 明细见 |
|---|---|---|---|
| 1 | 组织主干仓（FlagRT 组织下，成员直接 clone） | 6 个（5 子库 + runtime-team） | §1.1 |
| 2 | GitHub 账号（SSH key + 组织成员 write 权限） | 1 个 | §1.1 |
| 3 | 基础镜像 | 2 个（3 个 tag） | §2 |
| 4 | venv311 运行组合（容器内） | 1 套 | §3 |
| 5 | 运行环境变量 | 4 个 | §4 |
| 6 | 容器启动参数（子方向专属） | 见 memory 子方向 `runtime-memory/README.md` | 子方向目录 |

## 1. 开源代码仓（5 个，fork 基线 2026-08-17 锁定）

> 基线 commit 与上游一致（见 §1 表格）；本地已验证修改随组织仓推送（成员直接 clone FlagRT 组织仓，无需 fork）。
> **注**：PyTorch-Plugin-FL 上游已改名 **Torch-FL**（2026-08-18 确认，旧名 301 重定向）；本地目录/容器路径沿用 PyTorch-Plugin-FL（clone 时指定目录名）。

| 仓（fork 名） | 上游链接 | 基线 | 版本标识 | 角色 | 依赖（源码核实） |
|---|---|---|---|---|---|
| Torch-FL（本地目录 PyTorch-Plugin-FL） | github.com/flagos-ai/Torch-FL | main@caefaae | torch_fl 0.1.0 | 设备接入 + 显存池主战场（csrc/runtime/allocator/） | torch 2.10 固定（setup.py TORCH_PIN `>=2.10,<2.11`）；Python≥3.8；triton/flag_gems 由平台注入（不装 PyPI triton） |
| FlagCX | github.com/flagos-ai/FlagCX | main@0a747f6 | flagcx 0.13.0 | 多卡通信 + KV 传输 | torch（构建期自动检测，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`）；源码构建需 make + git submodule（plugin/torch 为 torch 插件） |
| FlagGems | github.com/flagos-ai/FlagGems | master@c22f8eb | flag_gems 0.0.0（editable） | 昇腾 triton 算子库 | packaging≥26.0、PyYAML==6.0.1、sqlalchemy==2.0.48、numpy；ascend 组合 torch==2.10.0+cpu（官方 extra 含 torch-npu，**本团队不用 torch-npu**，以 torch_fl 替代） |
| vllm-plugin-FL | github.com/flagos-ai/vllm-plugin-FL | main@db9afd6 | vllm-plugin-fl 0.0.0（editable） | 推理插件（KV Cache 挂载点、Platform 层） | 运行时 pyyaml；配套 vllm==0.20.2；Python 3.10~3.13；构建需 torch≥2.7.1 |
| FlagTree | github.com/flagos-ai/FlagTree | 分支 triton_v3.2.x | triton_ascend 3.2.1（import 报 3.2.0，已知差异） | 编译层（triton kernel 编译） | 构建 setuptools/wheel/cmake≥3.18/ninja≥1.11.1；triton_ascend 3.2.1 wheel 无 PyPI 发行，**从 vllm-ascend 镜像拷出**（cp311） |

### 1.1 协同开发链接（FlagRT 组织主干）

> 协作模式（同仓分支 PR，GitHub Flow）：成员 clone FlagRT 组织仓（无需 fork）→ 本地分支（`<名字>/<功能>`）→ push 组织仓同名分支 → 同仓 PR；`dev-1.0` 为当前公共开发集成分支（宽松，日常合并），`main` 为稳定主线（唯一变更来源：fork sync + dev-1.0 验收 PR）；flagos-ai 上游由组织仓 Sync fork 统一同步，成员无需各自配置

| 仓 | 上游（只读参照） | 组织仓（共同开发主干） | 主干分支 |
|---|---|---|---|
| Torch-FL（本地目录 PyTorch-Plugin-FL） | github.com/flagos-ai/Torch-FL | github.com/FlagRT/Torch-FL | main + dev-1.0 |
| FlagCX | github.com/flagos-ai/FlagCX | github.com/FlagRT/FlagCX | main + dev-1.0 |
| FlagGems | github.com/flagos-ai/FlagGems | github.com/FlagRT/FlagGems | main + dev-1.0 |
| vllm-plugin-FL | github.com/flagos-ai/vllm-plugin-FL | github.com/FlagRT/vllm-plugin-FL | main + dev-1.0 |
| FlagTree | github.com/flagos-ai/FlagTree | github.com/FlagRT/FlagTree | triton_v3.2.x（与上游同步） |

## 2. 基础镜像（2 个，3 个 tag）

| 用途 | 地址 | tag | 大小 | 获取方式 |
|---|---|---|---|---|
| 开发基础镜像（harbor 源） | harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl | manual-20260807-ascend-dev | 11.2GB | 私有 harbor，需 BAAI 账号 |
| 开发基础镜像（本地重建，推荐） | flagos-dev/pytorch-plugin-fl | manual-20260807-ascend-dev-hostnet | 13.7GB | 2026-08-13 快照重建（同 commit），CANN 9.0.0 + Python 3.12 + 昇腾工具链全套 |
| 对照/工具镜像 | quay.io/ascend/vllm-ascend | v0.20.2rc1-a3 | 18GB | 公开可直接 pull；triton_ascend wheel 来源、torch_npu 生态参照、上线容器基础 |

## 3. venv311 运行组合（容器内，2026-08-17 实测验证）

python 3.11.15 + torch 2.10.0+cpu + vllm 0.20.2 + triton_ascend 3.2.1
+ flag_gems + torch_fl（安装 ACCELERATOR=ascend）+ flagcx + vllm-plugin-FL

## 4. 运行必需环境变量

- `GEMS_VENDOR=ascend`
- `TRITON_ENABLE_TASKQUEUE=false`
- `FLAGCX_PATH=/workspace/FlagCX/plugin/torch`
- `DO_NOT_TRACK=1`

## 5. 变更记录

- 2026-08-18：上游 PyTorch-Plugin-FL 改名 **Torch-FL**（旧名 301 重定向）；组织仓/文档链接改用 Torch-FL，本地目录与容器路径沿用 PyTorch-Plugin-FL
