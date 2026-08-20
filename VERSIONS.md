# VERSIONS.md —— 物资起始清单与版本事实源

> 用途：**凭此清单可准确复现开发环境**

## 0. 物资起始清单（一屏）

| # | 物资 | 数量 | 明细见 |
|---|---|---|---|
| 1 | 组织主干仓（FlagRT 组织下，成员直接 clone） | 6 个（5 子库 + runtime-team） | §1 |
| 2 | GitHub 账号（SSH key + 组织成员 write 权限） | 1 个 | §1 |
| 3 | 基础镜像 | 2 个（3 个 tag） | §2 |
| 4 | venv311 运行组合（容器内） | 1 套 | §3 |
| 5 | 运行环境变量 | 4 个 | §4 |
| 6 | 容器启动参数（子方向专属） | 见 memory 子方向 `dev/memory/README.md` | 子方向目录 |

## 1. 开源代码仓（5 个）

> 组织仓 = `github.com/FlagRT/<仓名>`，成员直接 clone，无需 fork；上游 flagos-ai 只读参照。
> **基线 = fork 仓当前同步状态**（2026-08-20 标定；Sync fork 后需更新下表）。协作模式（main 只同步 / dev-1.0 公共开发 / 按需合上游）见主 README「分支模型（fork 三线）」。
> **注**：PyTorch-Plugin-FL 上游已改名 **Torch-FL**（旧名 301 重定向）；本地目录/容器路径沿用 PyTorch-Plugin-FL（clone 时指定目录名）。

| 仓（fork 名） | 上游（只读） | 基线分支@commit | 版本标识 | 角色 | 依赖（源码核实） |
|---|---|---|---|---|---|
| Torch-FL（本地目录 PyTorch-Plugin-FL） | github.com/flagos-ai/Torch-FL | main@af50297 | torch_fl 0.1.0 | 设备接入 + 显存池主战场（csrc/runtime/allocator/） | torch 2.10 固定（setup.py TORCH_PIN `>=2.10,<2.11`）；Python≥3.8；triton/flag_gems 由平台注入（不装 PyPI triton） |
| FlagCX | github.com/flagos-ai/FlagCX | main@a46d0d8 | flagcx 0.13.0 | 多卡通信 + KV 传输 | torch（构建期自动检测，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`）；源码构建需 make + git submodule（plugin/torch 为 torch 插件） |
| FlagGems | github.com/flagos-ai/FlagGems | master@f7ae8e6（fork 默认分支非 main） | flag_gems 0.0.0（editable） | 昇腾 triton 算子库 | packaging≥26.0、PyYAML==6.0.1、sqlalchemy==2.0.48、numpy；ascend 组合 torch==2.10.0+cpu（官方 extra 含 torch-npu，**本团队不用 torch-npu**，以 torch_fl 替代） |
| vllm-plugin-FL | github.com/flagos-ai/vllm-plugin-FL | main@885aaef | vllm-plugin-fl 0.0.0（editable） | 推理插件（KV Cache 挂载点、Platform 层） | 运行时 pyyaml；配套 vllm==0.20.2；Python 3.10~3.13；构建需 torch≥2.7.1 |
| FlagTree | github.com/flagos-ai/FlagTree | **triton_v3.2.x**@343e4f1（例外：基准非 main） | triton_ascend 3.2.1（import 报 3.2.0，已知差异） | 编译层（triton kernel 编译） | 构建 setuptools/wheel/cmake≥3.18/ninja≥1.11.1；triton_ascend 3.2.1 wheel 无 PyPI 发行，**从 vllm-ascend 镜像拷出**（cp311） |

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

> 均为容器内路径约定（`/workspace` = 公共仓根挂载点）；前 3 个 + HCCL 端口由 `dev/compose.base.yml` 注入，`DO_NOT_TRACK=1` 需容器内 export（待 compose 补注入）。

- `GEMS_VENDOR=ascend`
- `TRITON_ENABLE_TASKQUEUE=false`
- `FLAGCX_PATH=/workspace/FlagCX/plugin/torch`
- `DO_NOT_TRACK=1`
- `HCCL_NPU_SOCKET_PORT_RANGE=16666,16676`（多进程 HCCL 必需，compose 已注入）

## 5. 变更记录

- 2026-08-20：基线改为对齐 fork 仓 origin/main（原上游 commit 锁定作废；Sync fork 后更新 §1）
- 2026-08-18：上游 PyTorch-Plugin-FL 改名 **Torch-FL**（旧名 301 重定向）
- 2026-08-18：挂载/启动机制定稿——compose.base.yml 相对路径 `../` 解析公共仓根 + `-f` 多文件合并（WORKSPACE_HOST / include 机制废弃）
