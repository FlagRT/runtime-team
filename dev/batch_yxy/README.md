# FlagScheduler（开发目录：batch_yxy）

节点级统一推理调度引擎与 CLI。当前实现框架/Conda 环境发现、vLLM scheduler 定位与版本锁定启动计划、常驻请求队列、本机 API、可插拔执行后端及十五类模型契约。优先 NVIDIA + vLLM，后续通过同一应用 API 接入 TUI。

```bash
cd runtime-team/dev/batch_yxy  # 从工作区根目录进入
python -B -m flagscheduler --help
python -B -m flagscheduler discover
python -B -m flagscheduler scheduler inspect --python /path/to/conda/envs/vllm/bin/python
python -B -m flagscheduler scheduler list --python /path/to/conda/envs/vllm/bin/python
python -B -m flagscheduler run --config examples/flagscheduler.mock.json --input examples/flagscheduler.requests.jsonl
python -B -m flagscheduler serve --config examples/flagscheduler.mock.json
# 另一终端
python -B -m flagscheduler status
python -B -m flagscheduler submit --request examples/flagscheduler.request.json
```

源码运行不需要安装；`python -B -m flagscheduler` 与安装后的 `flagscheduler` 使用同一入口。Mock 与控制面只使用 Python 标准库，真实 vLLM 执行需要在对应环境运行。

`scheduler list` 显示脚本文件和 `(Default)` / `(Currently)`；初始原生脚本同时具有两个标记。用 `scheduler select flagscheduler --python /path/to/python` 选择项目脚本，用 `scheduler select default --python /path/to/python` 选回原生脚本。原文件始终保留，并在本目录 `.state/schedulers/originals/` 留存只读校验快照。选择用于后续新启动，不修改已经运行的进程。

- [使用指南](docs/flagscheduler_使用指南.md)：命令、启动与回退、NVIDIA 验收。
- [架构与开发路线](docs/flagscheduler_架构与开发路线.md)：边界、生命周期、模型矩阵、维护规则。
- [适配器开发](docs/flagscheduler_适配器开发.md)：Backend / Planner 插件及 TUI 接口。
- [验证记录](docs/flagscheduler_验证记录.md)：171 项测试、wheel 构建、Default/Currently 选择及真实 vLLM 插件接口检查；尚未进行 GPU 模型验收。
- 当前节点只读检查：Conda `vllm` 为 **0.21.0**，具有 V1 同步/异步 scheduler；NVIDIA 设备清单为 4 张 RTX A6000。设备清单不代表空闲或已完成模型验收。
- 十五类模型均有契约与 mock 链路，真实执行目前提供 `vllm-embed` 接入代码。其他框架的发现不代表已实现执行适配器。内部 scheduler 插件目前继承原生行为，用作后续算法开发入口。
- 全部文件修改必须限定在 `batch_yxy/` 内。外部环境与框架保持只读，部署/安装命令仅供在明确指定的目标环境中使用。

下列内容保留原动态组批与昇腾分支的历史看板；其测试数、环境和验收状态不代表 FlagScheduler 新节点运行时的设备验证结果。

## 原 Batch 项目看板

> **状态：🔄 进行中** ｜ 本文档 = 任务看板入口，供运行时组全员维护
> 对齐起点速查：vLLM 0.20.2（vllm-ascend v0.20.2rc1-a3 镜像）· V1 引擎 · 910C 单卡

## 目标（一句话）

Qwen3-Embedding-0.6B 动态组批与长度感知分桶：可外部调用的 `BatchCoordinator` 决策核心 + vLLM 上层适配（`LLM.embed` 外层分组，不替换内部调度），A/B/C 对照验证。
在此之上构建与模型、推理框架、芯片解耦的调度引擎 `sched_engine/`（四层架构：调度核心/模型画像/框架适配器/设备画像）。

## 现状（快照）

- 2026-09-23：核心模块 + 53 个单元测试 + fake/NPU 双执行器 bench 完成；接入路径与控制边界见 `docs/note_动态组批接入说明.md`
- 2026-09-23：**发现 vllm-ascend 0.20.2rc1 pooling 跨调用非确定缺陷**（复现探针 `probes/qwen3_embed_nondeterminism_probe.py`），阻塞 NPU 数值一致性验收，待独立上报
- 2026-10-14：调度引擎 `sched_engine/` 落地（四层架构，15 模型/6 芯片画像注册表，优先级队列+恰好一次结果回填+原路径开关）；`BatchCoordinator` 向后兼容扩展 priority；测试 117 全绿（容器内）。**910C 真机联调待有空闲卡后执行**（`examples/sched_engine_910c_vllm.py`）；架构与矩阵见 `docs/note_调度引擎四层架构说明.md`

## 目录约定（本子方向，位于 dev/batch_yxy/ 下）

```
dev/batch_yxy/
├── README.md           # 本文档（看板）
├── docker-compose.yml  # 容器配置（-f ../compose.base.yml 合并公共配置）
├── .env.example        # 环境变量模板（cp 成 .env 按需调整）
├── requirements.txt    # Python 依赖（setup_env.sh 安装）
├── dynamic_batching/   # 组批决策核心 + 执行器（vLLM 零侵入）
├── sched_engine/       # 调度引擎（四层：core/models/frameworks/devices）
├── scripts/            # 环境初始化/建库/入库等脚本
├── docs/               # 调研笔记、方案摘录、执行记录（按需建）
├── probes/             # 探针/画像脚本（按需建）
└── benchmarks/         # A/B 对比与负载脚本（按需建）
```

代码改造主战场不在本目录：本任务零侵入 vLLM（组批核心独立成包 `dynamic_batching/` + `sched_engine/`，仅上层调用 `LLM.embed`）

## 任务看板

| # | 任务 | 负责人 | 状态 | 依赖 | 出口标准 |
|---|------|--------|------|------|----------|
| 1 | T0 环境与基线核对（vLLM 0.20.2/V1/embed 入口/tokenizer/本地权重） | xianyiyuan | ✅ | — | `docs/note_动态组批接入说明.md` §1 |
| 2 | T1 接口与配置（types/errors/config） | xianyiyuan | ✅ | 1 | 边界输入返回确定错误 |
| 3 | T2 基础动态组批（预算/定时封口/flush/close） | xianyiyuan | ✅ | 2 | 单请求/满批/未满批/不重复发出测试 |
| 4 | T3 长度分桶（固定桶/相邻合并/Padding 约束） | xianyiyuan | ✅ | 3 | 桶边界/低流量不饥饿测试 |
| 5 | T4 vLLM 适配（VllmEmbedExecutor + A/B 开关） | xianyiyuan | ✅ | 4 | 结果按 ID 无错位映射 |
| 6 | T5 对照与交付（bench A/B/C + 接入说明） | xianyiyuan | ✅ | 5 | 结果在 `benchmarks/results/`；NPU 数值验收被平台缺陷阻塞（见接入说明 §7.1，组批检测关键输入） |
| 7 | T6 调度引擎四层架构（队列/优先级/统一组批/结果回填/路径开关；15 模型与 6 芯片画像契约） | xianyiyuan | 🔄 | 6 | 代码+117 测试完成；**910C 真机 32 请求联调待有空闲卡**（`examples/sched_engine_910c_vllm.py`），口径见 `docs/note_调度引擎四层架构说明.md` §8 |

> 状态图例：⬜ 待认领 ｜ 🔄 进行中 ｜ ✅ 完成 ｜ ❌ 取消

## 启动容器（宿主侧）

```bash
cd dev/batch_yxy
cp .env.example .env    # 按需调整专属开关（默认值即可直接启动）
docker compose --env-file .env -f ../compose.base.yml -f docker-compose.yml up -d runtime-dev
docker ps | grep flagos-batch-yxy-dev-910c    # 确认 Up
```

本机 docker 20.10.8 无 compose v2 插件（`!override` 也需 compose ≥2.24），可用等价脚本启动（单卡 davinci0）：

```bash
npu-smi info    # 先确认 davinci0 空闲
CONFIRM_DEVICE0_IDLE=yes bash dev/batch_yxy/scripts/start_container.sh
```

## 常用命令（环境速查）

```bash
# 进开发容器
docker exec -it flagos-batch-yxy-dev-910c bash
# 容器内验证挂载（应看到 6 个子库 + dev/ 等公共仓内容）
docker exec -it flagos-batch-yxy-dev-910c bash -c "ls /workspace"
```

## 容器内开发环境（git + venv + opencode）

首次进容器（或容器重建后）执行初始化脚本，装 ssh-client、配 git 身份、建 venv、装 opencode：

```bash
docker exec -it flagos-batch-yxy-dev-910c bash
bash /workspace/dev/batch_yxy/scripts/setup_env.sh
```

之后日常使用：

```bash
# 1. 激活 venv（在 /workspace 挂载盘，跨容器重建保留）
source /workspace/dev/batch_yxy/.venv/bin/activate

# 2. git（key 通过 compose 挂载宿主 ~/.ssh/id_rsa，身份由脚本配置）
cd /workspace
git status            # runtime-team 仓库，分支 xianyiyuan/batch_yxy

# 3. opencode（已加入 /root/.bashrc PATH，重新进容器或 source ~/.bashrc 后可直接用）
opencode
```

要点：
- venv / 代码都在 `/workspace` 挂载盘，容器重建后仍在；ssh-client / opencode 装在容器内，重建后重跑 `setup_env.sh` 即可
- 环境可复现：`scripts/setup_env.sh` + `requirements.txt` 均已入 git，`.venv/` 已 ignore
- git 身份默认 `YoannFang`，可用 `GIT_USER_NAME` / `GIT_USER_EMAIL` 环境变量覆盖

## 运行测试（容器内）

```bash
docker exec flagos-batch-yxy-dev-910c bash -c \
  "source /workspace/dev/batch_yxy/.venv/bin/activate && cd /workspace/dev/batch_yxy && python -m pytest tests/ -q"
# 2026-10-14：117 passed（dynamic_batching 53 + sched_engine 64）
```

## 工作原则

- 遵循"不预实现"原则：先有真实问题数据，再动手优化
- 公共红线（不改宿主配置/驱动、多卡前 npu-smi 确认、DrvMng 上限≈3）见主 README「红线」节
