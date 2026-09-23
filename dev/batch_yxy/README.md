# Batch（batch_yxy）项目

> **状态：🔄 进行中** ｜ 本文档 = 任务看板入口，供运行时组全员维护
> 对齐起点速查：vLLM 0.20.2（vllm-ascend v0.20.2rc1-a3 镜像）· V1 引擎 · 910C 单卡

## 目标（一句话）

Qwen3-Embedding-0.6B 动态组批与长度感知分桶：可外部调用的 `BatchCoordinator` 决策核心 + vLLM 上层适配（`LLM.embed` 外层分组，不替换内部调度），A/B/C 对照验证。

## 现状（快照）

- 2026-09-23：核心模块 + 53 个单元测试 + fake/NPU 双执行器 bench 完成；接入路径与控制边界见 `docs/note_动态组批接入说明.md`
- 2026-09-23：**发现 vllm-ascend 0.20.2rc1 pooling 跨调用非确定缺陷**（复现探针 `probes/qwen3_embed_nondeterminism_probe.py`），阻塞 NPU 数值一致性验收，待独立上报

## 目录约定（本子方向，位于 dev/batch_yxy/ 下）

```
dev/batch_yxy/
├── README.md           # 本文档（看板）
├── docker-compose.yml  # 容器配置（-f ../compose.base.yml 合并公共配置）
├── .env.example        # 环境变量模板（cp 成 .env 按需调整）
├── requirements.txt    # Python 依赖（setup_env.sh 安装）
├── scripts/            # 环境初始化/建库/入库等脚本
├── docs/               # 调研笔记、方案摘录、执行记录（按需建）
├── probes/             # 探针/画像脚本（按需建）
└── benchmarks/         # A/B 对比与负载脚本（按需建）
```

代码改造主战场不在本目录：本任务零侵入 vLLM（组批核心独立成包 `dynamic_batching/`，仅上层调用 `LLM.embed`）

## 任务看板

| # | 任务 | 负责人 | 状态 | 依赖 | 出口标准 |
|---|------|--------|------|------|----------|
| 1 | T0 环境与基线核对（vLLM 0.20.2/V1/embed 入口/tokenizer/本地权重） | xianyiyuan | ✅ | — | `docs/note_动态组批接入说明.md` §1 |
| 2 | T1 接口与配置（types/errors/config） | xianyiyuan | ✅ | 1 | 边界输入返回确定错误 |
| 3 | T2 基础动态组批（预算/定时封口/flush/close） | xianyiyuan | ✅ | 2 | 单请求/满批/未满批/不重复发出测试 |
| 4 | T3 长度分桶（固定桶/相邻合并/Padding 约束） | xianyiyuan | ✅ | 3 | 桶边界/低流量不饥饿测试 |
| 5 | T4 vLLM 适配（VllmEmbedExecutor + A/B 开关） | xianyiyuan | ✅ | 4 | 结果按 ID 无错位映射 |
| 6 | T5 对照与交付（bench A/B/C + 接入说明） | xianyiyuan | ✅ | 5 | 结果在 `benchmarks/results/`；NPU 数值验收被平台缺陷阻塞（见接入说明 §7.1，组批检测关键输入） |

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

## 工作原则

- 遵循"不预实现"原则：先有真实问题数据，再动手优化
- 公共红线（不改宿主配置/驱动、多卡前 npu-smi 确认、DrvMng 上限≈3）见主 README「红线」节
