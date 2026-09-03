# Batch（batch_yxy）项目

> **状态：⬜ 待启动** ｜ 本文档 = 任务看板入口，供运行时组全员维护
> 对齐起点速查：……

## 目标（一句话）

……

## 现状（快照）

- ……

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

代码改造主战场不在本目录：**<子库路径>**（如 torch_fl `csrc/runtime/allocator/`，按需填写）

## 任务看板

| # | 任务 | 负责人 | 状态 | 依赖 | 出口标准 |
|---|------|--------|------|------|----------|
| 1 | 示例任务 | TBD | ⬜ | — | 完成标准 |

> 状态图例：⬜ 待认领 ｜ 🔄 进行中 ｜ ✅ 完成 ｜ ❌ 取消

## 启动容器（宿主侧）

```bash
cd dev/batch_yxy
cp .env.example .env    # 按需调整专属开关（默认值即可直接启动）
docker compose --env-file .env -f ../compose.base.yml -f docker-compose.yml up -d runtime-dev
docker ps | grep flagos-batch-yxy-dev-910c    # 确认 Up
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
