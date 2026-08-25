# 分布式运行时（distribution_ljy）项目

> **状态：⬜ 待启动** ｜ 本文档 = 任务看板入口，供运行时组全员维护
> 对齐起点速查：……

## 目标（一句话）

……

## 现状（快照）

- ……

## 目录约定（本子方向，位于 dev/distribution_ljy/ 下）

```
dev/distribution_ljy/
├── README.md           # 本文档（看板）
├── docker-compose.yml  # 容器配置（-f ../compose.base.yml 合并公共配置）
├── .env.example        # 环境变量模板（cp 成 .env 按需调整）
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
cd dev/distribution_ljy
cp .env.example .env    # 按需调整专属开关（默认值即可直接启动）
docker compose -f ../compose.base.yml -f docker-compose.yml up -d
docker ps | grep flagos-distribution_ljy-dev-910c    # 确认 Up
```

## 常用命令（环境速查）

```bash
# 进开发容器
docker exec -it flagos-distribution_ljy-dev-910c bash
# 容器内验证挂载（应看到 6 个子库 + dev/ 等公共仓内容）
docker exec -it flagos-distribution_ljy-dev-910c bash -c "ls /workspace"
```

## 工作原则

- 遵循"不预实现"原则：先有真实问题数据，再动手优化
- 公共红线（不改宿主配置/驱动、多卡前 npu-smi 确认、DrvMng 上限≈3）见主 README「红线」节
