# __NAME_CN__（__NAME__）项目

> **状态：⬜ 待启动** ｜ 本文档 = 任务看板入口，供运行时组全员维护
> 对齐起点速查：……

## 目标（一句话）

……

## 现状（快照）

- ……

## 目录约定（本子方向，位于 dev/__NAME__/ 下）

```
dev/__NAME__/
├── README.md           # 本文档（看板）
├── docker-compose.yml  # 运行环境：x-flagrt 声明 + extends 复用镜像 demo，只写本方向差异
├── env.baseline.yaml   # 运行环境基线，由 baseline.py export 生成，不手工编辑
├── .env.example        # 环境变量模板（cp 成 .env 按需调整）
├── docs/               # 调研笔记、方案摘录、执行记录（按需建）
├── probes/             # 探针/画像脚本（按需建）
└── benchmarks/         # A/B 对比与负载脚本（按需建）
```

本目录即一个运行环境目录，规范见 [`dev/ENV-SPEC.md`](../ENV-SPEC.md) §3.2。需要不同配置组合时，
在本目录下另建子目录（各含 `docker-compose.yml` 与 `env.baseline.yaml`），相对路径相应加一级 `../`。

代码改造主战场不在本目录：**<子库路径>**（如 torch_fl `csrc/runtime/allocator/`，按需填写）

## 任务看板

| # | 任务 | 负责人 | 状态 | 依赖 | 出口标准 |
|---|------|--------|------|------|----------|
| 1 | 示例任务 | TBD | ⬜ | — | 完成标准 |

> 状态图例：⬜ 待认领 ｜ 🔄 进行中 ｜ ✅ 完成 ｜ ❌ 取消

## 启动容器（宿主侧，仓根执行）

```bash
cp dev/__NAME__/.env.example dev/__NAME__/.env          # 按需调整专属开关
python3 dev/lib/baseline.py export dev/__NAME__         # 首次及 x-flagrt / 挂载 repo 变更后执行
docker ps                                               # 昇腾驱动对带卡容器独占：确认无其他带卡容器
FLAGRT_MODELS_DIR=<模型目录> dev/lib/up.sh dev/__NAME__ # 验收或出结论的运行加 --strict
docker ps | grep flagos-__NAME__-dev-910c
```

`up.sh` 先按 `env.baseline.yaml` 核对镜像 id 与挂载 repo，再启动；默认只报告不一致，`--strict`
时拒绝启动。镜像、设备与公共挂载继承自
[`ascend-train-comm/v3/demo`](../images/ascend-train-comm/v3/demo/docker-compose.yml)；以可编辑方式
挂载 repo 的写法见 `dev/ENV-SPEC.md` §5。

修改 repo 代码的流程（挂载覆盖、开发分支提交并推送、`export --require-clean` 记录 commit、固化进新镜像）见
[运行时基座-使用与维护指南](../../docs/运行时基座-使用与维护指南.md) §4。

## 常用命令（环境速查）

```bash
# 进开发容器
docker exec -it flagos-__NAME__-dev-910c bash
# 容器内验证挂载（应看到 6 个子库 + dev/ 等公共仓内容）
docker exec -it flagos-__NAME__-dev-910c bash -c "ls /workspace"
```

## 工作原则

- 遵循"不预实现"原则：先有真实问题数据，再动手优化
- 公共红线（不改宿主配置/驱动、多卡前 npu-smi 确认、DrvMng 上限≈3）见主 README「红线」节
