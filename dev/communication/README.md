# communication — 多卡通信项目

> **状态：🟡 接口收敛与基线复验中（2026-09-02）** ｜ 本文档 = 任务看板入口，供运行时组全员维护
> 对齐起点速查：FlagCX 0.13.0；Ascend/HCCL；单机 16 卡；代码主战场 `/workspace/FlagCX`。

## 目标（一句话）

建立 FlagCX 在 Ascend/HCCL 上可复现的构建、正确性与性能基线，并支持多卡集合通信和 KV 传输的定位、优化与回归。

## 现状（2026-08-18 快照）

- `VERSIONS.md` 已锁定 FlagCX 0.13.0，运行组合中 `FLAGCX_PATH=/workspace/FlagCX/plugin/torch`。
- 公共 Compose 已挂载 16 张 Ascend 910 卡，并注入 `HCCL_NPU_SOCKET_PORT_RANGE=16666,16676`。
- FlagCX 官方提供 Ascend 构建入口 `make USE_ASCEND=1`、host API 性能测试及 Torch process-group 测试。
- Compose 合并配置与实际容器启动已通过验证；多卡 collective 测试尚未执行。

## 2026.09 最小交付口径

- 依据组内实施方案，9 月交付聚焦“基础通信适配原型、通信组与 Collective 基础测试”，不扩展独立通信系统。
- 最小 Backend 契约覆盖通信组生命周期、AllReduce/AllGather/ReduceScatter/Broadcast、Send/Recv、异步完成语义和能力声明。
- 接口、验收矩阵、跨方向复用资产及退出标准见 [`docs/MINIMUM_BACKEND_CONTRACT_202609.md`](docs/MINIMUM_BACKEND_CONTRACT_202609.md)。
- `device-context` 已有 TP、Stream/Event 与异步完成语义验证资产，`memory` 已有双卡 FlagCX AllReduce 探针；本方向负责收口为统一通信基线，避免重复造轮子。

## 环境验证记录（2026-08-24）

- `flagos-communication-dev-910c` 已在共享服务器启动，Compose 标签为独立的 `project=flagos-communication`，重启计数为 0。
- 容器可见 16 个 `/dev/davinci*` 设备及 Ascend driver 25.5.0；FlagCX、Torch-FL 与 communication 配置挂载正常。
- `FLAGCX_PATH`、`FLAGCX_DEBUG`、`HCCL_NPU_SOCKET_PORT_RANGE`、`GEMS_VENDOR` 等关键环境变量已核对。
- 服务器测试副本尚未同步 FlagPerf 工作树，`/workspace` 六仓完整性验证待补；不影响本轮 FlagCX communication 启动验证。

## 目录约定（本子方向，位于 `dev/communication/` 下）

```text
dev/communication/
├── README.md           # 本文档（看板）
├── docker-compose.yml  # 子方向配置，与 ../compose.base.yml 合并
├── .env.example        # FlagCX 专属环境变量模板
├── docs/               # 调研、方案与执行记录（按需建）
├── probes/             # 正确性、拓扑与故障定位探针（按需建）
└── benchmarks/         # 集合通信和 KV 传输基准（按需建）
```

代码改造主战场不在本目录：**FlagCX**（核心通信库与 `plugin/torch/`）；KV 传输联调按需涉及 `vllm-plugin-FL`。

本子方向通过 Compose 顶层 `name: flagos-communication` 使用独立 project；不得删除，否则公共 service 名 `runtime-dev` 会与其他子方向发生重建冲突。

## 任务看板

| # | 任务 | 负责人 | 状态 | 依赖 | 出口标准 |
|---|------|--------|------|------|----------|
| 1 | 容器与代码挂载验证 | lianzhongyou | 🔄 | Docker 权限 | 容器 Up；`/workspace` 可见 6 个子库 |
| 2 | FlagCX Ascend 构建基线 | lianzhongyou | ⬜ | #1 | `make USE_ASCEND=1` 成功，记录 commit、命令与耗时 |
| 3 | 2 卡 AllReduce 正确性冒烟 | lianzhongyou | ⬜ | #2 | 结果正确、无残留进程，日志可追溯 |
| 4 | 16 卡集合通信性能基线 | lianzhongyou | ⬜ | #3 | 覆盖 AllReduce/AllGather/ReduceScatter，归档带宽与时延 |
| 5 | Torch FlagCX process group 回归 | lianzhongyou | ⬜ | #2 | 关键 collective 用例通过 |
| 6 | KV 传输链路与瓶颈画像 | TBD | ⬜ | #3 | 形成链路图、基准数据与优化清单 |
| 7 | 2026.09 最小 Backend 契约与验收矩阵 | lianzhongyou | ✅ | 实施方案 | 接口边界、复用资产、退出标准入库 |
| 8 | 跨方向通信探针收口 | lianzhongyou | 🔄 | #3、device-context、memory | 统一入口，双卡正确性与异步语义结果可追溯 |

> 状态图例：⬜ 待认领 ｜ 🔄 进行中 ｜ ✅ 完成 ｜ ❌ 取消

## 启动容器（宿主侧）

启动前先确认 16 张卡空闲，并检查 DrvMng 容器并发数：

```bash
npu-smi info
docker ps

cd dev/communication
cp .env.example .env
docker compose -f ../compose.base.yml -f docker-compose.yml config
docker compose -f ../compose.base.yml -f docker-compose.yml up -d
docker ps --filter name=flagos-communication-dev-910c
```

## 常用命令（环境速查）

```bash
# 验证公共仓和 6 个子库挂载
docker exec flagos-communication-dev-910c bash -lc 'ls /workspace'

# 检查关键环境变量
docker exec flagos-communication-dev-910c bash -lc \
  'env | grep -E "^(FLAGCX|HCCL_NPU_SOCKET_PORT_RANGE)="'

# FlagCX Ascend 构建入口（容器内）
cd /workspace/FlagCX
git submodule update --init --recursive
make USE_ASCEND=1 -j"$(nproc)"
```

FlagCX 的性能和 Torch API 测试入口见 `/workspace/FlagCX/docs/getting_started.md`。基准前固定 commit、卡数、消息大小、预热次数和迭代次数；测试后检查并清理残留进程。

## 工作原则

- 先做 2 卡正确性冒烟，再扩大到 16 卡性能测试；多卡前必须复查卡空闲。
- 默认日志保持 `WARN`；定位时在 `.env` 临时启用 `INFO/TRACE`，避免高日志级别污染性能结果。
- 不修改宿主驱动或系统配置；依赖安装和实验均在容器内完成。
- 共享分支只 merge、不 rebase；个人分支经 PR 合入 `dev-1.0`。
