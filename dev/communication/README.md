# communication — 多卡通信项目

> **状态：🟡 双卡正确性基线已完成，训练腿联合收口中（2026-09-09）** ｜ 本文档 = 任务看板入口，供运行时组全员维护
> 统一基座：`dev/stack.lock.910c.v1.yaml`；训练腿使用 FlagCX 0.13.0；代码主战场为独立 FlagCX 仓库。

## 目标（一句话）

建立 FlagCX 在 Ascend/HCCL 上可复现的构建、正确性与性能基线，并支持多卡集合通信和 KV 传输的定位、优化与回归。

## 当前进展

- 本月统一目标是锁定训练镜像内的 Qwen3-Embedding-0.6B 双卡微调；训练腿使用 `flagos` 设备后端，通信由镜像内 FlagCX 实现。
- 双卡通信统一探针已覆盖 AllReduce、AllGather、P2P、异步 AllReduce：2 Rank × 2 dtype × 20 轮，共 320 次调用，320/320 PASS。
- 异步 AllReduce 的 `Work.is_completed()` 在 80/80 次立即查询中返回 `true`；等待并设备同步后的结果全部正确，跨流可消费时序仍需联合验证。
- 执行记录：[`docs/COMMUNICATION_CORRECTNESS_20260909.md`](docs/COMMUNICATION_CORRECTNESS_20260909.md)。

## 2026.09 最小交付口径

- 依据组内阶段目标，通信方向与设备方向共同交付双卡微调训练腿；通信侧负责三类正确性对照、通信接口约定和历史 FlagCX 缺陷的干净提交。
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
├── probes/             # 正确性、拓扑与故障定位探针
├── results/            # 结构化原始结果
└── benchmarks/         # 集合通信和 KV 传输基准（按需建）
```

代码改造主战场不在本目录：**FlagCX**（核心通信库与 `plugin/torch/`）；KV 传输联调按需涉及 `vllm-plugin-FL`。

本子方向通过 Compose 顶层 `name: flagos-communication` 使用独立 project；不得删除，否则公共 service 名 `runtime-dev` 会与其他子方向发生重建冲突。

## 任务看板

| # | 任务 | 负责人 | 状态 | 依赖 | 出口标准 |
|---|------|--------|------|------|----------|
| 1 | 容器与代码挂载验证 | lianzhongyou | 🔄 | Docker 权限 | 容器 Up；`/workspace` 可见 6 个子库 |
| 2 | FlagCX Ascend 构建基线 | lianzhongyou | ⬜ | #1 | `make USE_ASCEND=1` 成功，记录 commit、命令与耗时 |
| 3 | 2 卡通信正确性冒烟 | lianzhongyou | ✅ | 锁定训练镜像 | AllReduce、AllGather、P2P、异步 AllReduce 共 320/320 通过 |
| 4 | 16 卡集合通信性能基线 | lianzhongyou | ⬜ | #3 | 覆盖 AllReduce/AllGather/ReduceScatter，归档带宽与时延 |
| 5 | Torch FlagCX process group 回归 | lianzhongyou | ⬜ | #2 | 关键 collective 用例通过 |
| 6 | KV 传输链路与瓶颈画像 | TBD | ⬜ | #3 | 形成链路图、基准数据与优化清单 |
| 7 | 2026.09 最小 Backend 契约与验收矩阵 | lianzhongyou | ✅ | 实施方案 | 接口边界、复用资产、退出标准入库 |
| 8 | 跨方向通信探针收口 | lianzhongyou | ✅ | #3、device-context、memory | 统一入口，双卡正确性与异步语义结果可追溯 |
| 9 | 历史 FlagCX 缺陷整理为干净提交 | lianzhongyou | 🔄 | FlagCX 仓库 | 编译、单测通过且改动可独立审查 |

> 状态图例：⬜ 待认领 ｜ 🔄 进行中 ｜ ✅ 完成 ｜ ❌ 取消

## 统一基座启动方式

结论性验证只使用 `dev/stack.lock.910c.v1.yaml` 中的锁定训练容器 `flagos-proto-train-910c`。启动前确认目标卡空闲，且带卡容器总数不超过 3：

```bash
npu-smi info
docker ps
docker start flagos-proto-train-910c
docker ps --filter name=flagos-proto-train-910c
```

本目录原有 Compose 仅保留为历史开发环境，不用于当前原型的结论性验证，也不得据此另起第三套验收容器。

## 常用命令（环境速查）

```bash
# 双卡正确性探针（脚本需在容器内可见）
docker exec flagos-proto-train-910c bash -lc \
  'TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchrun --nproc_per_node=2 \
   /tmp/communication_correctness.py --iterations 20 \
   --out-dir /tmp/comm-results-lianzhongyou'
```

FlagCX 的性能和 Torch API 测试入口见 `/workspace/FlagCX/docs/getting_started.md`。基准前固定 commit、卡数、消息大小、预热次数和迭代次数；测试后检查并清理残留进程。

## 工作原则

- 先做 2 卡正确性冒烟，再扩大到 16 卡性能测试；多卡前必须复查卡空闲。
- 默认日志保持 `WARN`；定位时在 `.env` 临时启用 `INFO/TRACE`，避免高日志级别污染性能结果。
- 不修改宿主驱动或系统配置；依赖安装和实验均在容器内完成。
- 共享分支只 merge、不 rebase；按 `dev/sync_to_dev.sh` 的组内流程自行合入 `dev-1.0`，不额外创建 PR。
