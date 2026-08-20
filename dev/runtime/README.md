# runtime（设备运行时统一封装）项目

> **状态：🟡 开发期启动（2026-08-19）** ｜ 本文档 = 任务看板入口，供运行时组全员维护
> 对齐起点速览：Torch-FL 已有设备接入基础（csrc/runtime/）；本子方向聚焦 **Runtime 接口封装 + Stream/异步/同步语义**，以单机 2 卡 Qwen2.5-1.5B 训练作为功能验证。

## 目标（一句话）

在 FlagOS 运行时层把不同芯片（昇腾/寒武纪/昆仑芯/平头哥）的 Runtime 差异封装成统一接口：**统一设备、内存、执行句柄与生命周期管理**，并**统一 Stream 语义、Host/Device 异步传输、页锁定内存与双缓冲流水线、同步语义、错误码翻译与设备状态恢复**。

## 现状（2026-08-19 快照）

- Torch-FL（本地目录 PyTorch-Plugin-FL）**已有设备接入基础**：`csrc/runtime/`（设备句柄、显存池 allocator、Stream/Event 抽象）
- 910C 环境：CANN 8.5.0、16 chip（davinci0-15）、NPU 全空闲；仓库已 clone（5 子库对齐）
- **待验证**：容器内 torch_fl 设备注册 → 显存分配 → Stream/Event → 双卡 HCCL 跨卡 → 训练闭环

## 需求映射（科研任务 → 代码位置）

| 需求 | 内容 | 代码主战场 |
|------|------|-----------|
| 需求 1 | 封装不同芯片 Runtime 接口：统一设备/内存/执行句柄/生命周期 | **Torch-FL `csrc/runtime/`**（设备接入 + allocator 显存池） |
| 需求 2 | 统一 Stream 语义/异步传输/页锁定/双缓冲/同步/错误码翻译/状态恢复 | Torch-FL 设备 API 层（csrc/runtime 内） |
| 需求 3+4 | 张量并行/流水线并行下的跨卡同步与执行编排 | FlagCX（通信）+ 训练侧验证（本子方向用 2 卡训练验证 1+2 正确性） |

## 目录约定（本子方向，位于 dev/runtime/ 下）

```
dev/runtime/
├── README.md           # 本文档（看板）
├── docker-compose.yml  # 容器配置（-f ../compose.base.yml 合并公共配置）
├── .env.example        # 环境变量模板（cp 成 .env 按需调整）
├── docs/               # 调研笔记、方案摘录、执行记录
├── probes/             # 探针/画像脚本（只读）
└── benchmarks/         # A/B 对比与负载脚本
```

## 任务看板

| # | 任务 | 负责人 | 状态 | 依赖 | 出口标准 |
|---|------|--------|------|------|----------|
| 1 | 容器启动 + venv311 组合验证 | Kistich | ⬜ | docker 权限 + 镜像 | 容器 Up；/workspace 见 5 子库；venv311 各组件 import 通过 |
| 2 | torch_fl 设备注册与基础算子验证（需求 1） | Kistich | ⬜ | #1 | torch_fl.flagos 设备可用；显存分配/释放/生命周期 OK |
| 3 | Stream/Event/异步传输验证（需求 2） | Kistich | ⬜ | #2 | 双流并发、Event 同步、页锁定传输实测通过 |
| 4 | 单机 2 卡 Qwen2.5-1.5B 训练（需求 3+4 验证 1+2） | Kistich | ⬜ | #3 | 双卡 HCCL 跑通；loss 下降；记录全部踩坑 |
| 5 | 错误码翻译与设备状态恢复验证（需求 2 延伸） | Kistich | ⬜ | #2 | 注入错误场景，统一错误码 + 恢复路径 OK |
| 6 | 完整方案文档 + PR 提交 | Kistich | ⬜ | #4/#5 | docs/ 方案定稿；PR 合入 dev-1.0 |

> 状态图例：⬜ 待认领 ｜ 🔄 进行中 ｜ ✅ 完成 ｜ ❌ 取消

## 启动容器（宿主侧）

```bash
# 从公共仓根进入子方向
cd dev/runtime
cp .env.example .env    # 按需调整专属开关（默认值即可直接启动）
docker compose -f ../compose.base.yml -f docker-compose.yml up -d
docker ps | grep flagos-runtime-dev-910c    # 确认 Up

# 容器内验证挂载（应看到 5 个子库 + dev/ 等公共仓内容）
docker exec -it flagos-runtime-dev-910c bash -c "ls /workspace"
```

## 常用命令（环境速查）

```bash
# 进开发容器
docker exec -it flagos-runtime-dev-910c bash
# venv311 里跑探针
/root/vllm-venv311/bin/python /workspace/dev/runtime/probes/xxx.py
```

## 工作原则

- 遵循"不预实现"原则：先有真实问题数据，再动手优化
- 公共红线（不改宿主配置/驱动、多卡前 npu-smi 确认、DrvMng 上限≈3）见主 README「红线」节
- 所有安装与实验在容器内进行；个人调试记录默认收拢 personal/ 不上传

## 重要发现（2026-08-19，容器验证过程中）

### 🔴 P0：torch_fl Ascend 裸设备初始化 aclInit 失败（500000）

**现象**：容器内 `torch_fl`（tf-venv-integration，torch 2.10.0+cpu）调用设备接口时
`[flagos-ascend] aclInit failed: 500000`（ACL_ERROR_INTERNAL_ERROR），device_count=0。
手动 ctypes 直调 `aclInit` 同样失败——非 torch_fl bug，是环境层问题。

**对比实验**（决定性）：
- 宿主直接 aclInit：✅ 成功（ret=0，识别 16 卡）→ 驱动/硬件正常
- 容器内 aclInit（含 memory 组员 flagos-fl-dev-910c 容器，同一镜像）：❌ 均失败 500000

**根因**（官方兼容矩阵核实）：
- 容器镜像 CANN 9.0.0 要求 Ascend HDK（驱动）≥ **25.5.1 / 25.5.2**
- 910C 当前宿主驱动 **25.5.0**（低一个补丁版）→ 版本不匹配

**影响**：所有基于 `manual-20260807-ascend-dev-hostnet`（CANN 9.0.0）镜像的容器，torch_fl 裸设备层（aclInit）均不可用 → 阻塞需求 1（设备 Runtime 封装）的设备初始化验证。

**候选方案**（待组内决策）：
- A. 升级宿主驱动 25.5.0 → 25.5.2+（涉及宿主，需 IT/管理员，触碰"不改宿主"红线边界）
- B. 容器内降级 CANN 9.0.0 → 8.5.2（与 25.5.0 驱动匹配；容器内操作符合红线；需重建镜像或容器内安装）
- C. 验证 vllm-ascend 对照镜像（quay.io/ascend/vllm-ascend）的 torch_npu 链路能否绕过（memory 组员 V1 画像疑似走此路径）

**上报建议**：作为组级环境问题上报（参照 memory 组员 P0 上报模式），影响全组容器化验证。
