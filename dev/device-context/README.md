# device-context（设备执行上下文）项目

> **状态：🟡 开发期启动（2026-08-19）** ｜ 本文档 = 任务看板入口，供运行时组全员维护
> 对齐起点速览：Torch-FL 已有设备接入基础（csrc/runtime/）；本子方向聚焦 **设备执行上下文（Backend 插件 + 统一三大句柄 + Stream/异步/同步语义）**，以单机 2 卡 Qwen2.5-1.5B 训练作为功能验证。

## 目标（一句话）

**设备执行上下文**负责统一管理不同芯片设备的初始化、上下文创建、执行队列、Stream/Event、设备间同步、Host 与 Device 数据传输、错误捕获和状态恢复。

对于昇腾、平头哥、寒武纪、壁仞、燧原、昆仑芯等不同芯片，运行时层通过 **Backend 插件**封装厂商 Runtime 接口，向上提供统一**设备句柄、内存句柄和执行句柄**。该机制保证同一模型部署产物在不同芯片上能够通过一致的调用方式运行，降低上层模型服务和运维系统的适配复杂度。

## 现状（2026-08-19 快照）

- Torch-FL（本地目录 PyTorch-Plugin-FL）**已有设备接入基础**：`csrc/runtime/`（设备句柄、显存池 allocator、Stream/Event 抽象）
- 910C 环境（**分层版本，注意区分**）：
  - **宿主 CANN toolkit 8.5.0**（/usr/local/Ascend/ascend-toolkit/latest，与宿主驱动匹配，宿主直调 aclInit 正常）
  - **宿主驱动 HDK 25.5.0**（npu-smi driver version）
  - **容器镜像内置 CANN 9.0.0**（flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev-hostnet）
  - 16 chip（davinci0-15）、NPU 全空闲；仓库已 clone（5 子库对齐）
- **待验证**：容器内 torch_fl 设备注册 → 显存分配 → Stream/Event → 双卡 HCCL 跨卡 → 训练闭环

## 需求映射（科研任务 → 代码位置）

| 需求 | 内容 | 代码主战场 |
|------|------|-----------|
| 需求 1 | 封装不同芯片 Runtime 接口：统一设备/内存/执行句柄/生命周期 | **Torch-FL `csrc/runtime/`**（设备接入 + allocator 显存池） |
| 需求 2 | 统一 Stream 语义/异步传输/页锁定/双缓冲/同步/错误码翻译/状态恢复 | Torch-FL 设备 API 层（csrc/runtime 内） |
| 需求 3+4 | 张量并行/流水线并行下的跨卡同步与执行编排 | FlagCX（通信）+ 训练侧验证（本子方向用 2 卡训练验证 1+2 正确性） |

## 目录约定（本子方向，位于 dev/device-context/ 下）

```
dev/device-context/
├── README.md           # 本文档（看板）
├── docker-compose.yml  # 容器配置（-f ../compose.base.yml 合并公共配置）
├── .env.example        # 环境变量模板（cp 成 .env 按需调整）
├── docs/               # 调研笔记、方案摘录、执行记录
├── probes/             # 探针/画像脚本（只读）
└── benchmarks/         # A/B 对比与负载脚本
```


**同事复现与验证补充（2026-08-20，6 容器实测）**：
- 6 个容器重跑 ctypes 直调：新起的容器全部 `ACLINIT_FAIL ret=500000`，日志关键行 `get platform info failed, drvErr=87`；旧容器（8-13 前注册的老客户端）`ACLINIT_OK`
- 证伪版本论：同 CANN 9.0.0 库文件 md5 完全一致，旧容器成功新容器失败；同事容器混挂宿主 CANN 8.5.0 也失败；用旧镜像（1382b18ac660）起全新容器照样失败 → **与镜像/CANN 版本无关**
- 分界线 = **容器启动时间**：8-13 前起的容器（已注册客户端，重复 init 放行）成功，之后新起的全部被拒；错误发生在 aclInit 向驱动注册客户端、查询平台信息这一步（drvErr=87）
- **名额机制**：停 1 个长驻容器 → 释放 1 名额 → 恰好 1 个新注册放行（1 名额 = 1 新注册）
- **协作约定**：容器用完即停；多卡测试前先 `docker ps` 清点挂设备容器；不要挂宿主 CANN mount（避免 8.5.0/9.0.0 混装）


### ✅ FlagCX Ascend 适配（2026-08-20，需求 2/3 开发成果）

**问题**：FlagCX plugin/torch 的 ascend 事件/流实现硬编码 torch_npu（`event_flagcx.hpp`/`stream_guard_flagcx.hpp`/`backend_flagcx.cpp` include `NPUEvent.h`/`NPUStream.h`，`_build_config.py` 强制 `import torch_npu`）——与团队"torch_fl 替代 torch_npu"路线冲突，编译失败。

**根因定位**：通信层（FlagCX）未消费运行时层（torch_fl `csrc/runtime/guard.h` 已实现的 EventCreate/EventRecord）统一接口，直接借用了 torch_npu。

**修复**（已推送 FlagRT/FlagCX 分支 `kistich/ascend-flagcx-adapt`）：
- `_build_config.py`：ascend 构建去掉 torch_npu，用 torch cpp_extension + CANN 头
- `event_flagcx.hpp`：`flagcxCannEvent` 改用 CANN ACL（`aclrtCreateEvent/RecordEvent/StreamWaitEvent/DestroyEvent` + `aclrtCtxGetCurrentDefaultStream`）
- `stream_guard_flagcx.hpp`：NPUStream → `aclrtStream`
- `backend_flagcx.cpp`：getStreamByIndex 去掉 c10_npu，用 devHandle streamCreate
- torch_fl `process_group.py` `_resolve_view`：flagcx 后端 + view=None（ascend）返回恒等视图（flagcx 原生消费 privateuseone data_ptr）

**验证**：flagcx 0.13.0 编译安装成功；`import flagcx` + `flagcx backend registered: True`；**单机单卡 Qwen2.5-1.5B bf16 训练闭环跑通**（loss 1.7-3.0 正常波动，显存 14.6GB，~530 tok/s）；双卡待 DrvMng 名额。

**DrvMng 名额机制要点（组内协作约定）**：
- 客户端名额按进程计（≈3），docker stop 优雅退出释放；SIGKILL（force kill）残留占位
- 多进程训练（torchrun N 进程 = N 客户端）需先确认名额
- 容器异常退出后重启注册被拒 = 残留未释放，等超时或协调释放


### ⏳ 双卡训练阻塞根因（2026-08-20）：flagcx 核心 hccl adaptor commId 格式缺陷

**现象**：双卡 DDP（torchrun 2 进程）在 flagcxCommInitRank 阶段报 `flagcxInvalidArgument`（错误码 4），`flagcxComm is not fully initialized`。

**根因**（flagcx 核心源码定位）：
- `flagcxGetUniqueId` 生成 256B `flagcxUniqueId`（内容是 **bootstrap handle**，`bootstrapGetUniqueId`）
- `hcclAdaptorCommInitRank`（flagcx/adaptor/ccl/hccl_adaptor.cc:98）直接强转 `HcclCommInitRootInfo(nranks, (HcclRootInfo*)commId, ...)`——**HCCL 期望的是 `hcclGetRootInfo` 生成的 RootInfo，收到的是 bootstrap handle → 格式不匹配 → InvalidArgument**
- `hcclAdaptorCommInitRank` 的 bootstrap 参数被注释忽略（`/*bootstrap*/`），无 root info 生成/交换逻辑

**结论**：flagcx 核心对 ascend HCCL 通信集成是"半成品"（第三个 FlagCX ascend 适配缺口，前两个：torch_npu 事件依赖、plugin 构建）。修复方向：hccl adaptor 在 rank0 用 `hcclGetRootInfo` 生成 RootInfo 并经 bootstrap 交换，再 `HcclCommInitRootInfo`。

**影响**：双卡/多卡训练阻塞；单卡训练闭环已通（本页前述）。此任务列入需求 3（跨卡通信）开发。


**重要补充**：FlagRT/Torch-FL 源码（main）已是新版——`_VendorProfile("ascend", ..., flagcx_native=True)` 且 `_resolve_view` 已实现 flagcx 原生消费（`return None`）。**镜像内置 torch_fl 0.1.0 是旧版**（无此逻辑）；本会话对 site-packages 的 view patch 是旧版临时适配。**长期方案：用源码仓库重装 torch_fl**（pip install /workspace/PyTorch-Plugin-FL），无需修改源码。双卡阻塞的 hccl adaptor commId 缺陷在 **flagcx 核心库**（flagcx/adaptor/ccl/hccl_adaptor.cc），列入需求 3 开发任务。


### ✅ 双卡训练打通（2026-08-21）：FlagOS 全栈首个 910C 双卡闭环

**目标**：修复 flagcx 核心 hccl adaptor 的 commId 缺陷，跑通单机 2 卡 Qwen2.5-1.5B DDP 训练。

**四层根因链（源码级定位，全在 FlagCX ascend 适配层）**：

| # | 根因 | 修复 |
|---|------|------|
| 1 | (4108B) > (256B)：GetUniqueId 缓冲区溢出 + allGather 只传 256B →  InvalidArgument(4) | flagcx.cc  传 bootstrap state；hccl_adaptor.cc thread_local 存 RootInfo，CommInitRank 时 rank0 生成 +  全量分发 4108B |
| 2 | HCCL 传输选错网卡（bond4）→  /  失败 | 同节点走 HCCS 不需网卡；无需 HCCL_IF_NAME（验证无效） |
| 3 | 当前 ACL 设备与 comm 不匹配 →  E_PARA | 训练脚本 （torch_fl 默认只在 device 0） |
| 4 | collective 不在调用者当前 stream → API ret=0 但数据全 0（无 happens-before） |  改返回 （PyTorch 标准语义）；stream_guard ascend 分支移除无效 SetCurrentStream（libflagos 旧版无此符号）；plugin 链接 libflagos.so 解决 RTLD_LOCAL undefined |

**验证结果**：
- 双进程 allgather 数据正确（[1,2]、[10,11]）
- 双卡训练 loss 1.4-2.9 正常波动；吞吐 ~2000 tok/s（单卡 530 的 4 倍）；显存 14.64GB/卡
- 全链路：torch_fl(flagos 设备) → DDP(flagos backend) → ProcessGroupFlagOS → FlagCX(homoRunner) → HCCL(HCCS 机内互联)

**改动文件**（FlagCX）：
- ：flagcxHomoCommInit 传 bootstrap state（1 处）
- ：GetUniqueId 防溢出 + CommInitRank bootstrap 分发 RootInfo
- ：getStreamByIndex(0) 用调用者当前 stream
- ：ascend 分支修正（无操作，当前 stream 语义）
- ：ascend 分支链接 libflagos.so

**工具**：（纯 C 双进程 HCCL 冒烟测试，FP32/BFP16/INT64 全通过）+  + （allgather 数据验证）

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
docker ps | grep flagos-device-context-dev-910c    # 确认 Up

# 容器内验证挂载（应看到 5 个子库 + dev/ 等公共仓内容）
docker exec -it flagos-device-context-dev-910c bash -c "ls /workspace"
```

## 常用命令（环境速查）

```bash
# 进开发容器
docker exec -it flagos-device-context-dev-910c bash
# venv311 里跑探针
/root/vllm-venv311/bin/python /workspace/dev/device-context/probes/xxx.py
```

## 工作原则

- 遵循"不预实现"原则：先有真实问题数据，再动手优化
- 公共红线（不改宿主配置/驱动、多卡前 npu-smi 确认、DrvMng 上限≈3）见主 README「红线」节
- 所有安装与实验在容器内进行；个人调试记录默认收拢 personal/ 不上传

## 重要发现（2026-08-19，容器验证过程中）

### ✅ 已解决：aclInit 500000 根因 = DrvMng 容器客户端上限（非版本问题）

**现象**：容器内 aclInit 报 500000（ACL_ERROR_INTERNAL_ERROR），device_count=0。

**根因（2026-08-20 证实）**：DrvMng（驱动侧管理进程）对**同时挂载 davinci 设备的容器客户端数量有上限（实测 ≈3）**。槽位占满后，任何新容器/新进程调用 aclInit 都报 500000——与容器配置、torch_fl、CANN 版本均无关。

**验证过程**（决定性）：
- 5 个挂设备容器时：所有容器 aclInit 均 500000（含 memory 组员容器）
- 停掉 2 个闲置容器（剩余 3 个 = 上限）：同一容器 aclInit 立即恢复 `ret=0`、`device_count=16`
- torch_fl 全链路验证：`is_available=True`、16 卡枚举、`flagos` 设备上真实矩阵乘计算 OK

**之前误判为"CANN 9.0.0 vs 驱动 25.5.0 版本不匹配"——已纠正**。官方兼容矩阵是"官方支持的最低驱动版本"而非"能否运行"；真正版本不兼容会报明确的版本校验错误，而非 500000 通用运行时错误；且 9.0.0+25.5.0 组合在 8-17 曾真实推理成功。

**协作提醒**：
- 多卡测试前检查挂设备容器数：`docker ps` + 数一下 --device davinci 的容器
- DrvMng 上限 ≈3：超过需先停闲置容器（停他人容器前在群里打招呼）
- 无需升级宿主驱动（25.5.0 → 25.5.1/2 为主机级变更，按上述证据大概率白干，仅需进入官方支持矩阵时才考虑）
