# memory — 显存与缓存管理（2.4）项目

> **状态：🟢 已定稿（2026-08-17），开发期启动** ｜ 权威方案：《docs/02-调研与方案/显存与缓存管理-2.4-调研与实施方案.md》（定稿版）
> 本文档 = 任务看板入口，供运行时组全员维护。
> 对齐起点速览：调研验证已完成（分配器✅ V1画像✅ vLLM机制✅）；开发期主线 = **P0 长序列 prefill 定位修复（跨组）** → V3 分层缓存原型（并行） → V2 回归 → V4 SSD → 执行计划感知。

## 目标（一句话）

在 FlagOS 运行时层把显存从"够用就行"做到"按计划复用 + 分层可控"：降低峰值显存、减少分配开销、让低优先级张量（主要是 KV Cache）能按需释放、能溢出到 Host/SSD。

## 现状（2026-08-17 快照）

- 显存池第一层（torch_fl CachingDeviceAllocator）**已有未验证**——开关 `FLAGOS_USE_CACHING_ALLOCATOR`，源码 `PyTorch-Plugin-FL/csrc/runtime/allocator/`
- KV Cache 块级管理由 vLLM 自带（已在推理闭环中工作）
- **执行计划感知分配、分层缓存/可控溢出：未做**

## 目录约定（本子方向，位于 dev/memory/ 下）

```
dev/memory/
├── README.md           # 本文档（看板）
├── docker-compose.yml  # 本子方向容器配置（-f ../compose.base.yml 合并公共配置）
├── .env.example        # 环境变量模板（cp 成 .env 按需调整）
├── docs/               # 调研笔记、方案摘录（结论性数据归档到 workspace/docs/03-）
├── probes/             # 探针/画像脚本（只读，不改造，待入库）
└── benchmarks/         # A/B 对比与负载脚本
```

代码改造主战场不在本目录：**torch_fl `csrc/runtime/allocator/`**（已有多后端结构，加层不破坏）。

## 任务看板

| # | 任务 | 负责人 | 状态 | 依赖 | 出口标准 |
|---|------|--------|------|------|----------|
| 1 | V1 显存画像（长序列+并发负载） | xliu969 | ✅ 2026-08-17（部分） | — | docs/V1-显存画像报告-20260817.md：加载 31.9G/KV 170k tokens；**发现长序列 prefill P0 性能问题** |
| 1b | 长序列 prefill 性能问题上报（P0，跨组） | TBD | 🔄 待上报 | 例会/编译组 | 定位结论或缓解方案 |
| 2 | torch_fl 分配器现状画像（开关默认值/统计接口） | xliu969 | ✅ 2026-08-17 | — | docs/allocator-画像报告-20260817.md：默认开启、接口齐全、复用/切分/合并实测通过、碎片<5% |
| 3 | 调研 vLLM `--cpu-offload-gb` / `--kv-transfer-config` 0.20.2 实现 | xliu969 | ✅ 2026-08-17 | — | docs/vllm-offload-调研笔记-20260817.md：offloader 双后端、evict_blocks 挂载点、UVA 待验证项 |
| 4 | 向编译组书面确认执行计划内存规划接口 | xliu969 | 🔄 草稿待发出 | 阶段0 契约延伸 | docs/执行计划-显存规划-跨组确认-20260817.md |
| 5 | P0 算子清单确认 | TBD | ⬜ | 算子组 | 清单归档 |
| 6 | V2 显存池 A/B 对比（开关开/关） | TBD | ⬜ 暂缓 | #1b（长序列阻塞） | 对比数据 + 改造清单 |
| 7 | V3 分层缓存原型（KV 按需释放 + Host 溢出） | TBD | ⬜ 可并行 | #3 | 原型验证报告（短序列可验，不受 #1b 阻塞） |
| 8 | V4 SSD 层评估（NVMe 实测） | TBD | ⬜ | — | 立项结论 |

> 状态图例：⬜ 待认领 ｜ 🔄 进行中 ｜ ✅ 完成 ｜ ❌ 取消

## 重要发现（2026-08-17，V1 过程中）

- **首次 attention 调用极慢（13+ 分钟）**：vLLM 加载 24s 极快（triton cache 热后），但第一个请求的 prefill 卡在 attention kernel 首次初始化（AICore 91% 忙、无新编译、cache 不增长），疑似 flag_gems/flagtune autotune + event-timing 回退。→ **后续所有基准必须先跑短请求预热**，否则首 token 时间失真。
- EngineCore 是 spawn 子进程：主进程读不到 `torch_fl.flagos.memory_stats`（allocated/reserved=0）→ V1 画像依赖**设备级** HBM 采样（aclrtGetMemInfo/npu-smi）+ vLLM 日志（GPU KV cache size 行）。
- `docker exec` 被 kill 时 EngineCore 子进程会残留占卡 → 重跑前必须 `pkill -f "VLLM::EngineCore"`。
- vLLM usage 上报线程在容器内解析 cpuinfo 报错 → 设 `DO_NOT_TRACK=1`。

## 启动容器（宿主侧）

```bash
# 从公共仓根进入子方向
cd dev/memory
cp .env.example .env    # 按需调整专属开关（默认值即可直接启动）
docker compose -f ../compose.base.yml -f docker-compose.yml up -d
docker ps | grep flagos-fl-dev-910c    # 确认 Up

# 容器内验证挂载（应看到 6 个子库 + dev/ 等公共仓内容）
docker exec -it flagos-fl-dev-910c bash -c "ls /workspace"
```

## 常用命令（环境速查）

```bash
# 进开发容器
docker exec -it flagos-fl-dev-910c bash
# venv311 里跑探针
/root/vllm-venv311/bin/python /workspace/dev/memory/probes/xxx.py
# 推理负载参考（阶段4 已验证链路）
#   见 docs/01-阶段执行记录/推理插件接入-阶段4执行记录.md §3
```

## 工作原则

- 遵循"不预实现"原则：先有真实问题数据，再动手优化（8.5）
- 公共红线（不改宿主配置/驱动、多卡前 npu-smi 确认、DrvMng 上限≈3）见主 README「红线」节
