# memory — 显存与缓存管理（路线 A）项目

> **状态：🟢 A 线新方向启动（2026-08-22，设备层路线变更 B→A）** ｜ 权威方案：《[路线A-显存与缓存管理-方案-20260822](docs/路线A-显存与缓存管理-方案-20260822.md)》｜ 旧案（B 线 2.4 方案）已归档：[docs/archive/](docs/archive/README.md)
> 本文档 = 子方向**入口**（目标、环境、操作）；**进展/待办/时间线见 [PROGRESS.md](PROGRESS.md)**，文档收拢在 `docs/`。

## 目标（一句话）

在 FlagOS 官方栈 A 线（厂商设备插件 + FlagGems + FlagCX + vllm-plugin-FL）上做显存分层管理与可控溢出：降低峰值显存、减少分配开销、让低优先级张量（主要是 KV Cache）能按需释放、能溢出到 Host/SSD。

## 双线状态（2026-08-22 起）

- **A 线 = 生产主线**：各芯片厂商设备插件 + FlagGems + FlagCX + vllm-plugin-FL（FlagOS 官方发布配置，有官方镜像/安装流程/CI 支撑，昇腾与昆仑芯均已实测跑通 dense 推理）
- **B 线 = 预研支线**：torch_fl 设备层（保留验证资产，继续推进上游合入，不承担交付责任、不作前置依赖）
- 本子方向按 **A 线**推进；变更详情见 [FlagOS设备层路线变更指南](docs/FlagOS设备层路线变更指南.md)

## 目录约定

```
dev/memory/
├── README.md               # 本文档（入口：目标、环境、操作）
├── PROGRESS.md             # 项目进展时间线（待办 + 完成，含日期）
├── docs/                   # 子方向文档（权威方案、画像报告、执行记录）
├── docs/archive/           # 已归档旧案（B 线废案：2.4 方案、B 栈画像与执行记录）
├── dev-prep.sh             # 开发前置一键准备（fetch 全部仓 + 切本地 <用户名>/dev 分支，--dry-run 先看计划）
├── docker-compose.yml      # 昇腾 910c 容器配置（A 线昇腾 + B 线预研共用）
├── docker-compose.p800.yml # 昆仑芯 P800 容器配置（A 线）
├── .env.example            # 环境变量模板（cp 成 .env 按需调整）
├── probes/                 # 探针/画像脚本（只读，不改造，待入库）
└── benchmarks/             # A/B 对比与负载脚本
```

代码改造主战场：**vLLM 层**（显存池/缓存管理在 vLLM 内做）；厂商 torch 自带分配器为底座（P800 上 torch_plugin 无 torch_fl 显存池）。

## 重要发现（环境坑，实测）

**昇腾 910c（B 栈经验，A 线昇腾仍适用）**

- **首次 attention 调用极慢（13+ 分钟）**：vLLM 加载 24s 极快（triton cache 热后），但第一个请求的 prefill 卡在 attention kernel 首次初始化（AICore 91% 忙、无新编译、cache 不增长），疑似 flag_gems/flagtune autotune + event-timing 回退。→ **基准必须先跑短请求预热**。
- EngineCore 是 spawn 子进程：主进程读不到 `torch_fl.flagos.memory_stats` → 画像依赖**设备级** HBM 采样（aclrtGetMemInfo/npu-smi）+ vLLM 日志。
- `docker exec` 被 kill 时 EngineCore 子进程会残留占卡 → 重跑前 `pkill -f "VLLM::EngineCore"`。
- vLLM usage 上报线程在容器内解析 cpuinfo 报错 → 设 `DO_NOT_TRACK=1`。

**昆仑芯 P800（A 线，2026-08-21~22 实测）**

- **triton 版本偏差致 FlagGems GEMM 崩溃**：dev 容器 triton 3.6.0（flagtree 0.6.1+xpu3.6）≠ 官方发布镜像 3.0.0，FlagGems mm/bmm/addmm 在 dev 容器 SIGABRT（编译期 make_llir），官方发布镜像同 commit 全过 → **结论性测试在官方发布镜像内做**。
- **MoE 双阻塞**：`xpudnn::causal_conv1d_update ret=1`（厂商算子库缺口，混合注意力模型昆仑芯不可用）；`flag_gems._kunlunxin.topk_softmax` 缺 `renormalize` 参（组件版本配对，纯 MoE 必崩）→ 见 [昆仑芯问题反馈清单-20260822](docs/昆仑芯问题反馈清单-20260822.md)。
- **KV 预分配 P800 69.22GiB / 504k tokens**（Qwen3-4B，gpu_mem_util 0.9，加载合计 84.5s、含 graph 71.5s）。

## 启动容器（宿主侧）

**昇腾 910c（A 线昇腾 + B 线预研共用）**

```bash
# 从公共仓根进入子方向
cd dev/memory
cp .env.example .env    # 按需调整专属开关（默认值即可直接启动）
docker compose -f ../compose.base.yml -f docker-compose.yml up -d
docker ps | grep flagos-fl-dev-910c    # 确认 Up

# 容器内验证挂载（应看到 6 个子库 + dev/ 等公共仓内容）
docker exec -it flagos-fl-dev-910c bash -c "ls /workspace"
```

**昆仑芯 P800（A 线）**

```bash
docker compose -f ../compose.base.yml -f docker-compose.p800.yml up -d
docker ps | grep flagos-fl-dev-p800    # 确认 Up
# ⚠️ 注意：dev 容器 triton 3.6.0 与官方发布镜像 3.0.0 有偏差，FlagGems GEMM 会编译崩溃；
#    结论性测试在官方发布镜像内做，本 dev 容器仅用于开发调试。
```

## 常用命令（环境速查）

```bash
# 昇腾 910c
docker exec -it flagos-fl-dev-910c bash
# venv311 里跑探针
/root/vllm-venv311/bin/python /workspace/dev/memory/probes/xxx.py

# 昆仑芯 P800（A 线）
docker exec -it flagos-fl-dev-p800 bash
conda activate python310_torch29_cuda
```

## 工作原则

- 遵循"不预实现"原则：先有真实问题数据，再动手优化（8.5）
- A 线测试在官方发布镜像内进行（dev 容器仅用于开发调试；triton 版本偏差会污染结论）
- 公共红线（不改宿主配置/驱动、多卡前 npu-smi 确认、DrvMng 名额有限）见主 README「红线」节
