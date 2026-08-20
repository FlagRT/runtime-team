# memory — 显存与缓存管理（2.4）项目

> **状态：🟢 已定稿（2026-08-17），开发期启动** ｜ 权威方案：《[docs/显存与缓存管理-2.4-调研与实施方案.md](docs/显存与缓存管理-2.4-调研与实施方案.md)》（定稿版）
> 本文档 = 子方向**入口**（目标、环境、操作）；**进展/待办/时间线见 [PROGRESS.md](PROGRESS.md)**，文档收拢在 `docs/`。

## 目标（一句话）

在 FlagOS 运行时层把显存从"够用就行"做到"按计划复用 + 分层可控"：降低峰值显存、减少分配开销、让低优先级张量（主要是 KV Cache）能按需释放、能溢出到 Host/SSD。

## 目录约定

```
dev/memory/
├── README.md           # 本文档（入口：目标、环境、操作）
├── PROGRESS.md         # 项目进展时间线（待办 + 完成，含日期）
├── docs/               # 子方向文档（权威方案、画像报告、执行记录，已收拢）
├── dev-prep.sh         # 开发前置一键准备（fetch 全部仓 + 切本地 <用户名>/dev 分支，--dry-run 先看计划）
├── docker-compose.yml  # 本子方向容器配置（-f ../compose.base.yml 合并公共配置）
├── .env.example        # 环境变量模板（cp 成 .env 按需调整）
├── probes/             # 探针/画像脚本（只读，不改造，待入库）
└── benchmarks/         # A/B 对比与负载脚本
```

代码改造主战场不在本目录：**torch_fl `csrc/runtime/allocator/`**（已有多后端结构，加层不破坏）。

## 重要发现（环境坑，V1 过程中实测）

- **首次 attention 调用极慢（13+ 分钟）**：vLLM 加载 24s 极快（triton cache 热后），但第一个请求的 prefill 卡在 attention kernel 首次初始化（AICore 91% 忙、无新编译、cache 不增长），疑似 flag_gems/flagtune autotune + event-timing 回退。→ **基准必须先跑短请求预热**。
- EngineCore 是 spawn 子进程：主进程读不到 `torch_fl.flagos.memory_stats` → 画像依赖**设备级** HBM 采样（aclrtGetMemInfo/npu-smi）+ vLLM 日志。
- `docker exec` 被 kill 时 EngineCore 子进程会残留占卡 → 重跑前 `pkill -f "VLLM::EngineCore"`。
- vLLM usage 上报线程在容器内解析 cpuinfo 报错 → 设 `DO_NOT_TRACK=1`。

## 启动容器（宿主侧）

```bash
# 从公共仓根进入子方向
cd dev/memory
cp .env.example .env    # 按需调整专属开关（默认值即可直接启动）
docker compose -f ../compose.base.yml -f docker-compose.yml up -d
docker ps | grep flagos-fl-dev-910c    # 确认 Up

# 容器内验证挂载（应看到 5 个子库 + dev/ 等公共仓内容）
docker exec -it flagos-fl-dev-910c bash -c "ls /workspace"
```

## 常用命令（环境速查）

```bash
# 进开发容器
docker exec -it flagos-fl-dev-910c bash
# venv311 里跑探针
/root/vllm-venv311/bin/python /workspace/dev/memory/probes/xxx.py
# 推理负载参考（阶段4 已验证链路）
#   见 docs/推理插件接入-阶段4执行记录.md §3
```

## 工作原则

- 遵循"不预实现"原则：先有真实问题数据，再动手优化（8.5）
- 公共红线（不改宿主配置/驱动、多卡前 npu-smi 确认、DrvMng 名额有限）见主 README「红线」节
