# memory — 显存与缓存管理项目

> **状态：🟢 推进中** ｜ 权威方案：《[显存与缓存管理方案-20260822](docs/路线A-显存与缓存管理-方案-20260822.md)》
> 本文档 = 子方向**入口**（目标、环境、操作）；**进展/待办/时间线见 [PROGRESS.md](PROGRESS.md)**，文档收拢在 `docs/`。
> 历史归档（已冻结路线，勿作新工作基线）见 [docs/archive/](docs/archive/README.md)。

## 目标（一句话）

在 FlagOS 官方栈（各芯片厂商设备插件 + FlagGems + FlagCX + vllm-plugin-FL）上做显存分层管理与可控溢出：降低峰值显存、减少分配开销、让低优先级张量（主要是 KV Cache）能按需释放、能溢出到 Host/SSD。

## 目录约定

```
dev/memory/
├── README.md               # 本文档（入口：目标、环境、操作）
├── PROGRESS.md             # 项目进展时间线（待办 + 完成，含日期）
├── docs/                   # 子方向文档（权威方案、画像报告、执行记录）
├── docs/archive/           # 历史归档（已冻结路线，勿作新工作基线）
├── dev-prep.sh             # 开发前置一键准备（fetch 全部仓 + 切本地 <用户名>/dev 分支，--dry-run 先看计划）
├── docker-compose.yml      # 昇腾 910c 容器配置
├── docker-compose.p800.yml # 昆仑芯 P800 容器配置
├── .env.example            # 环境变量模板（cp 成 .env 按需调整）
├── probes/                 # 探针/画像脚本（只读，不改造，待入库）
└── benchmarks/             # 负载与对比脚本
```

代码改造主战场：**vLLM 层**（显存池/缓存管理在 vLLM 内做）；厂商 torch 自带分配器为底座（P800 上 torch_plugin 无独立显存池）。

## 重要发现（环境坑，实测）

**昇腾 910c**

- **首次 attention 调用极慢（13+ 分钟）**：vLLM 加载 24s 极快（triton cache 热后），但第一个请求的 prefill 卡在 attention kernel 首次初始化（AICore 91% 忙、无新编译、cache 不增长），疑似 flag_gems/flagtune autotune + event-timing 回退。→ **基准必须先跑短请求预热**。
- EngineCore 是 spawn 子进程：主进程读不到进程内显存计数 → 画像依赖**设备级** HBM 采样（aclrtGetMemInfo/npu-smi）+ vLLM 日志。
- `docker exec` 被 kill 时 EngineCore 子进程会残留占卡 → 重跑前 `pkill -f "VLLM::EngineCore"`。
- vLLM usage 上报线程在容器内解析 cpuinfo 报错 → 设 `DO_NOT_TRACK=1`。

**昆仑芯 P800（2026-08-21~22 实测）**

- **triton 版本偏差致 FlagGems GEMM 崩溃**：dev 容器 triton 3.6.0（flagtree 0.6.1+xpu3.6）≠ 官方发布镜像 3.0.0，FlagGems mm/bmm/addmm 在 dev 容器 SIGABRT（编译期 make_llir），官方发布镜像同 commit 全过 → **结论性测试在官方发布镜像内做**。
- **MoE 双阻塞**：`xpudnn::causal_conv1d_update ret=1`（厂商算子库缺口，混合注意力模型昆仑芯不可用）；`flag_gems._kunlunxin.topk_softmax` 缺 `renormalize` 参（组件版本配对，纯 MoE 必崩）→ 见 [昆仑芯问题反馈清单-20260822](docs/昆仑芯问题反馈清单-20260822.md)。
- **KV 预分配 P800 69.22GiB / 504k tokens**（Qwen3-4B，gpu_mem_util 0.9，加载合计 84.5s、含 graph 71.5s）。
- **decode 生成退化根因**：厂商 `patch_decode_attention`（decode 无条件替换为 prefix-cache prefill_attention）→ 禁用后正常、解码提速近 2x → [新线栈decode生成退化-根因定位-20260901](docs/新线栈decode生成退化-根因定位-20260901.md)
- **vllm 0.13 官方 KV CPU 卸载可用（P800）**：`--kv-offloading-size` 有 num_cpu_blocks=0 接线缺陷，须显式 KVTransferConfig；store/load 双向实测通过、吞吐代价 ~2.4% → [vllm-0.13-allocator与offload调研-20260822](docs/vllm-0.13-allocator与offload调研-20260822.md) §4
- **910C（vllm 0.20.2）官方 native KV 卸载不可用**：`is_cuda_alike()` 平台门 + `vllm._C` 缺 libcudart → [routeA-S4-KV卸载Host-910C尝试-20260903](docs/routeA-S4-KV卸载Host-910C尝试-20260903.md)

## 启动容器（宿主侧）

**昇腾 910c**

```bash
# 从公共仓根进入子方向
cd dev/memory
cp .env.example .env    # 按需调整专属开关（默认值即可直接启动）
docker compose -f ../compose.base.yml -f docker-compose.yml up -d
docker ps | grep flagos-fl-dev-910c    # 确认 Up

# 容器内验证挂载（应看到 6 个子库 + dev/ 等公共仓内容）
docker exec -it flagos-fl-dev-910c bash -c "ls /workspace"
```

**昆仑芯 P800**

```bash
docker compose -f ../compose.base.yml -f docker-compose.p800.yml up -d
docker ps | grep flagos-fl-dev-p800    # 确认 Up
# ⚠️ 注意：dev 容器 triton 3.6.0 与官方发布镜像 3.0.0 有偏差，FlagGems GEMM 会编译崩溃；
#    结论性测试在官方发布镜像内做，本 dev 容器仅用于开发调试。
```

## P800 运行环境（实测口径 2026-08-21/22）

**venv 组合**（官方发布镜像内）：python 3.10.18 + torch 2.9.0+cu129（xpytorch，CUDA 兼容 USE_CUDA=ON）+ vllm 0.13.0（昆仑芯版，插件强依赖做平台引导）+ vllm-plugin-fl 0.1.0 + flag_gems 4.2.1rc0（editable）+ flagcx 0.10.0（klx 适配器）+ flagtree 0.6.1+xpu3.6（dev，triton 3.6.0；官方发布镜像为 triton 3.0.0）+ xtorch_ops 0.1.2640 + torch_xray 2.0.4 + xmlir 1.0.0.1 + torch_plugin 0.1.0（仅 runtime 初始化，无独立显存池）

**运行环境变量**：

```bash
VLLM_PLUGINS=fl
VLLM_FL_PLATFORM=kunlunxin
VLLM_FL_PREFER=flagos|vendor    # vllm 0.13 口径；vllm 0.20.2 须单独 flagos，见问题清单 #5
USE_FLAGGEMS=1
GEMS_VENDOR=kunlunxin
KLX_USE_AUTOTUNE=0
CUDA_VISIBLE_DEVICES=<选卡>
DO_NOT_TRACK=1
FLAGCX_PATH=/workspace/FlagCX/plugin/torch
```

> 注：vllm 0.13 昆仑芯构建必须靠 vllm-plugin-FL 做平台引导——不带插件报 `Device string must not be empty`。

## 常用命令（环境速查）

```bash
# 昇腾 910c
docker exec -it flagos-fl-dev-910c bash
/root/vllm-venv311/bin/python /workspace/dev/memory/probes/xxx.py

# 昆仑芯 P800
docker exec -it flagos-fl-dev-p800 bash
conda activate python310_torch29_cuda
```

## 工作原则

- 遵循“不预实现”原则：先有真实问题数据，再动手优化（8.5）
- 测试在官方发布镜像内进行（dev 容器仅用于开发调试；triton 版本偏差会污染结论）
- 公共红线（不改宿主配置/驱动、多卡前 npu-smi 确认、DrvMng 名额有限）见主 README「红线」节
