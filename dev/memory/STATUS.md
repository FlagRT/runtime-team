# memory · STATUS

> 总组速览入口（战略文档 §8.2 格式）｜ 详细看板见 [PROGRESS.md](PROGRESS.md) / [README.md](README.md)
> 最近更新：2026-09-27

## 设备接入路线

- **当前路线：Route A**。memory 不走 `torch_fl` 设备层；统一基于各芯片厂商官方设备插件接入，再使用 FlagGems、FlagCX 与 vllm-plugin-FL 等 FlagOS 组件。
- 路线变更依据：[FlagOS 设备层路线变更指南](docs/common/policy_设备层路线变更指南.md)。该文档明确 2026-08-22 起生产交付由 B 切换为 A，2026-09-03 起 `torch_fl` 设备层路线冻结。
- `docs/goals/legacy-2.4-910c/` 下的 `torch_fl` 内容仅作历史验证资产，不作为 memory 当前开发基线。

## 当前阶段

- 对应 §3 验收标准**第 4 条**（显存池定义 + 显存占用峰值画像；KV 相关不作硬性指标）。
- §5 memory 任务清单进度：
  1. 锁定推理镜像显存画像 —— ✅ **完成**（2026-09-10，npu1-27 davinci-7，锁定镜像内实测）。
     环境核对 + embedding API surface 实测 + 加载阶段 HBM 分解 + 运行阶段峰值 sweep（batch×seq-len）+ A/B 三轴。
     报告：[docs/goals/proto-910c-202609/profile_显存画像_910c.md](docs/goals/proto-910c-202609/profile_显存画像_910c.md)（已从骨架转正）。探针已实测通过；
     `hbm-sampler_910c.py` 解析器按 npu-smi 25.5.0 A3 版式修过（按 Phy-ID 做键）。
  2. 显存池定义文档 + 对照数据 —— ✅ **完成**。[docs/goals/proto-910c-202609/design_显存池定义_910c.md](docs/goals/proto-910c-202609/design_显存池定义_910c.md)：
     两层池（torch_npu caching allocator 底座 + vLLM 层）、gpu_memory_utilization/pooling 预分配/
     ACLGraph capture、复用回收、A/B 对照表、**给 device-context/调度 的安全区间**
     （embedding 服务：gmu 0.35–0.45、max_num_seqs 64–128）。原始数据 `benchmarks/out/`。
  3. 分层缓存/Host 溢出 —— 本期非硬性指标，routeA-S4 留档即可（对 embedding 无意义：预分配池是余量非需求）。
  4. 生成式模型显存画像 —— ✅ **完成**（2026-09-23，npu1-27 davinci1，标准环境 `serve_standard.sh`，910C 不启用 FL）。
     主模型 `OpenOneRec/OneRec-8B`（revision `29f95b3d`，已落共享盘 `/data_lib/models/OneRec-8B`）；
     冒烟 PASS（model_list 首条生成记录）+ A(加载)/C(decode 增长)/E(归还闭环)/B(输入长度)/D(并发)/F(gmu 对照) 全矩阵。
     **双指标方法论**：设备 HBM（npu-smi）+ 引擎 KV 使用率（/metrics `kv_cache_usage_perc`）同步采集。
     报告：[docs/goals/proto-910c-202609/profile_生成式显存画像_910c.md](docs/goals/proto-910c-202609/profile_生成式显存画像_910c.md)；
     探针 [probes/910c/](probes/910c/) gen-{load,matrix,parse}；原始数据 `benchmarks/out/gen910c/`（gitignored，服务器本地）。

## 关键结论（供总组 / device-context / 调度）

- **embedding-on-vLLM 的 HBM 峰值 ≈ driver 2.82 + 权重 1.13 + `gpu_memory_utilization` × 空闲HBM(61.3 GiB) + 激活 0.2–0.57 + graph 0–0.1（GiB）**。
- **峰值几乎与 batch / seq-len / max_model_len 无关**；唯一大杠杆是 `gpu_memory_utilization`。
- vLLM 对 pooling runner 仍按 gmu 吃满空闲 HBM 建 "KV cache" 池（gmu0.9 → 53.79 GiB / 503,552 tok），
  但 embedding 永不使用 → 纯余量。**gmu 0.4 实测 HBM 峰值 42.8%（28 GiB），吞吐与 0.9 无差**。
- eager vs ACLGraph：显存差仅 0.2 GiB，是时延（load +15s）↔吞吐（+8%）权衡，非显存权衡。
- `expandable_segments:True` 对 embedding 无可测效果（池是整块分配，无碎片）。
- 预热 4.7s（旧栈生成式首次 attention 是 437s）。

**生成式（OneRec-8B，2026-09-23）追加**：
- **生成式账本与 embedding 同构**：HBM ≈ idle 2.82 + 权重 + gmu×可见HBM + 激活 ~0.3（GiB）。
  OneRec-8B 权重 15.66 GiB、KV 池 39.23 GiB / 285,568 tok、HBM 峰值 90.7%；激活两种负载下都是 0.2–0.6 GiB 零头。
- **双指标口径（判泄漏/判容量的标准）**：预分配使设备 HBM 恒平（实测波动仅 142 MiB），KV 块涨落只在
  `/metrics kv_cache_usage_perc` 可见（0→0.4%→0）；单看 npu-smi 会误判"无动态"或"泄漏"。
- **单 token KV 成本 0.14 MiB/token**（OneRec-8B / TP=1，结算行与压测两口径互证）→ 最大并发 ≈ 池 tokens ÷ 单请求上下文；
  gmu（动分子）与 max_model_len（动分母）对并发是等比例的两个旋钮。
- **gmu 语义随负载变化**：embedding 的池是纯余量（可放心压 0.4）；生成式的池是真实并发容量
  （0.9→0.4 并发 69.7×→15.3×）。共享机实验建议 gmu0.4，独占卡生产 0.9。
- 归还闭环健康：完成/客户端取消均回收 KV 块、无累积；标准栈上 Memory 侧暂无适配缺口（调度器抢占未测）。

## 给下游子方向的建议（Action Items，待接收）

> 以下三项是 memory 本期产出里**别的方向用得上、但目前只写在 memory 自己文档里**的部分。仓库里没有找到 `调度`/`监控` 的子方向目录（截至 2026-09-11，`dev/` 下只有 `communication` / `device-context` / `memory` / `performance` / `rag_ljy`），无法直接对接到具体目录，先在此列清楚，等总组确认接收方或该方向建目录后对接。

| # | 给谁 | 内容 | 状态 |
|---|---|---|---|
| 1 | **device-context**（段二显存参数） | 段二 embedding 服务建议起服务参数：`gpu_memory_utilization=0.35~0.45`（推荐 0.4）、`max_num_seqs=128`、`enforce_eager=False`（默认）。依据见 [design_显存池定义_910c.md](docs/goals/proto-910c-202609/design_显存池定义_910c.md) §6。不采纳的实际后果：默认 `gmu=0.9` 会把单卡 HBM 打到 91%，带卡容器抢名额/显存时（本次画像就因此被卡三轮）没有回旋余地。 | 待接收 |
| 2 | **调度**（请求调度侧） | 同上参数里 `max_num_seqs=128` 直接影响调度侧的并发/排队策略；HBM 峰值几乎不随 batch/并发变化（见同文档 §3.4），排队策略不必为"显存不够"保守让路。 | 待接收（该方向暂无仓库目录） |
| 3 | **监控**（§5 监控任务 2：内存采集脚本） | 建议直接复用 [`probes/910c/hbm-sampler_910c.py`](probes/910c/hbm-sampler_910c.py)（宿主侧轮询 npu-smi，已针对本机 npu-smi 25.5.0 版式修过解析 bug，采过 HBM+AICore），而不是各写一份——避免口径不一致、避免监控重踩已经踩过的解析器坑。错误注入闭环若涉及 OOM 类注入，此脚本可直接复用抓 HBM 证据。生成式服务请**同时采 npu-smi 与 /metrics `kv_cache_usage_perc`**（[`probes/910c/gen-load_910c.py`](probes/910c/gen-load_910c.py) 已实现，可直接复用）。 | 待接收（该方向暂无仓库目录） |
| 4 | **device-context / performance**（生成式服务参数） | OneRec-8B@910C 实测参数建议：独占卡 `gmu=0.9`（69.7 路 × 4096 tok，c64 无压力）；共享卡 `gmu=0.4`（15 路，HBM 42.9%，c8 轻负载实测够用）；若业务生成短，优先砍 `MAX_MODEL_LEN`（4096→1024 并发 ×4，零显存代价）。依据见 [profile_生成式显存画像_910c.md](docs/goals/proto-910c-202609/profile_生成式显存画像_910c.md) §10-11。 | 待接收 |

**给 performance（§5 任务 4：统一验收报告）的现成证据块**（可直接引用/粘贴，格式对齐 `dev/performance/README.md` 的「当前进展」表）：

| 能力 | 当前状态 | 已验证范围 | 交付状态 |
|---|---|---|---|
| 显存池定义 + 画像（910C 锁定镜像） | 已完成 | 锁定镜像 `vllm-ascend:v0.20.2rc1-a3`、`Qwen/Qwen3-Embedding-0.6B`，davinci-7 单卡；加载阶段 HBM 分解 + batch(8-256)×seqlen(128-512) 峰值 sweep + gmu/enforce-eager/alloc-conf 三轴 A/B，全部实测（2026-09-10/11） | 分支 `xliu969/memory-docs-reorg`（尚未合 `dev-1.0`），证据：[profile_显存画像_910c.md](docs/goals/proto-910c-202609/profile_显存画像_910c.md)、[design_显存池定义_910c.md](docs/goals/proto-910c-202609/design_显存池定义_910c.md) |
| KV 分层缓存 / Host 溢出 | 未验证（本期非硬指标） | 验收模型为 embedding，无生成式缓存增长场景，该能力对本次验收模型无意义 | 不在本期交付范围（§7 已注明），910C native 路径阻塞留档见 [note_KV卸载Host尝试_910c.md](docs/goals/proto-910c-202609/note_KV卸载Host尝试_910c.md) |
| 生成式显存画像（910C 标准环境） | 已完成 | `OpenOneRec/OneRec-8B`（revision `29f95b3d`），davinci1 单卡，`serve_standard.sh`（TP=1 / MAX_MODEL_LEN=4096 / EAGER=1）；加载基线 + decode KV 线性增长 + 完成/取消归还闭环 + 输入长度/并发对照 + gmu 0.9/0.4 对照，双指标（HBM + /metrics）全实测（2026-09-23） | 已合入 `dev-1.0`（merge `d7d6b1c`，2026-10-09），证据：[profile_生成式显存画像_910c.md](docs/goals/proto-910c-202609/profile_生成式显存画像_910c.md)、原始数据 `benchmarks/out/gen910c/` |

## 跨方向反馈（待总组收拢，本次画像顺带实测出的环境事实）

| # | 发现 | 建议处理方 | 处理建议 |
|---|---|---|---|
| 1 | 带卡容器并发**实测上限 2**（`x-benchmark` 跑多进程 sweep 时）：第 3 个容器 acl/dcmi init 报 `-8020 device is used` + `DrvMngGetConsoleLogLevel failed ret=4`，与 `stack.lock` rule #1 写的「上限 3」不符——本次因此被卡了三轮（起容器→排查→等窗口）。可能真实上限是「≈3 个 DrvMng 客户端」而非「3 个容器」，`x-benchmark` 多进程一次占多个槽。 | 总组 | 核实 + 更新 `stack.lock` rule #1 表述；跑结论性验证前先确认同机其他容器的进程数，不只数容器数 |
| 2 | house 起容器脚本（如 `distributed_inference/start_infer_container.sh` 一类）的驱动绑定用 `:ro`，在本机 driver 25.5.0 下会 `DrvMngGetConsoleLogLevel failed ret=4` / `device_count()=0`；实测须改 `:rw` 才能起。 | device-context | 检查并修正相关起容器脚本的驱动挂载模式 |
| 3 | 锁定推理镜像 `vllm-ascend:v0.20.2rc1-a3` 内 `transformers` 实测自带 **5.5.3**（战略文档 §7 提到的两处冲突记录 5.15.1 / 5.5.3 之一，本次是权威实测值）。 | device-context | 按 §7「device-context 负责固定版本并回写 stack.lock」处理 |

## 遗留 / 环境备忘

- `flagos-proto-infer-910c` 画像跑完已 `docker rm -f` 拆除，davinci-7 归还 idle。
- 画像期间为腾容器名额，`rag-ljy-vllm-910c` 被临时停过（memory 未重启，需 rag_ljy 或总组视情况处理）。

## 最近更新日期

2026-09-11
