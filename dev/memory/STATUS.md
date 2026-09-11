# memory · STATUS

> 总组速览入口（战略文档 §8.2 格式）｜ 详细看板见 [PROGRESS.md](PROGRESS.md) / [README.md](README.md)
> 最近更新：2026-09-10

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

## 关键结论（供总组 / device-context / 调度）

- **embedding-on-vLLM 的 HBM 峰值 ≈ driver 2.82 + 权重 1.13 + `gpu_memory_utilization` × 空闲HBM(61.3 GiB) + 激活 0.2–0.57 + graph 0–0.1（GiB）**。
- **峰值几乎与 batch / seq-len / max_model_len 无关**；唯一大杠杆是 `gpu_memory_utilization`。
- vLLM 对 pooling runner 仍按 gmu 吃满空闲 HBM 建 "KV cache" 池（gmu0.9 → 53.79 GiB / 503,552 tok），
  但 embedding 永不使用 → 纯余量。**gmu 0.4 实测 HBM 峰值 42.8%（28 GiB），吞吐与 0.9 无差**。
- eager vs ACLGraph：显存差仅 0.2 GiB，是时延（load +15s）↔吞吐（+8%）权衡，非显存权衡。
- `expandable_segments:True` 对 embedding 无可测效果（池是整块分配，无碎片）。
- 预热 4.7s（旧栈生成式首次 attention 是 437s）。

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
