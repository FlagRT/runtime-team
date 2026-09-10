# memory · STATUS

> 总组速览入口（战略文档 §8.2 格式）｜ 详细看板见 [PROGRESS.md](PROGRESS.md) / [README.md](README.md)
> 最近更新：2026-09-10

## 当前阶段

- 对应 §3 验收标准**第 4 条**（显存池定义 + 显存占用峰值画像；KV 相关不作硬性指标）。
- §5 memory 任务清单进度：
  1. 锁定推理镜像显存画像 —— **环境侦察完成**（栈版本已核对：py3.11.15 / vllm 0.20.2 / vllm_ascend 0.20.2rc1 / torch 2.10.0 / torch_npu 2.10.0 / CANN 9.0.0 / SOC ascend910_9391 / 单芯 HBM 64GiB），**探针已移植未测**（`infer910c_hbm_sampler.py` + `infer910c_mem_profile.py` + `infer910c_ab_matrix.py`，旧 flagos/xpytorch 探针不适用），报告骨架已建（`docs/910C-显存画像报告-骨架.md`）。**阻塞在带卡容器 slot**。
  2. 显存池定义文档 + 对照数据 —— A/B 矩阵脚本就绪，待起容器跑数。
  3. 分层缓存/Host 溢出 —— 本期非硬性指标，routeA-S4 留档即可。

## 关键阻塞

- **带卡容器并发上限 3，当前 3 个已占满**（stack.lock rule #1）。需总组协调一个 slot 起 `flagos-proto-infer-910c`（推理腿，串行即可；跑完即停）。在此之前所有画像/对照数据无法采集。

## 最近更新日期

2026-09-10
