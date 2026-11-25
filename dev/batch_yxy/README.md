# batch_yxy 信息分支（FlagScheduler 开发信息仓）

> **分工（2026-11-23 起）**：本分支只管理**开发信息**——各平台的测试数据、
> 平台环境事实、实验/排查记录、进度与待办；**代码交付**在独立仓
> [FlagRT/FlagScheduler](https://github.com/FlagRT/FlagScheduler)（main）。
> 交付基座对齐 runtime-team dev-1.0 的
> [`dev/images/USAGE_MATRIX.md`](https://github.com/FlagRT/runtime-team/blob/dev-1.0/dev/images/USAGE_MATRIX.md)
> （本分支不含该文件，链接指向 dev-1.0）。

## 目录导航（按平台分类——不同平台在不同机器上开发）

| 平台 | 目录 | 机器 |
| --- | --- | --- |
| 昆仑芯 P800 | [`platforms/kunlunxin-p800/`](platforms/kunlunxin-p800/) | P800-A（本机 10.20.20.4 / 公网 81.70.153.29:26008） |
| 昇腾 910C | [`platforms/ascend-910c/`](platforms/ascend-910c/) | 910C 机（历史数据） |
| NVIDIA A6000 | [`platforms/nvidia-a6000/`](platforms/nvidia-a6000/) | GPU 机（历史数据） |

- [`SUMMARY.md`](SUMMARY.md)：跨平台调度/组批数据汇总（迁自 FlagScheduler reports/）
- [`verify-records/`](verify-records/)：仓级验证记录（0.1.0：171 项测试、wheel 构建）
- [`AGENTS-DEV.md`](AGENTS-DEV.md)：平台相关的开发纪律与环境约束
- [`BOOTSTRAP.md`](BOOTSTRAP.md)：新机器迁移清单与两仓同步纪律
- [`DEV-REQUIREMENTS.md`](DEV-REQUIREMENTS.md)：开发要求——各平台的机器与镜像对照、启动核对、设备纪律、结论与证据口径
- [`status.md`](status.md)：拆分前的历史进度看板（**不再更新**，仅供追溯）

## 同步纪律

- 平台数据/进度更新 → 提交到本分支对应平台目录（`platforms/<平台>/`）
- 代码/测试交付 → FlagScheduler 仓（main）
- 本目录下的 `dynamic_batching/`、`sched_engine/`、`flagscheduler/`、`tests/`
  等是**拆分前的历史代码副本，不再维护**——一切以 FlagScheduler 仓为准
