# dev/issues —— 跨方向未决缺陷的 case 包

## 这是什么

本目录发布**由本组（运行时基座）定位、根治责任在其他方向**的缺陷。一个缺陷一个目录，内容是可直接
交付对方的 case 包：问题定义、单入口复现、修复验收判据、证据截选。

与 `dev/images/<name>/v<N>/` 的分工：镜像目录以镜像为单位组织重建档案与**该镜像**的 case；本目录的
缺陷不归属任何一个子方向、也不随镜像版本走，由本组统一发布，因此单独立目录。

## 规则

目录构成、README 三段式（问题与现状 / 复现 / 修复验收）、四条硬规则与自判定三态口径
（`overall: PASS`/0、`FAIL`/1、`ERROR`/2）统一见 [`../images/CASE-TEMPLATE.md`](../images/CASE-TEMPLATE.md)，
本文件不重复。每个缺陷所用的镜像、运行环境、卡数与其他前置条件，由各目录 README 的前置条件表声明；
从零到判定的逐步操作见各目录的 `NOTES-step-by-step.md`。

新增一个缺陷：建目录 → 按 CASE-TEMPLATE 写 README 与复现脚本 → 带卡验证并留证据截选 → 在下表追加一行。

**分版表态**：缺陷不整条关闭。状态一律按**坐标**陈述 —— 坐标 = 镜像血统 + 运行环境修订
（第 0 版 = 冻结的 `demo/`，第 1 版起 = `env/vN`；现行是哪一版以镜像索引与现行锁定声明为准）。
字段口径参照 Debian 的 Found / Fixed 为**版本集合**而非单值。**每发一个运行环境新版，所有未关闭缺陷
必须逐条重新表态**，三态之一：

1. **在该版仍命中**；
2. **在该版已修** —— 必须写明形态：挂载覆盖（启动时以 `pip-editable` 覆盖镜像内副本），还是镜像内固化；
   两种形态互不折算，各自只对自己那一版运行环境成立；
3. **未在该版测过** —— **必须显式写出来**。沉默会被读成「不命中」，这是环境声明失真的主要来源；
   没有该版实测就不得表态为已修或不命中。

各目录 README 的「版本集合」节是该缺陷的权威表态，下表只作摘要。

## 索引

| id | 芯片 | 归属方向 | 分版状态（摘要） | 目录 |
|---|---|---|---|---|
| `shmem-missing` | 昇腾 910C | 算子编译方向 | `ascend-train-comm v3`：第 0 版（`demo/`）**仍命中**；第 1 版（`env/v1`）**未在该版测过**。跟进状态：挂起、不跟进（现行两腿标准路径均不触发），不作为现行发版限制条目 | [`shmem-missing/`](shmem-missing/README.md) |
| `flagcx-c10d-collectives` | 昇腾 910C | FlagCX 通信方向 | `ascend-train-comm v3`：第 0 版（`demo/`）**仍命中**；第 1 版（`env/v1`）**已修 —— 形态为挂载覆盖**（宿主 FlagCX 须处于含修复的提交）。**镜像内固化形态未交付**，该形态未经验收 | [`flagcx-c10d-collectives/`](flagcx-c10d-collectives/README.md) |
| `kunlunxin-gemm-triton-compile` | 昆仑芯 P800 | 算子编译方向 | `kunlun-operator-runtime v2`：第 0 版（`demo/`）**仍命中**；`env/v1`（显式启用 GEMM compat 后）**编译维度已修**（12/12 编译 + pytest 18/99，2026-10-28；形态 = 挂载补丁版 FlagGems + 编译组 compat 包进程内加载）。`v1`：**未在该版测过**（指本 case 的判据） | [`kunlunxin-gemm-triton-compile/`](kunlunxin-gemm-triton-compile/README.md) |
| `kunlunxin-gemm-numeric-64cube` | 昆仑芯 P800 | 算子编译方向（口径裁定线归 performance） | 64×64×64 fp32 **非确定性乱值**（demo 与 `env/v1` 均命中——缺陷在 compiler/kernel 本体，与运行环境版本无关；同输入三次 rel 0.19~1.00、一次全零，vendor 原生 1.2e-5）+ 64³ bf16 超差 + 大形状 fp16/bf16 绝对容差口径之争（数据在案、转裁定）。未修复；已同步编译组、被动等进展（K-5 / B15） | [`kunlunxin-gemm-numeric-64cube/`](kunlunxin-gemm-numeric-64cube/README.md) |

> 问题定义、现状细节、规避手段、验收判据与完整的版本集合以各目录 README 为准；本表只摘要，
> 同一事实以各目录 README 的「版本集合」节为准。

## 共享机器

**设备模型按芯片实测，不要跨芯片照抄**——两款芯片在这点上恰好相反，照抄会得出错误的操作纪律。

| | 昇腾 910C | 昆仑芯 P800 |
|---|---|---|
| 带卡容器独占 | **是**。一台机器同时只能有一个持有 `/dev/davinci*` 的容器 | **否**（实测 9 个容器并存、同卡多容器同时映射仍可用） |
| 多个缺陷的检查 | 须**共用一个容器**，以 `ASCEND_RT_VISIBLE_DEVICES` 分区，不要各起一个 | 各起各的无妨；真实约束是显存与算力竞争 |
| 开工前 | `docker ps` 是**阻塞条件**：有他人带卡容器就停手、约时间 | `docker ps` 只是礼貌；用 `xpu-smi` 挑 0% 利用率 / 0MiB 显存的卡，跑负载时以 `CUDA_VISIBLE_DEVICES` 自律 |
| 卡号 | `ASCEND_RT_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES`（**设备按授予顺序从 0 密集重编号**：只授予子集时须写重编号后的逻辑号） |

两边共同的一条：**绝不**停止 / 重启 / `exec` 进入任何他人容器，用完把自己的收干净。
P800 的四条实测结论与证据见
[`../images/kunlun-operator-runtime/v2/demo/docker-compose.yml`](../images/kunlun-operator-runtime/v2/demo/docker-compose.yml) 文件头。

<!-- changelog:
2026-10-22 —— 依据指挥台 ADR-0008《运行环境目录版本化（env/vN）与 demo 冻结》：
  ① 「规则」段新增「分版表态」：缺陷不整条关闭，状态按坐标（血统 + 运行环境修订）陈述，
     每发一个运行环境新版，所有未关闭缺陷逐条重新表态，三态之一，且「未在该版测过」必须显式写出
     —— 沉默会被读成「不命中」，是环境声明失真的主要来源。字段口径参照 Debian 的 Found/Fixed
     为版本集合而非单值。
  ② 索引表「状态」列改为「分版状态（摘要）」，三行按新口径重写，与各目录 README 新增的
     「版本集合」节逐条对齐（flagcx：第 0 版命中 / env/v1 已修-挂载覆盖 / 镜像内固化未交付；
     shmem：第 0 版命中 / env/v1 未测过；kunlunxin：v2 第 0 版命中 / 该血统无 env/vN / v1 未测过）。
  本批无 docker，未重跑任何验证，表中读数全部沿用既有证据。
-->

