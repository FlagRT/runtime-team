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

## 索引

| id | 芯片 | 归属方向 | 状态 | 目录 |
|---|---|---|---|---|
| `shmem-missing` | 昇腾 910C | 算子编译方向 | 挂起，不跟进（现行两腿标准路径均不触发；缺陷路径仍在训练腿发布物镜像内） | [`shmem-missing/`](shmem-missing/README.md) |
| `flagcx-c10d-collectives` | 昇腾 910C | FlagCX 通信方向 | 未修复，部分可由调用方规避 | [`flagcx-c10d-collectives/`](flagcx-c10d-collectives/README.md) |
| `kunlunxin-gemm-triton-compile` | 昆仑芯 P800 | 算子编译方向 | 未修复，**无可用 workaround** | [`kunlunxin-gemm-triton-compile/`](kunlunxin-gemm-triton-compile/README.md) |

> 问题定义、现状细节、规避手段与验收判据以各目录 README 为准。

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
