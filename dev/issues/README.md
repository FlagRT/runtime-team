# dev/issues —— 跨方向未决缺陷的 case 包

## 这是什么

本目录发布**由本组（运行时基座）定位、根治责任在其他方向**的缺陷。一个缺陷一个目录，内容是可直接
交付对方的 case 包：问题定义、单入口复现、修复验收判据、证据截选。

与 `dev/images/<name>/v<N>/` 的分工：镜像目录以镜像为单位组织重建档案与**该镜像**的 case
（标准见 [`../images/CASE-TEMPLATE.md`](../images/CASE-TEMPLATE.md)）；本目录的缺陷不归属任何一个
子方向、也不随镜像版本走，由本组统一发布，因此单独立目录。目录构成与四条硬规则沿用同一标准。

## 索引

| id | 问题 | 归属方向 | 状态 | 目录 |
|---|---|---|---|---|
| `shmem-missing` | FlagTree triton（ascend 线）tle 顶层无条件 `import shmem`，任何含 for 循环的 Triton kernel 编译失败 | 算子编译方向 | 未修复；有临时规避（占位模块 + 算子黑名单），不覆盖 vLLM 自带 kernel | [`shmem-missing/`](shmem-missing/README.md) |
| `flagcx-c10d-collectives` | FlagCX c10d collective 两类缺陷：① 异步返回（同步后正确）；② 列表形式 `all_gather` 结果错位（同步无效） | FlagCX 通信方向 | 未修复；① 可由调用方显式同步规避，② 只能避开列表形式；原生 DDP 不可用 | [`flagcx-c10d-collectives/`](flagcx-c10d-collectives/README.md) |

## 统一判定约定

每个目录的复现脚本同时是修复验收脚本：同一条命令，当前退出码 1（问题存在），对方修复后退出码 0 即验收通过。
判定行与退出码（CASE-TEMPLATE §2 规则 3 的自判定要求）：

| 判定行 | 退出码 | 含义 |
|---|---|---|
| `overall: PASS` | 0 | 缺陷不存在 —— 修复验收通过 |
| `overall: FAIL` | 1 | 缺陷复现（明细行给出具体现象，如 `repro: CONFIRMED`） |
| `overall: ERROR` | 2 | 脚本无法下判定：环境不满足或撞到另一个问题，不计入复现与验收 |

执行者只读判定行与退出码，无需人工判读日志。

## 共同运行环境

两个缺陷都在 v3 完整镜像的 demo 运行环境上复现与验收（环境基线机制见 [`../ENV-SPEC.md`](../ENV-SPEC.md) §3.2/§4.3）：
[`../images/ascend-train-comm/v3/demo/`](../images/ascend-train-comm/v3/demo/docker-compose.yml)。
昇腾驱动对带卡容器独占，本机同时只能有一个带卡容器，两项检查共用一个容器、以
`ASCEND_RT_VISIBLE_DEVICES` 分区。从零到判定的逐步操作见各目录的 `NOTES-step-by-step.md`。
