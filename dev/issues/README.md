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

| id | 归属方向 | 状态 | 目录 |
|---|---|---|---|
| `shmem-missing` | 算子编译方向 | 未修复，有不彻底的临时规避 | [`shmem-missing/`](shmem-missing/README.md) |
| `flagcx-c10d-collectives` | FlagCX 通信方向 | 未修复，部分可由调用方规避 | [`flagcx-c10d-collectives/`](flagcx-c10d-collectives/README.md) |

> 问题定义、现状细节、规避手段与验收判据以各目录 README 为准。

## 共享机器

昇腾驱动对带卡容器独占，一台机器同时只能有一个带卡容器：多个缺陷的检查须共用一个容器，
以 `ASCEND_RT_VISIBLE_DEVICES` 分区，不要为每项检查各起一个容器。
