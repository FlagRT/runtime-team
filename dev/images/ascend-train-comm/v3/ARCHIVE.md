# ascend-train-comm v3 —— 离线归档

> 机器无关清单。**本阶段（phase 1，设计/起草）尚无任何镜像实体**——本文件与
> v1/v2 的 ARCHIVE.md 保持同构，是为 phase 2 真机构建之后立即可填的骨架。

## 压缩包（`docker load < <file>` 还原；gzip）

| 文件 | 大小 | sha256 | 内容 |
|---|---|---|---|
| （待 phase 2 构建后填）`ascend-train-comm-2.0.0-flagtree3.5-routeA.tar.gz` | 待填 | 待填 | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64` 本镜像原件 |

父层的离线包见 `../../ascend-operator-runtime/v3/ARCHIVE.md`（同样待 phase 2 补齐）。

## 重建依赖资产

与 v2 相同：本血统**不需要**离线回收资产——FlagCX 层从 `FlagRT/FlagCX` 组织仓
公开主干 commit `4e0e0cbcbf721169ca82348080f8353aebfe2c31`（与 v2 相同，未升级）
构建期直接源码编译，没有需要回收的已装包、没有私有 patch。

## 用途

构建机或 docker 镜像丢失时的兜底：`docker load` 直接还原原始镜像（**phase 2
构建完成后才有实体可 load**）；或按 `REBUILD.md` + `build.sh` 从零重建（推荐，
公开可复现）。
