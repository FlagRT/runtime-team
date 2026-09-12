# ascend-train-comm v2 —— 离线归档

> 机器无关清单。实体不在仓库；本机实体的绝对路径见同目录 `BACKUP.local`（不追踪）。

## 压缩包（`docker load < <file>` 还原；gzip）

| 文件 | 大小 | sha256 | 内容 |
|---|---|---|---|
| `ascend-train-comm-1.0.0-flagtree3.5.tar.gz` | 10.8G | `082aa0393b6a014cb5c3d836c8bf93919a2fac5f4f582806c3aad41a6467cb5d` | `flagrt/ascend-operator-runtime-comm:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64` 本镜像原件 |

父层的离线包见 `../../ascend-operator-runtime/v2/ARCHIVE.md`。

## 重建依赖资产

本血统**不需要**离线回收资产——FlagCX 层从 `FlagRT/FlagCX` 组织仓公开主干 tip
（commit `4e0e0cbcbf721169ca82348080f8353aebfe2c31`）构建期直接源码编译，没有
需要回收的已装包、没有私有 patch。这是相对 v1（依赖回收 `flagcx-vendor/` 已装包）
的核心简化，也是 v2 相比 v1 的可复现性改进点（见 `lock.yaml`）。

## 用途

构建机或 docker 镜像丢失时的兜底：`docker load` 直接还原原始镜像；或按
`REBUILD.md` + `build.sh` 从零重建（推荐，公开可复现）。
