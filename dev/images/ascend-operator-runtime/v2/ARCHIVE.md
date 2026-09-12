# ascend-operator-runtime v2 —— 离线归档

> 机器无关清单。实体不在仓库；本机实体的绝对路径见同目录 `BACKUP.local`（不追踪）。

## 压缩包（`docker load < <file>` 还原；gzip）

| 文件 | 大小 | sha256 | 内容 |
|---|---|---|---|
| `ascend-operator-runtime-1.0.0-flagtree3.5.tar.gz` | 10.7G | `47ce2d01ea00cd4e27914f77ad2f76fe46626d59de87e4971b4aa1e7e97a5518` | `flagrt/ascend-operator-runtime:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-arm64` 本镜像原件 |

父层（BAAI·FlagTree 官方 ascend3.5 预构建基座镜像）**未**单独离线归档：
`harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2`
本身是 BAAI·FlagTree 官方公开产物，公开 harbor + 官方离线包镜像
（`https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64.202608-torch2.10.0-vllm0.20.2.tar.gz`，
手册记载地址）双通道可得，不需要本组另存一份止血备份。

## 重建依赖资产

本血统**不需要**离线回收资产（`recovered/`）——所有 custom 层都是构建期从公开
commit / 公开分发桶现场拉取重建（见 `REBUILD.md`），没有"原镜像已删无法回收"
的资产依赖问题。这是相对 v1（依赖回收的 wheelhouse / mpich tar 等）的简化。

## 用途

构建机或 docker 镜像丢失时的兜底：`docker load` 直接还原原始镜像；或按
`REBUILD.md` + `build.sh` 从零重建（推荐，公开可复现）。
