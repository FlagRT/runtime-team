# ascend-operator-runtime v1 —— 离线归档

> 机器无关清单。实体不在仓库；本机实体的绝对路径见同目录 `BACKUP.local`（不追踪）。

## 压缩包（`docker load < <file>` 还原；均 gzip）

| 文件 | 大小 | sha256 | 内容 |
|---|---|---|---|
| `ascend-operator-runtime-0.2.0.tar.gz` | 6.1G | `413eb030340b71670f0a83e167bd609b28157f7d4db3ce6489255321e300b133` | 本镜像原件 |
| `cann-base-manual-20260807.tar.gz` | ~4.4G | `3ea6cfa837d677c638f895187fbfbb894d9615bc361aa40e69b6e3aba14ade70` | 父层 CANN 9.0.0 基座（`= sha256:a36a3022`）；此前只在 docker 本地 + BAAI harbor |

## 重建依赖资产（`recovered/`）

| 项 | 校验 | 说明 |
|---|---|---|
| `wheelhouse/`（36 wheel, 574M） | `assets/wheelhouse.sha256` | reproB 的离线 pip 源 |
| `mpich-4.1.3.tar.gz`（39M） | `assets/mpich-4.1.3.tar.gz.sha256` | MPICH 源码编译 |
| `GOLDEN-*` / `reproB-pipfreeze.txt` | — | 重建比对基线（也在 `assets/provenance/`） |

`recovered/` 全部可从 `*.tar.gz` 重生（`docker load` → `docker create` → `docker cp /opt/flagrt/wheelhouse`）。

## 用途

构建机或 docker 镜像丢失时的重建来源。配合 `REBUILD.md` + `build.sh` 使用。
