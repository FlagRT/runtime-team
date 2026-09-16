# ascend-train-comm v1 —— 离线归档

> 机器无关清单。实体不在仓库；本机实体的绝对路径见同目录 `BACKUP.local`（不追踪）。

## 压缩包（`docker load < <file>` 还原；gzip）

| 文件 | 大小 | sha256 | 内容 |
|---|---|---|---|
| `ascend-train-comm-0.1.3.tar.gz` | 6.1G | `db999c1ff325117a8a0f8aaef675f25167f9dd0569879a0de6bea82b90a23cac` | 本镜像原件 |

父层的离线包见 `../../ascend-operator-runtime/v1/ARCHIVE.md`。

## 重建依赖资产（`recovered/`）

| 项 | 校验 | 说明 |
|---|---|---|
| `flagcx-vendor/`（flagcx 已装包 + `flagcx-0.13.0.dist-info`, 31M） | `assets/provenance/flagcx-vendor.sha256` | reproB 的 FlagCX vendor 件；`.so` 与原镜像逐字节一致 |
| `flagcx-installed/` + `libflagos.so` | `recovered/flagcx-installed.sha256` | 取证副本 |
| `GOLDEN-*` / `reproB-pipfreeze.txt` | — | 重建比对基线（也在 `assets/provenance/`） |

`flagcx-vendor/` 可从 `*.tar.gz` 重生（`docker load` → `docker cp` site-packages 的 `flagcx` + `flagcx-0.13.0.dist-info`）。

## 用途

构建机或 docker 镜像丢失时的重建来源。配合 `REBUILD.md` + `build.sh` 使用。
