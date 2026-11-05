# thead-ppu-runtime v2 —— 离线归档

> 机器无关清单。实体不在仓库（`dev/images/.gitignore` 挡大二进制）；本机实体绝对路径见同目录
> `BACKUP.local`（不追踪）。

## 压缩包

| 文件 | 大小 | sha256 | 内容 |
|---|---|---|---|
| `thead-ppu-runtime-1.0.0-ppu3.6-triton3.6.0-x86_64.image.tar.gz` | 18,409,421,737 字节 = **17.15 GiB**（`docker save` \| `pigz -p 8`） | `6ab0dfb90525a5c55028811bff22ed91d0c8f993fb7981bc6c3c054f5370cff2` | `flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64`（image id `sha256:1ba7ead15378…`）完整镜像（含 layer 历史与元数据） |

产出方式（2026-10-27，本机 PTG_0 平头哥共享机）：

```bash
A=/bmcp_lvm_fs/hliu553/images-archive
IMG=flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64
docker save "$IMG" | pigz -p 8 > "$A/thead-ppu-runtime-1.0.0-ppu3.6-triton3.6.0-x86_64.image.tar.gz"
```

用时约 5 分钟；输出落**数据盘**（`/bmcp_lvm_fs`），根盘零占用（本机根盘长期 99%，见
`dev/device-context/910C/docs/STORAGE_MIGRATION_20261021.md` 里同类纪律）。

## 三重核验（2026-10-27 实测，全部 rc=0）

| # | 核验项 | 命令 | 读数 |
|---|---|---|---|
| 1 | 内容指纹 | `sha256sum <tar.gz>` | `6ab0dfb90525…cff2`（记入 `assets/` 与本文件） |
| 2 | 压缩完整 | `pigz -t <tar.gz>` | **`pigz_rc=0`**（无损坏） |
| 3 | **可还原** | `docker load -i <tar.gz>` | **`load_rc=0`** ⇒ `Loaded image: flagrt/thead-ppu-runtime:1.0.0-…`；**回读后 image id 与发布物逐字相同**（`sha256:1ba7ead15378…`） |

第 3 项是关键：它同时证明「归档不是空壳」和「还原后就是发布物本身」，不靠推断。

## ⚠️ 与 TODO #21 的关系（重要，别被旧结论误导）

[`../../TODO.md`](../../TODO.md) #21 记的是「**巨型镜像 `docker save` 会静默导出空 tar**（14 KB 级）」——
那是 **昆仑芯机** 上的实测，其 Docker 用 **containerd snapshotter** 后端（`Storage Driver: overlayfs` /
`driver-type: io.containerd.snapshotter.v1`）。

**本机（PTG_0）是传统 `overlay2` 后端**（`docker info` → `Storage Driver: overlay2`），
`docker save` **正常**（本例 17.15 GiB、blob 完整、可 `load`）。

⇒ 该缺陷是**存储后端相关**，不是「镜像大就必然失败」。判定顺序：先 `docker info | grep "Storage Driver"`，
`overlay2` 直接 `save`；`overlayfs`+containerd snapshotter 才需 `docker export` 兜底。
已把这条更正写回 TODO #21。

## 还原

```bash
# 目标机需 ≥ ~42 GB 空闲（解包后的镜像层）
docker load -i thead-ppu-runtime-1.0.0-ppu3.6-triton3.6.0-x86_64.image.tar.gz
docker inspect flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64 \
  --format '{{.Id}}'      # 期望 sha256:1ba7ead15378…；若不符则归档被改过
```

还原后按 [`../v2/assets/verify_ppu_v2.sh`](assets/verify_ppu_v2.sh) 复验（四条判据一次跑完）。

## 另一条路径：不用离线包，从配方重建

离线包只是「不求人」的兜底。有该 wheel 的机器可直接：

```bash
WHL_ROOT=/bmcp_lvm_fs/A/data/flagtree-src/dist bash build.sh
```

配方指纹见 [`assets/provenance/baked-versions.txt`](assets/provenance/baked-versions.txt) §6；重建后 image id
**会不同**（本组镜像未 push、无 registry digest），跨机一致性靠「wheel sha256 + 配方指纹」保证，不靠 id。
