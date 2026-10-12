# kunlun-operator-runtime v2 —— 离线归档

> 机器无关清单。实体不在仓库；本机实体的绝对路径见同目录 `BACKUP.local`（不追踪）。

## ⚠️ 还原方式是 `docker import`，不是 `docker load`

与 [`../v1/ARCHIVE.md`](../v1/ARCHIVE.md) 同理：本机 Docker Engine 29.1.3 使用 containerd
snapshotter 存储后端（`docker info` → `Storage Driver: overlayfs` /
`driver-type: io.containerd.snapshotter.v1`）。该后端下 `docker save` 对本镜像只导出
**manifest blob 一个**，产出 tar 仅 14336 字节，`docker load` 无法重建镜像。
2026-10-11 与 2026-10-12 两轮实测均如此，且**与「镜像是不是 `docker commit` 产物」无关**
（v2 是正规 `docker build` 产物、同样失败）。机制定位进展见
[`../v1/lock.yaml`](../v1/lock.yaml) `known_issues: docker-save-broken-on-this-host`
与 [`../TODO.md`](../../TODO.md) #21。

因此本条目用 `docker export`（导出容器展平后的 rootfs 为单层 tar）+ `docker import --change`
重放 `ENV`/`CMD`/`WORKDIR`/`LABEL` 元数据。代价：镜像 layer 历史丢失（`docker history`
只会显示一层），但 rootfs 内容与运行行为与原镜像一致。若目标机器的 Docker 不使用
containerd snapshotter 后端，`docker save`/`docker load` 应正常工作，可按
[`REBUILD.md`](REBUILD.md) + `build.sh` 重新构建后另存一份标准 `docker save` tar。

## 压缩包

| 文件 | 大小 | sha256 | 内容 |
|---|---|---|---|
| `kunlun-operator-runtime-2.0.0-rootfs.tar.gz` | 33065409282 字节 = 30.80 GiB（gzip 压缩后；`docker export` 展平 rootfs \| `pigz -p 32`） | `2083766411e2a048b9ce55d9f03cd44857f7f1191f03401fafd81cd46412d3d5` | `flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`（image id `sha256:cac5ddae5f62…`）的完整 rootfs |

产出方式（2026-10-12，本机 VM-0-2-ubuntu）：

```bash
# 专用容器，docker create 后不启动 —— 导出的是镜像层展平后的 rootfs，不含任何运行痕迹
docker create --name flagrt-kunlun-v2-export-src \
  flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64 bash
docker export flagrt-kunlun-v2-export-src | pigz -p 32 > kunlun-operator-runtime-2.0.0-rootfs.tar.gz
docker rm flagrt-kunlun-v2-export-src            # 用完即删
```

与 v1 的一处差别：v1 导出的是它的验证容器 `xliu969-images-p800-repro`（跑过验证、含运行
痕迹）；v2 用 `docker create` 的纯净容器，导出内容更贴近镜像本身。

实测耗时约 4 分钟（`docker export` 流式送 `pigz`，不在 `/var/lib/docker` 产生临时文件 ——
导出全程 `/var/lib/docker` 可用量未变）。

## 还原命令

```bash
zcat kunlun-operator-runtime-2.0.0-rootfs.tar.gz | docker import \
  --change 'ENV PATH=/root/.cargo/bin:/usr/local/go/bin:/usr/local/cuda-12.8/bin:/root/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
  --change 'ENV TZ=Asia/Shanghai' \
  --change 'ENV LANG=zh_CN.UTF-8' \
  --change 'ENV CONDA_DEFAULT_ENV=python310_torch29_cuda' \
  --change 'WORKDIR /root' \
  --change 'CMD ["bash"]' \
  --change 'LABEL org.opencontainers.image.title="FlagRT Kunlun operator runtime"' \
  --change 'LABEL org.opencontainers.image.version="2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64"' \
  --change 'LABEL io.flagrt.python.version="3.10.18"' \
  --change 'LABEL io.flagrt.flagtree.version="0.6.1+xpu3.6"' \
  --change 'LABEL io.flagrt.flag_gems.commit="73c5aff1e304c293ab525ecb03e9449644496fbb"' \
  - flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64-restored
```

元数据取自 `docker inspect … --format '{{json .Config}}'`（2026-10-12 实读）。
`--change` 只重放版本锚相关与运行必需的几项；基座自带的 `maintainer` / `description` /
`version` / `org.opencontainers.image.ref.name` 四个厂商标签不重放（与版本锚无关）。
注意 `docker import` 产物**不带** `io.flagrt.*` 之外的厂商标签，也不带 `Shell`
（原镜像为 `["/bin/bash","-o","pipefail","-c"]`）。

还原后校验：

```bash
docker run --rm flagrt/kunlun-operator-runtime:...-restored \
  bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda && \
    python3 -m pip freeze | sort' | diff - <(sort assets/provenance/repro-pipfreeze-v2.txt)
```

**本轮未执行还原验证**，原因是真实约束而非遗漏：`docker import` 会把这 30.8 GiB 的 tar
在 `/var/lib/docker` 里展开成约 105GB，而该盘只剩 104G（99% 满，共享机）。
在共享机上做这一步会把余量压到危险水位，代价大于「还原路径未实测」。
已完成的校验是 `gunzip -t`（rc=0）与 `sha256sum`（见上表，已写入同目录 `.sha256` 旁文件）。
与 v1 的校验深度一致（见 `../v1/BACKUP.local` 的 `last_verified`）。

## 磁盘实测（2026-10-12，`df -h`）

**更正此前「本机磁盘不允许」的结论**：那个判断把「`/var/lib/docker` 满」等同于
「本机没地方落盘」，实测不成立 —— 本机有**两块独立 NVMe**：

| 挂载点 | 设备 | 容量 | 已用 | 可用 | 使用率 |
|---|---|---|---|---|---|
| `/var/lib/docker`（= `/data1`） | `/dev/nvme0n1p1` | 5.9T | 5.8T | 104G | 99% |
| `/data2` | `/dev/nvme1n1p1` | 5.9T | 390G | 5.5T | 7% |
| `/` | `/dev/vda3` | 98G | 94G | 187M | 100% |

归档实体落在 `/data2`，**与 `/var/lib/docker` 不同物理盘**，不占用镜像存储余量；
导出全程 `/var/lib/docker` 可用量未变（105G → 104G 的 1G 变化来自本轮
`docker create` / 探针镜像构建，已全部回收）。
这也是 v1 归档实体一直所在的盘（见 `../v1/BACKUP.local`：
`/data2/xliu969/flagrt-image-backup/20260922/`）—— 所以「磁盘不允许」从一开始就只对
`/var/lib/docker` 成立，对归档落盘不成立。

`/` 只剩 187M（100%），任何会往 `/`、`/tmp` 写大文件的动作都要避开。

## 为什么仍然推荐「重建」而不是「用离线包」

离线包是兜底，不是首选途径：

- **本版本可从公开可达源头重建**，不依赖任何 owner 私有资产，**也不依赖宿主的任何
  checkout**：重建输入只有两样 —— 公开 harbor 的父基座 + 公开索引的 flagtree wheel。
  单入口 `build.sh <构建上下文目录>`，首轮（含 3.3GB wheel 下载）实测约 15 分钟。
  这与 v1 的处境不同：v1 的归档镜像是 `docker commit` 产物、其配方从未实机构建通过
  （见 [`../v1/lock.yaml`](../v1/lock.yaml) `known_issues: v1-image-is-docker-commit-not-build`），
  所以 v1 的离线包是**唯一**的还原途径；v2 的配方本身就是还原途径。
- 离线包因 `docker export` 的变通而**丢失 layer 历史**，重建产物在这点上更忠实于配方。

离线包的价值在于：构建机或 docker 镜像丢失、或迁移到无法访问
`harbor.baai.ac.cn` / `resource.flagos.net` 的环境时的兜底。

## 还原（当前途径，按优先级）

```bash
# 首选：从零重建（公开源头，~15 分钟，基座已在本机时需 /var/lib/docker ~12GB 余量）
dev/images/kunlun-operator-runtime/v2/build.sh <构建上下文目录>

# 兜底：从离线包 docker import（见上「还原命令」，需 /var/lib/docker ~105GB 余量）

# 两条路都要核对重建/还原产物与本条目记录的版本事实一致
python3 dev/lib/baseline.py check --image dev/images/kunlun-operator-runtime/v2
```

重建产物的 image id 与 [`image.baseline.yaml`](image.baseline.yaml) 的 `image.id` 不保证
逐字节相同（pip 安装时间戳等非确定性因素），`check` 核对的是 image id 与各版本锚标签；
内容等价性的判据见 [`README.md`](README.md)「验证」与 [`REBUILD.md`](REBUILD.md)「pip freeze 对比」。
`docker import` 还原产物是单层镜像，image id 必然与原镜像不同，故另起 `-restored` 标签。

## 重建依赖资产

本血统不需要离线回收资产（`recovered/`）——基座为公开 harbor 直接可拉，flagtree 为公开
索引直接可装，FlagGems 随基座而来（本层不装），均可从 `Dockerfile.repro` + `build.sh`
联网重建，无 owner 私有资产依赖。
