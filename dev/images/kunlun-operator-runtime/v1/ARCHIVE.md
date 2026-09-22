# kunlun-operator-runtime v1 —— 离线归档

> 机器无关清单。实体不在仓库；本机实体的绝对路径见同目录 `BACKUP.local`（不追踪）。

## ⚠️ 还原方式是 `docker import`，不是 `docker load`

本机 Docker Engine 29.1.3 使用 containerd snapshotter 存储后端
（`docker info` → `driver-type: io.containerd.snapshotter.v1`）。该后端下
`docker save`（对本镜像、对未改动的 BAAI·FlagTree 官方基座镜像均如此，
非本镜像特有缺陷）只导出 manifest/config blob，**不导出任何 layer blob**，
产出 tar 仅 ~14KB，`docker load` 后镜像不完整。详见
`lock.yaml known_issues: docker-save-broken-on-this-host`。

因此本条目改用 `docker export`（导出容器展平后的 rootfs 为单层 tar）+
`docker import --change` 重放 `ENV`/`CMD`/`WORKDIR`/`LABEL` 元数据。代价：
镜像 layer 历史丢失（`docker history` 只会显示一层），但 rootfs 内容与
运行行为与原镜像完全一致（元数据逐项核对见下）。若目标机器的 Docker 不使用
containerd snapshotter 后端，`docker save`/`docker load` 应正常工作，可
自行按 `Dockerfile.repro` + `build.sh` 重新构建后另存一份标准 `docker save` tar。

## 压缩包

| 文件 | 大小 | sha256 | 内容 |
|---|---|---|---|
| `kunlun-operator-runtime-1.0.0-rootfs.tar.gz` | 33.16G（gzip 压缩后；`docker export` 展平 rootfs \| `pigz -p 32`） | `8bebbb7e1533a315b29e5fdf3da934d6beeea257aa31ecdedc93ee9873eebf36` | `flagrt/kunlun-operator-runtime:1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64` 容器 `xliu969-images-p800-repro` 的完整 rootfs |

## 还原命令

```bash
zcat kunlun-operator-runtime-1.0.0-rootfs.tar.gz | docker import \
  --change 'ENV PATH=/root/.cargo/bin:/usr/local/go/bin:/usr/local/cuda-12.8/bin:/root/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
  --change 'ENV TZ=Asia/Shanghai' \
  --change 'ENV LANG=zh_CN.UTF-8' \
  --change 'ENV CONDA_DEFAULT_ENV=python310_torch29_cuda' \
  --change 'WORKDIR /root' \
  --change 'CMD ["bash"]' \
  --change 'LABEL org.opencontainers.image.title="FlagRT Kunlun operator runtime"' \
  --change 'LABEL org.opencontainers.image.version="1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64"' \
  --change 'LABEL io.flagrt.xpu3.6.version="3.6"' \
  --change 'LABEL io.flagrt.python.version="3.10.18"' \
  --change 'LABEL io.flagrt.torch.version="2.9.0+cu129"' \
  --change 'LABEL io.flagrt.flagtree.version="0.6.1+xpu3.6"' \
  --change 'LABEL io.flagrt.flag_gems.commit="73c5aff1e304c293ab525ecb03e9449644496fbb"' \
  - flagrt/kunlun-operator-runtime:1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64-restored
```

还原后校验：
```bash
docker run --rm flagrt/kunlun-operator-runtime:...-restored \
  bash -lc 'source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda && \
    python3 -m pip freeze | sort' | diff - <(sort assets/provenance/repro-pipfreeze.txt)
```
（`docker import` 产物是单层镜像，`pip freeze` 内容应与归档时逐行一致；
无需 `--device` 挂载即可通过静态检查，动态真机验证见 `REBUILD.md`。）

## 重建依赖资产

本血统不需要离线回收资产（`recovered/`）——基座为公开 harbor 直接可拉，
flagtree 为公开索引直接可装，FlagGems 源码为公开上游仓库直接可
clone/checkout，均可从 `Dockerfile.repro` + `build.sh` 联网重建，无
owner 私有资产依赖问题（与 ascend v2 血统同理）。

## 用途

构建机或 docker 镜像丢失、或需要迁移到无法访问
`harbor.baai.ac.cn`/`resource.flagos.net` 的环境时的兜底：`docker import`
直接还原（见上）；或按 `REBUILD.md` + `build.sh` 从零联网重建（推荐，公开
可复现，且能在支持 `docker save`/`docker load` 的正常环境下产出标准归档格式）。
