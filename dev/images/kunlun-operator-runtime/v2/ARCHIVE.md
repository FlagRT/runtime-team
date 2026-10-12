# kunlun-operator-runtime v2 —— 离线归档

> 机器无关清单。实体不在仓库。

## 结论：本版本**暂无**离线归档实体

当前没有 `docker save` / `docker export` 产出的离线包，因此本目录也没有 `BACKUP.local`。
这是一个有意的待办，不是漏填——理由与前置条件如下。

**为什么暂时不做**：

1. **本版本可从公开可达源头重建**，不依赖任何 owner 私有资产，**也不依赖宿主的任何
   checkout**：重建输入只有两样——公开 harbor 的父基座 + 公开索引的 flagtree wheel。
   2026-10-11 配方简化后（FlagGems 改为直接用基座自带那份，不再 COPY + 重装）这件事
   **比原先更简单了**：原先还需要宿主仓根有 `FlagGems/` 且 pin commit 已推送到远端
   （`build.sh` 会检查，未推送即拒绝构建），现在这条前置整条消失，构建上下文只剩一个
   `Dockerfile`。单入口 `build.sh <构建上下文目录>`，首轮（含 3.3GB wheel 下载）实测约
   15 分钟。这与 v1 的处境不同：
   v1 的归档镜像是 `docker commit` 产物、其配方从未实机构建通过
   （见 [`../v1/lock.yaml`](../v1/lock.yaml) `known_issues: v1-image-is-docker-commit-not-build`），
   所以 v1 的离线包是**唯一**的还原途径；v2 的配方本身就是还原途径。
2. **本机磁盘不允许**：`/var/lib/docker` 使用率 99%（5.9T 中剩 ~105G，共享机），
   该镜像 105GB、v1 的 rootfs 压缩包就占 33GB。在共享机上再压一份 30GB+ 的归档会把
   余量压到危险水位，影响同机其他人。

**要补归档时怎么做**：

- **`docker save` 在本机对本镜像不可用，必须沿用 v1 那套 `docker export` + `docker import
  --change` 重放元数据的变通。** 2026-10-11 实测（`docker save <ref> | head -c 20M | wc -c`）：
  本镜像 **14336 字节 / 1 blob**（只有 config blob，零 layer blob）；未改动的官方基座镜像
  （94GB，从 harbor 拉取）**同样 14336 字节 / 1 blob**；而本机 `docker build` 产出的 50MB
  合成镜像完整产出 **52453376 字节 / 3 blob（含 layer）**。
  所以这与「镜像是不是 `docker commit` 产物」无关（v2 是正规 build 产物、同样失败），
  只与这几个巨型厂商派生镜像相关，详见 [`../v1/lock.yaml`](../v1/lock.yaml)
  `known_issues: docker-save-broken-on-this-host`（该条原诊断成立）。
  变通做法与注意事项照抄 [`../v1/ARCHIVE.md`](../v1/ARCHIVE.md)：`docker export` 导出容器
  展平后的 rootfs 单层 tar，再用 `docker import --change` 重放 `ENV`/`CMD`/`WORKDIR`/`LABEL`；
  代价是 layer 历史丢失（`docker history` 只剩一层），rootfs 内容与运行行为等价。
- 落盘后按 [`../v1/ARCHIVE.md`](../v1/ARCHIVE.md) 的表格形态在本文件补：文件名、大小、
  sha256、内容、`docker load` 用法；并新建 `BACKUP.local` 记本机实体绝对路径与校验日期
  （该文件不追踪，见 [`../README.md`](../../README.md) §2）。

已登记到 [`../TODO.md`](../../TODO.md)。

## 还原（当前途径）

```bash
# 从零重建（公开源头，~15 分钟，基座已在本机时需 /var/lib/docker ~12GB 余量）
dev/images/kunlun-operator-runtime/v2/build.sh <构建上下文目录>

# 核对重建产物与本条目记录的版本事实一致
python3 dev/lib/baseline.py check --image dev/images/kunlun-operator-runtime/v2
```

重建产物的 image id 与 [`image.baseline.yaml`](image.baseline.yaml) 的 `image.id` 不保证
逐字节相同（pip 安装时间戳等非确定性因素），`check` 核对的是 image id 与各版本锚标签；
内容等价性的判据见 [`README.md`](README.md)「验证」与 [`REBUILD.md`](REBUILD.md)「pip freeze 对比」。
