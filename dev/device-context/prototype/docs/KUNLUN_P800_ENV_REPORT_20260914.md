# 昆仑芯 P800 基础环境汇总（2026-09-14）

> 采集方式：从本机经 SSH（`P800` = `43.180.254.67:26008`，用户 `hliu553`）执行**只读探测**，
> 未在目标机做任何写操作、未启动/停止任何进程与容器。
> 用途：任务 1 交付物；同时作为「任务 2 资源是否足够」的判定依据。

---

## 0. 结论速览

| 维度 | 判定 | 说明 |
|---|---|---|
| XPU 算力 | ✅ **充足** | 8× P800 OAM，每卡 96 GB，**全部空闲（0% 利用率、0 MiB 占用）** |
| 主机内存 | ✅ **充足** | 1.5 TiB，可用 1.4 TiB |
| CPU | ✅ **充足** | 2× AMD EPYC 9K84 96 核 = 384 线程 |
| **专用数据盘** | ✅ **物理上存在** | `/data1`、`/data2` 各 5.8 TB NVMe（xfs） |
| **数据盘可用性** | ❌ **无权限** | 两块盘顶层均 `root:root 755`，`hliu553` **不可写** |
| **根分区** | ❌ **将满** | `/dev/vda3` 98 GB 已用 91 GB，**仅剩 2.9 GB** |
| **容器权限** | ❌ **无权限** | `hliu553` **不在 `docker` 组**；`sudo` 需密码 |

> **一句话**：算力与内存充裕，**卡在「权限 + 磁盘」两件事上**——任务 2 目前无法启动（详见 §9）。

---

## 1. 主机与系统

| 项 | 值 |
|---|---|
| 主机名 | `VM-0-2-ubuntu` |
| OS | Ubuntu 24.04.2 LTS |
| 内核 | `6.8.0-87-generic` (#88-Ubuntu SMP PREEMPT_DYNAMIC, x86_64) |
| 当前时间 | 2026-09-14 09:39 CST |
| 运行时长 | 36 天 22 小时 |
| 负载 | load average 1.57 / 1.52 / 1.53（相对 384 线程，很轻） |
| 在线用户 | **25 个** → 多人共享机器 |
| 僵尸进程 | **272 个** |
| Swap | **0 B（未配置）** → 内存不可超卖 |

> 提醒：25 人在线 + 22 个容器 shim 在跑，说明这是**团队共享机**，不是独占机。
> 我们的容器命名与资源占用应保持可识别、可回收。

---

## 2. 计算资源

| 项 | 值 |
|---|---|
| CPU 型号 | AMD EPYC 9K84 96-Core Processor |
| 插槽 / 核 / 线程 | 2 socket × 96 core × 2 线程 = **384 线程**（`nproc=384`） |
| NUMA 节点 | 2（node0 = CPU 0-95,192-287；node1 = CPU 96-191,288-383） |
| 内存总量 | **1.5 TiB** |
| 已用 / 可用 | 85 GiB / **1.4 TiB** |
| buff/cache | 230 GiB |

---

## 3. XPU（昆仑芯 P800）资源

| 项 | 值 |
|---|---|
| 卡数 | **8 × P800 OAM** |
| 单卡显存 | **98304 MiB = 96 GB**（合计 768 GB） |
| 驱动版本（宿主 `xpu-smi`） | **5.0.21.47** ｜ XPU-RT 5.0.21 |
| 功耗上限 | 400 W（实测空载 91–95 W） |
| 温度 | 37–44 °C |
| 当前利用率 | **全部 0%**，Memory-Usage **全部 0 MiB** |
| ECC | 0（无不可纠正错误） |
| SR-IOV | Disabled |
| 设备节点 | `/dev/xpu0` … `/dev/xpu7`（char 195:0–7）+ `/dev/xpuctrl`（195:255），`crw-rw-rw-` 全局可读写 |
| 内核模块 | `kunlun`（5.0.21，引用 76 次）、**`kunlun_peermem`（RDMA peer memory）** |
| 管理工具 | `/usr/local/bin/xpu-smi`、`xpu_smi` |
| SDK 路径 | `/usr/local/xpu-5.0.21.47`、`/usr/local/xpu`（bin/include/lib/so/tools/profiler） |

XPU UUID 列表（`xpu-smi -L`）：

```
XPU 0: P800 OAM  GPU-ed733bb8-c4a8-5bb5-9431-14512597fc12
XPU 1: P800 OAM  GPU-b3509946-0bc6-5744-bd7c-a0b87dadef02
XPU 2: P800 OAM  GPU-09d75c76-1b13-5686-a3d2-a9647e48f619
XPU 3: P800 OAM  GPU-9429f5bd-7d13-5cc5-b1b4-c05ddc572f57
XPU 4: P800 OAM  GPU-b7942319-362e-5a53-a8eb-a2d06fbe84f6
XPU 5: P800 OAM  GPU-f396486d-9850-50e4-81c4-50f2e9f6ca87
XPU 6: P800 OAM  GPU-9e24d168-db57-5cd9-8cfb-aea4ae898897
XPU 7: P800 OAM  GPU-4922595e-feb9-5762-9082-d8fd7f11abf5
```

---

## 4. XPU 互联拓扑与 RDMA 网络（对多卡/分布式验证关键）

`xpu-smi topo -m` 摘要：

| 关系 | 链路 | 备注 |
|---|---|---|
| XPU0 ↔ XPU1/2/3 | **XL** | 组内私有高速链路 |
| XPU4 ↔ XPU5/6/7 | **XL** | 组内私有高速链路 |
| XPU0-3 ↔ XPU4-7 | **SYS** | 跨 NUMA 走 CPU 互联 |
| XPU0-3 归属 | NUMA **0** | CPU 0-95,192-287 |
| XPU4-7 归属 | NUMA **1** | CPU 96-191,288-383 |
| NIC0 ↔ XPU0 / NIC1 ↔ XPU1 / NIC2 ↔ XPU2 / NIC3 ↔ XPU3 | **PIX** | 网卡与卡同 PCIe 桥，最短路径 |
| NIC4 ↔ XPU4 / NIC5 ↔ XPU5 / NIC6 ↔ XPU6 / NIC7 ↔ XPU7 | **PIX** | 同上 |
| NIC6 ↔ XPU7 / NIC7 ↔ XPU6 | PHB | 次优 |

**网络设备**：

| 项 | 值 |
|---|---|
| 管理网 | `eth0` = 10.235.0.2/20（**NAT 后**，公网映射 43.180.254.67） |
| RDMA 网卡 | **`mlx5_bond_0` … `mlx5_bond_7`（8 个，16 个物理口绑定）** |
| 速率 | **200 Gb/sec (4X HDR)** |
| 固件 | 28.43.2566 |
| 端口状态 | 全部 `PORT_ACTIVE (4)` |
| 链路层 | **Ethernet（即 RoCE）**，非 InfiniBand |
| 设备节点 | `/dev/infiniband/uverbs0-7`、`umad0-7`、`rdma_cm` |

> **对设备上下文方向的价值**：`kunlun_peermem` 已加载 → XPU 侧存在**类 GDR 的网卡直访显存**能力
> （对标 NVIDIA `nvidia_peermem`）。这条线与我们在 910C 上追问的「昇腾 HIXL 是否提供 GDR 等价能力」
> 是同一命题的另一个厂商答案，**建议纳入跨芯片原语调研素材**。
> 另：卡与网卡 PIX 直连 + 200G RoCE，是单机 8 卡做多流/集合通信验证的良好条件。

---

## 5. 存储（**本次最关键的一项**）

### 5.1 文件系统全景

| 设备 | 类型 | 容量 | 已用 | 可用 | 使用率 | 挂载点 |
|---|---|---|---|---|---|---|
| `/dev/vda3` | ext4 | 98 G | 91 G | **2.9 G** | **98%** | `/`（含 `/home`、`/var/lib/docker`） |
| `/dev/vda1` | vfat | 537 M | 6.2 M | 531 M | 2% | `/boot/efi` |
| `/dev/nvme0n1p1` | xfs | 5.9 T | 4.4 T | **1.5 T** | 76% | **`/data1`** |
| `/dev/nvme1n1p1` | xfs | 5.9 T | 193 G | **5.7 T** | 4% | **`/data2`** |

### 5.2 专用数据盘：**有，但当前不可写**

| 目录 | 容量 | 顶层权限 | `hliu553` 可否写入 |
|---|---|---|---|
| `/data1` | 5.8 T（剩 1.5 T） | `drwxr-xr-x root root` | ❌ **NO** |
| `/data2` | 5.8 T（剩 5.7 T） | `drwxr-xr-x root root` | ❌ **NO** |
| `/data` | — | `drwxr-xr-x root root` | ❌（且 **`/data` 只是根分区上的普通目录，不是挂载点**） |
| `/tmp`、`/var/tmp` | 落在 98% 满的 `/` 上 | 可写 | ⚠️ 名义可写但仅剩 2.9 G |
| `/data1/songchao` | 落在 `/data1` | `drwxrwxrwx` | ⚠️ **可写**——但属他人目录，不建议占用 |

`/data1` 下已按用户划分（12 个顶层目录）：`chenyunlong / daizijian / dinghaisong / huyongquan /
kangkai / kfc / songchao / user_cache / xianghuang / zhenghaojia` 等，
**其中没有 `hliu553`**。

### 5.3 根分区为什么满

`du -xhd1 /` 可读部分合计仅 **11 G**（`/usr` 8.8 G、`/var` 1.5 G…），
但 `df` 显示根分区**已用 91 G** → 约 **80 G 落在无权限读取的目录**中。
最大嫌疑是 `/var/lib/docker`（`drwx--x--- root root`）与 `/var/lib/containerd`，
即 **docker 镜像层堆在只剩 2.9 G 的根分区上**。

---

## 6. 软件栈与容器运行时

| 项 | 值 |
|---|---|
| 系统 Python | **3.12.3**（`/usr/bin/python3`、`/usr/bin/pip3`） |
| Docker | **29.1.3**（`/usr/bin/docker`，`dockerd -H fd://`，配套 containerd） |
| containerd | 运行中，`/var/lib/containerd`（`drwx------ root root`，无权读取） |
| Singularity | **singularity-ce 4.3.7**（`/usr/local/bin/singularity`、`run-singularity`） |
| XPU SDK | `/usr/local/xpu-5.0.21.47`、`/usr/local/xpu` |
| `ldconfig` 中的 xpurt/kunlun | 未登记（需运行时配置 `LD_LIBRARY_PATH` 或由镜像自带） |
| `/etc/docker/daemon.json` | **不存在** → docker 采用默认 data-root `/var/lib/docker` |
| `/etc/profile.d/` 中的 XPU 变量 | 无 → XPU 环境变量需在容器内/脚本中自行设置 |

---

## 7. 账号与权限（**第二个阻塞点**）

| 项 | 值 |
|---|---|
| 用户 / UID | `hliu553` / 1017（gid 1018） |
| 所属组 | **仅 `hliu553` 一个组** |
| `docker` 组成员 | `xliu969`、`daizijian`（**不含 `hliu553`**） |
| `docker ps` / `docker images` | ❌ `permission denied ... /var/run/docker.sock` |
| `sudo` | ❌ `sudo: a password is required`（非 NOPASSWD） |
| 结论 | **当前身份无法使用 docker，也无法自行提权** |

---

## 8. 机器上已有的可复用资产（他人目录，**只读可访问**）

| 资产 | 路径 | 状态 |
|---|---|---|
| **flagtree xpu3.6 镜像包（32 GiB）** | `/data1/dinghaisong/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04.**202606**-base.tar.gz` | 33,504,153,783 B（32 GiB），**`-rw-rw-r--` 全局可读**，gzip 魔数校验通过 → 可直接 `docker load`（无需联网 pull 59.9 G） |
| FlagTree 源码 | `/data1/dinghaisong/FlagTree` | 可读（19 项） |
| FlagGems 源码 | `/data1/dinghaisong/FlagGems` | 可读（19 项） |
| 共享 HF 模型缓存 | `/data1/dinghaisong/hf_cache`（**1.7 T**） | 可读 |
| ├ **Qwen3-Embedding-0.6B** | `hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B` | **1.2 G，实测 `config.json` 可读** — **正是原型验收模型** |
| ├ Qwen3-Embedding-4B / 8B、Qwen3-Reranker-0.6B/4B/8B、gte-Qwen2-1.5B/7B 等 | 同上 | 可读 |
| 他人 docker 数据目录外移先例 | `/data1/xianghuang/docker-data`（另有 overlay 挂载落在 `/data1/...`） | 说明**本机已有「docker data-root 放到 /data1」的既有做法** |

> ⚠️ 版本差异：本机现成镜像包为 **`202606-base`**，而官方手册（User manual for xpu）给出的是
> **`202608-base`**。二者相隔一期，**须在使用前确认是否满足 xpu3.6 要求**，不可默认等同。

---

## 9. 资源充足性判定（对照任务 2 的实际需求）

任务 2 需要：拉/载入 **32–60 GB** 镜像、在容器内装 flagtree + FlagGems、跑通算子与训练。

| 需求 | 现状 | 判定 |
|---|---|---|
| 8 卡 XPU 可用 | 8 卡全空闲 | ✅ |
| 内存 ≥ 数百 GB | 1.4 TiB 可用 | ✅ |
| CPU 核数 | 384 线程 | ✅ |
| **docker 使用权限** | 不在 `docker` 组、`sudo` 需密码 | ❌ **阻塞** |
| **镜像落盘空间** | docker data-root 在 `/var/lib/docker`（即 98% 满的根分区，剩 **2.9 G**）；需 **32 G+**（load）或 **59.9 G**（pull） | ❌ **阻塞** |
| **模型/工作目录空间** | `/data1`、`/data2` 均不可写 | ❌ **阻塞** |
| 联网拉依赖（pip / 源码） | 未测（容器未起） | ⏳ 待验证 |

**结论：当前资源**（权限 + 磁盘）**不足以启动任务 2**。
XPU / 内存 / CPU 三项充裕，瓶颈完全在**账号权限**与**存储规划**。

---

## 10. 需要的协调（提交平台/总组）

请按优先级推进以下三项。**三项均需落实后任务 2 方可启动**：① 决定「能不能用 docker」，② 决定「代码/模型/产物放哪儿」，③ 决定「镜像有没有地方落」。

| # | 诉求 | 命令（供管理员执行） | 目的 |
|---|---|---|---|
| **①** | 将 `hliu553` 加入 `docker` 组 | `sudo usermod -aG docker hliu553`（执行后**重新登录**生效） | 获得 `docker` 使用权限 |
| **②** | 在 `/data2`（5.7 T 空闲）下开可写目录 | `sudo mkdir -p /data2/hliu553 && sudo chown hliu553:hliu553 /data2/hliu553` | 存放代码、模型与实验产物 |
| **③** | 将 docker data-root 迁到数据盘（**影响全机，需平台决策**） | 参考本机既有先例（`/data1/xianghuang/docker-data`）：`systemctl stop docker` → 迁移 `/var/lib/docker` 至 `/data2/docker` → 在 `/etc/docker/daemon.json` 写 `{"data-root":"/data2/docker"}` → 重启 docker | 否则 32–60 GB 镜像将撑爆只剩 2.9 G 的根分区 |

**关于「用已有 32 GiB 本地包省空间」的澄清**：`/data1/dinghaisong/...202606-base.tar.gz` 全局可读，
用 `docker load` 可省掉 **59.9 GB 的联网 pull**；但 `load` 与 `pull` **都要把镜像写入 docker data-root**，
**省不掉那 32 GB 落盘需求** —— 即 ③ 无法被绕开。另需确认 `202606` 版本是否满足 xpu3.6 要求。

---

## 11. 与本次适配工作的关联结论

1. **算力不是问题**：8 卡 96 GB 全空闲，单机即可完成原型要求的单卡/双卡与多卡验证。
2. **网络条件优于 910C**：200 G RoCE × 8 且与卡 PIX 直连，另加载了 `kunlun_peermem`，
   为「多流 + 跨设备同步 + 分布式」提供了良好底座，**也填补了我们在 910C 上追查的 GDR 类能力对标**。
3. **两处硬约束需写进本方向约束清单**（待总组裁定是否入 `stack.lock`）：
   - 昆仑芯机器的 **docker 权限与数据盘挂载方式**（与 910C 的「带卡容器并发上限 3」同类：属**基座级使用约束**）；
   - **根分区容量告警**：任何默认 data-root 的 docker 使用都会撞墙，
     **镜像必须落在数据盘**——这条对全组在 P800 上的工作都成立。
4. **诚实标注**：本报告仅覆盖**静态环境**。并发上限、可见设备变量、有界同步能力等
   **动态行为约束必须在容器启动后用实测回答**（见适配方案 §3.2 / §6）。
