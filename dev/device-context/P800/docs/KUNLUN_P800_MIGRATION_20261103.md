# 昆仑芯 P800 服务器迁移记录（2026-11-03）

> 触发：组内公告 —— 昆仑芯**新增两台服务器**，原 `43.180.254.67:26008`（VM-0-2-ubuntu）**2026-11-07 到期**，
> 到期后原机清理释放。本文件记录「搬了什么、搬到哪、还差什么、怎么复核」。
>
> 结论先行：**数据与模型已迁至新机 `/data2/hliu553`（与老机同路径），新机端到端复验通过**
> （离线判据 `43/0/1`、推理腿 `INFER_LEG_PASS 13/13`、区分度 `0.6392` 与老机归档逐字一致）。
> ⚠️ 仅余一件需外部配合：新机**卡 1/卡 2 分配显存挂死**的排查/报厂商（§5）。

## 1 服务器与凭据（脱敏记录）

| 角色 | 地址 | 别名 | 用户 | 备注 |
|---|---|---|---|---|
| **老机（11-07 到期）** | `43.180.254.67:26008` | `P800` / `P800-old` | `hliu553` | VM-0-2-ubuntu；384 核；到期**停止使用** |
| **新机 1（现行主用）** | `81.70.153.29:26008` | `P800-new1` / `P800-1` | `hliu553` | VM-20-4-ubuntu；384 核 / 1.5 TB；数据落此 |
| **新机 2（备用）** | `152.136.33.169:26008` | `P800-new2` / `P800-2` | `hliu553` | VM-20-14-ubuntu；同样 384 核 / 1.5 TB；**本次未放数据** |

- ⚠️ **端口不是 22，是 26008**（22 端口三台都关闭）。`~/.ssh/config` 已加 `Port 26008` 与三个别名。
- 认证：**沿用原密钥**（三台均可免密登录，实测 `BatchMode=yes` 通过）。公告说"账号密码都是原来的"，本次**未用到密码**。
- 新机 `/data1`、`/data2` 顶层均 `root:root 755` ⇒ **普通用户建不了一级目录**。
  ✅ **已解决（2026-11-03 16:15）**：`sudo` 口令可用（**与用户名相同**；口令本身不落仓），
  已用 `sudo` 正式执行 `mkdir + chown` 并通过验证；§5.1 另记一条**无需口令**的备用手法。

## 2 迁了什么、落在哪

新机路径（`P800-new1`）：

```
/data2/hliu553/             # ← 与老机**同名同路径**（2026-11-03 归位完成，5.1 TiB 可用盘）
├── prototype/              # 原型代码副本
├── dc_*/  out_*/ ...       # 历次验收/复跑的证据目录
└── models/
    └── Qwen3-Embedding-0.6B/   # 11 个文件、1.2 GiB（**普通目录形态**，可直接作 DC_MODEL）
```

> 迁移中转站 `$HOME/p800-migrate-20261103/` 已清空（`/` 盘当时只剩 15 GB，不宜久留）。

| 资产 | 老机位置 | 新机处理 | 校验 |
|---|---|---|---|
| **代码与证据**（`/data2/hliu553`） | 18 MB / 1360 文件（含 `__pycache__`） | 全量 rsync（排除 `__pycache__`） | 两侧同口径文件数 **1286 = 1286** |
| **模型** `Qwen3-Embedding-0.6B` | `/data1/dinghaisong/hf_cache/...`（**别人**的只读共享目录） | 新机**自建一份**，改为普通目录形态 | `model.safetensors` sha256 = **`0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`**，与老机/910C/MLU590 **逐字节相同** |
| **容器镜像（老容器用的）** | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`（38.3 GB，id `sha256:cd53efa40eb7…`） | **未搬**（见 §3） | — |
| **容器镜像（我方 repo）** | `flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-…-flaggems73c5aff1-x86_64` | 新机**已预置**（105 GB） | 同名 tag 已存在 |
| 老容器 `hliu553-device-context-p800` 规格 | — | 导出留证 `probes/migration_20261103_out/container_spec_old.json` | 8×xpu + `/dev/xpuctrl` + `/dev/fuse`、shm 64 GiB |

## 3 为什么没搬 38 GB 镜像：新机已有等价镜像

新机预置的 `flagrt/kunlun-operator-runtime:2.0.0-…` 里**有同名同版本的 conda 环境**
（`/root/miniconda/envs/python310_torch29_cuda`，torch `2.9.0+cu129`），与老容器的运行环境等价。
实测：在该镜像里跑我方原型离线判据 **43 通过 / 0 失败 / 1 跳过**，推理腿 **13/13 PASS**。

⇒ **不需要 `docker save` 搬 38 GB**（按老机→新机链路实测约 1.2 MB/s，搬 38 GB 要 9 小时以上，不现实）。

### 3.1 建容器的**两个必备细节**（少一个就「设备不可用」）

```bash
docker run -d --name <name> \
  --device /dev/xpu0 … /dev/xpu7 \
  --device /dev/xpuctrl --device /dev/fuse \      # ← 关键：缺 xpuctrl 会 torch.cuda.is_available()==False
  --shm-size=64g \
  -v /data2/hliu553:/workspace \
  <镜像> sleep infinity
```

- ⚠️ **只挂 `/dev/xpu0..7` 不够**：实测缺 `/dev/xpuctrl` 时 `torch.cuda.is_available()=False`、`count=0`
  （看起来像"没有卡"，实际是缺控制设备）；补上后 `count=8`。
- 解释器用 `/root/miniconda/envs/python310_torch29_cuda/bin/python3`（默认 `python3` 无 torch）。

## 4 新机可用性实测（2026-11-03）

| 判据 | 读数 |
|---|---|
| 离线契约自检（`--backend kunlun`） | **43 通过 / 0 失败 / 1 跳过** |
| 最小算子（卡 3） | `alloc+sync 2.1 s` · `matmul 2.1 s` · `nn.Linear 2.1 s`（含首次编译） |
| **推理腿端到端（卡 3）** | **`INFER_LEG_PASS 13/13`** · dim 1024 · **区分度 `0.6392`** · 52.59 句/s · p50 56.67 ms |
| 与老机归档对照 | 老机归档区分度同为 **`0.6392`** ⇒ 逐字一致 |

证据：`probes/migration_20261103_out/new1_infer_leg_result.json`。

## 5 完成情况与未完成项

### 5.1 ✅ `/data2/hliu553` 已建并归位（2026-11-03 16:15）

`/data2` 顶层是 `root:root 755` ⇒ 普通用户建不了一级目录。**两条路径都实测过**：

**A. sudo（正式执行，`rc=0`）** —— 该机 `sudo` 口令**与用户名相同**，权限 `(ALL : ALL) ALL`：

```bash
echo '<口令>' | sudo -S -p '' bash -c \
  'mkdir -p /data2/hliu553 && chown -R hliu553:hliu553 /data2/hliu553'
# 验证：属主不符 0 个 / 属组不符 0 个
find /data2/hliu553 ! -user hliu553 | wc -l   # 0
find /data2/hliu553 ! -group hliu553 | wc -l  # 0
```

> ⚠️ 口令属敏感信息，**只存本机 skill，不进本仓**。

**B. 备用手法（口令缺失 / 未开通 sudo 时）** —— 该机在 **docker 组**，用容器内 root 做等价动作：

```bash
# 注：新机 uid=1017 / gid=1018（用 id 核过），不是通用的 1000
docker run --rm -v /data2:/data2 --entrypoint /bin/bash <我方镜像> \
  -lc "mkdir -p /data2/hliu553 && chown -R 1017:1018 /data2/hliu553"
```

⭐ 通用手法：**宿主机拿不到 sudo 口令、但能跑 docker 时**，`docker run -v <目标父目录>:<同路径>` 即可用容器 root 做
`mkdir/chown/rm`。⚠️ 两点纪律：① 只对**自己那条目录**操作（同盘还有 `chenyuanxin`/`dengyuzhi`/`xiliu569` 等他人目录）；
② `uid:gid` 必须实测（`id`），别心算 1000。
> 本次实际顺序：**先用 B 完成**（当时不知口令），**随后确认口令可用，再用 A 正式执行一遍**（幂等，`rc=0`）。
> 两条路径结果一致 ⇒ 结论不依赖手法。

归位结果（已逐项核对）：

| 项 | 读数 |
|---|---|
| 目录属主 | `hliu553:hliu553`（`drwxr-xr-x 1017 1018`），本人 `touch` 写入实测通过 |
| 文件数 / 字节数 | **1337 = 1337** / **1,221,715,797 = 1,221,715,797**（rsync 后核对，再删中转副本） |
| 模型权重 sha256 | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`（与老机/910C/MLU590 逐字节相同） |
| 容器挂载 | 已重建为 `-v /data2/hliu553:/workspace`（**与老机一致**） |
| 中转目录 | `$HOME/p800-migrate-20261103/` 已清空（其中 root 属主残留同样用容器 root 清） |

### 5.1b 归位后的端到端复验（同一容器、同一镜像）

| 判据 | 读数 |
|---|---|
| 离线自检（kunlun） | **43 通过 / 0 失败 / 1 跳过**（与迁移前逐项相同） |
| 推理腿（卡 3） | **`INFER_LEG_PASS 13/13`** · dim 1024 · **52.58 句/s** · p50 55.92 ms · 区分度 **0.6392** |
| 设备可见性 | `torch.cuda.is_available()=True` · `count=8` |

### 5.2 ⭐ 新机 **卡 1、卡 2 分配显存时挂死**（需排查/报厂商）

逐卡最小算子实测（同一镜像、同一容器）：

| 卡 | 结果 |
|---|---|
| 卡 0 | ✅ 通过 |
| **卡 1** | ❌ 卡在 `torch.zeros(... device=...)`，`timeout -s KILL 90` 才结束（`set_device` 正常、`cuda.available=True`） |
| **卡 2** | ❌ 同上 |
| 卡 3 | ✅ 通过（推理腿 13/13 就是在卡 3 上跑的） |

- 现象与老机记录的「**物理卡 1 故障**」同型（`xpu-smi` 显示空闲、控制面正常、计算内核永不完成）。
- ⚠️ **纪律**：`xpu-smi` 显示 0 MiB **不代表能用** —— 取卡前必须跑一次最小算子/事件探针。
- 待办：把卡 1/卡 2 的 `xpu-smi` 序列号与现象提交厂商/管理员（我方**不对该现象下归因结论**，只记录可复现现象与判据）。

## 6 复核方法（第三方可自查）

```bash
# 1) 连通与身份
ssh P800-new1 'hostname; id -un'

# 2) 数据完整性：两侧同口径文件数应相等（老机上执行）
ssh P800-old 'find /data2/hliu553 -type f ! -path "*__pycache__*" | wc -l'      # 1286
ssh P800-new1 'find /data2/hliu553 -type f ! -path "*models*" ! -path "*out_new*" | wc -l'

# 3) 目录属主（应为 hliu553:hliu553）
ssh P800-new1 'ls -ld /data2/hliu553'

# 4) 模型权重指纹
ssh P800-new1 'sha256sum /data2/hliu553/models/Qwen3-Embedding-0.6B/model.safetensors'
# 期望 0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd

# 5) 逐卡可用性（**必须**，别只看 xpu-smi）
ssh P800-new1 'for c in 0 1 2 3; do echo -n "card$c: "; docker exec -e CUDA_VISIBLE_DEVICES=$c \
  hliu553-dc-probe-new1 bash -lc "timeout -s KILL 90 /root/miniconda/envs/python310_torch29_cuda/bin/python3 \
  -c \"import torch;x=torch.zeros(2,2,device=\\\"cuda\\\");print(float(x.sum().item()))\"" ; done'
```

## 7 附：模型获取方式（新机可复现）

老机那份模型在别人的只读共享目录里，新机没有 ⇒ 改为从 **hf-mirror** 自取（HF 直连在本机不可达，镜像站实测 **2.86 MB/s**）：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-Embedding-0.6B \
  --local-dir <目标目录>       # 11 个文件 ≈ 1.2 GiB
```

⚠️ 取完**必须**核 `model.safetensors` 的 sha256（见 §6.3）—— 镜像站内容也要验，不能只看下载成功。
