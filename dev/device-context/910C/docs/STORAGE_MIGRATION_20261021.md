# 910C（npu1-27）存储盘点与 RAID5 重建前数据迁移（2026-10-21）

> 背景：IT 将重建 npu1-27 的 RAID5，重建期间本地 RAID 不可用。按要求把本方向在该机的数据整体迁到**共享存储**。
> 本文自包含：拓扑、清单、读数、校验、恢复步骤都在正文里。**只做复制、源未删除**（等 RAID 恢复并核对后再清）。

---

## 0 一页结论

| 项 | 结果 |
|---|---|
| 本地 RAID5 是什么 | `/dev/md127`（3 × 3.5T nvme 组 RAID5，11T，xfs）挂在 **`/mnt/raid`** —— **本次重建对象** |
| ⚠️ 关键发现 | **Docker 数据根目录也在 RAID 上**（`/mnt/raid/docker`，overlay2）⇒ 镜像/容器层**无法 rsync**，只能靠清单重建 |
| 迁移目标 | **`/tgm_rent_data/data/hliu553/migrate_20261021/`**（共享存储 dtfs，50T 池，49T 空闲；写入实测 **7.6 GB/s**） |
| 迁移量 | **72G**：`raid_hliu553` 66G（**45126 文件 / 68,808,179,582 B**）+ `home_hliu553` 6.6G + git bundle 46M + 清单 1.1M |
| 校验 | 文件数 **45126 = 45126**、字节数 **逐字节相等**（差 0）；抽样 sha256 一致；4 个 git bundle `git bundle verify` **全部有效**；家目录**仅源有 0 个** |
| 未迁（有理由） | `venvs` 7.8G（绝对路径不可移植，迁了也用不了）、`pipcache` 3.7G（纯缓存） |
| 权限缺口（已补） | **1091 个 root 拥有的文件 / 67.4MB**（1060 个在 `vllm-ascend/csrc/build/**`，其余是 lock 与 `.msc` 缓存标记）⇒ 用 **root 容器只读挂载**打包补捞（**0 失败**），再以本用户解包 |
| 凭据 | 家目录复制**排除** `.ssh` / `.gnupg` / `.cache` / keyrings；复制的 `.bash_history` 扫描 `ghp_`/`hf_` **命中 0**；无 `.git-credentials` / `.docker/config.json` / `gh/hosts.yml` |
| 待你决定 | ① IT 动工前是否**停掉我们的两个空转容器**（`repro-910c-a1e`、`repro-910c-aug`，CPU 0.01–0.04%）；② RAID 恢复后是否**回迁**到 `/mnt/raid`（还是继续在共享存储上跑） |

---

## 1 这台机器的存储拓扑（实测 `df -hT` + `lsblk`）

| 挂载点 | 设备 | 类型 | 容量 / 已用 | 是否在本次重建范围 |
|---|---|---|---|---|
| **`/mnt/raid`** | `/dev/md127`（nvme0n1+nvme1n1+nvme2n1，3×3.5T RAID5） | xfs | 11T / 4.0T（38%） | ⚠️ **是 —— 本次重建对象** |
| **`/mnt/raid/docker`** | 同上（同一 md127） | overlay2 | 读不到（root 拥有） | ⚠️ 是 —— 镜像/容器/卷全在这里 |
| `/home` | `/dev/sda5` | ext4 | 239G / 175G（78%） | 否（另一块盘，不在 md127 内） |
| `/` | `/dev/sda3` | ext4 | 196G / 143G（77%） | 否 |
| `/data_lib` | `/dev/nvme3n1p1` | xfs | 3.5T / 3.3T（93%） | 否（独立 nvme；`root` 拥有：FlagTree / models / lib / ustc，非本方向数据） |
| **`/tgm_rent_data`** | dtfs（共享存储） | dtfs | 50T / 2.0T（4%） | ✅ **迁移目标**（49T 空闲） |
| `/opt/dpc_om/{package,log,export}` | NFS `10.120.69.x` | nfs | 2G / 2T / 30G | 否（IT 系统区） |

> 结论：**真正要救的是 `/mnt/raid`**（我们的数据目录 + Docker 全部数据）。`/home`、`/`、`/data_lib` 是其他物理盘，不在 md127 内。

---

## 2 我们在这台机上的数据（迁移前实测体积）

`/mnt/raid/hliu553` 合计约 **75G**，按体积排序（`du -sh`）：

| 路径 | 体积 | 性质 | 处置 |
|---|---|---|---|
| `runtime-team.stale-856b24e-20260930-hostcopy` | 45G | 09-30 的旧仓整机副本（含 5 个 fork 的 `.git`） | ✅ 全量迁（虽 0 未推送提交，仍整体保留） |
| `models` | 12G | 模型权重（Qwen2.5-1.5B / Qwen3-4B / Qwen3-Embedding-0.6B 等） | ✅ 迁 |
| `venvs` | 7.8G | Python 环境 | ❌ 不迁（绝对路径绑定 `/mnt/raid`，迁了也不可用）→ 留清单 |
| `pipcache` | 3.7G | pip 缓存 | ❌ 不迁（纯缓存） |
| `repro_a1e` | 3.4G | A1 复现实验目录（含被改过的 FlagCX 工作树） | ✅ 迁 |
| `dc_r10_20261008` | 1.2G | 第 10 轮实验原始日志/结果 | ✅ 迁 |
| `vllm-ascend` | 673M | vllm-ascend 仓 + 构建产物 | ✅ 迁 |
| `flagcx-build-20261016` · `dc-ws` · `dc_p9_20261016` · `dc_env_20261016` | 每项 32–175M | 构建目录 / 工作区（含 FlagCX checkout）/ 缺陷复现证据 | ✅ 迁 |
| `runtime-team` | 11M | 主仓 host 副本 | ✅ 迁（另出 bundle） |
| 其余 `dc_*_2026*` / `out_*` / `*.log` / `*.sh` / `backup_*.tar.gz` | 合计约 20M | 各轮实验原始输出、本地脚本 | ✅ 全迁 |

`/home/hliu553` = **5.9G**（其中 `runtime-team` 3.0G 是另一份仓副本）。

---

## 3 迁移做了什么（分四段，全部只复制、不删源）

```bash
# 目标结构
/tgm_rent_data/data/hliu553/migrate_20261021/
├── raid_hliu553/          # /mnt/raid/hliu553 的完整镜像（排除 venvs / pipcache）
├── home_hliu553/          # /home/hliu553（排除 .ssh / .cache / .gnupg / 密钥环 / .local/lib / snap）
├── git_bundles/           # 4 个 git bundle + 补捞的 root 文件 tar
└── _manifest/             # 迁移日志、Docker 清单、stash/diff、不可读清单、校验报告
```

| 段 | 内容 | 手段 | 结果 |
|---|---|---|---|
| PHASE1 | 关键数据（排除 `venvs`/`pipcache`/45G 旧副本） | `tar -cf - \| tar -xf -`（**本机无 rsync**） | 18G，rc=2（因 root 文件不可读，见 §4） |
| PHASE2 | `runtime-team.stale-…hostcopy` 45G | 同上 | 完成 |
| HOME | `/home/hliu553` | 同上 + 排除密钥/缓存 | 6.6G |
| MANIFEST | git bundle + Docker/环境清单 | `git bundle create --all`、`git stash show -p`、`docker images/ps/volumes` | 4 bundle + 25 份清单 |

> 为什么用 `tar` 而不是 `rsync`：该机 **未安装 `rsync`**（`which rsync` 为空），且无 sudo 装不了。`tar` 流式拷贝同样保权限/时间戳/符号链接，代价是中断后需重跑该段。

---

## 4 权限缺口：1091 个 root 文件（已补捞，0 失败）

`tar` 以 `hliu553` 身份读取时出现 **1199 条 `Permission denied`**（去重后 **1091 个文件 / 67.4MB**）——成因是**此前往容器里以 root 跑构建**，产物写回 bind-mount 的工作区（归属变成 root）：

| 归属位置 | 文件数 |
|---|---|
| `vllm-ascend/csrc/build/**`（构建产物、autogen 头文件） | **1060** |
| `runtime-team.stale-…hostcopy` | 15 |
| `repro_a1e` | 15 |
| `models/Qwen3-Embedding-0.6B/.msc`（ModelScope 缓存标记，0600 root） | 1 |

**补捞办法**（不装 sudo、不动源）：起一个 **root 容器**、把源**只读**挂进去打包，再以本用户解包（避免共享存储上留一堆 root 文件）：

```bash
docker run --rm --network none --user 0:0 --entrypoint bash \
  -v /mnt/raid/hliu553:/src:ro \
  -v /tgm_rent_data/data/hliu553/migrate_20261021:/dst \
  -v /tmp/unreadable_list.txt:/list.txt:ro \
  <任一已有镜像> -c 'tar -C /src -czf /dst/git_bundles/unreadable_extra_20261021.tar.gz -T /list.txt'
# ⇒ rc=0、0 失败、29M；随后以 hliu553 解包到 raid_hliu553/
```

抽查 `.msc`：**目标侧 sha256 = tar 包内 sha256**（`0cb811c9…`）⇒ 补捞件可信。

---

## 5 本地独有内容（不在任何远端的东西）与留存方式

| 仓 | 未推送提交 | stash | 未提交/未跟踪 | 留存方式 |
|---|---|---|---|---|
| `dc-ws/FlagCX` | **5** | 0 | 0 | `git_bundles/dc-ws__FlagCX.bundle`（含 5 个本地分支） |
| `repro_a1e/FlagCX` | **3** | **1** | **242 项** | bundle + `*.stash0.patch`（543 行）+ `*.dirty-list.txt` + `*.dirty-diff.patch` |
| `runtime-team.stale-…hostcopy` | 0 | **1** | **1276 项** | bundle + `*.stash0.patch`（2716 行）+ 清单 + diff |
| `runtime-team` | 0 | 0 | 0 | 干净（bundle 仍留一份） |

> 这 5 + 3 个"未推送提交"多为 **rebase 前的老副本**（提交信息与已推送版本相同、sha 不同），但**不靠推断**，一律用 bundle 原样留存；bundle 已 `git bundle verify` 全部有效。

---

## 6 Docker：不能靠搬文件，只能靠清单重建

`Docker Root Dir = /mnt/raid/docker`（overlay2，在 RAID 上）。内容**无法用 tar/rsync 语义搬运**（overlay 层 + 引用计数），正确做法是留清单重建：

| 清单 | 条数 | 说明 |
|---|---|---|
| `docker_images.txt` | **57** | `repo:tag` / 大小 / 创建时间 / ID |
| `docker_containers.txt` | **119** | 名字 / 镜像 / 状态（含其他同事的容器） |
| `docker_volumes.txt` | **7** | 含 `flagos-rag-ljy-elasticsearch-data`（**非本方向**） |

- 我方需重建的镜像以 `flagrt/ascend-operator-runtime*`、`flagrt/ascend-operator-runtime-comm` 为主，**配方在仓内 `dev/images/**`**，可离线包或按 `Dockerfile.repro` 重建。
- 我方容器 `repro-910c-a1e` / `repro-910c-aug` 的数据都在 bind-mount 的 `/mnt/raid/hliu553/repro_a1e`（已迁）；容器层只是环境，可重建。
- ⚠️ **Docker 是整机共享**：镜像/容器列表里混有 `cgu135-*` / `rrj-*` / `kelvin-*` / `rag-ljy-*` 等他人资产，本次**未动任何人**的容器与卷。

---

## 7 校验读数（`_manifest/verify_report_*.txt`，逐条可复核）

| 判据 | 读数 |
|---|---|
| 文件数（源排除 venvs/pipcache vs 目标） | **45126 = 45126**（差 0） |
| 字节数 | **68,808,179,582 = 68,808,179,582**（差 0） |
| 抽样 sha256（脚本/日志/模型标记/代码） | 一致（`.msc` 的"DIFF"是**源不可读**导致，已用 tar 内 sha256 证明目标正确） |
| 4 个 git bundle | `git bundle verify` **全部 ✅ 有效** |
| 家目录逐文件比对 | **仅源有 0 个**；仅目标多 1 个 `.local/state/gh/device-id`（无害） |
| 家目录字节 | 目标 6,227,053,878 B（源 6,227,053,842，+36B 来自 `.local` 排除粒度） |
| 目标池余量 | 迁移后 `/tgm_rent_data` 仍有 ~48T 空闲 |

---

## 8 恢复步骤（RAID 重建完成后）

```bash
# 0. 确认 RAID 已就绪且可写
df -hT /mnt/raid && touch /mnt/raid/.wtest && rm -f /mnt/raid/.wtest && echo RAID-OK

# 1a. 回迁到原位置（推荐，所有既有脚本/基线的绝对路径都不用改）
mkdir -p /mnt/raid/hliu553
cp -a /tgm_rent_data/data/hliu553/migrate_20261021/raid_hliu553/. /mnt/raid/hliu553/
cp -a /tgm_rent_data/data/hliu553/migrate_20261021/home_hliu553/. /home/hliu553/   # 按需，注意别覆盖新产生的文件

# 1b. 或者：不回迁，直接在共享存储上干活（改 WORKSPACE_ROOT 即可）
export WORKSPACE_ROOT=/tgm_rent_data/data/hliu553/migrate_20261021/raid_hliu553

# 2. Python 环境重建（venvs 未迁，清单在 _manifest/venvs_inventory.txt）
#    按清单在原路径重建，或改用容器内环境

# 3. Docker 重建（清单独在 _manifest/）
#    - 组内镜像：docker load 离线包，或按 dev/images/**/Dockerfile.repro 重建
#    - 官方镜像：docker pull quay.io/ascend/vllm-ascend@sha256:5cf8a2b6…
#    - 校验：docker images 条数与 docker_images.txt 对得上

# 4. git 侧核对（可选，确认无丢失）
cd /mnt/raid/hliu553/repro_a1e/FlagCX && git bundle verify /tgm_rent_data/.../git_bundles/repro_a1e__FlagCX.bundle
git stash list   # 与 _manifest/*.stash-list.txt 对照

# 5. 清共享存储（⚠️ 确认第 1–4 步都核对完再删）
# rm -rf /tgm_rent_data/data/hliu553/migrate_20261021
```

---

## 9 待办 / 需你决定

| # | 事项 | 说明 |
|---|---|---|
| 1 | **IT 动工前是否停掉我方空转容器** | `repro-910c-a1e`、`repro-910c-aug`（CPU 0.01–0.04%，非在跑任务）；不停也会在重建时被强行中断，提前 `docker stop` 更干净 |
| 2 | **RAID 恢复后回迁还是继续用共享存储** | 回迁则所有既有绝对路径不变（推荐）；留在共享存储则所有 `WORKSPACE_ROOT` / 基线路径要改 |
| 3 | 共享存储副本何时清 | 建议**回迁并核对完成后**再删，本次未删除任何源数据 |
| 4 | 向 IT 确认范围 | 我们的盘点前提是"只重建 md127"；若 IT 连 `/home`（sda5）或 `/data_lib`（nvme3n1）也要动，需要再做一次迁移 |
