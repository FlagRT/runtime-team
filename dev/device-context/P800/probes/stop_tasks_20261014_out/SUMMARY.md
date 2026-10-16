# P800 · 停止我方任务 · 执行记录（2026-10-14）

> 起因：用户要求「停止 P800 上的任务，需要迁移数据」。
> 定位：**操作记录**（不是验证报告）。只作「我停过什么/我有没有动到别人」的可复核入口。
> 原始快照（含全机容器清单）留在机上：`P800:/data2/hliu553/stop_tasks_20261014/`（不随仓库分发）。

## 0 结论先行

| 项 | 结果 |
|---|---|
| **我停的对象** | **仅 1 个**：`hliu553-device-context-p800`（我方唯一在跑的容器，停前 `Up 6 hours`、**内部只有 PID 1 的 `bash`、计算类进程数 = 0**）⇒ 停它**不打断任何取数** |
| **停止时刻** | **2026-10-14 16:50:57**（北京）= `08:50:57Z` |
| **停后状态** | `Exited (137)` · 我方在跑的容器数 = **0** · 宿主**无**我方残留进程 · 设备侧我方占用 **0** |
| 另一个我方容器 | `hliu553-dc-base-p800` 停前即为 `Exited (137) 8 days ago`（未动） |
| **是否动到他人** | ⛔ **未动任何他人容器**；⚠️ 但同一分钟内另有他人容器退出，**时序上均早于我的动作** ⇒ 见 §3 |

## 1 停前的只读盘点（先分清「我的」与「他人的」）

```bash
# 我方容器
docker ps -a --format '{{.Names}}|{{.Status}}' | grep -E '^hliu553'
#   hliu553-dc-base-p800|Exited (137) 8 days ago
#   hliu553-device-context-p800|Up 6 hours          ← 唯一在跑的
# 容器内进程（确认空转）
docker top hliu553-device-context-p800
#   UID  PID     PPID     C  STIME  TTY    TIME      CMD
#   root 2740753 2740721  0  10:54  pts/0  00:00:00  /bin/bash   ← 仅此一条，无 python/torch/vllm
```

我方容器挂载映射（宿主 → 容器）：`/data2/hliu553 → /workspace` · `/data1/dinghaisong/hf_cache → /hf_cache`（他人共享 HF 缓存，只读复用）。

## 2 执行与终态核验

```bash
docker stop hliu553-device-context-p800          # 16:50:57  → Exited (137)
docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}}' hliu553-device-context-p800
#   exited exit=137   （PID 1 是 bash、不处理 SIGTERM ⇒ 10 s 宽限后 SIGKILL，属正常，不是崩溃）
docker ps --format '{{.Names}}' | grep -c '^hliu553'      # → 0
ps -eo pid,user,cmd | grep -E '/data2/hliu553|serve_standard|[v]llm serve'   # → 空（无残留）
xpu-smi   # 卡 0: 1048 MiB → 166 MiB（其余 0；卡 3 的 10928 MiB 是他人作业）
```

## 3 ⭐ 「同一分钟他人容器也退出」的定性（**不是我的动作造成的**）

`docker ps -a` 快照比对发现 2 个他人容器由 `Up` 变 `Exited (137)`。逐条取时间戳后的判读：

| 容器 | 结束时刻（UTC） | 归属 |
|---|---|---|
| `daizijian-flaggems` | `08:50:29` | 他人，**早于我 28 s** |
| `bahejiang-desktop` | `08:50:47` | 他人，**早于我 10 s**（`OOMKilled=false`、exit 137） |
| **`hliu553-device-context-p800`** | **`08:50:57`** | **我方（本次动作）** |
| `liyanz-flaggems-p800` | `08:51:00` | 他人，**晚于我 3 s** |

三条依据说明与我无关：
1. `docker stop <名字>` **只作用于被命名的容器**，不可能连带停别人的；
2. 两个他人容器**在我之前**就已退出 ⇒ 时间上不可能是我的结果；
3. 该机当时正跑**自动编排**：`python3 -B operation/run.py run --vendor kunlunxin --runtime-profile p800-xpytorch29 --oplib both --workload daily …`
   且同一分钟内还**新建了 3 个** `flagperf-op-*` 容器 ⇒ 机器本身在持续起停容器。

⇒ 结论：这批并发退出属**机器级/他人自己的停机**（若这正是你为迁移做的准备，则与本次请求同源）。
**本记录不声称"他人容器由我停止"，也不声称"与我无关"是猜测 —— 依据是上面的时间戳与 docker 语义。**

## 4 我方数据足迹（迁移清单）

| 位置 | 大小 | 说明 |
|---|---|---|
| `/data2/hliu553`（容器内 `/workspace`） | **15 MB** | 全部我方产出：`dc_t1re_20261014/`（本轮 T1-re）· `dc_t1_20261013/` · `dc_e1_20261008/` · `dc_regress_20260929/` · `dc_audit_20261008/` · `prototype/` · `P800/` · `backup_proto_before_acceptance_20260922.tar.gz` · `dcfix.tgz` · `dc_runtime.tgz` · 若干 `out_*`/`*.log` 小文件 |
| `/home/hliu553` | 32 KB | 登录相关 |
| `/tmp` | 7 项 | 我方早期遗留（`dc_r10_emit_patch.py` · `dc_r10_ranges2.json` · `dc_r10_chunk_hash.py` · `x.txt` 等），可删 |
| 我方容器镜像 | —— | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`（**FlagTree 共享镜像**，非自建；按需保留） |

## 5 机器磁盘现状（与「迁移数据」相关的背景）

| 挂载 | 大小 | 已用 | 可用 | 用率 |
|---|---|---|---|---|
| `/` (`/dev/vda3`) | 98 G | 89 G | 5.4 G | **95%** |
| `/data1` = `/var/lib/docker`（`/dev/nvme0n1p1`） | 5.9 T | 5.8 T | **83 G** | **99%** |
| `/data2`（`/dev/nvme1n1p1`） | 5.9 T | 539 G | 5.3 T | 10% |

⚠️ **`/data1` 同时是 `/var/lib/docker`** ⇒ docker 镜像/容器层与 `/data1` 上的数据**共用一个 99% 满的文件系统**；
若迁移目的是腾这个盘，**我方 15 MB 不是主要占用者**（大头在他人数据集/镜像层）。

## 6 复现

```bash
bash -s < probes/survey_before_stop_20261014.sh   # 只读盘点（不改动任何东西）
bash -s < probes/stop_tasks_p800_20261014.sh      # 停我方容器 + 停前/停后比对他人状态
# 恢复（容器是保留可复用的，未删除）：
ssh P800 'docker start hliu553-device-context-p800'
```
