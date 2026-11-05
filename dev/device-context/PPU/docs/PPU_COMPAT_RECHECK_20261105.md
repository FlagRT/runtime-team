# PPU 兼容性对照表 · 归档复核与待上机清单（2026-11-05）

> 本轮目标：按 `docs/CHIP_COMPAT_MATRIX_20261104.md` §6.1「三级采集」与 §6.2「剩余待核项」，把 PPU 六格逐一对到
> **可自查的归档证据**，并写清上机后要跑什么。
> ⚠️ **本轮没有取得任何新的真机读数**：2026-11-05 15:44–15:46 CST 三个 SSH 入口全部不可达（§1）。
> 因此下文所有「已验证」都指**历史归档实测**，不得读作「2026-11-05 上机复核」。

## 0 结论先行

| # | 结论 | 性质 |
|---|---|---|
| 1 | 本机到 PPU 的链路**未建立**：`192.168.5.112:22`、`:2222` 均 `Connection refused`，旧入口 `:36444` `Operation timed out`，且路由走 `en0` 默认网关而非 VPN 隧道 ⇒ 待用户确认 VPN / 新入口 | **事实 + 时间戳**，不归因到厂商或网络层 |
| 2 | 「宿主 OS / 内核**未采集**」这个表述**不准确**：宿主内核有 10-10 实测原文 `5.10.134-17.2.al8.x86_64`；宿主发行版有归档文字记录「Alibaba Cloud Linux 3」，但**缺宿主侧 `/etc/os-release` 命令原文** | 归档复核 ⇒ P0 待核项**收窄** |
| 3 | 「容器基座 Ubuntu 24.04」**不是只靠镜像名推断**：10-10 preflight 有 `/etc/os-release` 实测 `NAME="Ubuntu"` / `VERSION="24.04.2 LTS (Noble Numbat)"` | 归档复核 ⇒ 推断升级为实测（但属 v1 基座容器，v2 需回读） |
| 4 | 驱动 / SDK / 集合通信版本在 v2 验收日志里**与 10-10 逐字一致**，未发现漂移 | 归档复核 ⇒ 三格 ✅ 维持 |
| 5 | PPU 验收路径**不引用 FlagGems**：`PPU/probes/` 下 `flag_gems / use_gems / enable()` 零命中；`serve_standard.sh` 的 `ppu` 分支明确不需要任何厂商插件或平台环境变量 | 离线代码核对 ⇒ 厂商算子路径格补证据 |
| 6 | 六格的状态判定（✅ / 🟡）**本轮一个都不改**：没有新真机读数就不改判定，只改表述与证据 | 判据纪律 |

## 1 本轮连通性实测（只读探测，逐条带时间）

采集时间 **2026-11-05 15:44:19 – 15:46:19 CST**，本机 macOS，命令与读数如下：

| 时间 | 命令（原文） | 读数 |
|---|---|---|
| 15:44:19 | `ssh -o BatchMode=yes -o ConnectTimeout=10 PPU-2 'hostname'` | `ssh: connect to host 192.168.5.112 port 22: Connection refused` |
| 15:44:19 | `ssh -o BatchMode=yes -o ConnectTimeout=10 PPU-2-alt 'hostname'` | `ssh: connect to host 192.168.5.112 port 2222: Connection refused` |
| 15:44:19 | `ssh -o BatchMode=yes -o ConnectTimeout=10 PPU 'hostname'` | `ssh: connect to host ssh-yichang.puhui.chengfengerlai.com port 36444: Operation timed out` |
| 15:44:19 | `ping -c 2 -t 3 192.168.5.112` | `1 packets transmitted, 0 packets received, 100.0% packet loss` |
| 15:44:19 / 15:45:25 | `route -n get 192.168.5.112` | `gateway: 172.25.207.254` · `interface: en0`（**未走 VPN 隧道**） |
| 15:45:25 | `ssh -G PPU-2 \| grep -E '^(hostname\|port\|user)'` | `hostname 192.168.5.112` · `port 22` · `user hliu553`（配置无误） |
| 15:45:25 / 15:46:19 | 重试 `22`、`2222`（`ConnectTimeout=20`、两次） | 仍 `Connection refused` |

⇒ **结论只到「不可达」为止**：本机没有到 PPU 的可用链路，端口拒绝与超时同时存在，无法据此判断机房侧服务状态；
按「外部依赖结论分三层」纪律，不升格为「厂商/网络层不足」。历史口径要求「须先连 gmisp VPN → 内网 192.168.5.112」，
本轮路由未进隧道，需用户确认 VPN 是否连上或给出新入口。

## 2 六格归档复核（证据自包含）

### 2.1 操作系统格 —— 两处表述更正

10-10 的 `PPU/probes/std_steps_20261010_out/preflight_env_ppu_20261010.log` 前 20 行原文：

```
=========== 新芯片接入 · 环境普查 2026-10-10 14:23:20 ===========
host=ai-server  user=root  vendor_arg=（未指定）
-- uname:
Linux ai-server 5.10.134-17.2.al8.x86_64 #1 SMP Fri Aug 9 15:49:42 CST 2024 x86_64 x86_64 x86_64 GNU/Linux
-- os-release:
NAME="Ubuntu"
VERSION="24.04.2 LTS (Noble Numbat)"
-- python（多解释器并列）:  python3 Python 3.12.3 (/usr/local/bin/python3)
-- 容器可见性:  docker 不可用        -- sudo: 需密码 / 不可用
```

⭐ **先自证视角，再谈版本**（这是本轮最容易出错的一步）：这份 preflight 是在**容器内**跑的，五条自证——
① `docker 不可用`；② `sudo 需密码 / 不可用`；③ `python3 = /usr/local/bin/python3 3.12.3`（运行手册记录的**容器内**解释器）；
④ 根文件系统是 `overlay` 且 `/workspace` 挂 11 T 共享盘；⑤ `bond0 192.168.5.112/26`（host 网络）。
⇒ 容器与宿主共享内核，所以 **`uname` 读到的是宿主内核、`os-release` 读到的是容器内发行版**，两者不是冲突，是两层。
⛔ 绝不能把 `Ubuntu 24.04.2 LTS` 写成 PPU 宿主 OS（它与 P800 宿主 OS 恰好同名，极易误填）。

| 项 | 归档事实 | 证据 | 判定 |
|---|---|---|---|
| 宿主内核 | `5.10.134-17.2.al8.x86_64`（`al8` 系） | 上述 preflight `uname` 原文 | ✅ 实测（10-10） |
| 宿主发行版 | `Alibaba Cloud Linux 3` · 192 线程 / 2015 GB | `PPU/README.md` **L61**、`PPU/docs/PPU_BACKEND_ONBOARDING_20261010.md` **L31** | ⚠️ **归档文字记录，缺宿主侧 `/etc/os-release` 命令原文** ⇒ 不写「已验证」 |
| 容器内 OS | `Ubuntu 24.04.2 LTS (Noble Numbat)` | 上述 preflight `os-release` 原文 | ✅ 实测（10-10，**v1 基座容器**） |
| v2 容器 OS | 继承同一基座，`image.baseline.yaml` 基座名含 `ubuntu24.04` | `dev/images/thead-ppu-runtime/v2/image.baseline.yaml` **L18–L21** | ⚪ **未在 v2 上回读** ⇒ 上机补 |

⇒ 表中「宿主 OS/内核 **未采集**」应改为「宿主内核已实测 / 宿主发行版缺直接原文 / 容器 OS 已实测」；
状态仍 **🟡 部分验证**，但待核理由从「整格未采集」**收窄为「只缺宿主 `/etc/os-release` 原文 + v2 容器回读」**。

### 2.2 驱动 / 开发包 / 框架格

| 层 | 归档读数（含原文） | 表中现值 | 本轮判定 |
|---|---|---|---|
| 驱动 | preflight §2：`nvidia-smi -> /usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi` 输出首段 `PPU-SMI 1.28 / Driver Version: 1.5.1-1d747a / HGGC Version: 13.0`；11-03 v2 训练腿日志 `PPU/probes/v2_acceptance_20261103_out/acceptance/s07-trainleg.log` **L24**：`PCCL version 2.1.0-11-accl-p, COMMIT_ID bc333f0, BUILD_TIME 2026-04-28 11:09:52, HGGC Runtime version 13.0, HGGC Driver version 13.0` | KMD `1.5.1-1d747a` / HGGC `13.0` ✅ | ✅ **维持**；v2 与 10-10 读数一致 ⇒ 未漂移。上机按 §6.1 回读 `ppu-smi` 首段 |
| 开发包 | `SDK 2.1.0-a5f865`（`PPU_BACKEND_ONBOARDING_20261010.md` **L33**）；`image.baseline.yaml` **L58–L64**：`ppu_sdk v2.1.0`，装在 `/usr/local/PPU_SDK`（CUDA wrapper cu130），`/opt/accl-p` = 集合通信库 libpccl；preflight §3 列出 `/usr/lib/x86_64-linux-gnu/libnccl-net.so`、`libnccl-net-accl-lite.so` | SDK `2.1.0-a5f865` · PCCL `2.1.0` ✅ | ✅ **维持**，可把 PCCL 精确版补成 `2.1.0-11-accl-p`（COMMIT_ID `bc333f0`）；⚠️ 文件路径属**登记/存在性**证据，不是运行时 dispatch 证据 |
| 框架 | preflight：`torch: 2.10.0 · /usr/local/lib/python3.12/site-packages/torch/__init__.py`、`python3 3.12.3`；`image.baseline.yaml` **L21** `provides: {python: 3.12.3, torch: 2.10.0, vllm: 0.19.0+cu130, transformers: 4.57.0, ppu_sdk: v2.1.0}`；11-03 v2 全量 13 段 `overall: PASS` | torch `2.10.0` · py `3.12.3` · transformers `4.57.0`，🟡（weakref 平台缺陷） | 🟡 **维持**：版本事实已确定，🟡 的理由是**平台侧退出期缺陷（能力缺口）**，不是「版本不确定」；表中措辞应显式区分，别被读成「版本没测」 |

### 2.3 两行算子库格

**「实际使用 · 厂商」✅ —— 补三条离线证据**

| 证据 | 命令 / 读数 | 说明 |
|---|---|---|
| 代码层不启用 | `grep -rInE 'flag_gems\|use_gems\|flaggems\.enable\|VLLM_FL_PREFER\|USE_FLAGGEMS' dev/device-context/PPU/probes --include='*.sh' --include='*.py'` → **0 命中** | PPU 的全部验收探针不引用 FlagGems |
| 服务端配置 | `prototype/scripts/serve_standard.sh` **L201–L215** 的 `ppu)` 分支原注释：「**不需要任何厂商插件/平台环境变量**，也不踩 910C 的坑 A5（`VLLM_PLUGINS`）与 P800 的 `PYTHONPATH`/FlagGems 硬前置」；分支实际只设 `MODEL`、`SERVED_NAME`、`CUDA_VISIBLE_DEVICES`、`SKIP_EXTRA=0`、`DEV_API=cuda` | 与 910C「unset ≠ 禁用」、P800「必须启用插件」两条口径都不同 |
| 通信库名 | 底层库 **PCCL / libpccl**（`/opt/accl-p`），但 PyTorch 后端名是 **`nccl`**（`pccl` 等 ⇒ `Unknown backend type`）；`PPU/README.md` **§2 结论 3** | 「库名 ≠ 后端名」，表中「通信＝nccl」正确 |

⇒ 表中「具体算子库名**未采集**」应改为「厂商 SDK 算子路径已给到登记路径（`/usr/local/PPU_SDK`、CUDA wrapper cu130；通信 libpccl `/opt/accl-p`）」；
⚠️ **运行时 dispatch 归属（谁真正实现了算子）仍需上机三层证据**，可复用 910C §3.1e 的做法：`_dispatch_dump_table` 看注册来源 / `'flag_gems' in sys.modules` / 服务日志平台行。本轮不上机 ⇒ 不写「已实测归属」。

**「替代 · FlagGems 系」🟡 —— 维持，但更正一处易混**

- 表中现值 `FlagGems 5.4.0rc2 · triton 3.6.0（v2 起由 flagtree 0.7.0+ppu whl 固化）` 与 v2 归档一致：
  `image.baseline.yaml` **L25–L37** `flagtree 0.7.0+ppu.gite223f42c`，wheel sha256 `b57daff739…`，`provides: [triton 3.6.0, flagtree]`；
  `lock.yaml` 写明官方基座**不含 triton**（10-27 空跑 `import triton` ⇒ `ModuleNotFoundError`）。
- ⚠️ **别混用 10-10 的旧值**：`PPU_BACKEND_ONBOARDING_20261010.md` **L35** 曾写「`flagtree 0.6.1+ppu3.6` · triton 3.6.0」，
  `PPU/README.md` **§3（2026-10-14 更正）** 已判为**从镜像名 `-3.6-base` 推断的误读**并作废。
- ⇒ 本格 🟡 的含义按 §5.4 口径＝「**能不能替换掉厂商**尚未做结论性验证」；v2 修好的是「服务化不再因缺 triton 起不来」，
  **不等于** FlagGems 算子数值/可用性已通过。要转 ✅ 需上机跑逐算子对照（见 §3），属算子域，本方向不单独下结论。

## 3 上机后要跑的清单（对齐 §6.1 三级采集）

> 一键脚本：`PPU/probes/collect_ppu_compat_20261105.sh`（只读；按 `RUNID` 分目录、只追加、不覆盖旧证据）。
> 下列命令原文已写进脚本，**每条读数都要带采集时间 + 机器名 + 命令原文**（§6.3）。

| 级别 | 要补的具体项 | 命令（脚本内） | 期望 / 判据 |
|---|---|---|---|
| **L0 宿主** | ① **宿主 `/etc/os-release` 原文**（唯一 P0 缺口）② 内核 ③ 驱动/固件 ④ 共享盘水位 | `grep -E '^(PRETTY_NAME\|NAME\|VERSION\|ID)=' /etc/os-release` · `uname -r` · `ppu-smi`/`nvidia-smi` 首段 · `df -h /bmcp_lvm_fs / /data` | 与 §2 归档逐字对齐；**不一致即标「版本已漂移」并改表**。⚠️ 11-03 记录 `/bmcp_lvm_fs` 100% 满（可用 0），本轮必须重测 |
| **L1 容器** | v2 容器 OS + 包版本回读（**不挂设备**即可） | `docker run --rm <v2 镜像>` 内：`/etc/os-release`、`python3 -c 'import torch;print(torch.__version__)'`、`pip list \| grep -Ei 'torch\|triton\|flag\|vllm\|transformers'`、`import triton, flag_gems` | 与 `image.baseline.yaml` **L21 provides** 逐项一致；`import triton` 必须成功（v2 修法的核心） |
| **L2 行为** | 设备可见性 + 最小算子 + 本层 smoke（**显式开关，默认不跑**，避免占共享机卡） | `torch.cuda.device_count()` · `zeros+matmul+sync` · `runtime/smoke_runtime.py --backend ppu`（`DC_ROOT=<原型根>`、cwd=原型根、`CUDA_VISIBLE_DEVICES` 用容器内 0 基） | `device_count>=1`、判据 PASS **且退出码 0**；⚠️ PPU 有平台侧退出期 weakref 缺陷，判据与退出码要同时看 |
| **补 1** | 11-03 遗留：共享盘释放后把 v2 脚本/证据补写到 `/workspace` 归档路径 | `df -h /bmcp_lvm_fs` 判断后再 `rsync`/`cp` | 11-03 因盘满只落在容器 `/tmp` 并标注「非归档路径」，补齐后需在证据索引里改口径 |
| **补 2** | 替代算子库的结论性判据（§5.4 口径） | 复用 `910C/probes/op_parity_20261104/op_parity_ascend.py` 的逐算子重放思路，或最小 GEMM 数值判据 | 属算子/编译域；本方向只保证「判据可复现 + 现象可追踪」，不单独宣称 FlagGems 可用 |

⚠️ **容器前置（照运行手册，缺一就会以「像别的问题」的形态失败）**：`-e NCCL_SOCKET_IFNAME=bond0`、
`--add-host=ai-server:127.0.0.1`、只挂自己用的 `/dev/alixpu*`、`--shm-size`；能连的是 `PPU-2`（`PPU-2-alt` 历史不可用）。

## 4 边界声明

1. 本轮**零新真机读数**；§2 所有读数均注明取得日期（10-10 / 10-27 / 11-03）与文件行号，不写「11-05 实测」。
2. 「不可达」只记事实 + 时间戳（§1），**不归因**；链路恢复后本节需按实测结果改写。
3. 六格状态判定（✅17 / 🟡6 / ⚪1 中的 PPU 部分）**本轮不变**；表格只做表述与证据更正，
   最终按交付要求「合入公共开发分支」应在上机数据齐后一次性收口。
4. 10-10 报告 L35 的 `flagtree 0.6.1+ppu3.6` 属**已作废的推断值**，不得再被引用为 v2 版本事实。
