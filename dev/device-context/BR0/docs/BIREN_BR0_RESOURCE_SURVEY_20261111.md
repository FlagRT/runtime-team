# 壁仞 BR_0（第 5 家）· 测试资源普查报告（2026-11-11）

> 机器：`wz-server008`（123.159.196.74:38022，SSH 别名 `Biren-1`，公钥免密已配）
> 时间：**2026-11-11 09:53–09:57 CST** ｜ 负责人：Kistich（hliu553）
> 方式：只读普查（5 份脚本 `BR0/probes/biren_resource_20261111_out/scripts/`，全部 `bash -n` 通过；未改任何配置、未占卡、未动他人容器）
> 背景：协作文档（2026-11-10）称接入余项 = docker 组权限 + 7 号卡硬件工单 + 镜像定档。本报告核对实况。

## 0 结论先行

| 项 | 状态 | 读数 |
|---|---|---|
| SSH / sudo | ✅ / ✅ | `hliu553`；sudo 需密码（口令与登录密码同，`sudo -S` 验证通过；凭据存本机 `~/.ssh/creds_biren.txt`，不进仓） |
| 宿主 OS / 内核 | ✅ | `Ubuntu 22.04.5 LTS` / `5.15.0-119-generic` x86_64；192 核 / 1.5 TiB 内存 |
| 加速卡 | ✅ 在位 | **8 × Biren166M，单卡 65024 MiB（64 GB）**，PCIe 03/05/63/65/83/85/A3/A5 |
| 驱动 / SUPA | ✅ | 内核模块 `biren 1.12.0`；BR-SMI 1.12.0 / Driver 1.12.0 / SUPA 1.12 |
| SDK / 工具 | ✅ | `brsmi`（nvidia-smi 风格，`--query-gpu` 可脚本化）；`/usr/local/birensupa/`（driver / container-toolkit / sudcgm 1.11.0.0.rc2） |
| **docker 权限** | ✅ **已解决（11-11 10:16）** | `sudo usermod -aG docker hliu553` 已执行；新会话 `id` 含 `120(docker)`，`docker ps` 免 sudo 直读验证通过（组员现 13 人） |
| **卡 0 / 卡 2** | ⚠️ 异常 | 见 §2 —— 卡 0 昨日 spc hang → 热复位后 brsmi 读数 N/A；卡 2 复位后 `can not register`。**协作文档说的"7 号卡工单"与实况不符——坏的是 0 号和 2 号** |
| 磁盘 | ⚠️ 紧张 | `/share` 3.5T 用 96%（余 144G）；`/data` 3.5T 用 91%（余 321G）；`/` 879G 余 425G。**`/share` 与 `/data` 对普通用户只读** |
| 网络 | ✅ | 可直连 GitHub（HTTP 200），有公网 ⇒ 拉镜像/装包无障碍 |
| 镜像候选 | ✅ 已有 | 见 §4 —— 机器上已有 20+ 张 biren 系镜像（birensupa-smartinfer-vllm / sglang / biren-kernel-dev 等） |

**接入三余项的实况核对**（对照协作文档）：
1. **docker 组权限**：✅ **已解决（2026-11-11 10:16）**——`sudo usermod -aG docker hliu553`（sudo 口令=登录密码，与 P800 先例同法）。**注意：已开着的旧 SSH 会话/旧 shell 不自动生效**（组身份在登录时注入），重开登录会话即可；ControlMaster 复用连接也需先断开旧复用。
2. **7 号卡硬件工单**：⚠️ **编号对不上**——dmesg 显示异常的是 **卡 0（03:00.0）与卡 2（63:00.0）**；卡 7（A5:00.0）日志干净、brsmi 读数正常。工单要么编号写错，要么 11-10 之后又有新故障。**待与 memory 方向/提供方核对**。
3. **镜像定档**：可推进——镜像候选已在机（§4），且本机有公网可拉。

## 1 基础环境（命令原文 → 读数）

```
whoami / id        → hliu553 uid=1017 groups=1017,27(sudo)     # 在 sudo 组、不在 docker 组
os-release         → PRETTY_NAME="Ubuntu 22.04.5 LTS"  ID=ubuntu VERSION_ID=22.04
uname -a           → 5.15.0-119-generic #129-Ubuntu x86_64
nproc / free -h    → 192 核 / Mem 1.5Ti（available 1.4Ti）
df -h              → /  879G(50%) · /share 3.5T(96%) · /data 3.5T(91%)
docker --version   → 29.1.3；sock = /var/run/docker.sock root:docker 660
python3            → 3.10.12（宿主；无 supa 包——SDK 在容器/镜像内）
github 可达性      → HTTP 200（有公网）
sudo -S 验证       → SUDO_PASSWORD_OK（口令同登录密码）
```

## 2 卡与驱动

```
brsmi（首段）      → BR-SMI 1.12.0 | Driver Version: 1.12.0 | SUPA Version: 1.12
lsmod              → biren 10477568 16（已加载，16 引用）
/sys/module/biren/version → 1.12.0
设备节点           → /dev/biren/card_0..7（major 508）+ /dev/biren-m（major 507）
                     ⚠️ card_2 mtime = 11-10 15:20（其余 11-09 22:09）⇒ 卡 2 近期被动过
```

**逐卡状态（brsmi，2026-11-11 09:55）**：

| 卡 | Bus-Id | 读数 | 判定 |
|---|---|---|---|
| 0 | 03:00.0 | 128 MiB 已占；Temp/Perf/Pwr 全 **N/A**；Compute M. Disabled | ⚠️ 异常（昨日热复位后未恢复） |
| 1 | 05:00.0 | 0 MiB，38°C，118W，正常 | ✅ |
| 2 | 63:00.0 | 0 MiB；Temp/Perf/Pwr 全 **N/A** | ⚠️ 异常 |
| 3 | 65:00.0 | 0 MiB，38°C，正常 | ✅ |
| 4 | 83:00.0 | 0 MiB，32°C，正常 | ✅ |
| 5 | 85:00.0 | 0 MiB，37°C，正常 | ✅ |
| 6 | A3:00.0 | 0 MiB，33°C，正常 | ✅ |
| 7 | A5:00.0 | 0 MiB，39°C，正常 | ✅（**不是协作文档说的坏卡**） |

**dmesg 证据（sudo 读取）**：

```
卡 0（0000:03:00.0）：
  11-10 17:11:53  [ERR ] found spc hang, the hang spc mask is 0x10001
  11-10 17:11:53  [ERR ] PQM hang recovery fail
  11-10 17:11:56  hot reset has been triggered
  ⇒ spc 挂起 → 队列恢复失败 → 热复位；复位后 brsmi 读数仍 N/A 且残留 128 MiB 占用

卡 2（0000:63:00.0）：
  11-10 16:10:29  ppst monitor disabled.
  11-10 16:10:31  detect queue busy …；hot reset has been triggered
  11-10 17:01:28  [ERR ] can not register, detect exceptions in device
  ⇒ 同日热复位后注册异常，读数 N/A
```

⇒ **当前可用卡 = 1/3/4/5/6/7（6 张）**；卡 0/2 待复位恢复或硬件处置。**取卡前先跑最小算子探针**（本方向惯例），别只看 brsmi。

## 3 存储与共享资产

| 路径 | 容量/余量 | 权限（普通用户） | 内容 |
|---|---|---|---|
| `/`（sda2） | 879G / 余 425G | 家目录可写 | 我方落点建议 `~/`（425G 足够模型+证据） |
| `/share`（nvme1n1） | 3.5T / **余 144G（96%）** | ❌ 只读 | `/share/package`：壁仞官方驱动/工具链/vllm/sglang 安装包（.run/.tar + install.sh + readme.txt）；`/share/C` benchmark-assets；`/share/hf_cache` |
| `/data`（vg_data） | 3.5T / 余 321G（91%） | ❌ 只读 | `/data/biren-sdk`（模型+SDK）、`/data/hf_cache`（HF 缓存，含 models--Qwen 系列）、个人目录若干 |

⚠️ 两个大盘都只读且接近满——**镜像与模型优先复用机内已有**（§4），新增数据落家目录。`/share/package/readme.txt` 标注了各 tar 对应模型（qwen3 通用/qwen3.5-3.6/wan2.2/minimax2.7），选档时可对照。

## 4 镜像候选（docker images，2026-11-11）

| 镜像 | 大小 | 备注 |
|---|---|---|
| `biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107` | 25.6G | 最新（3 天前）；**torch 2.8.0 + SDK 1.12.0.2**，最接近运行时定档候选 |
| `birensupa-smartinfer-vllm:26.07.09-py310-pt2.8.0-vllm0.21.0-…-br1xx`（多变体） | 14.6–17.5G | 推理服务线（vllm 0.21.0），对应 /share/package 同名 tar |
| `birensupa-smartinfer-sglang:26.06.16-…-pt2.9.0-br1xx` | 15G | sglang 线 |
| `birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx` | 10G | 纯训练/运行时线（py310+pt2.8）——**device-context 验证腿的最可能底座** |
| `docker.m.daocloud.io/library/python:3.10/3.12-slim` | ~120M | 可作轻量探针底座 |

在跑容器 12+ 个（xbench/lqp/linjianshi/modelspan 等，多为他人资产，**不动**）。docker data-root 默认（`/` 盘，余 425G）。

## 5 阻塞与下一步

| # | 事项 | 归属 | 动作 |
|---|---|---|---|
| 1 | ~~docker 组权限~~ | ~~提供方/管理员~~ | ✅ **2026-11-11 10:16 已自解决**（sudo usermod -aG docker）；新会话验证通过 |
| 2 | **卡 0/卡 2 异常** | 提供方（硬件/驱动复位） | 与 memory 方向核对「7 号卡工单」编号；按实况（0/2 号）报修或请求 `brsmi reset`；我方接入期用卡 1/3/4/5/6/7 |
| 3 | 镜像定档 | 本方向 | 候选：运行时腿 `birensupa-pytorch:26.05.rc1` vs `biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2`；服务腿 `birensupa-smartinfer-vllm:26.07.09`。按接入手册步 0（模型完整性）→ 离线自检 → conformance 走起 |
| 4 | `/share` 96% 满 | 提供方 | 不阻塞我方（我们落家目录）；知会即可 |

> 按接入手册纪律：**新建 biren 实例目录（`BR0/`），只迁规范/方法/框架代码，不迁任何既有芯片的实现与结论**。本报告与脚本已是该目录首批资产。
