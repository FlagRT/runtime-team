# 寒武纪 MLU590 测试机 · 环境报告（第三实例 · 第 0 步）

> 日期：2026-09-22 ｜ 负责人：Kistich（hliu553）｜ 方向：设备抽象与执行上下文 + 多流 Stream
> 对象：两台测试机 `Mlu-1`（10.1.1.21）· `Mlu-2`（10.1.1.22）
> 方式：**只读探测**（未安装任何包、未改动任何配置、未启停任何服务）
> 原始证据：`probes/preflight_env_mlu1_20260922.log`、`probes/preflight_env_mlu2_20260922.log`
> （由 `prototype/scripts/preflight_env.sh` 一键产出；**本报告是该脚本首次在真机上的实跑结果**）

---

## 1. 结论速览

| 项 | 结论 |
|---|---|
| 登录 | ✅ 两台公钥认证免密可用（`ssh Mlu-1` / `ssh Mlu-2`） |
| 加速卡 | ✅ 每台 **8 × MLU590-M9**，驱动 v6.2.29 / 固件 v1.5.0，健康 `Good`，**单卡 98304 MiB（= 96 GB）** |
| 数据盘 | ✅ 每台一块 **11T** 挂在 `/srv`（Mlu-1 余 **6.7T** · Mlu-2 余 **9.6T**） |
| **docker 镜像是否已在 11T 盘** | ✅ **本来就在**：`/var/lib/docker` 是 **→ `/srv/var/lib/docker` 的符号链接**，无需改 `daemon.json` |
| **能否创建 `/srv/hliu553`** | ❌ **不能**：`/srv` 属主 `root:root 755`，实测 `mkdir: Permission denied` |
| 能否用 docker CLI | ❌ **不在 `docker` 组**（组成员 `gpfs, liangfan1, daizijian, huangxiang, qiyiyan, leihuhu`） |
| 能否 `sudo` | ⚠️ 在 `sudo` 组但**需密码**（`sudo -n` 不可用） |
| 两机共享目录 | ❌ 无；`/srv` 是各自本地盘，数据需分别放 |

**⇒ 硬件与软件环境都具备，当前唯一阻塞是 root 权限（见 §4）。**

---

## 2. 环境明细（两机并列）

| 项 | `Mlu-1` | `Mlu-2` |
|---|---|---|
| 主机名 | `tza-0a06-ai01-em9` | `tza-0a06-ai02-em9` |
| 系统 / 内核 | Ubuntu **20.04.6 LTS** / `5.15.0-139-generic` | 同 |
| CPU / 内存 | 128 核 / **2004 GiB** | 128 核 / **2004 GiB** |
| 账户 / 组 | `hliu553`（uid 1022）/ `hliu553 sudo` | 同 |
| sudo | 需密码 | 需密码 |
| 加速卡 | 8 × **MLU590-M9**（`/dev/cambricon_dev0..7`） | 8 × MLU590-M9 |
| 驱动 / 固件 | `v6.2.29` / `v1.5.0` | 同 |
| 单卡显存 | **98304 MiB**（≈96 GB） | 同 |
| 探测时占用 | 卡 0 被他人占 **33654 MiB**；卡 1–7 空闲 | **8 张全空闲** |
| 厂商工具 | `cnmon` = **CNMON v6.2.29**（`/usr/bin/cnmon`） | 同 |
| 驱动模块 | `cambricon_drv` + `cambricon_peermem` + `cambricon_gdrdrv` | 同 |
| 设备节点 | `/dev/cambricon_dev{0..7}`、`cambricon_ctl`、`cambricon_gdr`、`cambricon_ipcm{0..7}` | 同 |
| 系统盘 | `/dev/sdb5` 437G（余 208G） | `/dev/sdb5` 437G（余 220G） |
| 数据盘 | **`/dev/sda` 11T → `/srv`**（已用 3.3T / 余 **6.7T** / 33%） | **`/dev/sda` 11T → `/srv`**（已用 310G / 余 **9.6T** / 4%） |
| `/srv` 权限 | `root:root 755`（不可写） | 同 |
| docker | 25.0.3；`docker.sock` 属 `root:docker` | 同 |
| docker data-root | **`/var/lib/docker` →(符号链接)→ `/srv/var/lib/docker`** | 同 |
| 容器工具 | `docker` / `nerdctl` / `crictl` / `kubelet` 均在 | 同 |
| 编排 | **K8s 节点**：`kubelet` + `containerd` + `docker` 三服务均 active（有 `/etc/kubernetes`、`/var/lib/kubelet`） | 同 |
| 宿主 Python | `python3` = **3.8.10**；**无 conda** | 同 |
| NeuWare | ❌ 宿主**无 `/usr/local/neuware`** ⇒ MLU 软件栈走**容器镜像** | 同 |
| 私有镜像仓 | `daemon.json` insecure-registries 已配：`docker.cambricon.com`、`docker-user.cambricon.com:30080`、`docker-user.extrotec.com:30080` | 同 |

**网络与源**：两台 `resource.flagos.net` / `pypi.org` / `mirrors.aliyun.com` 均可达（详见原始日志 §5）。

---

## 3. 两处"意外发现"（对接入很有用）

### 3.1 `/srv/hliu553` 建不了，但 docker 镜像**本来就在这块盘上**

探测结论链（证据在原始日志 §4）：
1. `docker info --format '{{.DockerRootDir}}'` → **`未取得`**（我们不在 docker 组，读不到 daemon）；
2. 但 `readlink -f /var/lib/docker` → **`/srv/var/lib/docker`** ⇒ 它是符号链接；
3. `/proc/mounts` 中 docker overlay2 的 `lowerdir/upperdir` 路径全部形如
   `/srv/var/lib/docker/overlay2/...` ⇒ **镜像与容器层实际落在 `/dev/sda`（11T）上**。

⇒ **用户"docker 镜像数据也保存到 `/srv`"这一要求已天然满足，不需要改 `daemon.json` 的 `data-root`。**
（`data-root` 改动需要重启 docker，风险大于收益，**不建议动**。）

### 3.2 ⭐ 他人脚本直接给出了寒武纪 MLU 栈的**确切版本组合**

`/srv/data/build_base_venv.sh`（**他人（`x-benchmark` 负载）的资产，只读可读、不可写**）标题即
"寒武纪 MLU 基础 venv 重建（torch-mlu + triton-mlu + flag_gems）"，内容明确：

```text
python           3.12（脚本注释原文："neuware472 需 py3.12"）
NEUWARE_HOME     /usr/local/neuware
寒武纪私有 wheel 源  https://resource.flagos.net/repository/flagos-pypi-cambricon/simple
寒武纪专属包        torch==2.11.0+cpu
                  torch-mlu==1.33.1+torch2.11.0
                  torch-mlu-ops==1.12.1+torch2.11.0
                  triton==3.4.0+mlu2.1.1        ← 注意是 triton-mlu 变体
通用包            走 mirrors.aliyun.com（numpy 2.2.6 / pandas 3.0.5 / sympy / networkx / jinja2 / ninja …）
FlagGems          codeload 拉 master 源码 → /opt/FlagGems，`--no-deps -e` 安装
目标 venv         /opt/flag_gems_cambricon_venv
```

**价值**：直接回答接入前必须确认的分叉点——
① `torch_mlu` 对 **PyTorch 2.11.0** 的配套组合已明确（此前我们按 torch_fl 的 2.10 口径担心过版本对不上）；
② 依赖获取是**厂商私有源 + 通用源混合**，不是裸 `pip install torch-mlu`。
⚠️ 该脚本里出现 `/workspace/volume/data/...` 这类**容器内路径** ⇒ 它是在容器里执行的，与"环境走容器镜像"一致。

### 3.3 `/srv` 下现有内容属他人，不得触碰

`/srv` 下有 `data/`（**仅 Mlu-1 有**）、`dragonfly/`、`faster-storage/`、`mist/`、`var/`（docker/containerd/kubelet 在用）、`lost+found/`。
其中 `/srv/data/` 是他人资产：`hf_cache/`（**521 个 HF 模型**，**只读可复用**）、`B/`、`C/`（空）、
`build_base_venv.sh`、`setup_envs_314.log`（314 个模型 venv 的构建日志）、`prefetch_314.py`、`postprocess_mlu.py`。

⇒ 我们只**读**（可复用其 HF 模型缓存），绝不写、不删。

---

## 4. 开通需求（**需 root 在两台各执行一次**）

| # | 需求 | 不做的后果 |
|---|---|---|
| **1** | 建数据目录 **`/srv/hliu553`** 并 `chown hliu553:hliu553` | **阻塞**。否则数据只能塞 `/home`（仅 208G / 220G，且两机不共享） |
| **2** | 把 `hliu553` 加入 **`docker` 组** | **阻塞**。我们的接入验证（conformance / 两条腿 / 错误闭环）全部在**带卡容器**内进行 |
| **3** | 确认**容器镜像名与获取方式** | 宿主无 NeuWare ⇒ 必须用寒武纪官方镜像（`daemon.json` 已配私有仓，待确认镜像） |

```bash
# 两台各执行一次
sudo mkdir -p /srv/hliu553
sudo chown -R hliu553:hliu553 /srv/hliu553
sudo chmod 750 /srv/hliu553
sudo usermod -aG docker hliu553        # 执行后需重新登录 SSH 生效

# 复核
ls -ld /srv/hliu553                    # 期望 drwxr-x--- hliu553 hliu553
id -nG | tr ' ' '\n' | grep -qx docker && echo "docker 组 OK"
```

**降级方案**（若管理员只肯给最小权限）：第 1 条是硬需求；第 2 条可退化为"sudo 免密 docker"或"由他人代跑 docker 命令"。

**授权后我们将遵守的数据放置约定**：

```text
/srv/hliu553/
├── prototype/   # 原型代码与脚本（从仓库同步；不在远端直接改）
├── out/         # 阶段产物（conformance / 两条腿 / 错误闭环证据）
├── models/      # 模型（优先复用 /srv/data/hf_cache 只读缓存）
└── images/      # 如需离线搬运镜像 tar（docker 数据本身已在 /srv/var/lib/docker）
```

---

## 5. 下一步（接入手册 8 步的落点）

| 步 | 内容 | 状态 |
|---|---|---|
| 0 | 环境普查（7 项） | ✅ **本报告完成**（可一键复跑 `prototype/scripts/preflight_env.sh`） |
| 1 | 厂商栈判别 + 实机确认 `torch.mlu.device_count() == 8` | 🟨 版本组合已由 §3.2 间接确认；实机验证**需容器** |
| 2 | 镜像就绪（5 条判据） | ⛔ **待 §4 第 3 条** |
| 3 | `prototype/runtime/backends/cambricon/` 落地（`name="cambricon"` / `device_type="mlu"` / `vendor="cambricon"`） | ⏳ |
| 4 | conformance 13 例 + 推理 6 例 | ⏳ |
| 5 | 多流 16 项基线逐项比对 | ⏳ |
| 6 | 训练腿：2 卡 DDP + FlagCX(CNCL) + 三类通信对照 | ⏳ |
| 7 | 产出并入接入手册 SOP + 接口修订建议 | ⏳ |

**当前唯一阻塞 = §4 第 1、2 条（root 权限）**；解除后接入动作本身预计仍在 ≤5 人天口径内
（第二实例实测单日完成）。

---

## 6. 探测边界（如实说明）

- **只读**：未装任何包、未改任何配置、未创建共享位置、未启停服务；探测期间**未占用任何加速卡**
  （只跑 `cnmon info` / `df` / `ls` / `find` / `readlink` 等只读命令）；
- **未取得**：docker 镜像清单与容器清单（无 docker 权限）、`DockerRootDir` 的 CLI 返回值
  （但符号链接关系已由 `readlink -f` + `/proc/mounts` 双重确认）；
- **未验证**：`torch.mlu` 在实机上是否可用（宿主无 MLU Python 栈，须进容器才能验）。

---

## 附：一键复跑命令（供后续复核）

```bash
# 在两台上各跑一次；产出 <OUT>/preflight_env_<date>.log
scp prototype/scripts/preflight_env.sh Mlu-1:/home/hliu553/
ssh Mlu-1 'OUT=/home/hliu553/dc_preflight bash /home/hliu553/preflight_env.sh'
ssh Mlu-1 'cat /home/hliu553/dc_preflight/preflight_env_*.log'
```
