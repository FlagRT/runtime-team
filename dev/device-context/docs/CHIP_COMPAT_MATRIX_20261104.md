# 国产芯片兼容性对照表（设备执行上下文层 · v0.8）

> 起草：2026-11-04 ｜ 责任人：Kistich（hliu553）｜ 状态：**910C 两格 🟡 已处理（§3.1c 框架格 env 层转 ✅ · §3.1d 逐算子对照）+ §3.1e 专项核对「算式子实现归属」（v0.8）；P800 核验完成；MLU590 / PPU 待核**
> 范围：**设备执行上下文层（device-context / 多流 Stream）** 在四款国产芯片上的软件栈兼容性。
> 口径：**只登记本方向实际跑过的组合**；没跑过的一律标 🟡/⚪，**不按厂商宣传或镜像名推断**。

## 0 怎么读这张表

四态判据见 §1。先看这三句：

| 你要问的 | 答案 |
|---|---|
| **哪些是确定的（可直接对外声明）** | **13 格 ✅**（v0.7）：910C 操作系统·驱动·开发包·**框架（`2.10.0.post7`，env 层，形态须随声明一并写出）**；**P800 操作系统·驱动·开发包·框架**；MLU590 驱动·开发包·框架；PPU 驱动·开发包 |
| **哪些要上机核验** | **6 格 🟡 + 1 格 ⚪**。**910C 与 P800 事实项已核完**（910C 算子库已有 184 条逐算子证据，见 §3.1d）；剩余集中在 **MLU590 与 PPU**（两家本轮网络不可达） |
| **哪些是不支持的** | **§5 共 9 条**。真正「芯片级不支持」只有 2 条：MLU590 的 **FP8/MX 量化**（设备能力 `(5,0)` 不满足）与 **FlagTree 无寒武纪线** |

⚠️ **本表不声明的事**：算子库的**逐算子数值正确性**、性能、以及「另三家做不到」这类横向推断。这些属算子/数值域，超出本方向职责（见 §7）。

### 0.1 v0.2 的显式更正（旧值作废，勿再引用）

| 项 | v0.1 记的 | **v0.2 实测** | 说明 |
|---|---|---|---|
| **910C 驱动** | `25.5.0`（ascendhal `7.35.23`），来源 `910C/docs/DIAG_TRAIN_IMAGE_NPU_20260908.md` L64 | **`26.1.1`**（ascendhal `7.35.23`、`package_version=26.1.1`） | ⚠️ **v0.1 的 `25.5.0` 是 09-08 单次采集的历史值，已过期**。本次实测驱动已升级到 `26.1.1` ⇒ 该旧值**作废**。另：`910C/docs/OFFICIAL_RUNTIME_COUNTERPART_20260922.md` L149「现网驱动未采集」的表述**当时是对的**，本条冲突已闭合 |
| **910C 操作系统** | 「宿主 OS **未采集**」 | **`openEuler 22.03 (LTS-SP4)`** / kernel `5.10.0-216.0.0.115.oe2203sp4.aarch64` | ⚠️ **不是 Ubuntu**。宿主是 openEuler aarch64，容器基座才是 Ubuntu 22.04 —— 两者不可混写 |
| **P800 宿主** | 「新机未采集」（现有值全来自 11-07 到期的老机） | **新机 `VM-20-4-ubuntu`：Ubuntu 24.04.2 LTS / kernel `6.8.0-87-generic` / 驱动 `5.0.21.47`** | ✅ 新机与老机**三项完全一致** ⇒ 老机到期不再影响这两格依据 |
| **P800 开发包** | ✅（直接依据基线 `provides`）→ **v0.2 曾误降 🟡** → **v0.3 恢复 ✅** | **`XTDK 3.6.0.1` / `XRE 5.13.0.0` / `XCCL 20260209_9` 有自证文件**：容器内 `<conda env>/lib/python3.10/site-packages/` 下的 torch_xmlir/_versions.txt，2026-11-04 在锁定镜像内 `cat` 实测，三项**逐字一致** | ⚠️ **v0.2 的降级是错的**：当时只搜了 `/usr/local`、`/opt`、`/etc`、`/env`，**没搜 site-packages**；而 `dev/images/kunlun-operator-runtime/v2/lock.yaml` **L13 早就写明**该文件位置。已在 `dev/images/kunlun-operator-runtime/v2/image.baseline.yaml` 的 `provides` 行补上指针，避免再踩 |

## 1 状态定义（四态）

| 标记 | 含义 | 判据（三者同时满足才算 ✅） |
|---|---|---|
| ✅ **已验证** | 该版本组合下，本方向跑过**完整验收**且 PASS | ① 有单入口全量或多流/两腿判据 PASS；② 证据文件在仓内、可复现；③ 版本号来自**实测读取**（非镜像名推断） |
| 🟡 **部分验证** | 能用，但**证据不全**或**带已知限制** | 以下任一：只跑了部分判据 / 版本号来自声明而非实测 / 存在已登记缺陷限定使用面 |
| ⚪ **待核验** | **从未验证**，且当前无任何实测依据 | 只有镜像声明或上游文档，本方向零实测 |
| ⛔ **不支持** | 实测不可用 / 厂商官方明确无此线 / 本层明确不承诺 | 有明确失败记录或厂商矩阵排除依据 |

> **⚠️ 未采集 ≠ 不支持。** 本表用 ⚪ 区分「没测过」与「测过不行」，因为把前者写成后者会让别人白白返工。
> 🟡 与 ⚪ 都属「**上机要核验的**」，区别是 🟡 已有部分证据、⚪ 一片空白。

## 2 主表：四芯片 × 五层

> 每格 = **版本 ｜ 状态**。状态是对「该版本在本层可用」的判定，不是对芯片厂商的判定。
> 证据指针在 §3。**路径基准**：`910C/` `P800/` `MLU590/` `PPU/` 开头的相对 `dev/device-context/`；
> `dev/images/…` 开头的相对**仓库根**。`实测 2026-11-04` 指 §6.4 的采集记录。

| 层 | **昇腾 910C** | **昆仑芯 P800** | **寒武纪 MLU590** | **平头哥 PPU（真武 ZW810E）** |
|---|---|---|---|---|
| **操作系统** | 宿主 **openEuler 22.03 (LTS-SP4)**<br>kernel `5.10.0-216.0.0.115.oe2203sp4.aarch64`<br>**裸金属 `Atlas 800T A3`**（非虚拟化）<br>容器 **Ubuntu 22.04.5 LTS**<br>✅ 已验证 | 宿主 **Ubuntu 24.04.2 LTS**<br>kernel `6.8.0-87-generic`<br>容器 **Ubuntu 22.04.5 LTS**<br>✅ 已验证 | 宿主 **Ubuntu 20.04.6 LTS**<br>kernel `5.15.0-139-generic`<br>容器 **Ubuntu 22.04**（**由包名推断**）<br>🟡 部分验证 | 容器基座 **Ubuntu 24.04**（x86_64）<br>宿主 OS/内核 **未采集**（本轮不可达）<br>🟡 部分验证 |
| **驱动** | **`26.1.1`**<br>ascendhal `7.35.23` · npu-smi `26.1.1`<br>安装于 **2026-09-16**（早于 10-27 验收）<br>固件：平台不支持该查询（见 §3.1b）<br>✅ 已验证 | 宿主 `5.0.21.47`（XPU-RT `5.0.21`）<br>新机与老机**同版本**<br>✅ 已验证 | **`v6.2.29`** / 固件 `v1.5.0` / CNMON `v6.2.29`<br>✅ 已验证 | KMD **`1.5.1-1d747a`** / HGGC `13.0`<br>✅ 已验证 |
| **开发包（SDK）** | 容器 **CANN `9.0.0`**（验收即在它上面）<br>宿主 **CANN `9.1.0`**（`cann -> cann-9.1.0`，另有 `8.5.0`）<br>✅ 已验证 | 宿主 userspace **`xpu-5.0.21.47`**（`/usr/local/xpu/version.txt`）<br>容器 **XTDK `3.6.0.1`** · **XRE `5.13.0.0`** · XCCL `20260209_9`<br>（自证 容器内 torch_xmlir/_versions.txt，容器内）<br>✅ 已验证 | **NeuWare `4.4.3`**<br>cntoolkit `4.4.3-1` · cnnl `2.1.829` · cnnlextra `2.2.7` · mluops `1.8.1` · cncl `1.29.4` · cnclep `1.1.1`<br>✅ 已验证 | **PPU SDK `2.1.0-a5f865`**（CUDA wrapper `cu130`）· PCCL `2.1.0`<br>✅ 已验证 |
| **框架（PyTorch）** | torch **`2.10.0+cpu`** · torch_npu `2.10.0`（镜像）→ **`2.10.0.post7`**（**env 层 overlay，已实测**）· py `3.11.15`<br>post7 下 weakref 回调**触发**、退出码 `0`、区分度 **0.6380 不变**（§3.1c）<br>⚠️ 形态＝**挂载覆盖（环境层）**，镜像内仍 `2.10.0`（缺陷未固化修）<br>✅ 已验证（env 层） | torch **`2.9.0+cu129`** · py `3.10.18`<br>设备 API 走 **`torch.cuda`**（实测 `torch.xpu.is_available()=False`）<br>✅ 已验证 | torch **`2.7.1+cpu`** · torch_mlu **`1.29.2+torch2.7.1`** · py `3.10`<br>退出期 weakref **未中招**<br>✅ 已验证 | torch **`2.10.0`** · py `3.12.3` · transformers `4.57.0`<br>⚠️ 退出期 weakref UAF 为**平台缺陷**，我方运行时已自保<br>🟡 部分验证 |
| **算子库** | flag_gems `0.0.0+f7ae8e6b934a`<br>flagtree `0.6.0+ascend.git15ec1a6c` ⇒ triton **`3.5.1`**<br>flagcx `0.13.0`<br>**逐算子对照已建**：18 个算子 / **184 条重放一致、0 条数值不一致**（§3.1d）<br>**但启用后真实前向失败**：triton 编译期缺 `shmem`；`triu` 可独立复现（§3.1d）<br>🟡 部分验证（证据已硬、缺口已定位） | flag_gems `4.2.1rc0`（`/env/FlagGems`）<br>flagtree **`0.6.1+xpu3.6`**<br>xtorch_ops `0.1.2640`<br>⚠️ GEMM **数值缺口未解**（`K-5`，禁用于关键路径）<br>**路径：vendor 原生，不经过它**（见 §5.4-C）<br>🟡 部分验证 | flag_gems **`5.4.0rc2.post3`**<br>triton **`3.2.0+mlu1.7.2`**（厂商 triton-mlu 变体）<br>torch_mlu_ops `1.8.0`<br>⚪ **本组未做结论性验证** | FlagGems **`5.4.0rc2`**<br>triton **`3.6.0`**（v2 起由 flagtree `0.7.0+ppu` whl 固化）<br>⚠️ v1 无 triton ⇒ 服务化必挂，v2 已修<br>🟡 部分验证 |

### 2.1 计数

| 状态 | v0.1 | v0.2 | v0.3 | **v0.7** | 变化（v0.7） |
|---|---|---|---|---|---|
| ✅ 已验证 | 8 | 11 | 12 | **13** | **+910C 框架格**（`2.10.0.post7` 环境层已实测：回调触发 + `rc=0` + 区分度不变，见 §3.1c） |
| 🟡 部分验证 | 11 | 8 | 7 | **6** | −910C 框架格；剩 910C 算子库 · P800 算子库 · MLU590 OS · PPU OS·框架·算子库 |
| ⚪ 待核验 | 1 | 1 | 1 | **1** | MLU590 算子库（不变） |
| ⛔ 不支持 | 0（格级） | 0 | 0 | **0** | 能力级不支持见 §5 |
| 合计 | 20 | 20 | 20 | **20** | — |

⚠️ **910C 框架格的 ✅ 只对「env 层」成立**：镜像内仍是 `torch_npu 2.10.0`（缺陷版）。
按「挂载覆盖 vs 镜像固化**不可折算**」的纪律，对外声明必须带形态，不得读作「镜像已修」。

## 3 逐格证据指针

> 原则：**每格都要能落到一个文件**。落不到的，就是 🟡/⚪ 而不是 ✅。

### 3.1 昇腾 910C（宿主 `npu1-27`，实测 2026-11-04 09:34）

| 层 | 版本事实来源 | 验收证据 |
|---|---|---|
| OS | **宿主**：`grep -E "^PRETTY_NAME=" /etc/os-release` ⇒ `openEuler 22.03 (LTS-SP4)`；`uname -r` ⇒ `5.10.0-216.0.0.115.oe2203sp4.aarch64`<br>**容器**：`/etc/os-release` ⇒ `Ubuntu 22.04.5 LTS` | `910C/docs/ASCEND_910C_RERUN_20261027.md`（离线 92/0/2 · 审计 73/0/5 · 推理 14/14 · 训练 50 步 6/6） |
| 驱动 | `cat /usr/local/Ascend/driver/version.info` ⇒ `Version=26.1.1` / `ascendhal_version=7.35.23` / `package_version=26.1.1`；`/usr/local/Ascend/version.info` ⇒ `version=26.1.1`；`npu-smi` 自报 `26.1.1` | 同上；⚠️ 固件版本 `npu-smi info -t board` 报 `NA` ⇒ **未获取**，不声明 |
| 开发包 | 容器 `ls /usr/local/Ascend/` ⇒ `cann-9.0.0`；宿主 `ls -l /usr/local/Ascend/` ⇒ `cann -> cann-9.1.0`，另有 `cann-8.5.0` | 同上（验收在容器 CANN 9.0.0 上完成） |
| 框架 | 容器 `pip list` ⇒ `torch 2.10.0+cpu` / `torch_npu 2.10.0`（`pip show` 的 `Version:` 无 `post` 后缀 ⇒ **post7 修复未进镜像**） | `910C/probes/weakref_check_20261103_out/MANIFEST.md`；`910C/docs/ASCEND_WEAKREF_EXIT_UAF_20261103.md` |
| 算子库 | 容器 `pip list` ⇒ `flag_gems 0.0.0+f7ae8e6b934a`、`flagtree 0.6.0+ascend.git15ec1a6c`、`flagcx 0.13.0`；`import triton` ⇒ **3.5.1**；`import flagtree` 自报 **3.5.1**（跟随 triton 线号，同类说明见 `dev/images/thead-ppu-runtime/v2/image.baseline.yaml`） | 随全量验收间接跑通（无逐算子对照）；构建期需 `FLAGGEMS_KERNEL=OFF` 补丁，见 `910C/docs/PROGRESS_20260822.md` **L31** |

### 3.1b 昇腾 910C 深核补充（2026-11-04 第二轮，专查「还有哪些不确定」）

> 目的：把 910C 上**仍标「未获取 / 未核」的项**逐一定性。结论：**没有新增不确定项，但纠正了两处认知**。

| # | 项 | 实测（`npu1-27`，2026-11-04 10:0x） | 定性 |
|---|---|---|---|
| 1 | **机器形态** | `systemd-detect-virt` = **none**；`/sys/class/dmi/id/product_name` = **`Atlas 800T A3`**；`sys_vendor` = `Huawei`；`bios_version` = `32.85`；**640 核 / 2013 GB 内存** | ⭐ **裸金属，不是虚拟机** ⇒ 「固件查不到」不能再用「VM 直通看不到」解释 |
| 2 | **硬件口径** | `npu-smi info -l` ⇒ `Total Count 8`，`NPU ID 0..7` 每个 `Chip Count 2` ⇒ **8 卡 / 16 chip**；`ls /dev/davinci[0-9]*` = **16**；每 chip HBM **65,536 MB** ⇒ 合计 **1,024 GB** | ✅ 闭合（⚠️ 旧文档 `910C/distributed_training/docs/flagcx_ascend_aline_validation_20260824.md` 写的「4 NPU / 8 chip」是 **08-24 当时环境**，非现行口径，已加注） |
| 3 | **整机型号与序列** | `-t board -i 0` ⇒ `Product Name: IT22HMDA_2_S` · `Serial Number: 10255B485124` · `Board ID 0x71` · `Main Board ID 0x14` · `PCIe Bus 0000:9D:00.0` | ✅ 记录（供报障/报厂商时定位） |
| 4 | **固件版本** | `Firmware Version: **NA**` —— **8 个 NPU 全部一致**；`npu-smi info -t firmware` **不是合法类型**（`-t` 合法表里无 firmware）；`-t product` 与 `-t first-power-on-date` 均返回 **`This device does not support querying …`**；`Compatibility` 同为 **`NOT SUPPORT`** | ⭐ **判定为「该平台不支持此查询」，不是「固件不兼容」**。三重佐证：三种查询同源报不支持；8 卡行为一致（非单卡异常）；机器是裸金属（非直通限制）。⚠️ `/usr/local/Ascend/firmware` 是 `dr-xr-x--- root` ⇒ **非 root 不可读**，若确需固件版本须 root 或 BMC |
| 5 | **验收证据是否适用于当前栈** | `driver/version.info` mtime = **2026-09-16 12:47**；`cann-9.1.0` = **2026-10-16**；锁定镜像 `Created = 2026-10-08T23:35Z`；最近一次 910C 全量验收 = **2026-10-27** | ⭐ **三者都早于验收日** ⇒ 10-27 的证据**确实跑在当前栈上**，✅ 成立（按 mtime 推断，非事后重装） |
| 6 | **8 卡健康** | `-t health` ⇒ **8 × 2 chip 全 `OK`**；空闲 die 基线占用 ~2.9–3.2 GB / 65,536 MB | ✅ 无坏卡（与 P800 新机卡 1/2 挂死形成对照） |
| 7 | host CANN vs 容器 CANN | 宿主 `cann -> cann-9.1.0`（另有 `8.5.0`），容器内 `cann-9.0.0` | 两版本并存，验收在容器 9.0.0 上；结论**只对 9.0.0 成立**（已入 §2 表） |

**仍未闭合的（如实标明）**：固件**具体版本号**——用户态不可得，需 root 或 BMC 读取；本方向不为此专门申请权限，除非厂商要求提供。

### 3.1c 910C 框架格：weakref UAF 的修复与实测（2026-11-04，两格之一）

> 问题不是「查不清」，而是**镜像内版本确实带缺陷**。本轮把「怎么修」做实了：**不重建镜像也能修，且已实测修成**。

**缺陷与官方修复**：`torch_npu 2.10.0` 的子类 `tp_dealloc` 覆写后不链式调父类清理 ⇒ weakref 悬空 ⇒ 退出期 SIGSEGV；
Ascend PR #41596 在 **`2.10.0.post7`** 补上 `PyObject_ClearWeakRefs`。

**修复手段 = env 层 overlay（不动镜像）**：把 post7 的**完整包树**放到 `PYTHONPATH` 之前即可生效。
介质已在本机（2026-11-03 A/B 时落地，**本机无外网也能装**）：

```
/home/hliu553/dc-rerun-1027/pkgs_npu_post7/      # 139 MB
├── torch_npu/                                   # 含 _C.cpython-311-aarch64-linux-gnu.so
├── torch_npu-2.10.0.post7.dist-info/
├── torchnpgen/
└── bin/
```

**A/B（同容器、同镜像、同 die2，唯一变量＝`PYTHONPATH`）**

| 臂 | `torch_npu` | weakref 回调 | 进程退出码 | 推理腿 | 区分度 |
|---|---|---|---|---|---|
| **A 基线**（镜像自带） | `2.10.0` | **False**（`called=0`） | **139** | `INFER_LEG_PASS` 14/14 | **0.6380** |
| **B overlay** | **`2.10.0.post7`** | **True**（`called=1`） | **0** | `INFER_LEG_PASS` 14/14 | **0.6380** |

- **空白对照**：普通 Python 对象的 weakref 回调两臂均 `True` ⇒ 排除探针自身问题（判据不是「回调不触发」就是缺陷，得先证明探针能触发）。
- **数值不变**：区分度两臂都是 **0.6380**，与归档值逐字相同 ⇒ 升级**不动读数**。
- ⚠️ **吞吐单跑差异 78.94 → 66.16 句/s（−16%）不作结论**：每臂仅单次、且同机其他 die 有他人负载（die4+ 各占 ~56 GB）。
  若要定论需重复测 + 交错（A/B/A/B）；本轮**不主张性能结论**。

**结论与边界**

| 项 | 结论 |
|---|---|
| 修复是否可行 | ✅ **可行且已验证**（回调触发 + `rc=0` + 读数不变，三判据齐） |
| 形态 | ⚠️ **挂载覆盖（环境层 overlay）** —— 镜像内仍 `2.10.0`，**镜像固化未交付** |
| 我方能否自行固化 | ⛔ 不能：`torch_npu` 版本属基座，**只由总组裁定**（我方只消费 `stack.lock`）。本方向提供「已有官方修复版本 + 升级零读数变化」两项事实作为依据 |
| 待办 | 总组决定是否把 `post7` 纳入基座/下一次镜像重建；纳入后本格可从「env 层 ✅」升级为「镜像固化 ✅」 |

**证据**：`910C/probes/fw_post7_20261104/`（`weakref_out/` 下两臂 env/回调/空白对照日志、`910C/probes/fw_post7_20261104/weakref_out/A_rc.txt`=`139` / `910C/probes/fw_post7_20261104/weakref_out/B_rc.txt`=`0`、推理腿两臂结果、执行脚本 `910C/probes/fw_post7_20261104/run_wr1104.sh` · `910C/probes/fw_post7_20261104/run_fw_ops.sh` · `910C/probes/fw_post7_20261104/run_fw_ops_A.sh`）

### 3.1d 910C 算子库格：逐算子对照（2026-11-04，两格之二）

> 目的：把「只有端到端间接证据」补成**逐算子实测**。结论是**证据变硬了，同时也查出了确证缺口**。

**方法**（新探针 `910C/probes/op_parity_20261104/op_parity_ascend.py`，可在任一家复用）

1. **记录**：在 **vendor 原生路径**下跑真实前向（Qwen3-Embedding-0.6B，3 条文本），用 `TorchDispatchMode`
   捕获**实际执行的每个 aten 算子**及其输入/输出张量；
2. **重放**：`flag_gems.enable()` 后，用**同一份输入**重放每个算子，与步骤 1 的 vendor 输出比对（`rel` + `allclose(1e-3)`）；
3. 纪律：可变算子（in-place）跳过（重放会污染输入）；每算子限 3 个不同签名；记录内存预算 600 MB。

**结果（`910C/probes/op_parity_20261104/out/op_parity.tsv` · `910C/probes/op_parity_20261104/out/summary.json`）**

| 项 | 读数 |
|---|---|
| 我方前向实际执行的唯一 aten 算子 | **27** 个 |
| 重放比对条数 | **304**（`mul` 1104 次、`_to_copy` 696、`add` 678、`matmul` 591 …） |
| 比对结论 | **PASS 184 / FAIL 0 / SKIP 120** |
| 被 flag_gems 覆盖且**数值一致**的算子 | **18** 个：`slice` `neg` `view` `t` `mul` `unsqueeze` `transpose` `rsqrt` `mean` `matmul` `lift_fresh` `expand` `cat` `all` `add` `pow` `sin` `embedding` |
| 数值不一致 | **0**（这 18 个算子上 rel 全部在 `1e-3` 容差内） |

**SKIP 120 条必须分三类看（混写会得出错误结论）**

| 类别 | 条数 | 定性 |
|---|---|---|
| `aten.silu`(22) + `aten.lift_fresh`(56) 重放报错 | 78 | **探针自身姿势问题，不是缺陷**：最小独立复验 `910C/probes/op_parity_20261104/op_min.py` 证明 `gems.silu`、`gems.lift_fresh(空张量)` **均正常**（`ok=True`，与 vendor 同值）⇒ 不得计为算子缺陷 |
| `aten.triu`(21) `CompilationError` | 21 | **确证缺陷**：`910C/probes/op_parity_20261104/op_min.py` 最小脚本**可独立复现**（`gems.triu` → `CompilationError: at 16:4`，vendor 正常） |
| `npu.npu_fusion_attention_v3`(20) | 20 | flag_gems **未覆盖**（厂商融合算子）⇒ 走 vendor 回落，属预期 |
| 其余（`sig_cap`/`budget`/`arg_type`） | — | 探针限流，非缺陷 |

**端到端判定：启用 FlagGems 后我方模型前向直接失败**

```
[gems-forward] FAILED  CompilationError: at 44:4:
   ModuleNotFoundError("No module named 'shmem'")
```
（`910C/probes/op_parity_20261104/out/gems_forward.log`；同一进程内 vendor 路径正常）

**归因（已定位到具体依赖，不泛指「不兼容」）**：`shmem` 这个模块在
`flag_gems/**.py` 与 `triton/language/**` 里**都搜不到** ⇒ 它由 **flagtree/triton-ascend 在编译期引入**，
属**后端栈依赖缺口**；镜像内既无该 Python 包、也无 wheel，且本机**无外网**（`pypi` 全部 `http=000`）。

**结论**

| 项 | 结论 |
|---|---|
| 910C 上 FlagGems 能否承载我方模型前向 | ⛔ **当前不能**（编译期缺 `shmem`；`triu` 另可独立复现失败） |
| 「我方不启用 FlagGems」的性质 | **既是设计选择，也是当前的必要** —— 装了也用不起来 |
| 该格状态 | **仍 🟡，但性质已变**：从「缺证据」→「**有 184 条逐算子证据 + 缺口定位到具体依赖**」 |
| 修复路径 | ① **补 `shmem`**（来源需厂商/算子组确认，镜像内无、网络不可达）→ 重跑本探针；② **`triu` 编译失败**单独立项给算子/编译组；③ 我方声明面不变（vendor 原生路径，见 §5.4-C） |

**证据**：`910C/probes/op_parity_20261104/`（`910C/probes/op_parity_20261104/op_parity_ascend.py` · `910C/probes/op_parity_20261104/op_min.py` · `910C/probes/op_parity_20261104/gems_forward.py` · `910C/probes/op_parity_20261104/out/console.log` · `910C/probes/op_parity_20261104/out/op_parity.tsv` · `910C/probes/op_parity_20261104/out/summary.json` · `910C/probes/op_parity_20261104/out/op_min.log` · `910C/probes/op_parity_20261104/out/gems_forward.log`）

### 3.1e 910C 到底谁在实现算子：厂商算子库 vs FlagGems（2026-11-04 专项核对）

> 问题：910C 上我方路径是否真的全走**厂商算子库**？「不用 FlagGems」是声明还是事实？
> 结论：**是事实，且有三层独立证据**；但核对过程同时**推翻了脚本里一句错误因果**（`unset VLLM_PLUGINS` 不等于「禁用插件」）。

#### A. 证据一（注册层，不需要设备）：谁注册了 `PrivateUse1` 的实现

`torch._C._dispatch_dump_table(op)` 直接看 dispatch 表里 `PrivateUse1` 的实现来源：

| 算子 | 我方路径（只 `import torch, torch_npu`） | 手动 `flag_gems.enable()` 后 |
|---|---|---|
| `aten::mm` | `torch_npu/csrc/aten/RegisterNPU.cpp:33542` | **`flag_gems/__init__.py:62`**（被接管） |
| `aten::add.Tensor` | `torch_npu/csrc/aten/RegisterNPU.cpp:33542` | **`flag_gems/__init__.py:62`**（被接管） |
| `aten::silu` | `torch_npu/csrc/aten/RegisterNPU.cpp:33542` | 同左（未列出覆盖） |
| **`aten::matmul`** | `torch_npu/csrc/aten/RegisterNPU.cpp:33542` | **仍是 `torch_npu`** ⇒ ⭐ **flag_gems 未实现 `matmul`**，即使 enable 也回落厂商 |

⇒ 我方路径下 **所有**算子都注册在厂商 C++ 侧；且顺带查到一个事实：**`matmul` 不在 flag_gems 覆盖范围内**（只有 `mm`）。

#### B. 证据二（运行期，进程级）：我方各腿**根本不 import** `flag_gems`

- 21 个入口文件静态检查（§5.4-C）：`import flag_gems` / `use_gems()` / `enable()` **全为 0**；
- 运行期：跑真实前向时 `'flag_gems' in sys.modules` = **False** ⇒ **连模块都没加载**，不是「加载了但没用」。

#### C. 证据三（服务端）：插件**加载 ≠ 接管**，且"unset"的语义与脚本注释不符

服务端 `prototype/scripts/serve_standard.sh` 的 ascend 分支执行 `unset VLLM_PLUGINS`，原注释写的是"禁用 vllm-plugin-FL"。**实测三态对照**（`VLLM_PLUGINS` 取不同值后调 `load_general_plugins()`）：

| `VLLM_PLUGINS` | `vllm_fl` 是否加载 | `flag_gems` 是否进 `sys.modules` | 平台解析结果 |
|---|---|---|---|
| **unset**（我方现行） | **是** | **是** | 探针里为 `vllm_fl.platform.PlatformFL` |
| **`""`（空串）** | **否** | **否** | `UnspecifiedPlatform` |
| `fl` | 是 | 是 | `PlatformFL` |

vLLM 源码语义（容器内 vllm/plugins/__init__.py L32/L56）：`allowed_plugins is None or plugin.name in allowed_plugins`
⇒ **`VLLM_PLUGINS` 未设（None）= 加载全部插件**；**空串才是一个都不加载**。
⇒ ⚠️ **「unset = 禁用插件」是错的因果**，已修正脚本头注释与该分支注释（代码行为未改）。

**但「加载」不等于「接管」**：手动 `load_general_plugins()` 让 `vllm_fl`/`flag_gems` 都进来之后，
`aten::mm / add / silu / matmul` 的 `PrivateUse1` 实现**一条都没变**（仍全是 `RegisterNPU.cpp`）⇒ 插件**不调** `enable()`。

**真实服务日志（ground truth）**：`910C/probes/r10_serve_910c_npu_20261008_out/tp2_eager1/vllm_serve.log`

```
INFO [__init__.py:238] Platform plugin ascend is activated      ← 平台是 ascend，不是 fl
SERVE_STANDARD_PASS (ready=1 smoke=1)                            ← 同轮服务判据通过
```
且该日志中 `flag_gems` / `flaggems` / `vllm_fl` / `flagos` **零命中**。
机制：`fl` 的平台插件在 ascend 上返回 `None`（无 ascend 后端）⇒ 只有一个 out-of-tree 平台被激活 ⇒ `ascend` 胜出。

#### D. 结论与待办

| 问题 | 结论 |
|---|---|
| 910C 我方路径是否走厂商算子库 | **是**（三层证据：注册层全 `RegisterNPU.cpp` + 运行期不 import + 服务端平台 `ascend`） |
| 是否"曾经接管过" | 无证据：服务日志零 FlagGems 痕迹；各腿从不 enable |
| 「unset VLLM_PLUGINS = 禁用插件」 | ⛔ **错误因果，已更正为「加载全部插件」**；现行行为仍安全（平台是 ascend、不 enable） |
| 是否改用 `VLLM_PLUGINS=""` 真禁掉插件 | **候选，未做**：能连 `import` 都省掉，但属行为变更，需一次服务回归（`SERVE_STANDARD_PASS` + 消费方 10/10）才可改默认值 |
| 该结论对算子库格的影响 | 支持 §3.1d 的判断：**910C 上 FlagGems 既未被使用、当前也无法使用**（缺 `shmem`） |

### 3.2 昆仑芯 P800（新机 `VM-20-4-ubuntu` / `P800-new1`，实测 2026-11-04 09:34）

| 层 | 版本事实来源 | 验收证据 |
|---|---|---|
| OS | `grep PRETTY_NAME /etc/os-release` ⇒ `Ubuntu 24.04.2 LTS`；`uname -r` ⇒ `6.8.0-87-generic`；容器 ⇒ `Ubuntu 22.04.5 LTS` | `P800/docs/KUNLUN_P800_MIGRATION_20261103.md` **§5.1b**（离线 43/0/1 · 推理 13/13 · 区分度 0.6392） |
| 驱动 | `xpu-smi` 首行 ⇒ `Driver Version: 5.0.21.47  XPU-RT Version: 5.0.21`；`/usr/src/kunlun-5.0.21.47`、`/usr/local/xpu-5.0.21.47` 存在 | 同上；⭐ 与老机 `P800/docs/KUNLUN_P800_ENV_REPORT_20260914.md` **L68** 的 `5.0.21.47` **一致** |
| 开发包 | 宿主：`/usr/local/xpu/version.txt` ⇒ `5.0.21.47`（老机与新机同）<br>容器：`cat <conda env>/lib/python3.10/site-packages/torch_xmlir/_versions.txt` ⇒ **`XTDK: 3.6.0.1` / `XRE: 5.13.0.0` / `XCCL: 20260209_9`**（另含 `XPYTORCH: XMLIR--bc1b1dc6f-dev+2026032411`、`XHPC: 20770323`、`CUPTI: 1.1.0.0`、`compile_date: 20260324-11:53:49`、`arch: x86_64`） | 与 `dev/images/kunlun-operator-runtime/v2/image.baseline.yaml` **L16** 的 `provides` **逐字一致** ⇒ ✅ |
| 框架 | 容器 ⇒ `torch 2.9.0+cu129`（`torch.version.cuda=12.9`）、py `3.10.18`；`hasattr(torch,"xpu")=True` 但 **`torch.xpu.is_available()=False`**（另有 `torch_xray 2.0.4` 负责符号重写） | 同 §5.1b；命名空间结论与 `P800/README.md` **L74** 一致 |
| 算子库 | 容器 `pip list` ⇒ `flag_gems 4.2.1rc0`（editable `/env/FlagGems`）、`flagtree 0.6.1+xpu3.6`、`flagcx 0.10.0`、`xtorch_ops 0.1.2640+9bc36a18`、`xmlir 1.0.0.1` | GEMM 编译维度已在 `env/v1` 修复；**数值缺口 `K-5` 未解**，见 `dev/stack.lock.kunlunxin.v1.yaml` known_issues |

### 3.3 寒武纪 MLU590（本轮网络不可达，沿用 09-22 / 11-03 记录）

| 层 | 版本事实来源 | 验收证据 |
|---|---|---|
| OS / 驱动 | `MLU590/docs/CAMBRICON_MLU_ENV_REPORT_20260922.md` **L33**（`Ubuntu 20.04.6 LTS / 5.15.0-139-generic`）、**L38**（驱动 `v6.2.29` / 固件 `v1.5.0`）、**L41**（CNMON `v6.2.29`） | `MLU590/docs/CAMBRICON_MLU_RERUN_20261103.md`（单入口 9 段 `overall: PASS`：离线 103/0/0 · smoke 46/0 · 审计 63/0/15） |
| 开发包 | `dev/images/cambricon-mlu-runtime/v1/image.baseline.yaml` **L72–L82**（neuware `4.4.3` + 六个包级版本，均 `.ubuntu22.04`） | 同上 |
| 框架 | 同上 **L25–L43**（torch `2.7.1+cpu` / torch_mlu `1.29.2+torch2.7.1` / torch_mlu_ops `1.8.0`） | 同上；weakref 未中招见 `MLU590/probes/weakref_check_20261103_out/MANIFEST.md` |
| 算子库 | 同上 **L45–L58**（triton 在 `/opt/triton`、flag_gems `5.4.0rc2.post3`；文件原话：**「本组尚未在本芯片上做结论性验证」**） | **无** ⇒ ⚪ |

### 3.4 平头哥 PPU（本轮网络不可达，沿用 10-10 / 11-03 记录）

| 层 | 版本事实来源 | 验收证据 |
|---|---|---|
| OS / 开发包 | `dev/images/thead-ppu-runtime/v2/image.baseline.yaml` **L18–L21**（基座 `…-ubuntu24.04:202607-3.6-base`；`provides: {python: 3.12.3, torch: 2.10.0, vllm: 0.19.0+cu130, transformers: 4.57.0, ppu_sdk: v2.1.0}`） | `PPU/docs/PPU_V2_ACCEPTANCE_20261103.md`（13 段 `overall: PASS`：训练 loss 11.1530 · 推理 13/13 · 区分度 0.6391 · 审计 73/0/5） |
| 驱动 | `PPU/docs/PPU_BACKEND_ONBOARDING_20261010.md` **L33**（`KMD 1.5.1-1d747a` · SDK `2.1.0-a5f865` · HGGC `13.0` · PCCL `2.1.0`） | 同上 |
| 框架 | image.baseline **L40–L55** | 同上；weakref 平台缺陷见 `PPU/docs/PPU_VENDOR_WEAKREF_ISSUE_20261026.md` |
| 算子库 | image.baseline **L25–L37**（flagtree `0.7.0+ppu.gite223f42c` whl 提供 triton 3.6.0） | ⚠️ **v1 曾无 triton**（`import triton` 报 `ModuleNotFoundError`，见 `PPU/README.md` **L105**）⇒ v2 加派生层修复 |

## 4 补充：推理服务栈（vLLM 线）单列

> 单独列出的理由：四家差异最大、踩坑最多，且**它与「框架」不是同一件事**。

| 项 | 910C | P800 | MLU590 | PPU |
|---|---|---|---|---|
| vLLM | **`0.20.2+empty`**（editable `/vllm-workspace/vllm`） | **`0.13.0`** | **`0.20.2`** | `0.19.0+cu130` |
| 插件 | `vllm-plugin-fl 0.0.0+g8b059122e`（+1 补丁） | `vllm-plugin-fl 0.1.0`（editable `/env/xvllm-plugin-FL`） | `vllm-plugin-FL **0.2.2**` | 无（镜像自带） |
| **必设环境变量** | ⚠️ **`unset VLLM_PLUGINS`**（现行做法；**注意它不是「禁用插件」**——见 §3.1e） | ⚠️ 必须 **`VLLM_FL_PREFER=vendor`** + `PYTHONPATH=/env/FlagGems/src` | ⚠️ 必须 **启用**插件（勿套 910C 的 unset） | 两者都**不需要** |
| 服务化状态 | ✅ `SERVE_STANDARD_PASS` | ✅ `SERVE_LEG_PASS 10/10` | ✅ `SERVE_STANDARD_PASS`（ready=1 smoke=1） | ✅ `SERVE_LEG_PASS 10/10` |

⭐ **这四行是反的**：910C 要求 **关闭**插件、MLU590 要求 **开启**插件。**「按芯片相反」这条已写成实测结论，别照抄隔壁家。**

## 5 不支持 / 已知限制清单（能力级）

> **分三类，别混写**：① 芯片级不支持（厂商能力边界）② 我方不承诺（职责边界）③ 已知缺陷但可规避。

### 5.1 芯片级不支持（厂商能力边界，不可通过我方规避）

| # | 芯片 | 项 | 依据 | 影响 |
|---|---|---|---|---|
| N-1 | MLU590 | **FP8 / MX 量化不可用** | `torch.mlu.get_device_capability()` 返回 **`(5, 0)`**；厂商 vLLM 手册标 FP8 需 ≥`(6,0)`、MX 需 ≥`(6,1)` | 量化路径只能用 SmoothQuant 一类；**别按 910C/PPU 的量化方案平移** |
| N-2 | MLU590 | **无 FlagTree 构建物线** | BAAI·FlagTree 官方手册**无寒武纪条目**；镜像内 triton 是厂商 `triton-mlu` 变体 | 不能用 FlagTree 统一编译层；版本锚只能靠 Harbor digest |

### 5.2 我方不承诺（职责边界，不是芯片缺陷）

| # | 项 | 说明 |
|---|---|---|
| N-3 | **算子逐算子数值正确性** | 仍属我方**不承诺**项（只保证「上下文层/多流」行为）。⚠️ **2026-11-04 更新**：**910C 已补逐算子对照**（18 算子 / 184 条重放 / 0 数值不一致，见 §3.1d）；**P800 / MLU590 / PPU 仍未做** —— 别把 910C 的结论外推到另三家 |
| N-4 | **性能 / 吞吐横向可比** | 四家基准模型相同（`Qwen3-Embedding-0.6B`），但**卡型、卡数、单请求延迟量级差异极大**（MLU590 单请求 p50 约 645 ms、P800 约 57 ms）⇒ **吞吐数字不可横比** |
| N-5 | **「另三家做不到」类推断** | 禁止从两家结论外推第三家。每家结论只对本机、本镜像、本卡成立 |

### 5.3 已知缺陷（可规避，已登记）

| # | 芯片 | 缺陷 | 规避 | 状态 |
|---|---|---|---|---|
| N-6 | 910C | `torch_npu 2.10.0` 退出期 weakref UAF（`rc=139`） | 升 `2.10.0.post7`（Ascend PR #41596）；或我方运行时侧已不依赖弱引用语义 | **镜像内仍是 `2.10.0` ⇒ 固化未交付**；但 **2026-11-04 已实测 env 层修法有效**（post7 overlay：回调触发 + `rc=0` + 区分度不变，见 §3.1c）。固化属总组基座裁定 |
| N-7 | PPU | `torch 2.10.0` 同类 weakref UAF（平台侧） | 运行时等价补丁（`PPU/probes/weakref_check_20261103_out/wrfix.c`）—— **止血，不进镜像/不进交付包** | 已对外提交厂商，**未修** |
| N-8 | P800 | FlagGems GEMM **数值缺口**：64³ fp32 非确定性乱值（同输入 rel 0.29~1.02） | 禁用于关键路径；可用面限 pytest 18/99 矩阵 + minirepro 12 组 | **未解**，`K-5` |
| N-9 | P800 | 流优先级**退化单点**：`range=(0,0)`，「能设置 ≠ 有效果」 | 不做基于优先级的调度决策 | 已登记 `KUNLUN-STREAM-PRIORITY-SINGLE-LEVEL` |

### 5.4 「算子库」这格怎么才能变 ✅ —— 判定口径与三条路径（2026-11-04）

> 针对 P800 这格停在 🟡（K-5 GEMM 数值缺口）。**先说清性质：它不是「没测清」，而是「测清了、确实有缺陷」。**

#### A. 现状是「已确定」而非「未确定」

| 项 | 状态 |
|---|---|
| 判据 | **已存在且可自判定**：`dev/issues/kunlunxin-gemm-numeric-64cube/assets/verify_numeric_64cube.py`（三态：`overall: PASS/0` = 三次同输入 rel ≤ 1e-3 且一致；`FAIL/1` = 超差或漂移；`ERROR/2` = 环境不满足） |
| 最近一次实测 | **2026-10-28，env/v1 卡 0：FAIL** —— 三次 trial `rel = 0.97 / 0.79 / 0.98`（同输入漂移），vendor 原生 `max_abs` ~1e-5 |
| 归属 | **算子/编译域**（FlagTree `@edca42a`，编译组）；已提交含 8 行最小复现 + 排查建议（split-K/atomic_add 写回、64 整倍 tile 边界） |
| 我方职责 | 按 §5.2 **N-3**，算子逐算子数值正确性**不在本方向承诺内**；本方向只负责「判据可复现 + 现象可追踪」 |

⇒ **再跑一遍就能确认当前状态**，不需要新设计判据。若要在新机上跑，见 §5.4-D。

#### B. 三条路径（选一条才可能从 🟡 → ✅）

| 路径 | 做什么 | 谁能做 | 代价 / 风险 |
|---|---|---|---|
| **A 等上游修** | 编译组修 split-K / 64 整倍 tile 边界 → 复跑判据 `overall: PASS` → 删 K-5、本格转 ✅ | 编译组（我方被动等） | 时间不可控；期间本格只能维持 🟡 |
| **B 收窄声明面**（推荐） | 把「算子库已验证」重新定义为「**本方向承诺的算子使用面**内已验证」，并显式列出**排除项**（FlagGems GEMM 的 64 整倍边界等） | **我方**（口径决定，需你拍板） | ⚠️ **必须四家统一改口径**，否则表不可比；且对外承诺范围变小，需要写清「不含 FlagGems GEMM 关键路径」 |
| **C 兜底路由** | 按形状把命中 64 整倍边界的 GEMM 路由到 vendor 原生实现（与 P800 已有的 `VLLM_FL_PREFER=vendor` 同思路），使**关键路径不受影响**并可被实测证明 | 我方（工程） | 只能把「关键路径可用」变成实测结论，**不能**让 FlagGems 缺陷消失；多一层路由即多一处需维护 |

⭐ **我的建议：B + C 并行，A 作追踪项。** 理由：K-5 属算子域、修复权不在我们手里；而本表要回答的问题是「**设备执行上下文层在这款芯片上能不能用**」，不是「FlagGems 每个形状都对」。收窄声明面 + 兜底路由能把这个问题的答案变成 ✅，同时把 K-5 完整保留在 §5.3（不隐藏）。

#### C. 各腿/判据的算子路径逐条核对（K-5 是否落在我方路径上）

> 结论：**训练腿、推理腿、多流、conformance、服务化消费方全部走厂商原生算子路径，从不进 FlagGems 的 GEMM。**
> 依据不是推断，而是**逐文件核对**（2026-11-04，`prototype/` 内 21 个入口文件全部扫过，**启用 FlagGems 的 = 0**）。

| 腿 / 判据 | 入口文件 | `import flag_gems` | `use_gems()` / `enable()` | 实际算子路径 |
|---|---|---|---|---|
| **训练腿**（50 步微调） | `prototype/runtime/proto/proto_train_leg.py` | 无 | 无 | **vendor 原生** |
| **推理腿**（13 项前向） | `prototype/runtime/proto/proto_infer_leg.py` | 无 | 无 | **vendor 原生** |
| **多流**（语义/配额/优先级/图捕获，14 个探针） | `prototype/probes/probe_stream_*.py` · `prototype/probes/audit_stream_priority_*.py` · `prototype/probes/probe_graph_capture_stream_v2.py` | 无 | 无 | **vendor 原生** |
| **conformance 13 例 + 推理 6 例 + 契约不变式 I1–I4** | `prototype/runtime/conformance/`{`runner`,`cases`,`infer_cases`,`contract_invariants`}`.py` | 无 | 无 | **vendor 原生** |
| **服务化消费方**（10/10 判据） | `prototype/runtime/proto/proto_infer_serve.py` | 无 | 无 | **vendor 原生** |
| **服务化服务端**（vLLM 进程） | `prototype/scripts/serve_standard.sh` | 脚本本身不 import | **配置层显式关闭** | `VLLM_FL_PREFER=vendor` + **`USE_FLAGGEMS=0`**（**L166**）⇒ dispatch 强制 vendor |
| 离线自检 / 跨后端对称性 / 两腿口径 | `prototype/scripts/backend_offline_check.py` · `prototype/scripts/model_path_parity_check.py` · `prototype/scripts/run_legparity_dcmodel.sh` | 无 | 无 | 纯契约校验，不跑算子 |
| 错误恢复 / 压力 | `prototype/runtime/proto/proto_error_recovery_loop.py` · `prototype/probes/recover_*.py` | 无 | 无 | **vendor 原生** |

⚠️ **必须分清两种「不走 FlagGems」**（性质不同，不可混写）：

1. **代码层不启用**（前 5 行 + 探针）：我方脚本**从不 import `flag_gems`、从不调 `use_gems()`/`enable()`** ⇒ 天然走 vendor 原生。
2. **配置层强制关闭**（服务化服务端）：vLLM 的默认 dispatch **会**优先 flagos（FlagGems），所以必须显式
   `VLLM_FL_PREFER=vendor` + `USE_FLAGGEMS=0` 才走 vendor —— 这不是"不启用"，是**主动关掉**（与 K-2 的规避同一条配置）。

⚠️ **一个容易被误读的点（先自证）**：`prototype/scripts/serve_standard.sh` **L164** 确实把 `PYTHONPATH=/env/FlagGems/src` 挂上了，
看起来像"在用 FlagGems"。但脚本原注释（L162–L163）写明它的作用是 **import 完整性**——
镜像 site-packages 里的 `flag_gems` 子模块不完整，`vllm_fl` import 会失败并报 `Failed to infer device type`
（看着像设备问题，其实是包问题）。**它只为"能 import"服务，dispatch 走哪条路由 L166 的 `USE_FLAGGEMS=0` 决定。**
⇒ 判定时不能只看 PYTHONPATH 有没有它，要看 `USE_FLAGGEMS` / `VLLM_FL_PREFER`。

**间接佐证**：四家推理区分度逐字接近（910C `0.6380` / P800 `0.6392` / MLU590 `0.6391` / PPU `0.6391`）——
若 P800 的 GEMM 走了坏路径（rel 0.29~1.02 的乱值），读数不可能与另三家对齐到 1e-3 量级。

⇒ **K-5 对「我方验收路径」的实际影响 = 未观测到，且代码/配置两层都可自证不经过它**；
但它对「**用 FlagGems 跑 GEMM 的下游**」是真实风险，**必须保留在 §5.3 不隐藏**。
⇒ 这正是路径 B（收窄声明面）可成立的技术依据：我方承诺面 = **vendor 原生算子路径**，FlagGems GEMM 不在其中。

#### D. 要在新机复跑判据，缺什么（老机 11-07 到期，有归档风险）

`env/v1` 需要**宿主仓检出** `FlagGems/`（分支 `xliu969/kunlun-gemm-compat`，基点 `73c5aff1` + patch 04/05/06），
新机 `P800-new1` **没有这份**。可重建（不需要老机）：

```bash
# 1) 基础树：锁定镜像内 /env/FlagGems 就是 73c5aff1，直接取出
docker run --rm -v /data2:/data2 --entrypoint /bin/bash <锁定镜像> \
  -lc "cp -a /env/FlagGems /data2/hliu553/FlagGems && chown -R 1017:1018 /data2/hliu553/FlagGems"
# 2) 打补丁 04/05/06（仓内已有）
cd /data2/hliu553/FlagGems && for p in 04-flaggems-mm-block-pointer 05-flaggems-bmm-block-pointer 06-flaggems-addmm-composition; do
  git apply --3way /workspace/dev/images/kunlun-operator-runtime/v2/env/v1/compat/legacy_sdnn/$p.patch; done
# 3) 跑判据（gemm_compat 会校验 triton compiler sha256 = 7133710c…）
docker exec -e CUDA_VISIBLE_DEVICES=3 <容器> <python> \
  /workspace/dev/issues/kunlunxin-gemm-numeric-64cube/assets/verify_numeric_64cube.py
```

⚠️ **归档风险（P1）**：补丁树与编译组的 numerics 镜像（`flagtree-xpu-numerics:20261030-v3`，125 GB）都在**会清理的老机**上。
好在**补丁可从仓内 patch + 镜像内基础树重建**（上面三步，已验证路径存在），所以**不依赖老机存活**；
但**建议在 11-07 前实跑一遍上面三步**，确认"新机可复现 K-5"——否则这条判据在新机上只是"理论上能跑"。

## 6 上机核验清单

### 6.1 三级采集（每家都跑同一套）

| 级别 | 采集内容 | 命令（示例，按芯片替换） | 期望/判据 |
|---|---|---|---|
| **L0 宿主** | OS / 内核 / 驱动 / 固件 | `grep PRETTY_NAME /etc/os-release` · `uname -r`<br>910C: `cat /usr/local/Ascend/driver/version.info`<br>P800: `xpu-smi`（首行）<br>MLU590: `cnmon`（首段）<br>PPU: `ppu-smi`（首段） | 与表中版本逐字对齐；**不一致即标「版本已漂移」并更新表** |
| **L1 容器** | 基础镜像 OS / py / torch / SDK | `grep PRETTY_NAME /etc/os-release`<br>`python3 -c "import torch;print(torch.__version__)"`<br>`pip list` 过滤 torch/triton/flag | 镜像声明的版本必须与容器内 `pip`/`import` 实测一致 |
| **L2 行为** | 设备可见性 + 最小算子 + 本层判据 | `torch.cuda.device_count()`（P800/PPU）/ `torch.npu.device_count()`（910C）/ `torch.mlu.device_count()`（MLU）<br>最小 `zeros + matmul + sync`<br>`runtime/smoke_runtime.py --backend <x>` | 见 §6.2 |

⭐ **本轮学到的操作要点**：L0/L1 采集**不需要挂设备**。用 `docker run --rm` **不带 `--device`** 读包版本，既不触发 ACL 初始化、也不占带卡名额 —— 这是最省事的核验姿势。

### 6.2 剩余待核项（按优先级）

| 优先 | 芯片 | 要核的具体项 | 状态 |
|---|---|---|---|
| ~~P0~~ | ~~910C~~ | ~~宿主 OS + 驱动真实版本~~ | ✅ **2026-11-04 完成**：`openEuler 22.03` / 驱动 `26.1.1`；旧值 `25.5.0` 已作废 |
| ~~P0~~ | ~~P800~~ | ~~新机宿主 OS / 内核 / 驱动~~ | ✅ **2026-11-04 完成**：`Ubuntu 24.04.2 / 6.8.0-87 / 5.0.21.47`，与老机一致 |
| **P0** | PPU | 宿主 OS + 内核（现有只有容器基座 OS） | ⛔ 本轮网络不可达（`192.168.5.112:22` 超时），待恢复 |
| **P1** | MLU590 | ① 容器 `/etc/os-release`（现为**包名推断**）② **FlagGems 算子对照**（唯一 ⚪ 格） | ⛔ 本轮网络不可达（`10.1.1.21:22` 超时），待恢复 |
| ~~P1~~ | ~~P800~~ | ~~容器内 XTDK/XRE 的真实来源~~ | ✅ **2026-11-04 结案**：容器内 torch_xmlir/_versions.txt 实测 `XTDK 3.6.0.1 / XRE 5.13.0.0 / XCCL 20260209_9`，与基线逐字一致 |
| — | **P800 总收尾** | **事实项全部闭合**：OS / 内核 / 驱动 / 开发包 / 框架 / 各腿算子路径 六类均有实测依据；唯一遗留是**工程能力**（新机复现 K-5 判据，见 §5.4-D），不是事实缺口 | ✅ **P800 视为核验完成** |
| **P1** | P800 | **新机复现 K-5 判据的能力**（env/v1 的补丁版 FlagGems 不在新机） | 见 §5.4-D 三步重建；**建议 11-07 老机清理前实跑一遍**确认 |
| ~~P0~~ | ~~910C 框架~~ | ~~`torch_npu 2.10.0` weakref 退出期 UAF~~ | ✅ **2026-11-04 已修并实测**：`2.10.0.post7` 环境层 overlay ⇒ 回调**触发**、`rc=0`、区分度 **0.6380 不变**（§3.1c）。⚠️ 形态＝挂载覆盖；镜像固化待总组裁定 |
| **P1** | 910C 算子库 | FlagGems 实际不可用：编译期缺 `shmem`（`triu` 另可独立复现编译失败） | ① 补 `shmem`（来源待厂商/算子组确认，本机无外网）→ 重跑 `910C/probes/op_parity_20261104/op_parity_ascend.py`；② `triu` 单独立项；③ 我方声明面不变（vendor 原生） |
| ~~P2~~ | ~~910C~~ | ~~固件版本~~ | ✅ **2026-11-04 已定性**：不是「没查」而是「**该平台不支持此查询**」（三种查询同源报 `does not support`、8 卡一致、裸金属非直通、固件目录 `root` 不可读）⇒ 用户态不可得，需 root/BMC。**不再作为待核项** |
| **P2** | 全部 | 逐格把「随验收间接可用」升级为「逐算子对照」 | 属算子域，**建议与算子方向联合，本方向不单独下结论** |

### 6.3 核验记录写法（避免返工）

- 每条读数必须带 **采集时间 + 机器名 + 命令原文**（例：`npu1-27 · 2026-11-04 09:34 · cat /usr/local/Ascend/driver/version.info`）
- **同一事实出现两处冲突时，两处都保留并标「冲突待定」**，别悄悄改成新值；闭合后在 §0.1 显式记「旧值作废」
- 采到新值后：改本表 + 在 §8 记修订；**同时全仓扫旧值**（避免某份文档还留着老版本号）

### 6.4 2026-11-04 采集记录（910C / P800）

> 两台机器同一时段采集，均为**宿主命令原文**。容器侧均在**不挂设备**的 `docker run --rm` 内完成。

**昇腾 910C（`npu1-27`，09:34）**

```
grep -E "^PRETTY_NAME=" /etc/os-release    →  PRETTY_NAME="openEuler 22.03 (LTS-SP4)"
uname -r                                   →  5.10.0-216.0.0.115.oe2203sp4.aarch64
cat /usr/local/Ascend/driver/version.info  →  Version=26.1.1 / ascendhal_version=7.35.23
                                              package_version=26.1.1
                                              Innerversion=V100R001C10SPC009B220
cat /usr/local/Ascend/version.info         →  version=26.1.1
npu-smi info（首行）                        →  npu-smi 26.1.1   Version: 26.1.1
ls -l /usr/local/Ascend/ | grep cann       →  cann -> /usr/local/Ascend/cann-9.1.0
                                              cann-8.5.0（另存）
npu-smi info -t board -i 0                 →  Software Version 26.1.1 / Firmware Version NA
npu-smi info -l                            →  Total Count 8，每 NPU Chip Count 2（共 16 chip）
                                              ｜每 chip HBM 65,536 MB
容器（锁定镜像 comm:2.0.0-…）：
  /etc/os-release                          →  Ubuntu 22.04.5 LTS
  python                                   →  3.11.15
  torch / torch_npu                        →  2.10.0+cpu / 2.10.0（pip show 无 post 后缀）
  triton / flagtree                        →  3.5.1 / 3.5.1（自报，跟随 triton 线号）
  flag_gems / flagcx                       →  0.0.0+f7ae8e6b934a / 0.13.0
  vllm / vllm-plugin-fl                    →  0.20.2+empty / 0.0.0+g8b059122e
  ls /usr/local/Ascend/                    →  cann-9.0.0
```

**昇腾 910C 深核（同机 `npu1-27`，10:0x，第二轮）**

```
systemd-detect-virt                                  →  none（裸金属）
/sys/class/dmi/id/product_name                       →  Atlas 800T A3
/sys/class/dmi/id/bios_version                       →  32.85
nproc / free -g                                      →  640 核 / 2013 GB
npu-smi info -l                                      →  Total Count 8；NPU 0..7 每个 Chip Count 2
                                                        ⇒ 8 卡 / 16 chip
ls /dev/davinci[0-9]* | wc -l                        →  16
npu-smi info -t board -i 0                           →  Product Name: IT22HMDA_2_S
                                                        Serial Number: 10255B485124 / Board ID 0x71
                                                        Software Version: 26.1.1
                                                        Firmware Version: NA
                                                        Compatibility: NOT SUPPORT
npu-smi info -t board -i 1..7 | grep Compatibility   →  全部 NOT SUPPORT（8 卡一致）
npu-smi info -t firmware                             →  Error parameter of -t（**该类型不合法**）
npu-smi info -t product / -t first-power-on-date      →  This device does not support querying …
npu-smi info -t health -i 0..7                       →  8 × 2 chip 全 OK
ls -ld /usr/local/Ascend/firmware                    →  dr-xr-x--- root（**非 root 不可读**）
ls -l  /usr/local/Ascend/driver/version.info         →  Sep 16 12:47（驱动升级时间）
ls -ld /usr/local/Ascend/cann-9.1.0                  →  Oct 16 15:42
docker inspect <锁定镜像> --format {{.Created}}       →  2026-10-08T23:35:58Z
```

**昆仑芯 P800（新机 `VM-20-4-ubuntu`，09:34）**

```
grep -E "^PRETTY_NAME=" /etc/os-release    →  PRETTY_NAME="Ubuntu 24.04.2 LTS"
uname -r                                   →  6.8.0-87-generic
xpu-smi（首行）                             →  Driver Version: 5.0.21.47  XPU-RT Version: 5.0.21
ls /usr/src/ | grep kunlun                 →  kunlun-5.0.21.47
ls /usr/local/ | grep xpu                  →  xpu / xpu-5.0.21.47 / xpu_extension
xpu-smi | grep -c "P800 OAM"               →  8
容器（锁定镜像 kunlun-operator-runtime:2.0.0-…）：
  /etc/os-release                          →  Ubuntu 22.04.5 LTS
  python                                   →  3.10.18
  torch                                    →  2.9.0+cu129（torch.version.cuda=12.9）
  torch.xpu.is_available()                 →  False（hasattr(torch,"xpu")=True）
  flag_gems / flagcx                       →  4.2.1rc0 (/env/FlagGems) / 0.10.0
  flagtree / torch_xray                    →  0.6.1+xpu3.6 / 2.0.4
  vllm / vllm-plugin-fl                    →  0.13.0 / 0.1.0
  xmlir / xtorch_ops                       →  1.0.0.1 / 0.1.2640+9bc36a18
  cat <conda env>/lib/python3.10/site-packages/torch_xmlir/_versions.txt
                                           →  XPYTORCH: XMLIR--bc1b1dc6f-dev+2026032411
                                              XHPC: 20770323
                                              XTDK: 3.6.0.1        ← 与基线 provides 一致
                                              XRE: 5.13.0.0        ← 与基线 provides 一致
                                              CUPTI: 1.1.0.0
                                              XCCL: 20260209_9     ← 与基线 provides 一致
                                              compile_date: 20260324-11:53:49
                                              arch: x86_64
```

⭐ **找版本文件别只搜 `/usr/local` `/opt` `/etc`**：P800 的 `XTDK/XRE/XCCL` 自证在 容器内 `<conda env>/lib/python3.10/site-packages/` 下的 torch_xmlir/_versions.txt（**site-packages 下**）。
本轮 v0.2 就因为只搜了那三处，误判成「无自证」并错误降级 —— **先读 `dev/images/<name>/v<N>/lock.yaml` 里的指针，再决定找不到**。

**本轮不可达**：`PPU-2`（`192.168.5.112:22`）、`Mlu-1`（`10.1.1.21:22`）均 `Operation timed out` ⇒ 两家数据沿用历史记录，标注「本轮未复核」。

## 7 边界声明

1. **本表只覆盖设备执行上下文层**（init/context/queue、Stream/Event、D2D/H2D、错误恢复、多流）。算子库、集合通信、显存管理、调度、性能均**只登记不结论**。
2. **✅ 的含义是「本方向在该组合上跑通了完整验收」**，不是「该组合在所有场景都正确」。
3. **未采集不写不支持**；**不支持必须带失败记录或厂商矩阵依据**。
4. 四家判定**互不外推**：某芯片「能」不证明另一家「能」，某芯片「不能」也不证明另一家「不能」。

## 8 修订记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-11-04 | v0.1 | **初表**：四芯片 × 五层共 20 格，✅ 8 / 🟡 11 / ⚪ 1 / ⛔ 0；给出 §6 上机核验清单 |
| 2026-11-04 | **v0.8** | **新增 §3.1e「910C 到底谁在实现算子」专项核对**：三层独立证据证明**我方路径全走厂商算子库** —— ① 注册层 `torch._C._dispatch_dump_table`：`mm/add/silu/matmul` 的 `PrivateUse1` 全为 `torch_npu/csrc/aten/RegisterNPU.cpp:33542`；② 运行期 `'flag_gems' in sys.modules` = **False**；③ 真实服务日志 `Platform plugin ascend is activated` + `SERVE_STANDARD_PASS`，FlagGems 相关字样**零命中**。⭐ 顺带两个事实：(a) **`aten::matmul` 不在 flag_gems 覆盖内**（enable 后仍回落厂商，只有 `mm` 被接管）；(b) ⚠️ **更正一句错误因果** —— vLLM 里 `VLLM_PLUGINS` 未设（None）= **加载全部插件**，「unset = 禁用 vllm-plugin-FL」是错的（真禁用要给空串）；脚本头注释与 ascend 分支注释已改，**代码行为未改**（实测插件即使被 import 也不调 `enable()`，dispatch 不变）。⚠️ 20 格状态计数不变（仍 ✅13 / 🟡6 / ⚪1）| 
| 2026-11-04 | **v0.7** | **① 910C 框架格修复并实测（新增 §3.1c）**：用 **env 层 overlay** 把 `torch_npu` 2.10.0 换成 **`2.10.0.post7`**（介质在本机 139 MB 包树，**无外网可装**）⇒ weakref 回调 **False→True**、退出码 **139→0**、推理腿两臂均 `INFER_LEG_PASS` 14/14、区分度**两臂同为 0.6380**（升级不动读数）⇒ 该格转 **✅（仅 env 层，形态＝挂载覆盖，镜像未固化）**；⚠️ 吞吐单跑 78.94→66.16 句/s **不作结论**（单次 + 同机他人负载）。**② 910C 算子库格补齐逐算子对照（新增 §3.1d）**：新探针在 vendor 路径记录**我方前向实际执行的 27 个 aten 算子**，再启用 FlagGems 用**同一输入重放**比对 ⇒ **304 条重放 / 184 PASS / 0 数值不一致**，覆盖 **18 个算子**；并查出两条确证缺口：**启用后真实前向编译期失败**（`ModuleNotFoundError: No module named 'shmem'`，归因到 flagtree/triton 后端栈依赖）与 **`triu` 编译失败**（最小脚本可独立复现）。⭐ 同时如实剔除**探针自身造成的假缺陷**：`silu`/`lift_fresh` 的重放失败经最小脚本复验为**正常**，不计为算子缺陷。该格仍 🟡 但性质由「缺证据」变为「证据已硬、缺口已定位」。⚠️ 计数 ✅12→**13**、🟡7→**6**、⚪1，合计 20 |
| 2026-11-04 | **v0.6** | **① P800 收尾**：事实项全部闭合（六类均有实测依据），唯一遗留是工程能力（新机复现 K-5）不是事实缺口。**② 910C 第二轮深核（新增 §3.1b）**：⭐ 纠正两处认知 —— (a) 机器是**裸金属 `Atlas 800T A3`**（`systemd-detect-virt`=none）**不是虚拟机**，所以「固件查不到」不能再用 VM 直通解释，实为 **该平台不支持该查询**（`-t firmware` 类型不合法、`-t product`/`-t first-power-on-date` 均报 `This device does not support querying`、8 卡 `Compatibility` 同为 `NOT SUPPORT`、固件目录 `root` 不可读）；(b) 硬件口径 **8 卡 / 16 chip / 每 chip 65,536 MB = 1,024 GB**，旧文档「4 NPU / 8 chip」为 08-24 环境已加注；③ 新增两条闭合结论：**驱动 09-16 / CANN 10-16 / 镜像 10-08 三者的 mtime 都早于 10-27 验收** ⇒ 该验收证据确实跑在当前栈上；**8 卡 health 全 OK**（与 P800 新机卡 1/2 挂死对照）；④ 记录整机 `Product Name: IT22HMDA_2_S` / SN `10255B485124` 供报障定位。⚠️ 20 格状态计数仍 ✅12 / 🟡7 / ⚪1 |
| 2026-11-04 | **v0.5** | **§5.4-C 由一句话扩成「各腿/判据算子路径逐条核对表」**：训练腿 / 推理腿 / 多流（14 探针）/ conformance 13+6+I1–I4 / 服务化消费方 / 服务化服务端 / 离线自检 / 错误恢复共 8 类，逐条给出入口文件、是否 `import flag_gems`、是否 `use_gems()`，结论 **`prototype/` 内 21 个入口文件全部不启用 FlagGems（= 0）**；并辨析两种"不走"的性质（**代码层不启用** vs **服务端配置层 `USE_FLAGGEMS=0` 主动关闭**），另**先自证一个易误读点**：`prototype/scripts/serve_standard.sh` L164 的 `PYTHONPATH=/env/FlagGems/src` 是 **import 完整性前置**（缺它 `vllm_fl` 报 `Failed to infer device type`），**不代表在用 FlagGems**。P800 算子库格补指针「路径：vendor 原生，不经过它」。⚠️ **20 格状态计数仍为 ✅12 / 🟡7 / ⚪1**（一格仍一个标记）|
| 2026-11-04 | **v0.4** | **新增 §5.4「算子库这格怎么才能变 ✅」**：把 K-5 的性质讲清（**已定判据 + 已跑出 FAIL**，不是未测清）、给出三条路径（**A 等上游 / B 收窄声明面 / C 兜底路由**）与各自责任人和代价，建议 **B+C 并行、A 追踪**；并记入两条关键事实：① **本方向代码从不启用 FlagGems**（全仓只 grep 到注释）⇒ K-5 不在我方验收路径上，四家区分度 0.6380/0.6392/0.6391/0.6391 逐字接近为间接佐证；② **新机复现 K-5 的三步重建法**（镜像内 `/env/FlagGems` = `73c5aff1` 基础树 + 仓内 patch 04/05/06），并标出**老机 11-07 到期的归档风险**。⚠️ **20 格状态计数不变**（仍 ✅12 / 🟡7 / ⚪1）—— 本节只定口径，不改判定 |
| 2026-11-04 | **v0.3** | **纠正 v0.2 的一处误降**：① **P800 开发包恢复 ✅** —— `XTDK 3.6.0.1 / XRE 5.13.0.0 / XCCL 20260209_9` 的自证文件是 容器内 `<conda env>/lib/python3.10/site-packages/` 下的 torch_xmlir/_versions.txt，实测逐字一致；② ⚠️ 误降原因记为教训：v0.2 只搜 `/usr/local` `/opt` `/etc` `/env`，**漏了 site-packages**，而 `dev/images/kunlun-operator-runtime/v2/lock.yaml` **L13 早已写明**该路径；③ 已在 `dev/images/kunlun-operator-runtime/v2/image.baseline.yaml` 的 `provides` 行**补上自证指针**，避免后人重复踩；④ 计数 ✅ 11 → **12**、🟡 8 → **7** |
| 2026-11-04 | **v0.2** | **910C 与 P800 上机核验完成**：✅ 由 8 升至 **11**、🟡 由 11 降至 **8**。① **910C 驱动 `25.5.0` → `26.1.1`（旧值作废）**；② **910C 宿主 OS 首次采集 = openEuler 22.03 LTS-SP4**（此前完全空白，且非 Ubuntu）；③ 910C 宿主 CANN 9.1.0 与容器 CANN 9.0.0 分列；④ **P800 新机 OS/内核/驱动实测，与老机完全一致** ⇒ 老机到期不再影响这两格；⑤ **P800 开发包降为 🟡**（容器内无 XTDK/XRE 自证）；⑥ 新增 §0.1 更正表与 §6.4 采集记录；⑦ 910C 固件报 `NA`，如实标「未获取」；⑧ 修正 v0.1 §0 与 §2.1 的 🟡 格数不一致（§0 误写 10，实为 11） |
