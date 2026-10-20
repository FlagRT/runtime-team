# FlagCX c10d collective 缺陷：真机复现 + 我们分支修复验收（910C · 2026-10-16）

> 对上游 issue：`runtime-team` 仓 **`dev-1.0`** 的 `dev/issues/flagcx-c10d-collectives`（`README.md`「修复验收」）。
> 执行机：**npu1-27（10.120.72.27）**，容器 `dc-p9-910c-20261016`，用卡 **die5 + die0**（容器内 0,1）。
> 完整原始日志留执行机：`/mnt/raid/hliu553/dc_p9_20261016/`；仓内只放截选（`probes/p9_c10d_20261016_out/`）。

---

## 0 结论先行

| 问题 | 结论 |
|---|---|
| **issue 能否在本机复现** | ✅ **复现**。同一镜像 + 同一条命令 ⇒ `overall: FAIL`（退出码 1），八行图案与 issue 记录一致（`probes/p9_c10d_20261016_out/A_image_alt_r1_A.txt`） |
| **我方分支的修复是否修掉了它** | ✅ **修掉了，两项验收均通过**：① `flagcx_sync_test.py --iters 20` ⇒ **`overall: PASS`，八组全过，退出码 0**（3 次独立运行）；② 原生 `DistributedDataParallel` 双卡 ⇒ **`overall: PASS`**（`ddp_wrapped/train_steps/loss_all_finite/ranks_param_equal` 四项全 PASS，参数跨 rank 逐位一致） |
| **为什么 issue 里的镜像必然复现** | ⭐ **issue 所用镜像的 FlagCX 主干 `4e0e0cb` 正是我们分叉的起点**，此后 `dev-1.0` **一条我们的修复都没收到** ⇒ 该镜像必然同时携带 P6 与 P9。issue 写的"2026-10-11 第 4 次判定与前 3 次逐行一致"**不是缺陷稳定，而是镜像里的 FlagCX 从来没变过**（详见 §1） |
| **归属** | **FlagCX 侧**（我们分支已修）：socket 数据面 P9 + 设备回调层 P6 + plugin 层完成语义。⭐ 但**镜像/上游同步也有一半责任**：修复长期停在个人分支，未回流 `dev-1.0` |
| **能否原样构建我们的分支** | ❌ 不能，需 **3 个构建期前置**（§5）——其中两条是**我方分支自身的可编译性缺陷**，建议随修复一起上游 |

---

## 1 ⭐「镜像缺我们修复」的事实链（先把归属说清）

```bash
# FlagCX 仓（git@github.com:FlagRT/FlagCX.git）
git log -1 origin/dev-1.0                       # → 4e0e0cb（2026-08-20）Merge PR #1 from xliu969/flagcx-ascend-fix
git merge-base origin/dev-1.0 HEAD              # → 4e0e0cb 本身
git rev-list --count origin/dev-1.0..HEAD       # → 15（我们领先 15 个提交）
```

| 项 | 值 |
|---|---|
| 我们分支 | `kistich/ascend-flagcx-adapt`，tip **`8402a93`** |
| `dev-1.0` tip | **`4e0e0cb`**（2026-08-20）= **merge-base** |
| 关键修复提交 | `a1e7e0f`（P2+P6+P7+P8+P9，2026-08-28）· `d296824`（完成语义 + event RAII）· `0686805`/`60e0695`（A 线 stream 语义） |
| 镜像声明的 FlagCX | `4e0e0cbcbf721169ca82348080f8353aebfe2c31`（`dev/stack.lock.910c.v4.yaml` / `ascend-train-comm/v3`） |

⇒ **`dev-1.0` 停在分叉点**：以它为基础构建的镜像**必然**携带 P6/P9 与 plugin 层旧语义。
⚠️ **必须在文档中标注这条**（用户要求）：这不是"厂商缺陷反复出现"，而是**我们自己的修复没有推送上去**；
issue 侧"四次判定逐行一致"= 镜像内容不变的自然结果。

⚠️ **一处需登记的差异**：issue 记录 npu1-11 的 image id `e13a15d0d02b`；本机 npu1-27 同一 tag 的 image id = **`7028028bb62c`**
⇒ 同 tag、**不同 id**（两机镜像不是同一实体）。本机在**该镜像版本上同样复现**，故结论不受影响，但引用 image id 时须连机器一起引。

---

## 2 我们的修复 → issue 两个类别的映射

| issue 类别 | 现象 | 机制（我们的定位） | 我们的改动点（均在 HEAD 上） |
|---|---|---|---|
| **① 异步返回**：`broadcast` / `all_reduce` / `all_gather_into_tensor` 返回时数据未写完（`+sync` 后正确）；后果：**原生 DDP 不可用**（`_verify_params_across_processes` 报 "rank0 有 N 个参数、其他 rank 0 个"） | 调用方在当前流上立即读，读到未写完/陈旧数据 | **两处**：<br>· socket 数据面用**非阻塞完成查询**判 D2H 完成，而 CANN 的完成标记可能早于数据对 CPU 可见 ⇒ `isend` 发出未落位的 pinned buffer（**P9**）<br>· 设备回调层 `aclrtLaunchCallback` 不等 stream 前置任务（**P6**）<br>· 且 collective 跑在**自建流**上，与调用方当前流无 happens-before（**A 线 stream 语义**） | `flagcx/core/net.cc`：`eventQuery` → **`eventSynchronize`**（每 chunk 阻塞等 D2H 真完成）<br>`flagcx/adaptor/device/cann_adaptor.cc`：`aclrtLaunchCallback` 前 **`aclrtSynchronizeStream(stream->base)`**<br>`plugin/.../backend_flagcx.cpp`：`getStreamByIndex(0)` 改返回 **torch_npu 当前流**（`dlopen(RTLD_NOLOAD)` + `dlsym` 取 `c10_npu::NPUStream::stream()`，不链 libtorch_npu） |
| **② 列表 `all_gather` 读到上一次结果，显式同步也无效** | `outputFlattened`（`newLikeFlat`）被缓存分配器复用，in-flight 写入落到**下一次调用**的缓冲上 ⇒ 事后 `synchronize()` 也没用（**逻辑上的缓冲生命周期错误**，不只是时序窗口） | plugin 层：所有 collective 返回前**没有**等自己的完成事件；下一个 collective 的 `newLikeFlat` 可能复用上一轮仍在写的内存 | `plugin/.../backend_flagcx.cpp`：**每个 collective 在 `markCompleted` 前 `work->event_->synchronize()`**（17 处）<br>`plugin/.../include/event_flagcx.hpp`：`flagcxEvent::synchronize()` 虚接口 + CANN 实现 **`aclrtSynchronizeEvent`**；并补 **event RAII 析构**（`aclrtDestroyEvent`，原实现每次 collective 泄漏一个 ACL event，DDP 数百步后资源耗尽） |

> ⭐ 这解释了为什么**一个机制能同时盖住两个类别**：把"collective 返回"重新定义为**"数据已落定"**（而不是"已入队"），
> 类别 ① 的"未写完"与类别 ② 的"缓冲被复用"就都不成立了。
> ⚠️ 但**不是**同一个修复点：类别 ① 依赖 P9/P6 + 流语义，类别 ② 依赖 plugin 层 `event_->synchronize()`；
> 复查时**不要用一个修复去解释两个现象**（issue 里对类别 ② 的位置推断——`outputFlattened` 未 `recordStream`——方向对，但**不完整**）。

---

## 3 验收设计与执行

**判据（issue README「修复验收」原文）**：单入口命令退出码 0 且判定行 `overall: PASS`（八组全过）；
**另需**原生 `DistributedDataParallel` 双卡验证通过（该脚本 issue 侧标注"待补"⇒ 本轮补齐，见 §4b）。

**对照设计**（为什么不只用镜像跑一遍就下结论）：

| 臂 | 内容 | 切换方式 |
|---|---|---|
| **A 镜像原版** | 镜像自带的 FlagCX（主干 `4e0e0cb`） | 从镜像内解出 `flagcx` 包放 `/workspace/p9/orig`，运行时 `PYTHONPATH=/workspace/p9/orig` 影子化 |
| **B 我们分支** | 用我们分支源码在**同一镜像内**编译安装 | 默认加载 `site-packages/flagcx` |

- **交替序列 A/B/A/B**（2 轮）：项目纪律 —— 分组跑分不清「变量效应」与「时间漂移」。
- **加载自证**：每次运行前打印实际加载的 `flagcx` 路径 ⇒ `alt_r1_A` = `/workspace/p9/orig/flagcx`、`alt_r1_B` = `site-packages/flagcx`（影子确实生效）。
- **两套插件指纹**（证明不是"同一份插件跑两次"）：

| 臂 | `_C.*.so` sha256 | `libflagcx.so` sha256 |
|---|---|---|
| A 镜像原版 | `a6a4a8e0…9312f4` | `8196a720…cead86` |
| B 我们编译 | `72256b23…366c8c` | `8fe6bd4a…66fa4f1f` |

- **用卡**：die5 + die0（本轮**逐 die 全量探测**得出；其余 14 张被他人活跃容器占着）。容器只挂这 2 张（不挂全 16 张）。
- 命令（两臂完全同一条）：

```bash
docker exec -w /workspace/p9 -e ASCEND_RT_VISIBLE_DEVICES=0,1 [PYTHONPATH=/workspace/p9/orig] $C \
  bash -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh; cd /workspace/p9 && \
            torchrun --nproc_per_node=2 flagcx_sync_test.py --iters 20"
```

---

## 4 结果

### 4a `flagcx_sync_test.py`（八组 × `--iters 20`）

| 轮次 | A 镜像原版 | B 我们分支 |
|---|---|---|
| 首次（A 单跑） | `overall: FAIL` · rc=1 | — |
| 交替 r1 | `overall: FAIL` · rc=1 | **`overall: PASS` · rc=0** |
| 交替 r2 | `overall: FAIL` · rc=1 | **`overall: PASS` · rc=0** |
| 编译后单独复跑 | — | **`overall: PASS` · rc=0（八组全 PASS）** |

**A 臂逐组读数**（`alt_r1_A.txt`）：`broadcast+nosync` FAIL · `broadcast+sync` FAIL · `all_reduce+nosync` FAIL · `all_reduce+sync` PASS · `all_gather_list+nosync` FAIL · `all_gather_list+sync` **FAIL** · `all_gather_into_tensor+nosync` FAIL · `all_gather_into_tensor+sync` PASS ⇒ `overall: FAIL`。

**B 臂逐组读数**：**八组全 PASS**（含 `all_gather_list+sync` —— 类别 ② 也消失）。

⚠️ **对 issue 记录的一处细化（如实登记）**：issue 写"四次判定逐行一致"，但本机 A 臂首跑 `broadcast+sync` 为 **PASS**、
交替两轮为 **FAIL** ⇒ **`overall: FAIL` 稳定，但 `+sync` 列的个别子行会波动**。判定以 `overall` 为准（issue 的口径也是 `overall`），
但"逐行一致"这句在引用时应放宽为"**overall 一致**"。

### 4b 原生 DDP 双卡（issue 要求、此前"待补"的脚本）

脚本：`probes/910C/ddp_native_test.py`（自判定三态；`PASS/0`、`FAIL/1`、`ERROR/2`）。

| 轮次 | A 镜像原版 | B 我们分支 |
|---|---|---|
| r1 | `overall: FAIL` · rc=1 | **`overall: PASS` · rc=0** |
| r2 | `overall: FAIL` · rc=1 | **`overall: PASS` · rc=0** |

- **A 臂原始报错**（复现 issue 类别 ① 的症状原文）：

  ```
  [rank0] DDP 封装失败: RuntimeError: DDP expects same model across all ranks,
          but Rank 0 has 4 params, while rank 0 has inconsistent 0 params.
  ```
  （issue 里的原文是 "rank0 有 338 个参数、其他 rank 0 个" —— **同一失败形态**，只是模型规模不同。）
- **B 臂四项判据全 PASS**：`ddp_wrapped`（封装成功）· `train_steps`（10 步不崩）· `loss_all_finite`（2.065917 → 1.350363）· **`ranks_param_equal`**（参数跨 rank 逐位一致 ⇒ 梯度确实被 all-reduce，不是"恰好没报错"）。

---

## 5 ⚠️ 构建前置三件事（否则要么编不过、要么"看起来重编其实没重编"）

issue 镜像**不能直接编译我们的分支**。三条实测（截选 `probes/p9_c10d_20261016_out/build_prereq.txt`）：

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| **①** | 链接期没问题，但 import 时 `ImportError: libflagcx.so: undefined symbol: aclnnInplaceAddGetWorkspaceSize` | 我们 `makefiles/ascend.mk` **无条件**加了 DAG 持久 kernel 的链接项 `-lopapi -lnnopbase -L…/op_api/lib -lcust_opapi`；而镜像内 `opp/vendors/` 为空、**`libcust_opapi.so` 缺失**（自建算子包未装）<br>⚠️ **不能三个一起砍**：`aclnnInplaceAddGetWorkspaceSize`（P8 设备侧 reduce 用）**定义在标准库 `libopapi.so`**，砍掉即未解析 | 构建期**只去掉 `-lcust_opapi` 与其 `-L`**（`-lopapi -lnnopbase` 必须保留）。<br>判据：产物 `libflagcx.so` 的 `DT_NEEDED` 含 `libopapi.so`/`libnnopbase.so` ✅ |
| **②** | 编译期 `fatal error: cuda_runtime_api.h: No such file or directory`（来自 `torch/include/ATen/cuda/CUDAContextLight.h`） | 我们给 `backend_flagcx.cpp` **无条件**加了 `#include <ATen/cuda/CUDAContext.h>`，而它只被 `#ifdef USE_NVIDIA_ADAPTOR` 块里的 `at::cuda::getCurrentCUDAStream` 使用；ascend-only 镜像没有 CUDA 头 | **建议上游**：把该 include 放进同一个 `#ifdef USE_NVIDIA_ADAPTOR`（本轮构建期如此处理）⇒ **我方分支自身的可编译性缺陷** |
| **③** | `pip` 报 `Successfully installed`，但 `buildX.log` 里**零条编译命令**、产物 sha256 **逐字不变** | 增量构建缓存（旧 `.o`/`build/`）生效 ⇒ 实际**没有重链** | 构建前清 `build/`、`*.egg-info`、`*.o`、`*.so`，并 `--no-cache-dir`。<br>**判据不能只看 pip 退出码**：要看 `DT_NEEDED` 与新时间戳（本轮正是靠这条发现"假成功"） |

> 我方分支源码出处（构建副本自带 `SOURCE-PROVENANCE.txt`）：`FlagRT/FlagCX` @ `kistich/ascend-flagcx-adapt`，
> HEAD `8402a93` + **8 个文件未提交改动**（+180/−14，主要是 `cannAdaptorGetDeviceByPciBusId` 补实现）。
> ⭐ 已核：**完成语义相关的修复点全部在已提交的 HEAD 上**（`net.cc` `eventSynchronize`=3 · `cann_adaptor.cc` `aclrtSynchronizeStream`=2 ·
> `backend_flagcx.cpp` `event_->synchronize`=17 · `event_flagcx.hpp` `aclrtSynchronizeEvent`=2 · `RTLD_NOLOAD`=2）；
> 未提交的 WIP 只多一个 topo 反查 ⇒ **B 臂 PASS 可归因于已提交的修复**。

---

## 6 边界与未验证面（不许含糊）

| 项 | 状态 |
|---|---|
| **最小修复集** | ⬜ **未做二分**。本轮验证的是**整棵工作树**（HEAD + WIP）。哪个提交是"必要且充分"**未验证** |
| **跨机 / 8 卡 / 训练腿 50 步** | ⬜ 未跑。本轮只覆盖**单机 2 卡 c10d**（issue 的验收面） |
| **issue 类别 ② 的位置推断** | 我们的证据支持"plugin 层返回前未等完成事件 + 缓冲复用"，但**没有逐行定位到 `recordStream` 是否缺失** ⇒ 结论按我们实测写，不照抄 issue 的推断 |
| **性能** | 未作结论。我们的修法把 collective 变成**宿主等待**（`aclrtSynchronizeEvent`）⇒ 可能牺牲重叠；是否影响训练吞吐**需另行测量** |
| **本机 A 臂 `broadcast+sync` 波动** | 已如实登记（§4a）。**overall 稳定 FAIL**，子行会波动 |
| **他人容器** | 用卡前逐 die 探测；**未动任何他人容器**。die0 上有他人进程记录（4102 MB），但 INIT 与两卡小算子通过 |

---

## 7 建议动作

1. **上游（FlagCX / `dev-1.0`）**：把完成语义修复**最小集**回流 `dev-1.0`（个人分支已停 15 个提交、`dev-1.0` 一条未收）。
   建议候选：`0686805` + `60e0695`（A 线 stream 语义）· `d296824`（完成语义 + event RAII）· `a1e7e0f`（P2+P6+…，需**拆出** P6/P9 部分）·
   WIP 的 `cannAdaptorGetDeviceByPciBusId`。⚠️ **最小集尚需二分验证**（§6）。
2. **我方分支（可编译性，建议随修复一起上游）**：`#include <ATen/cuda/CUDAContext.h>` 加 `#ifdef USE_NVIDIA_ADAPTOR` 守卫；
   `ascend.mk` 的 DAG 链接项改为**条件化**（有 custom_opp 才加 `-lcust_opapi`，而不是无条件）。
3. **issue 侧文档**：在 `dev/issues/flagcx-c10d-collectives/README.md` 增补一句归属说明 ——
   "本文所用镜像的 FlagCX 主干 `4e0e0cb` 位于本组修复分叉点之前 ⇒ **镜像必然携带 P6/P9**；
   修复到货的判定应换成**含修复的新镜像**，而不是重复在同一镜像上复跑"。（该文件在 `dev-1.0`，需按子树同步纪律改）
4. **case 包**（`Fla/docs/06-delivery/flagcx-ascend-completion-semantics/`）：B2（带卡验证+证据）已补、B3（自判定三态）已补（`ddp_native_test.py`）、
   B1（单入口）由本轮命令固定、B4（补丁 .diff + sha256）部分；剩余见该包 README。

---

## 8 复现命令（本报告全部读数由这些命令产生）

```bash
# 0) 选卡：逐 die 全量探测（只挂单 die 试 INIT；不早退）
bash -s < probes/910C/probe_all_dies_910c_20261016.sh          # → FREE = 5 0

# 1) Arm A（镜像原版）+ 编译安装 + Arm B（一条脚本走完）
bash -s < probes/910C/run_p9_c10d_910c_20261016.sh            # 首轮（含 Arm A 复现）

# 2) 构建期修复 ① 的修正 + 清缓存重编 + Arm B
bash -s < probes/910C/run_p9_armB3_910c_20261016.sh           # 以 DT_NEEDED 为判据

# 3) 交替对照 A/B/A/B（影子切换 + 加载自证）
bash -s < probes/910C/run_p9_alternate_910c_20261016.sh

# 4) 原生 DDP 双卡（issue 要求的第二项验收）
bash -s < probes/910C/run_p9_ddp_910c_20261016.sh
```

**收尾**（本报告执行后已执行）：

```bash
ssh 910C 'docker stop dc-p9-910c-20261016'      # ✅ 已执行（2026-10-16 10:5x）：Exited (137)、我方无残留进程、die5/die0 已释放
# 容器**保留**（不删）便于复核；确认后如需彻底清理：docker rm dc-p9-910c-20261016
# 复核时重起即可：docker start dc-p9-910c-20261016（源码/构建产物/证据都在 /mnt/raid/hliu553/ 下）
```

---

## 9 证据索引

| 项 | 位置 |
|---|---|
| 完整原始日志（**不进仓**） | `npu1-27:/mnt/raid/hliu553/dc_p9_20261016/`（`A_image.out` · `B4_ours.out` · `alt_r*_{A,B}.out` · `ddp_r*_{A,B}.out` · `build{,2,3,4}.log`） |
| 判定截选 | `probes/p9_c10d_20261016_out/{A_image_verdict,B_ours_verdict,alt_r1_A,alt_r1_B,alt_r2_A,alt_r2_B,ddp_r1_A,ddp_r1_B,ddp_r2_A,ddp_r2_B,ddp_A_error_symptom}.txt` |
| 两套插件指纹 | `probes/p9_c10d_20261016_out/fingerprints.txt` |
| 构建前置三件事 | `probes/p9_c10d_20261016_out/build_prereq.txt` |
| 运行脚本 | `probes/910C/{probe_all_dies,pick_910c_pair,run_p9_c10d,run_p9_armB,run_p9_armB2,run_p9_armB3,run_p9_alternate,run_p9_ddp}_20261016.sh` · `probes/910C/ddp_native_test.py` |
| 被测源码出处（机上） | `npu1-27:/mnt/raid/hliu553/flagcx-build-20261016/SOURCE-PROVENANCE.txt` |

---

## 10 本轮踩坑（都属"看起来通过"家族）

| # | 坑 | 表现 | 处置 |
|---|---|---|---|
| 1 | **构建"成功"但没重编** | pip 报 `Successfully installed`，日志里零条编译命令、产物 sha256 不变 ⇒ 差点把"旧插件"的 FAIL 读成"我们分支没修好" | 清缓存 + `--no-cache-dir`；判据改为 `DT_NEEDED` 与产物 sha256 |
| 2 | **砍链接库砍过头** | 一次砍掉 `-lopapi -lnnopbase -lcust_opapi` ⇒ 新 `.so` 未解析 `aclnnInplaceAddGetWorkspaceSize`，Arm B 变成 `ERROR`（未出判定段） | **库级定位**：该符号在标准库 `libopapi.so`，只有 `aclnnFlagcxCollective` 属自建算子 ⇒ 只砍 `-lcust_opapi` |
| 3 | **`local r=$1 arm=$2 out=…${r}`** | `set -u` 下报 `arm: unbound variable`（同一 `local` 内先用后绑） | 拆成分行 `local` |
| 4 | **把"全部卡对都失败"读成"整机繁忙"** | 真因是**我挑的卡对里有被他人占的 die**；逐 die 探测后立刻找到 free 的 {5,0} | 逐 die 单独试 INIT（判据粒度要匹配"独占单位 = 单卡"） |
| 5 | **`grep` 用 `\|`（BSD grep 不支持）** | 静默 0 命中，差点把"已移除 DAG 链接项"的自证读成"未移除" | 多模式一律 `-E`；关键结论用**看定义行原文**复核 |
| 6 | **`nm -D --undefined-only` 的误读** | 我预设"未解析符号应为 0"；实际 `libopapi.so` 通过 `DT_NEEDED` 提供解析 ⇒ **未解析符号非 0 是正常的** | 判据改成"`DT_NEEDED` 是否含提供者"（并把这条预期更正写进文档） |
