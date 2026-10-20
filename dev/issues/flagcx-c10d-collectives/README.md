# FlagCX c10d collective：异步返回与列表形式 `all_gather` 结果错位

## 问题与现状

- **一句话**：FlagCX 的 c10d ProcessGroup 实现有两类缺陷——① `broadcast`、`all_reduce`、
  `all_gather_into_tensor` 返回时数据未写完（调用方按同步语义立即读取得到旧值或全零，显式
  `torch.npu.synchronize()` 后正确）；② 列表形式 `dist.all_gather` 读到**上一次调用**的数据，
  显式同步也无效。前者使 PyTorch 原生 `DistributedDataParallel` 不可用（DDP 启动检查
  `_verify_params_across_processes` 在 C++ 层命中类别 ①，报 "rank0 有 338 个参数、其他 rank 0 个"），
  后者没有调用方规避手段，只能避开列表形式。
- **归属方向**：FlagCX 通信方向。需求文本：
  [`决策备忘录 附录 B-2 及其 2026-10-10 复验修正段`](../../../docs/v3-决策备忘录与shmem跨组需求-20261008.md)。

| collective | 未同步即读取 | `torch.npu.synchronize()` 后读取 | 缺陷类别 |
|---|---|---|---|
| `broadcast` | FAIL | PASS | ① 异步返回 |
| `all_reduce` | FAIL | PASS | ① 异步返回 |
| `all_gather_into_tensor`（DDP 启动检查所用） | FAIL | PASS | ① 异步返回 |
| `all_gather`（输出为张量列表） | FAIL | **FAIL**：得到上一次调用的结果或全零 | ② 结果错位，同步无效 |

### 现状（2026-10-16 更新）

- **缺陷未修**（**镜像里**的那份没修），但**修复已存在并已合入组织的 FlagCX `dev-1.0`**：
  仓 `git@github.com:FlagRT/FlagCX.git`，分支 `kistich/ascend-flagcx-adapt` 提交 `b576fa6a`（2026-10-16），
  同日以 merge 提交 **`62fe615`** 合入 `dev-1.0`（tip 由 `4e0e0cb` → `62fe615`；现可从 `dev-1.0` 直接取到该修复）。
  ⚠️ **但镜像配方 pin 的仍是旧 commit `4e0e0cb`** ⇒ 对使用方而言，**「修复到货」的判据是镜像里换成含修复的
  commit（`b576fa6a`），不是在同一镜像上再跑一次**（镜像侧动作见 `dev/images/` 的晋升流程）。
- ⚠️ **旧口径更正**：本文件此前记「2026-10-11 第 4 次运行判定与前 3 次逐行一致」，并据此认为缺陷"稳定"。
  **真实原因不是缺陷稳定，而是镜像里的 FlagCX 从未变过**——实测：

  ```
  $ git merge-base 4e0e0cbcbf721169ca82348080f8353aebfe2c31 <本方向分支 HEAD>
  4e0e0cbcbf721169ca82348080f8353aebfe2c31          # 即镜像基线记的 FlagCX commit 本身
  $ git rev-list --count 4e0e0cb..<本方向分支 HEAD>
  17
  ```

  ⇒ 在同一镜像上反复复跑，判定当然逐行一致。**真正的"修复到货"判据是镜像里换成含修复的 commit，
  不是在同一镜像上再跑一次**；引用旧结论时应连这一层一起说明。
- 本组此前的规避是在每个 collective 之后显式 `torch.npu.synchronize()` 并不使用列表形式
  `all_gather`——2 卡手动 DDP 可训（Qwen2.5-1.5B 24 步 loss 2.8911 → 2.5528），原生 DDP 封装不可用。
  Python 层补丁（vllm-plugin-FL 覆盖层）覆盖不到 c10d 路径，修复必须落在 FlagCX 自己的 c10d 实现。
- **类别 ② 的定位（2026-10-16 实测定案，取代此前的代码阅读推断）**：修复面在 **Torch 插件层的缓冲生命周期**
  —— 列表型 `all_gather` **返回时结果还没落定**，紧随其后的读取就拿到上一次的缓冲。修复 = **每个 collective
  返回前 `event_->synchronize()`（插件层 17 处）+ event 走 RAII 析构**，等价于把「collective 返回」重定义为
  「数据已落定」。
  ⚠️ **类别 ① 与类别 ② 是两个独立修复点**（前者 = 完成语义与流语义，后者 = 缓冲生命周期），
  **不要用一个修复去解释两个现象**；完整报告见
  [`device-context/910C/docs/FLAGCX_C10D_COMPLETION_20261016.md`](../../device-context/910C/docs/FLAGCX_C10D_COMPLETION_20261016.md)。
  > 此前基于代码阅读的推断（`backend_flagcx.cpp` 的 `outputFlattened` 临时缓冲：`copy_(non_blocking=true)`
  > 未与 collective 所在流建立先后顺序、且未 `recordStream`）已被实测取代，保留于此仅供对照。
- **连带现象**：未同步的调用若在执行中途其张量被释放，迟到的写入会落到复用该内存的后续张量上
  （测试早期版本中观察到 `broadcast` 源 rank 的张量被上一次调用的数据覆盖）。
- 退出期另有已知的 `free(): invalid pointer` SIGABRT（低优先级，结果已在退出前输出，不影响判定；
  见 [`ascend-train-comm/v3/lock.yaml`](../../images/ascend-train-comm/v3/lock.yaml) known_issues `teardown-sigabrt`）。

## 复现

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | **现行训练腿发布物**：`flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`（镜像基线记的 FlagCX = 公开主干 `4e0e0cbcbf721169ca82348080f8353aebfe2c31`）—— [`dev/stack.lock.910c.v4.yaml`](../../stack.lock.910c.v4.yaml) `lock.train`，血统目录 [`dev/images/ascend-train-comm/v3`](../../images/ascend-train-comm/v3/README.md) | 按 [`ascend-train-comm/v3/REBUILD.md`](../../images/ascend-train-comm/v3/REBUILD.md) 构建，或从其 `ARCHIVE.md` 所列离线包 `docker load`；image id 由 `up.sh --strict` 按镜像基线核对 |
| 运行环境 | **复现臂**用镜像 demo [`dev/images/ascend-train-comm/v3/demo/`](../../images/ascend-train-comm/v3/demo/docker-compose.yml)；**修复验收臂**用 device-context 方向的运行环境目录 [`dev/device-context/910C/env/train/`](../../device-context/910C/env/train/docker-compose.yml)（它在同一镜像上把 FlagCX 以 `pip-editable` 挂载覆盖） | 随仓。两臂用的是同一个镜像；全部 repo 为固化资产，宿主检出不影响复现臂 |
| 卡 | **2 × 910C** | 启动前 `docker ps` 确认本机无其他带卡容器（昇腾驱动对带卡容器独占）。⚠️ 在共享机上另有一条更细的选卡口径：**「空闲卡」的依据是容器映射表而不是 `npu-smi` 进程表**（已 init 但空闲的容器在进程表里是隐形的），判据 = 逐 die 建容器试 ACL INIT。脚本 [`device-context/910C/probes/probe_all_dies_910c_20261016.sh`](../../device-context/910C/probes/probe_all_dies_910c_20261016.sh)，口径与实测 [`device-context/910C/docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md`](../../device-context/910C/docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md) §6 |
| 模型与数据 | **不需要**（脚本只跑 collective，不加载模型、不联网） | — |
| 模型目录变量 | `FLAGRT_MODELS_DIR` 指向一个实体目录 | demo 把它挂为 `/workspace/models:ro`；缺省值 `<仓根>/models` 在共享机上常是指向宿主个人目录的软链，容器内无法解析，故启动时须显式给值（本缺陷的复现不读模型，指向任一实体目录即可） |
| 被测修复源码（**仅修复验收臂需要**） | FlagCX `kistich/ascend-flagcx-adapt` @ `b576fa6a`，工作区干净、已推送 | `git clone git@github.com:FlagRT/FlagCX.git <仓根>/FlagCX && git -C <仓根>/FlagCX checkout b576fa6a`；挂载与生效方式由 [`dev/device-context/910C/env/train/docker-compose.yml`](../../device-context/910C/env/train/docker-compose.yml) 声明（`activation: pip-editable`，容器启动时执行 `USE_ASCEND=1 python3 -m pip install --no-deps --no-build-isolation -e /workspace/FlagCX`） |

### 单入口命令

**复现臂（镜像固化版，当前应 FAIL）**——容器内：

```bash
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0,1 flagrt-v3-demo-910c \
  torchrun --nproc_per_node=2 dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py --iters 20
```

**修复验收臂（挂载覆盖版，应 PASS）**——先按机制起 [`device-context/910C/env/train`](../../device-context/910C/env/train/docker-compose.yml)
的运行环境，再在容器内用一条命令跑完**两项**验收（① 复现脚本 ② 原生 DDP 双卡脚本）：

```bash
# 起环境（仓根执行）：export 与启动都走机制（本方向的 env 目录声明了 FlagCX 挂载覆盖）
python3 dev/lib/baseline.py export dev/device-context/910C/env/train --require-clean
FLAGRT_MODELS_DIR=<实体模型目录> dev/lib/up.sh dev/device-context/910C/env/train
#   ⚠️ npu1-27 上**不要**加 --strict：本机镜像 id（7028028bb62c…）与基线记录的制造机 id
#      （e13a15d0d02b…）不同 ⇒ check 恒 FAIL、--strict 恒拒绝启动（image_list.md 已登记"跨机
#      image id 不同"，但这条后果是 2026-10-16 才实测钉死的）。本机结论请注明"未满足 --strict"。
#      实测：不加 --strict 时 up.sh rc=0、容器内两项验收均 PASS（2026-10-16）。
```


```bash
bash dev/device-context/910C/probes/run_910c_p9_acceptance.sh
# 三态：overall: PASS / 退出码 0 = 修复到货；overall: FAIL / 1 = 缺陷仍在；ERROR / 2 = 无法下判定
```

被测脚本（**引用，不复制**）：
[`dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py`](../../images/ascend-train-comm/v3/assets/flagcx_sync_test.py)
——对 4 个 collective 各做「未同步即读取 / 同步后读取」两组，每次调用使用唯一数据（据此区分"本次结果"
与"上一次调用的遗留结果"），跨 rank（`all_reduce` MIN）、跨迭代汇总后判定；未同步组的失败为偶发，
单次可能通过，故取 `--iters 20`。
原生 DDP 脚本 [`assets/ddp_native_test.py`](assets/ddp_native_test.py)（本目录，
`DistributedDataParallel` 封装、**不含手动同步**、双卡自判定）。
容器启动与收尾见 [`NOTES-step-by-step.md`](NOTES-step-by-step.md)。

### 预期结果（问题存在时，退出码 1）

```
=== VERDICT (all ranks, iters=20) ===
  broadcast+nosync                 FAIL
  broadcast+sync                   PASS
  all_reduce+nosync                FAIL
  all_reduce+sync                  PASS
  all_gather_list+nosync           FAIL
  all_gather_list+sync             FAIL
  all_gather_into_tensor+nosync    FAIL
  all_gather_into_tensor+sync      PASS
overall: FAIL
```

判定段之后 torchrun 会打印 `ChildFailedError`（退出码 1 的正常表现），不是另一个错误。

## 修复验收

**判据**（两项都要）：① 复现脚本退出码 0、判定行 `overall: PASS`（八组全 PASS——调用方在当前流上
直接读取结果无需额外同步，且列表形式 `all_gather` 结果正确）；② 原生 `DistributedDataParallel`
双卡验证通过（`DistributedDataParallel` 封装、不含手动同步的自判定脚本，四项全 PASS）。
两项已合进 [`run_910c_p9_acceptance.sh`](../../device-context/910C/probes/run_910c_p9_acceptance.sh)
**一条命令**，并按 `CASE-TEMPLATE` 的三态口径给合并判定。

**2026-10-16 实测（npu1-27，die 5 + die 0，同一容器、同一条命令，只换加载哪一份 FlagCX）**：

| 臂 | 容器内实际加载的 FlagCX | ① `flagcx_sync_test.py --iters 20` | ② `ddp_native_test.py --steps 10` | 合并 |
|---|---|---|---|---|
| **B = 挂载覆盖臂**（`b576fa6a`，`pip-editable`） | `/workspace/FlagCX/plugin/torch/flagcx` | **`overall: PASS`**（八组全 PASS） | **`overall: PASS`**（四项全 PASS） | **PASS / rc=0** |
| **A = 镜像固化臂**（`4e0e0cbc`，`PYTHONPATH` 影子） | `/opt/baked-flagcx/flagcx` | `overall: FAIL`（八行图案与上文"预期结果"逐行一致） | `overall: FAIL` | FAIL / rc=1 |

装置自证：两臂的 `_C…so` / `libflagcx.so` sha256 前 16 位不同
（B `098f3227…`/`5af25cc9…`，A `a6a4a8e0…`/`8196a720…`），且每次运行打印实际加载路径；
挂载覆盖臂的安装自证为 `Successfully installed flagcx-0.13.0` 且 `import flagcx` 解析到
`/workspace/FlagCX/plugin/torch/flagcx`（宿主 checkout，非镜像内那份）。凭据见
[`assets/evidence/device-selfproof-20261016.txt`](assets/evidence/device-selfproof-20261016.txt)
与 [`assets/evidence/install-proof-20261016.txt`](assets/evidence/install-proof-20261016.txt)。

| 脚本 | 类别 | 命令 | 通过判据 | 最近一次结果与证据 |
|---|---|---|---|---|
| [`flagcx_sync_test.py`](../../images/ascend-train-comm/v3/assets/flagcx_sync_test.py)（引用，不复制） | 复现 / 验证 | 见「单入口命令」 | 退出码 0 且 `overall: PASS` | 2026-10-16 npu1-27：复现臂 `overall: FAIL`（rc=1，八行与"预期结果"逐行一致）；挂载覆盖臂 `overall: PASS`（rc=0）。证据 [`assets/evidence/A_sync_verdict-20261016.txt`](assets/evidence/A_sync_verdict-20261016.txt) · [`assets/evidence/B_sync_verdict-20261016.txt`](assets/evidence/B_sync_verdict-20261016.txt) |
| [`assets/ddp_native_test.py`](assets/ddp_native_test.py) | 验证 | 见「单入口命令」，或单独 `torchrun --nproc_per_node=2 assets/ddp_native_test.py --steps 10` | 退出码 0 且 `overall: PASS`（`ddp_wrapped` / `train_steps` / `loss_all_finite` / `ranks_param_equal` 四项全 PASS） | 2026-10-16 npu1-27：复现臂四项全 FAIL（`DDP expects same model across all ranks …`），挂载覆盖臂四项全 PASS。证据 [`assets/evidence/A_ddp_verdict-20261016.txt`](assets/evidence/A_ddp_verdict-20261016.txt) · [`assets/evidence/B_ddp_verdict-20261016.txt`](assets/evidence/B_ddp_verdict-20261016.txt) |
| [`run_910c_p9_acceptance.sh`](../../device-context/910C/probes/run_910c_p9_acceptance.sh) | 复现 / 验证 | `bash dev/device-context/910C/probes/run_910c_p9_acceptance.sh` | 合并退出码 0 = 两项均 PASS | 同上（本表前两行即它的两项） |

**完整日志**（不入仓，照 `CASE-TEMPLATE §2` 规则 4 注明机器名与绝对路径）：
`npu1-27:/mnt/raid/hliu553/dc_env_20261016/log/`（含 `dc_env_run.log`、`arm_{A,B}_final.log`）
与 `/mnt/raid/hliu553/dc-ws/acc/{A,B}/`；当轮驱动脚本
[`device-context/910C/probes/run_dc_env_910c_20261016.sh`](../../device-context/910C/probes/run_dc_env_910c_20261016.sh)，
本轮证据索引与装置说明
[`device-context/910C/probes/dc_env_20261016_out/SUMMARY.md`](../../device-context/910C/probes/dc_env_20261016_out/SUMMARY.md)。

### 修复到货后的收尾

两项验收通过后，`ascend-train-comm v3` 的 `repro_status` 由 🟡 升 🟢
（[`ascend-train-comm/v3/README.md`](../../images/ascend-train-comm/v3/README.md)），训练侧不再需要手动同步写法。

⚠️ **判据是"镜像里换成了含修复的 FlagCX"，不是"本 case 在旧镜像上又能 PASS"**：本轮已证明修复有效
且可复现，但**镜像尚未接收该修复** ⇒ 在镜像更新前**不得**改 `repro_status`，也不得删除已知限制
A-2（[`docs/基座发版说明-20261014.md`](../../../docs/基座发版说明-20261014.md)）。镜像接收修复的形态
（新建 `v4/`，还是给 v3 出补丁版）由镜像维护者裁定。
