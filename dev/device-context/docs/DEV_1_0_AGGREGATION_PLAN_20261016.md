# 910C 今日工作 → 新基座/汇聚规范 → 未推送内容与推送方案（2026-10-16）

> 本文自包含：数字与判定都带实测出处，命令可直接照抄。**分支与提交的事实以 2026-10-16 15:0x 实测为准**。
> 配套工具：[`../scripts/dryrun_merge_dev1.0.sh`](../scripts/dryrun_merge_dev1.0.sh)（合入前干跑预检，只读 + 自动回滚）。

---

## 0. 一页结论

| 项 | 状态 |
|---|---|
| 上午 · 910C `all_gather` 类缺陷（flagcx-c10d-collectives） | ✅ **真机复现 + 本分支修复验收通过**（A/B 交替 ×2 + 原生 DDP 双卡）；修复已于当日以 merge 提交 **`62fe615`** 合入 FlagCX `dev-1.0`（见 §7 执行记录；镜像里那份仍 pin 在 `4e0e0cb`） |
| 下午 · 新基座规范适配 | ✅ 运行环境目录 `910C/env/{train,infer}/` + 分层关系与执行标准成文 |
| 下午 · 机制路径 | ✅ 用户级补齐前置（compose v2.29.7 + pyyaml 6.0.2）→ `export --require-clean` 双腿 rc=0 |
| 下午 · `up.sh --strict` | ✅ 真机 rc=0；容器内两项验收 `overall: PASS`；`down` rc=0 无残留 |
| **待推送内容**（执行前快照） | **两条线**：FlagCX 分支领先 `dev-1.0` **17 提交**（可快进）；本仓 `kistich/device-context` 与 `dev-1.0` **118 领先 / 106 落后**，本方向独有 **1295 项** —— ⏹ **当日下午已全部推送完成，权威状态见 §7** |
| **推送通道** | **4 条**（§4）：① FlagCX `dev-1.0` ② 本仓 `dev-1.0` ③ `dev/issues` case 包 ④ 镜像侧晋升 —— **①②③ 已于 2026-10-16 执行完毕**（§7），④ 待镜像维护者 |
| **决策点** | **3 个 —— 已按建议裁定并执行**（§5 / §7）：`image.id` 取 (B) 回发布 id · `summary/` 取 (A) 随本次带入 · 草稿取 (A) 从个人分支移除 |

---

## 1. 上午：910C 的 all_gather 问题（问题 → 定位 → 修复 → 验收）

### 1.1 问题（上游 issue `dev/issues/flagcx-c10d-collectives`，两个类别）

| 类别 | 现象 |
|---|---|
| ① | collective **返回时数据还没写完** —— 后续依赖“已写好的数据”的逻辑读到陈旧张量 |
| ② | 列表型 `all_gather`（`all_gather_list`）**读到上次结果**，**显式 `synchronize` 也无效** |

### 1.2 归属事实链（本次最重要的判定）

| 项 | 实测 |
|---|---|
| `origin/dev-1.0` tip（FlagCX 仓） | `4e0e0cb`（2026-08-20） |
| `git merge-base origin/dev-1.0 HEAD` | **`4e0e0cb` 本身** ⇒ 正是我们分支的分叉点 |
| 我们领先 | **17 个提交，dev-1.0 一个都没收到** |
| ⇒ 推论 | **issue 所用镜像必然同时携带 P6/P9**；issue 里「第 4 次判定与前 3 次逐行一致」**不是缺陷稳定，而是镜像里的 FlagCX 从未变过** |
| ⚠️ 另需注意 | 同 tag 镜像在不同机器上 **image id 不同**：npu1-11 `e13a15d0d02b` vs npu1-27 `7028028bb62c`（引用 image id 要连机器一起引） |

### 1.3 我的复现与验收（npu1-27，die 5 + die 0，同一容器同一命令，只换加载哪份 FlagCX）

| 臂 | ① `flagcx_sync_test.py --iters 20` | ② 原生 DDP 双卡 |
|---|---|---|
| 镜像固化版（`4e0e0cb`） | **FAIL**（rc=1，八行图案与 issue 一致；`all_gather_list+sync` 亦 FAIL） | **FAIL** —— 复现 issue 原文 `DDP expects same model across all ranks, but Rank 0 has 4 params, while rank 0 has inconsistent 0 params` |
| 我们分支编译版 | **PASS**（八组全过，rc=0，3 次独立运行） | **PASS**（`ddp_wrapped`·`train_steps`·`loss_all_finite`·**`ranks_param_equal`** 四项全过） |

对照按纪律做：**交替 A/B ×2 轮** + 每次打印 `LOADED` 自证（A=镜像解出的包 / B=site-packages）+ **两套插件指纹**（A `_C=a6a4a8e0`/`libflagcx=8196a720`；B `_C=72256b23`/`libflagcx=8fe6bd4a`）。

### 1.4 修复面（都在**已提交**的 HEAD 上）

| 类别 | 机制 | 落点 |
|---|---|---|
| ① 完成语义 | `eventQuery → eventSynchronize`；回调前 `aclrtSynchronizeStream`；**collective 跑在 torch 当前流**（`getCurrentCUDAStream` / `torch_npu` 当前流） | `flagcx/core/net.cc` · `flagcx/adaptor/device/cann_adaptor.cc` · `plugin/torch/flagcx/src/backend_flagcx.cpp` |
| ② 缓冲生命周期 | **每个 collective 返回前 `event_->synchronize()`（17 处）** + event RAII 析构 ⇒ “collective 返回”被重定义为**数据已落定** | `plugin/torch/flagcx/**` |

⚠️ **纪律**：类别 ①/② 的失败**不是同一个修复点**，报告里特意写明「**不要用一个修复去解释两个现象**」。

### 1.5 在标准基座上构建我们的分支：三个前置（缺一个就编不过或“假重编”）

1. `makefiles/ascend.mk` 的 DAG 持久 kernel 链接项 `-lcust_opapi` 在镜像里不存在（无 `custom_opp_*.run`）⇒ **只砍 `-lcust_opapi`**，**`-lopapi`/`-lnnopbase` 必须留**（`aclnnInplaceAddGetWorkspaceSize` 在 `libopapi.so`，砍多了 import 期报 undefined symbol）。已改为开关 `FLAGCX_ASCEND_DAG`（默认 0）。
2. `plugin/torch/flagcx/src/backend_flagcx.cpp` 无条件 `#include <ATen/cuda/CUDAContext.h>` ⇒ ascend-only 镜像编译失败（**本分支自身引入的可编译性缺陷**）；已加 `#ifdef USE_NVIDIA_ADAPTOR` 守卫。
3. **`pip` 报 `Successfully installed` ≠ 真重编**（增量缓存会零编译输出）：判据看产物 `DT_NEEDED` + sha256，别只看退出码。

### 1.6 产出

`910C/docs/FLAGCX_C10D_COMPLETION_20261016.md`（主报告）· `910C/probes/run_910c_p9_acceptance.sh`（单入口，两项一起跑 + 合并三态）· `910C/probes/ddp_native_test.py`（原生 DDP 自判定，补齐 `dev/images/TODO.md` 第 2 项）· 13 份证据截选 `910C/probes/p9_c10d_20261016_out/`。

---

## 2. 新基座 / 汇聚规范要点（`dev-1.0` 上的现行文本）

### 2.1 分层：**两套编号不可混**

| 会议口述 | 规范里的东西 | 该走的动作 |
|---|---|---|
| “搭好基座让大家往上堆” | **镜像**（L1–L3 随 image id 冻结） | 使用方不直接改镜像 |
| “FlagCX 是基础库，算第二层” | 规范里它是 **L3 组件**；“第二层”指**堆叠顺序** | 要它变新 ⇒ **挂载覆盖**，不是等基座发版 |
| “子方向建自己的 package，第三四层” | **L4 挂载装配**下的子方向代码 | 放进运行环境目录 |
| “只 fix、commit，大家覆盖安装最新 FlagCX” | **下放**（固化 → 挂载） | 镜像不变，`env.baseline.yaml` 记 `overrides_baked` |

- **ENV-SPEC 的 L1–L4 是「职责层级」**，且与“从哪继承”**正交**（基座自带的 FlagCX 仍是 L3）。
- 跨界只有两个显式动作：**下放**（固化→挂载）/ **晋升**（挂载→固化，进下一版镜像配方）。⚠️ **包索引预编译 wheel 不可下放**（如昆仑芯 flagtree）；源码固化的（FlagCX）可。

### 2.2 两类基线 + 三条命令

| 类别 | 位置 | 生成/核对 |
|---|---|---|
| 镜像基线 `image.baseline.yaml` | `dev/images/<name>/v<N>/` | `baseline.py export --image …` / `check --image …` |
| 运行环境基线 `env.baseline.yaml` | 各方向的 `<env-dir>/` | `baseline.py export <env-dir> [--require-clean]` / `check` |
| 启动 | —— | `dev/lib/up.sh <env-dir> [--strict]` |

- **两类基线均由 `baseline.py` 生成和核对，不手工编辑**（ENV-SPEC §3）。
- `--strict`：`check` 不一致即**拒绝启动**，用于**验收与出结论**；⚠️ **只约束启动时刻**。
- `export --require-clean`：工作区不干净或 commit 未推送即**拒绝导出**（出结论的前提）。
- 四条硬规则 = **单入口 / 外部依赖声明 / 自验证 / 日志不进仓**；三态 = **PASS(0) / FAIL(1) / ERROR(2)**。

### 2.3 目录规范（三类产物各有形态）

| 产物 | 形态 | 模板 |
|---|---|---|
| 镜像 | `v<N>/`：`Dockerfile.repro` + `build.sh` + `image.baseline.yaml` + `demo/` + `ARCHIVE.md` + `lock.yaml` + `README.md` | `dev/images/README.md` §3 |
| 运行环境 | `<env-dir>/`：`docker-compose.yml` + `env.baseline.yaml`（+ `.env.example`） | `dev/_template/` |
| 缺陷 case 包 | `dev/issues/<name>/`：README（三段式）+ `NOTES-step-by-step.md` + `assets/`（脚本三类）+ `assets/evidence/`（截选） | `dev/images/CASE-TEMPLATE.md`；实例 `dev/issues/` 下三个 |

### 2.4 ⭐ 汇聚规范（本次重点，原文在 `dev-1.0` 的 `README.md` §分支模型 + `dev/sync_to_dev.sh`）

**分支模型（fork 三线 · GitFlow develop）**：`main` 只同步与验收 PR；**`dev-1.0` 是团队公共开发分支，所有个人分支在此汇合**；共享分支（main / dev-1.0）**只用 merge，不用 rebase**。

| 仓类型 | `dev-1.0` 接受什么 | 怎么提交 |
|---|---|---|
| **自建仓（runtime-team）** | 只接受**用户开发分支的 merge** | `dev/sync_to_dev.sh` **五步**：① `fetch` ② 本地 `dev-1.0` 快进 ③ **把 `dev-1.0` 合进个人分支**（冲突在此解决）④ 把个人分支**合进 `dev-1.0`** ⑤ 校验无分叉后 `push origin dev-1.0`（**无需 PR**） |
| **fork 子库（FlagCX 等 6 个）** | ① 用户开发分支的 merge ② `main` 的 merge | 同上（本地 merge 后 push） |

**⇒ 与我们此前的做法不同**：过去我们走「**只做子树同步、绝不整体推**」（把 `dev/device-context` 单独 checkout 成一条从 `dev-1.0` tip 出发的新提交）。新规范要求**走 merge**（个人分支 ↔ `dev-1.0` 双向合流）。两者差异会直接影响推送安全性（见 §3.3、§4.2）。

---

## 3. 我们未推送至 `dev-1.0` 的全量内容（实测清单）

### 3.1 FlagCX 仓（`FlagRT/FlagCX`）

| 项 | 值 |
|---|---|
| 分支 / HEAD | `kistich/ascend-flagcx-adapt` @ **`b576fa6a`**（工作区**干净**，已推送） |
| 相对 `origin/dev-1.0`（`4e0e0cb`） | **领先 17 · 落后 0**（`merge-base` = `4e0e0cb`）⇒ **快进关系** |
| 推送权限 | `git push --dry-run origin kistich/ascend-flagcx-adapt:dev-1.0` → `4e0e0cb..b576fa6` ✅ 可推 |
| 遗留检查 | 早期调试打点（`HETERO-DBG` = FlagCX 源码里的调试标记名，**非对外表述**）命中 **0** 个文件；无 `/mnt/raid` 等本机路径硬编码 |

**17 个提交按主题分组**：

| 主题 | 提交 |
|---|---|
| **完成语义（本次 all_gather 修复的核心）** | `a1e7e0f`（P2+P6+P7+P8+P9 五修复）· `d296824`（大模型 DDP step~765 退化：`flagcxCannEvent` 泄漏 aclrtEvent + 完成语义）· `0686805` / `60e0695`（collective 跑 torch_npu 当前流）· `4fbd9e9`（NVIDIA 侧 collective 跑 torch 当前 CUDA 流） |
| **跨厂商（cross-vendor）** | `08535e1`（O4 socket 协议 seq 校验 + 错误终结）· `e9eb435`（irecv recvSeq 递增）· `900f9d2`（uniRunnerAllReduce + launchHostFunc）· `41a0b44`（4090+910C 跨机 allgather 打通） |
| **昇腾适配** | `5f7ad78`（四层根因）· `984e37a`（UVA host-mapped + DAG 持久 kernel + 探测回退）· `4bbbae5`（`flagcxGetLastError` 存根） |
| **性能** | `8402a93`（`flagcx_collective` 向量化 reduce + 去数据缓冲 dcci） |
| **构建可编译性（本次新增）** | `5e44b3c`（CUDA 头宏守卫 + DAG 链接项开关化 + `.gitignore`） |
| **跨厂商结构性问题（本次新增）** | `b576fa6`（adaptor 表布局漂移 / 拓扑空回调 / 簇作用域 bootstrap） |
| **文档** | `60a1296` · `80967b8` |

### 3.2 本仓 `FlagRT/runtime-team`

| 项 | 值 |
|---|---|
| 分支 / HEAD | `kistich/device-context` @ **`2ffa3e9`**（工作区干净；`dev/issues/` 为**未跟踪**改动） |
| 与 `origin/dev-1.0`（`74197bf`，2026-10-15 21:41）的关系 | 分叉于 **`7313aef`**（2026-09-17「Merge PR #19」）⇒ **领先 118 · 落后 106**（双向） |
| **我们独有（`dev-1.0` 没有）** | **1299 项**：`dev/device-context/**` **1295** · `summary/**` **3** · `dev/stack.lock.910c.yaml` **1** |
| `dev-1.0` 独有（会流入我们） | **539 项**：`dev/communication` 145 · `dev/framework-adapter` 139 · `dev/memory` 84 · `dev/images` 80 · `dev/rag_ljy` 43 · `dev/issues` 12 · `dev/model-defs` 9 · `dev/batch_yxy` 7 · `dev/lib` 3 · `dev/performance` 2 · `docs/*` 4 |
| 合并冲突（干跑实测，基线 = 真 tip `74197bf`） | **34 项**：`dev/device-context` **22**（UU 12 + AA 10）· `dev/framework-adapter` **4** · `dev/memory` **5** · `dev/performance` **2** · `dev/communication` **1**；其中 **12 项是 modify/delete（DU）**—— 我们删了、对方改了 |
| 合并变更总量（干跑实测） | **A 305**（他方向/共用目录流入）· **M 14**（自动合并）· **UU 12 · AA 10 · DU 12** · **D 0**（⚠️ 没有任何“静默真删”—— 删别人目录都以 DU 冲突形式暴露出来） |

**我们目录新增的三个顶层子目录**（`dev-1.0` 侧只有 `.env.example / 910C / P800 / README.md / STATUS.md / prototype`）：

| 目录 | 内容 |
|---|---|
| `MLU590/` | 第 3 家寒武纪实例：接入结论、T1-re 复跑、服务化四形态（`docs/CAMBRICON_MLU590_SERVE_FOUR_FORMS_20261014.md`）等 |
| `PPU/` | 第 4 家平头哥实例：接入收官、T11 退出期段错误根因（`docs/PPU_T11_EXIT_SEGV_ROOTCAUSE_20261014.md`）等 |
| `docs/` | 方向级规范：`BASELINE_LAYERS_AND_ADOPTION_20261016.md`（分层关系/执行标准/能力边界/待办） |
| `910C/env/` | 新机制的运行环境目录（`train/` `infer/`）+ `pin_local_image_id.sh` |

### 3.3 ⚠️ 推送前必须处理的 4 处风险（都是干跑实测出来的）

| # | 风险 | 事实 | 处置 |
|---|---|---|---|
| 1 | **我们分支删掉了别人的目录** | 2026-09-22 的 5 个提交 `Delete dev/<dir> directory`（`communication`/`framework-adapter`/`memory`/`performance`/`rag_ljy`）+ `7d05d69`（路线 B 退出）。干跑实测：这些删除**不会静默生效**，而是呈现为 **12 项 modify/delete 冲突**（framework-adapter 4 · memory 5 · performance 2 · communication 1），且 D 项 = 0 | 合并冲突解决时**一律恢复为 `dev-1.0` 版本**（等于撤回这些删除）：`git checkout --theirs -- <path>` 或 `git checkout origin/dev-1.0 -- <path>`。**绝不能把这些删除合进 `dev-1.0`** |
| 2 | **夹带基座草稿** | `dev/stack.lock.910c.yaml` **仅我们分支有**（`dev-1.0` 无）⇒ 合并会把草稿带进公共分支 | 合并前从个人分支**移除**（纪律：草稿不 PR） |
| 3 | **`summary/` 在 `dev-1.0` 缺失但被引用** | `dev-1.0` 的我们目录里多份文档引用 `summary/DEVICE_ABSTRACTION_ROUTE_AB_SUMMARY_20260922.md`（如 `910C/docs/DC_STAGE_SUMMARY_20260909.md`、`910C/distributed_training/ascend_regression/*.md`），而 `dev-1.0` 顶层**没有 `summary/`** ⇒ 当前是**悬空引用** | 建议**随本次带入**（顺势修复悬空引用）；见 §5 决策 2 |
| 4 | **`env.baseline.yaml` 的 `image.id` 是本机 id** | 我们在 10-16 按本机（npu1-27 `7028028bb62c…`）对齐过，而镜像基线记制造机（npu1-11 `e13a15d0d02b…`）⇒ 合进共享分支后**所有人的 `check` 会 FAIL** | 见 §5 决策 1（**建议改回发布 id**） |

> ✅ **一条让人放心的结论**：`dev-1.0` 侧对我们目录的改动**只有 2 个提交**（`6af89c4` 顶层看板/STATUS 两实例口径 · `8cc9dbe` 对齐三个子树），**作者都是我们自己**（子树同步），且内容口径停留在 **09-23**（我们分支更新）⇒ **`dev/device-context/**` 整体取我们的版本不会丢掉别人的修正**。抽查三例：`README.md`（dev-1.0 侧仍是「两实例 阶段 0–5 完成」）· `STATUS.md`（同）· `prototype/runtime/backends/base.py`（dev-1.0 侧仅 14 行更旧表述，我们侧多 794 行改动）。

---

## 4. 如何按新规范推送（分通道 · 分步）

### 通道 ① FlagCX（fork 子库）—— 关系最简单，建议先做

```bash
cd FlagCX
export GIT_SSH_COMMAND="ssh -i ~/.ssh/id_kistich -o IdentitiesOnly=yes"
git fetch origin
git status --porcelain                 # 必须为空（已实测为空）
git checkout dev-1.0
git merge --ff-only origin/dev-1.0     # 本地 dev-1.0 先对齐远端
git merge --no-ff kistich/ascend-flagcx-adapt \
  -m "Merge kistich/ascend-flagcx-adapt into dev-1.0（昇腾完成语义 + 跨厂商修复，17 提交）"
git push origin dev-1.0
# 三项核验
git ls-remote origin refs/heads/dev-1.0        # tip 应 = b576fa6a…（含新 merge 提交则为其 sha）
git merge-base --is-ancestor origin/dev-1.0 dev-1.0 && echo 无分叉
git checkout kistich/ascend-flagcx-adapt
```
- **判据**：`git rev-parse origin/dev-1.0` 的祖先链含 `b576fa6a`；`git show origin/dev-1.0:makefiles/ascend.mk | grep -c FLAGCX_ASCEND_DAG` ≥ 1。
- ⚠️ **合入后必须通知各方向**：10-14 的口径是「FlagCX 是第二层基础库，各自 `fix`/`commit` 后**覆盖安装**最新 FlagCX」——他们需要重新 `pip install -e` 或拉新镜像。
- 失败处置：`--ff-only` 失败说明本地 `dev-1.0` 与远端分叉 ⇒ 重跑 `git fetch` + `--ff-only`；merge 冲突（预期无）⇒ 手工解决后 `git merge --continue`。
> ⚠️ **本次实际执行时该行提交信息写作「异构修复」**（旧口径）。FlagCX `dev-1.0` 的 merge 提交 `62fe615` 已推送到组织仓、不追改；
> 自本行起统一用「**跨厂商（cross-vendor）**」。

### 通道 ② 本仓 `runtime-team`（自建仓）—— 冲突最多，按下面顺序

**Step 0 预检（只读 + 自动回滚，不动任何分支）**
```bash
bash dev/device-context/scripts/dryrun_merge_dev1.0.sh
#   它在 `--detach` 的临时 worktree 里 merge --no-commit，打印冲突/新增/删除清单后 `merge --abort` 并移除 worktree
#   ⚠️ 脚本内有守卫：断言「当前目录 == 临时 worktree」才执行 merge
```

**Step 1 清理个人分支（一个提交，让后续冲突面变小且方向明确）**
```bash
git rm -q --cached dev/stack.lock.910c.yaml && rm -f dev/stack.lock.910c.yaml   # ① 移出草稿
git checkout origin/dev-1.0 -- dev/communication dev/framework-adapter dev/memory dev/performance dev/rag_ljy
                                                                              # ② 撤回 09-22 对别人目录的删除
git add -n dev/issues && git status --short dev/issues                        # ③ 先看 case 包会带进哪些改动
git add dev/issues                                                             #    确认后纳入跟踪（见通道 ③）
git commit -m "chore(sync): 剪除基座草稿；恢复被误删的他方向目录；纳入 flagcx-c10d case 包"
```

**Step 2 把 `dev-1.0` 合进个人分支（冲突在此解决）**
```bash
git fetch origin
git merge --no-ff origin/dev-1.0 -m "Merge dev-1.0 into kistich/device-context"
# 冲突解决策略（逐目录）：
#   本方向： git checkout --ours   -- dev/device-context   （必要时逐个文件看 diff 再定）
#   其他一切目录： git checkout --theirs -- <path> 或 git checkout origin/dev-1.0 -- <path>
git add -A && git commit
```

**Step 3 个人分支 → `dev-1.0` → push（等价于 `dev/sync_to_dev.sh` 的 4–5 步）**
```bash
git checkout dev-1.0
git merge --ff-only origin/dev-1.0
git merge --no-ff kistich/device-context -m "Merge kistich/device-context into dev-1.0"
git fetch origin
git merge-base --is-ancestor origin/dev-1.0 dev-1.0 && echo "无分叉，可推"
git push origin dev-1.0
git checkout kistich/device-context
# 或者：Step 1/2 完成、分支已吸收 dev-1.0 后，直接 `bash dev/sync_to_dev.sh` 走五步
```

**Step 4 四项核验（本仓）**
```bash
R=$(git ls-remote origin refs/heads/dev-1.0 | awk '{print $1}'); L=$(git rev-parse origin/dev-1.0)
[ "$R" = "$L" ] && echo "① tip 一致"
git fetch -q origin dev-1.0 && [ -z "$(git diff --stat origin/dev-1.0 | head -1)" ] || true   # ② 与本地一致
git ls-tree -r --name-only origin/dev-1.0 | grep -cE "^dev/device-context/(MLU590|PPU|docs)/" # ③ 新目录已在
git ls-tree -r --name-only origin/dev-1.0 | grep -c "^dev/memory/"                            # ④ 别人目录未丢
git ls-tree --name-only origin/dev-1.0:dev | grep -c "stack.lock.910c.yaml"                   # ⑤ 草稿未带入（应为 0）
```

### 通道 ③ `dev/issues/flagcx-c10d-collectives`（总组发布区）

- 现状：我们改好的 case 包**只在本机工作区**（本分支不跟踪 `dev/issues`）。
- **规范约束**：`dev-1.0` 只接受个人分支的 merge ⇒ **不能直接改 `dev-1.0`**。两条路：
  - **推荐 (a)**：在个人分支上把 `dev/issues/` 纳入跟踪（内容 = `dev-1.0` 版本 + 我们的改动）⇒ 随**通道 ②**一起合入。安全性已核：我们的工作区 `dev/issues/` 是从 `dev-1.0` 归档解出后编辑的，其余 issue 内容与 `dev-1.0` 一致，`git add` 只会带出 `flagcx-c10d-collectives/` 的改动（Step 1 的 `git add -n` 会先给清单）。
  - (b) 把改动打成 patch 交总组代改。
- 内容：更正旧口径（「第 4 次逐行一致」的真因）· 补修复位置与归属事实 · 复现臂/修复验收臂分列单入口 · 补 `assets/ddp_native_test.py` + 6 份证据截选 · NOTES 追加修复臂逐步操作。

### 通道 ④ 镜像侧「晋升」（不由我们执行，我们提供输入）

按指南 §4.3（把改动固化进基座）：
1. **我们已做**：修复合入组织 fork 的 `dev-1.0`（= 通道 ①），并留下验证证据。
2. **需镜像维护者做**：在 `dev/images/ascend-train-comm/v4/`（或给 v3 出补丁版）的 `Dockerfile.repro` 把 `ARG FlagCX_COMMIT=` pin 成 **`b576fa6a…`**，按指南 §3 发布（构建 → demo → `baseline.py export/check --image` → 带卡 `up.sh --strict` + 自判定脚本 → `ARCHIVE.md`/`lock.yaml`/`image_list.md`/`VERSIONS.md`）。
3. **发布后我们做**：运行环境改用新镜像并**删除 `mounted_repos` 的 FlagCX 挂载声明**（挂载只是过渡态）。
4. **判据**：新镜像 `image.baseline.yaml` 的 FlagCX `commit = b576fa6a…`；且**在 baked 镜像上直接跑** `flagcx_sync_test.py --iters 20` 即 PASS（不再依赖挂载覆盖）。
5. ⚠️ **A-2（已知限制）的解除权在镜像侧**：我们只提供「修复 + 验证证据」，不改 `repro_status`。

---

## 5. 需你裁定的 3 个决策点

| # | 决策 | 选项 | 我的建议 |
|---|---|---|---|
| 1 | `910C/env/train/env.baseline.yaml` 的 **`image.id`** | (A) 保留**本机** id（现状：仓内文件对 27 有效、对别人 FAIL）；(B) **改回镜像基线的发布 id**，本机用 `pin_local_image_id.sh` 对齐（工具与证据都留仓） | **(B)**。`dev-1.0` 是共享分支：机器专属 id 会让所有人的 `check` FAIL；而 `export` 本来就会把它抄回发布 id（本次已实测复现）。本机 `--strict` 验收证据已冻结，不因改回而失效（验收时的文件形态与证据都在 `probes/baseline_strict_20261016_out/`） |
| 2 | **`summary/`**（3 个文件：对外总结 + 周报 + 路线 A/B 摘要） | (A) 随本次带入 `dev-1.0`；(B) 移进 `dev/device-context/` 下并改引用；(C) 不带入 | **(A)**。`dev-1.0` 的我们目录多处**已在引用** `summary/…`（现为悬空），带入即修复；若总组认为顶层不宜新增目录，则退 (B) |
| 3 | 草稿 **`dev/stack.lock.910c.yaml`** | (A) 从个人分支移除（不入公共分支）；(B) 一并带入 | **(A)**。纪律是草稿不 PR；所需内容由总组的 `stack.lock.910c.v1–v4.yaml` 承担（干跑实测：它是**唯一**“仅我们分支有的 `dev/` 顶层文件”） |

**裁定结果（2026-10-16，用户裁定：按上表建议执行）**

| # | 裁定 | 落地位置（可复核） |
|---|---|---|
| 1 | **(B)** `image.id` 回**发布 id** `sha256:e13a15d0…f15`；本机用 `pin_local_image_id.sh` **临时**对齐（对齐后文件是脏的、跑完 `git checkout --` 复原，**不入仓**） | 提交 `f014442`；口径同步进 `910C/env/README.md` §2.1/§6 · `README.md` 头部 · `STATUS.md` |
| 2 | **(A)** `summary/` 随本次带入 `dev-1.0` ⇒ **悬空引用已修**（我们目录内有 16 个文件引用 `summary/…`） | 合并提交 `2e077fb`：`git ls-tree origin/dev-1.0 summary` = **3** 文件 |
| 3 | **(A)** 草稿从个人分支移除；同时把 5 处指向草稿的文档引用改为「已被取代 / 现存依据是总组定稿」，不留悬空指针 | 提交 `f014442`：`git ls-tree origin/dev-1.0:dev` 中 `stack.lock.910c.yaml` **= 0**（实测） |

⚠️ **决策 1 的连带影响（如实登记）**：仓内改回发布 id 后，**本机**跑 `check` / `--strict` 仍须先 `pin`（固定顺序 `export` → `pin` → `check` → `up --strict`），跑完 `git checkout -- <env-dir>/env.baseline.yaml` 复原，否则会把本机 id 提交上来。本机 10-16 的 `--strict` 验收证据不受影响（已冻结在 `910C/probes/baseline_strict_20261016_out/`）。

---

## 6. 附：本轮自查发现的操作风险（含我自己踩的一个）

| # | 事 | 说明 | 已做的加固 |
|---|---|---|---|
| 1 | ⚠️ **`git worktree add` 失败后命令会在真实工作树上执行** | 我第一次干跑时 `worktree add` 因「分支已在本仓检出」失败，`cd /tmp/...` 随之失败，`&&` 链断开但后续命令仍**在真实工作树**跑了 `git merge --no-commit`（83 项变更、27 冲突） | 已 `git merge --abort` 复原（HEAD 未变、工作树只剩未跟踪 `dev/issues/`、10 个文件完好）；并把干跑**固化成脚本**：`git worktree add --detach` + **断言 `pwd` == 临时 worktree 才执行 merge** + 结束 `merge --abort` + 移除 worktree |
| 2 | 判据写错会报假 FAIL | 驱动脚本用 `docker ps --format '{{.Names}}|{{.Status}}'` 的行做 `grep -E "^<名字>$"` ⇒ **永不命中**（行尾还有 `|Up …`） | 改 `docker ps --format '{{.Names}}' | grep -qx`，并另打 `名字|状态` 佐证 |
| 3 | ⚠️ **陈旧 remote-tracking ref 会低估冲突面** | `git fetch <remote> <branch>` **只写 `FETCH_HEAD`**，不动 `refs/remotes/origin/<branch>`。本仓 `origin/dev-1.0` 当时停在 `6af89c4`，而真 tip 是 `74197bf`（差 19 提交）⇒ 用陈旧 ref 干跑得 **27 冲突 / 53 A**，用真 tip 得 **34 冲突 / 305 A**，还多出 framework-adapter 与 performance 两组 DU | 预检脚本已加**不一致告警**，并在文件头写明「先 `git fetch origin dev-1.0:refs/remotes/origin/dev-1.0`，或把 `FETCH_HEAD` 当参数传入」 |
| 4 | ⚠️ **管道会吞掉 `&&` 链里的失败码** | 本次 `git checkout dev-1.0` 因「工作区有未提交改动」**失败**，但它后面接了 `2>&1 \| tail -2` ⇒ 整条管道退出码取 `tail` 的 **0**，链条继续往下跑（后续 `merge` 变成「把分支 merge 到自己」的 no-op，无副作用，但过程判断已失真） | 核验/流程链里**关键步骤一律不接管道**，改用 `set -e` + 逐条断言；本次重做时已如此 |

---

## 7. 执行记录（2026-10-16 下午 · 全部实测）

> ⏹ §0 / §3 / §5 是**执行前**的快照；**执行后的权威状态以本节为准**。

### 7.1 通道 ①：FlagCX（fork 子库）

| 项 | 值 |
|---|---|
| 操作 | 本地 `dev-1.0` 从 `origin/dev-1.0` 新建 → `merge --ff-only origin/dev-1.0` → **`merge --no-ff kistich/ascend-flagcx-adapt`** → `push origin dev-1.0` |
| 结果 | **`dev-1.0`：`4e0e0cb` → `62fe615`**（merge 提交双亲 = `4e0e0cb` + `b576fa6a`） |
| 三项核验 | ① `ls-remote` tip == 本地 `62fe615` ✅ ② 重新 fetch 后 `git diff HEAD FETCH_HEAD` **空**（逐字一致）✅ ③ **远端对象库直读**：`makefiles/ascend.mk` 含 `FLAGCX_ASCEND_DAG` **6** 处 · `plugin/torch/flagcx/src/backend_flagcx.cpp` 含 `USE_NVIDIA_ADAPTOR` **24** 处 · `git rev-list FETCH_HEAD..b576fa6a` = **0**（17 提交全在祖先链）✅ |
| ⚠️ 操作注意（已处理） | 合并后**必须把 FlagCX 工作树切回 `kistich/ascend-flagcx-adapt`**：`910C/env/train/env.baseline.yaml` 的 `mounted_repos.FlagCX.commit` 校验的是**工作树 HEAD**，而 `62fe615` 与 `b576fa6a` **树相同、commit 不同** ⇒ 留在 `dev-1.0` 上会让 `check` 判 FAIL |

### 7.2 通道 ② / ③：本仓 `runtime-team`

| 步骤 | 提交 | 内容 |
|---|---|---|
| 清理个人分支 | **`f014442`** | 决策 ①③ 落地（草稿移除 + `image.id` 回发布 id）+ 撤回 09-22 误删的 5 个他方向目录（**414 文件 / +73832 行**）+ 文档口径同步 |
| 把 `dev-1.0` 合进个人分支 | **`ad21b94`** | 冲突 **24 项全部落在 `dev/device-context`** ⇒ 取 ours（判据见该提交正文：dev-1.0 侧对该目录仅 2 个提交且作者均为我们自己；37 个双方都改的文件中 35 个「对方 blob = 我们历史快照」） |
| case 包（通道 ③） | **`d1c768d`** | `dev/issues/flagcx-c10d-collectives`：口径更正 + 7 个新文件（`ddp_native_test.py` + 6 份证据） |
| 禁用词核对 | **`5aa44c8`** | 推送前发现本轮会**新引入**「异构」5 文件（`dev-1.0` 侧原为 **0**）⇒ 4 处表述改「跨厂商（cross-vendor）」 |
| 个人分支 → `dev-1.0` | **`2e077fb`** | **`dev-1.0`：`74197bf` → `2e077fb`**（merge 提交双亲 = `74197bf` + `5aa44c8`）；树与个人分支**逐字一致** |

**推送前本地核对**：合并树 == 个人分支树（`git diff` 逐字一致）· `dev/device-context/{MLU590,PPU,docs}` **562** 文件 · 他方向目录齐全（communication 145 · framework-adapter 139 · memory 84 · performance 2 · rag_ljy 43 · images 140 · lib 3）· 草稿 **0** · env 基线 `image.id` = 发布 id。

**推送后远端核对**：① `ls-remote` tip == 本地 `2e077fb` ✅ ② 重新 fetch 后 `git diff HEAD FETCH_HEAD` **空** ✅ ③ 远端树直读：本方向 562 文件 · `dev/memory` 84 · 草稿 0 · `summary/` **3** 文件 · case 包 README 已含 `62fe615` 记录 ✅

> ⚠️ **本轮第二次踩「陈旧 remote-tracking ref」**：`git fetch origin dev-1.0`（不给目标 refspec）**只写 `FETCH_HEAD`**，`refs/remotes/origin/dev-1.0` 仍停在 `74197bf` ⇒ 用它查 `summary/` 得到「**0 文件**」的假象；改用 `git fetch origin dev-1.0:refs/remotes/origin/dev-1.0` 后为 **3 文件**（真实）。⇒ 核对远端**一律用 `FETCH_HEAD` / `ls-remote`**，或先做带 refspec 的 fetch。

### 7.3 收尾状态与下一步

| # | 事项 | 状态 |
|---|---|---|
| 1 | FlagCX `dev-1.0` | ✅ **`62fe615`**（含修复）⇒ 各方向可**覆盖安装**或换镜像 |
| 2 | 本仓 `dev-1.0` | ✅ 已接收本方向全部内容（**首次合并 `2e077fb`**；本 §7 记录随后以另一次无冲突 merge 带入）⇒ 四实例内容 + 910C 新基座适配 + case 包全部入库 |
| 3 | 本方向个人分支 | ✅ `kistich/device-context`（工作区干净、与远端一致） |
| 4 | **通道 ④ 镜像侧晋升** | ⏳ **待镜像维护者**：`dev/images/ascend-train-comm/v4`（或 v3 补丁版）的 `ARG FlagCX_COMMIT` pin 到 **`b576fa6a`**；发布后在 **baked 镜像上直接跑** `flagcx_sync_test.py --iters 20` 应为 PASS（不再依赖挂载覆盖） |
| 5 | 通知各方向 | ⏳ 待发：「FlagCX 已更新到 `62fe615`」—— 10-14 口径是「各自 `fix`/`commit` 后**覆盖安装**最新 FlagCX」 |
| 6 | 本方向运行环境 | ⏳ 镜像换新后**删除** `910C/env/train` 的 `mounted_repos.FlagCX` 声明（挂载只是过渡态，见 ENV-SPEC §2.3 晋升） |

**2026-10-16 收尾时的实测快照**（带日期 · 是那一刻的读数，**不是**"当前 tip"的永久声明）：FlagCX `dev-1.0` = **`62fe615`** · 本仓 `dev-1.0` = **`6e64bfe`** · 本方向个人分支 = **`2669f0c`**。

> 核对方式（不看工作树）：`git ls-remote origin refs/heads/<分支>` 与本地 `git rev-parse` **逐字比对**。
> ⚠️ **不要**直接拿 `refs/remotes/origin/*` 判：`git fetch <remote> <branch>`（不带目标 refspec）**只写 `FETCH_HEAD`**，
> 该远端跟踪 ref 会停在旧值 —— 本轮已因此误判一次（`summary/` 显示 0 文件，真值 3）。

