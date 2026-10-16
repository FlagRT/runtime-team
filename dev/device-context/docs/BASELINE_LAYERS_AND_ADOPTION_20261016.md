# 基座分层关系与执行标准（device-context 方向，2026-10-16）

> 本文回答四件事：① 基座与「第 N 层」到底怎么分（会议语汇与规范语汇的对应关系）；
> ② 现行机制的**执行标准**是什么、哪些写法已退役；③ 哪些动作属本方向、哪些不属；
> ④ 本方向已完成的 910C 产物按新规范落到哪一处、做完了没有。
>
> 规范依据（本文不重述其内容，只引用其结论并给出处）：[`dev/ENV-SPEC.md`](../../ENV-SPEC.md) ·
> [`dev/images/CASE-TEMPLATE.md`](../../images/CASE-TEMPLATE.md) ·
> [`dev/images/README.md`](../../images/README.md) ·
> [`docs/运行时基座-使用与维护指南.md`](../../../docs/运行时基座-使用与维护指南.md) ·
> [`docs/基座发版说明-20261014.md`](../../../docs/基座发版说明-20261014.md) ·
> [`dev/images/image_list.md`](../../images/image_list.md) ·
> [`dev/_template/`](../../_template/README.md)。

## 0. 一句话

**「层」是职责划分（谁负责什么），「固化 / 挂载」是交付方式（代码怎么进环境）——两者正交，
必须分开说。** 会议里说的"第一层基座、第二层 FlagCX、第三第四层子方向"讲的是**堆叠顺序**，
不是 `ENV-SPEC` 的 `L1–L4`；把两套编号混用会直接得出错误的动作（例如以为"第二层的 FlagCX"
要在基座里发版，而规范给的现行动作是**挂载覆盖**）。

## 1. 先分清两套编号

### 1.1 规范语汇：`L1–L4` 是**职责层级**（[`ENV-SPEC §2`](../../ENV-SPEC.md)）

基线文件里每项资产都带 `layer` 标注。以 910C 训练腿镜像
（`flagrt/ascend-operator-runtime-comm:2.0.0-…g4e0e0cb-arm64`）为例，逐层落位：

| 层 | 含义 | 910C 训练腿上的实例（取自该镜像的 `image.baseline.yaml`） |
|---|---|---|
| **L1 芯片底座** | 厂商基座自带、本组改不了的那部分：CANN / Python / torch / torch_npu 等**没有仓库身份、拿不到 commit** 的事实 | `base_image`：`harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2` @ `sha256:9ee5d835…`，`provides` 记 `cann 9.0.0 / python 3.11.15 / torch 2.10.0+cpu / torch_npu 2.10.0` |
| **L2 算子编译** | 提供算子/编译器能力的组件 | `FlagTree`（triton）`15ec1a6c…` · `FlagGems` `f7ae8e6b…` |
| **L3 Runtime 组件** | 运行时与通信/框架组件 | `vllm` `bc150f50…` · `Torch-FL` `162582d6…` · `vllm-plugin-FL` `8b059122…` · **`FlagCX` `4e0e0cbc…`** |
| **L4 挂载装配** | 运行环境目录里的 compose + 挂载声明（不进镜像） | `dev/device-context/910C/env/train/docker-compose.yml` 与其 `x-flagrt` 声明 |

⚠️ 两个易错点（都是规范里点名的）：

1. **`layer` 与「从哪继承」正交**：基座里自带的组件该是哪层就是哪层 —— 昆仑芯基座自带的
   `FlagGems` 是 L2、`FlagCX` 是 L3，不因来自基座就变成 L1；`base_image.provides` 里**只放**
   没有仓库身份、拿不到 commit 的那部分 L1 事实。
2. **同一层的资产既可固化也可挂载**（详见 §2）。

### 1.2 会议语汇：第 N 层 = **堆叠顺序 / 依赖关系**

| 会议里的说法 | 实际对应 | 按规范该走哪个动作 |
|---|---|---|
| "现在搭好基座，就是让大家往上堆的" | 基座 = **镜像**（L1–L3 随 image id 一起冻结） | 基座由镜像维护者按版本发布；使用方**不直接改镜像** |
| "flagCX 是基础库，会被其他功能依赖的，算是第二层" | FlagCX 在规范里是 **L3 组件**；"第二层"指的是它**依赖基座、被上层依赖** | 需要它变新 ⇒ **挂载覆盖**（`ENV-SPEC §4.2`），不是等基座发版 |
| "其他 runtime 子方向甚至可能建自己的 package，这是第三层第四层" | 子方向自己的代码（本方向 = `dev/device-context/prototype/`） | 放在**运行环境目录**里，靠仓根挂载可见；只有需要 pip 安装并生效时才登记进 `mounted_repos` |
| "不需要正式发版，只需要 fix、commit、然后大家把最新的 flagCX 覆盖安装一下" | = **下放（固化 → 挂载）** | 见 §2.3 |
| "基座不会这样敏捷迭代改，会变更的库自己在 docker-compose 起容器时 pip 覆盖安装、volume 挂载" | = `ENV-SPEC §2.2`「同一 repo 既固化又挂载」+ §5 `activation: pip-editable` | 见 §2.3 |

**所以本方向的正确姿势是**：基座（镜像）照用不动，`FlagCX` 用**挂载覆盖**叠上去；
子方向自己的原型代码按路径调用（仓根挂载已使其可见），**不登记**进 `mounted_repos`
（`ENV-SPEC §2.1`：仓根整体挂到 `/workspace` 只使代码可见、不使其生效，不自动收录）。

## 2. 执行标准：现行机制的硬性要求

### 2.1 版本事实只有一个来源，且由工具生成

| 对象 | 基线文件 | 回答什么 | 生成/核对 |
|---|---|---|---|
| 镜像 | `dev/images/<name>/v<N>/image.baseline.yaml` | 镜像固化了哪些资产、各自什么版本 | `baseline.py export/check --image <image-dir>` |
| 运行实例 | `<env-dir>/env.baseline.yaml` | 实例用哪个镜像、另外以**可编辑方式**挂了哪些 repo、各自在哪个 commit | `baseline.py export/check <env-dir>` |

**两类基线文件都不手工编辑**；`lock.yaml` 只记构建过程与已知问题，**不再重复版本事实**。
启动统一 `dev/lib/up.sh <env-dir> [--strict]`（先 `check` 再 `docker compose up -d`）。

### 2.2 固化还是挂载：**看代码从哪里来**，与产出方、变更频率无关

| | 固化（baked） | 挂载（mounted） |
|---|---|---|
| 定义 | 代码在**构建期**进入镜像文件系统 | 代码来自宿主目录，运行时经 volume 挂载并**以可编辑方式生效** |
| 版本归属 | 镜像（image id 唯一确定） | 运行实例（启动时宿主 repo 的 HEAD） |
| 记在哪 | `image.baseline.yaml` | `env.baseline.yaml` 的 `mounted_repos` |
| 变更动作 | 改配方 → 新建 `v<N+1>/` → 重建 → 重新 export | 改宿主 checkout → 重启容器（含 C++ 的组件须重新编译安装） |

### 2.3 只有两个跨越边界的显式动作

| 动作 | 触发条件 | 结果 |
|---|---|---|
| **下放**（固化 → 挂载） | 需要在**现有镜像**上快速迭代该组件 | 在运行环境 compose 声明挂载，形成覆盖关系；镜像不变。`env.baseline.yaml` 记 `overrides_baked: true` / `baked_commit`；镜像内固化版本带补丁时另记 `baked_patches`，`check` 会提示挂载副本不含这些补丁 |
| **晋升**（挂载 → 固化） | 该组件本阶段**验证冻结** | 写进下一版镜像的构建配方，构建后 export 镜像基线；运行环境改用新镜像并**删掉挂载声明** |

⚠️ **不可下放的例外**：从包索引装入的**预编译 wheel**（`baked_packages`，如昆仑芯的 `flagtree`）
没有源码 checkout，挂载源码也不会改变已装好的 wheel ⇒ 把它写进 `mounted_repos` 是**配置错误**，
`export` 直接报错。FlagCX 在本组是**源码固化**（`USE_ASCEND=1 pip install`，有 commit 锚）⇒ **可下放**。

### 2.4 四条硬规则（对 case 包与任何交付物都成立，[`CASE-TEMPLATE §2`](../../images/CASE-TEMPLATE.md)）

1. **单入口**：复现与验证各一条命令即可完成，前置准备纳入脚本或 compose，不写"先手动做若干步"；
2. **外部依赖必须声明**：镜像与挂载 repo 进基线文件（由 `up.sh --strict` 机器核对），
   基线覆盖不到的（模型权重、数据集、**卡数**）写在 README 前置条件表；
3. **自判定**：验证脚本以退出码 + 判定行给结论，统一三态口径
   `overall: PASS`/0 = 缺陷不存在 · `FAIL`/1 = 缺陷复现 · `ERROR`/2 = 无法下判定（**不计入前两者**）；
4. **日志不进仓**：完整日志留在执行机器并在 README 注明机器名与绝对路径，仓内只放
   `assets/evidence/` 下的截选，每份截选在文件头注明来源日志与截取范围。

### 2.5 `--strict` 的语义边界

`up.sh --strict` 只约束**启动时刻**：不一致即拒绝启动。启动之后宿主 checkout 的改动会
直接进入运行中的实例。⇒ 出结论的运行除了 `--strict`，还须用**已推送的干净 commit**
（`baseline.py export <env-dir> --require-clean`：工作区不干净或 commit 未推送即拒绝导出）。

### 2.6 已退役的写法（别用）

`compose.base.yml` + 各子方向 `docker-compose.yml` 用 `-f ../compose.base.yml` 叠加，
于 **2026-09-23 退役**，替代写法是 compose `extends` 复用镜像 demo。
[`使用与维护指南 §6`](../../../docs/运行时基座-使用与维护指南.md) 的名单里**点着本方向**
（`dev/communication/`、**`dev/device-context/`**、`dev/memory/`、`dev/rag_ljy/`、`dev/batch_yxy/`），
并写明"迁移属各方向自身工作，总组不代做"。本轮就是在做这件事。

一并退役的还有：`pins.<tag>.yaml` + `verify_env.py/.sh`（→ 基线文件）·
`dev/lib/.env_provenance.json`（→ `env.baseline.yaml` + `.last-check.json`）·
`VLLM_FL_FLAGOS_BLACKLIST=cos,sin` 作为 910C 推理规避配置（→ 推理腿改用华为昇腾官方
`vllm-ascend` + 起服务前 `unset VLLM_PLUGINS`）。

## 3. 谁做什么（能力边界）

| 动作 | 本方向（device-context） | 镜像维护者 / 总组 |
|---|---|---|
| 建/维护**运行环境目录**（compose + `env.baseline.yaml` + `.env.example`） | ✅ 本方 | — |
| 声明并维护**挂载覆盖**（FlagCX 等） | ✅ 本方 | — |
| 子方向试点代码（`dev/device-context/prototype/`）按路径调用 | ✅ 本方 | — |
| 出结论性验证（`--strict` + 干净已推送 commit + 证据截选） | ✅ 本方 | — |
| 缺陷 case 包（`dev/issues/`）的内容与判据 | ✅ **本方提供内容与证据**（跨方向缺陷由总组统一发布，目录归总组） | 发布与索引 |
| **发布新镜像版本**（改配方、建 `v<N+1>/`、export 镜像基线、更新 `image_list.md`） | ❌ 本方可提需求与证据，**不擅改** | ✅ |
| 改 `dev/stack.lock.*.yaml`（锁定声明） | ❌ 不动 | ✅ 总组 |
| 改宿主驱动 / 系统环境 | ❌ **红线** | 需专项裁定 |
| 动他人容器 | ❌ **红线**（含 stopped，须先问） | — |

## 4. 910C 已完成产物 → 按新规范的落位

| 已完成的产物 | 新规范里的位置 | 状态 |
|---|---|---|
| `prototype/`（原型：统一 runtime 层 + 四实例后端） | **L4 挂载装配**下的子方向代码，按路径调用；**不**进 `mounted_repos` | ✅ 已到位（仓根挂载使其可见） |
| 910C conformance / 多流 / 训推两条腿 / 职责审计的脚本与判据 | 各自 run-env 目录下的调用入口 + `<chip>/probes/` 证据 | ✅ 已到位（判据入口见 [`../910C/README.md`](../910C/README.md)、[`../prototype/docs/`](../prototype/docs/)） |
| **FlagCX c10d 完成语义 + 异构修复**（镜像固化的 `4e0e0cb` 不含） | **下放**：`910C/env/train/docker-compose.yml` 的 `x-flagrt.mounted_repos.FlagCX`（`pip-editable`） | ✅ 本轮完成（源码分支已推送、真机 A/B 已验证） |
| 缺陷 case（两类缺陷 + 修复验收判据 + 原生 DDP 脚本） | `dev/issues/flagcx-c10d-collectives/`（三段式 + 四条硬规则 + 三态口径） | ✅ 本轮更新（内容与证据由本方向提供，发布归总组） |
| 910C 训练腿 / 推理腿的**运行环境目录** | `910C/env/train/` 与 `910C/env/infer/` | ✅ 完整就位：compose + `.env.example` + README + **两份 `env.baseline.yaml`（`export --require-clean` 生成，rc=0）**；`up.sh` 起容器 rc=0、容器内两项验收 PASS（见 §6.1） |
| 原 `dev/device-context/README.md` 里的 `-f ../compose.base.yml` 写法 | 退役写法 | ✅ 本轮删除（改指运行环境目录） |

## 5. 机制前置：先缺后补（**已解决**，2026-10-16 当日）

`baseline.py` 与 `up.sh` 需要两项宿主能力，npu1-27 **原本都不满足**；当日按**用户级**方式补齐
（`sudo -n true` 实测报 `sudo: a password is required`，非交互 sudo 不可用，故未走 sudo）：

| 依赖 | 原本状态（实测） | 补齐方式 | 版本 |
|---|---|---|---|
| `docker compose` v2 插件 | `docker 20.10.8`；`/usr/libexec/docker/cli-plugins/` 只有 `docker-app`/`docker-buildx`；`~/.docker/cli-plugins/` 空 | 官方 Release 静态二进制 → `~/.docker/cli-plugins/docker-compose`（`install -m 0755`） | **v2.29.7**（与总组在 npu1-11 上验证的 Compose 同版本） |
| `pyyaml` | `/usr/bin/python3`(3.9.9) 报 `No module named 'yaml'`；另一解释器 `/usr/local/python3.7.5` 缺 `libpython3.7m.so.1.0` 起不来 | `pip install --user` 一份 aarch64 manylinux wheel（cp39） | 6.0.2 |

下载件已对官方 `.sha256` 逐字校验：`6e9fbd5daa20dca5d7d89145081ae8155d68ef2928b497d9f85b54fe0f9dbb2c`。
复现步骤写在 [`../910C/env/README.md`](../910C/env/README.md) §2。

> ⚠️ **npu1-11 拷不过来**：它是 `10.120.72.11`（在 27 的 `/etc/hosts` 里、ping 通、22 开），
> 但 `hliu553@10.120.72.11` 与本地三把私钥（`id_kistich`/`id_rsa`/`id_ed25519`）一律
> `Permission denied (publickey,password)`；从 27 直连同样被拒 ⇒ 走"**同版本官方件**"补齐
> （等价做法，且版本号与 11 上验证过的一致）。

**结果**：`export` 双腿 **rc=0**、`up.sh <env>` **rc=0** 起容器、容器内两项验收 **`overall: PASS`**
（见 §6 与 [`../910C/probes/baseline_mechanism_20261016_out/SUMMARY.md`](../910C/probes/baseline_mechanism_20261016_out/SUMMARY.md)）。
**唯一未通过的是 `--strict`** —— 原因不在本方向，见 §5.1。

### 5.1 ⚠️ `up.sh --strict` 在非制造机上**必然拒绝启动**（跨机镜像 id）

```
[FAIL] image: id 不一致: 基线 sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15
                         本机 sha256:7028028bb62ce06322f2b53379891c062caf8ac6b3f2b1a469be6dafe9af7913
```

`env.baseline.yaml` 的 `image.id` 抄录自镜像基线，而镜像基线记的是**制造机 npu1-11** 的 id。
[`image_list.md`](../../images/image_list.md) 早已登记"跨机 image id 不同属预期（内容同源）"，
但**它的后果此前没写下来**：`check` 在非制造机上恒 `FAIL` ⇒ `--strict` 恒拒绝启动。
本轮处置：本机的结论性运行改用 `up.sh`（spec 明文"默认只报告、不阻止启动"）并**在证据中注明未满足 `--strict`**；
其余各项照常核对（`[PASS] FlagCX: commit=b576fa6a…` 逐字通过）。正解属镜像/机制侧，见 §7.2 第 10 项。

## 6. 本轮实测结果

### 6.1 机制路径（`baseline.py` + `up.sh`，2026-10-16 12:00 npu1-27）

| 步骤 | 结果 |
|---|---|
| `export <env> --require-clean`（训练腿 / 推理腿） | **rc=0**，生成两份 `env.baseline.yaml` |
| `show` | 挂载覆盖关系清楚：`FlagCX b576fa6ad977 pip-editable 否 是 <- 4e0e0cbcbf72 /workspace/FlagCX` |
| `check` | **rc=1**，唯一 `FAIL` = 跨机镜像 id（§5.1）；`[PASS] FlagCX: commit=b576fa6ad977` |
| `up.sh <env> --strict` | rc=1，**因上述 `FAIL` 拒绝启动**（预期，见 §5.1） |
| `up.sh <env>`（不加 `--strict`） | **rc=0**，容器 `flagos-dc-train-910c-dev` Created→Started |
| `up.sh` 起的容器内两项验收（单入口） | **`overall: PASS` / rc=0**（八组 + 原生 DDP 四项全 PASS） |
| `docker compose down` | **rc=0**，容器 Removed，我方运行中容器 0、残留进程 0 |

### 6.2 挂载覆盖的 A/B 对照（同一容器、同一命令，只换加载哪份 FlagCX）

同一容器、同一条命令（`run_910c_p9_acceptance.sh`，两项一起、三态自判定），只换加载哪份 FlagCX：

| 臂 | 容器内实际加载的 FlagCX | ① `flagcx_sync_test.py --iters 20` | ② `ddp_native_test.py --steps 10` | 合并 |
|---|---|---|---|---|
| **B = 本环境**（`pip-editable` 挂载覆盖 `b576fa6a`） | `/workspace/FlagCX/plugin/torch/flagcx` | **PASS**（八组全 PASS） | **PASS**（四项全 PASS） | **PASS / rc=0** |
| **A = 镜像固化版**（`4e0e0cbc`，`PYTHONPATH` 影子） | `/opt/baked-flagcx/flagcx` | FAIL（图案与 issue 一致） | FAIL | FAIL / rc=1 |

- 挂载生效方式成立：容器启动时 `USE_ASCEND=1 python3 -m pip install --no-deps --no-build-isolation
  -e /workspace/FlagCX` → `Successfully installed flagcx-0.13.0`，`import flagcx` 解析到
  `/workspace/FlagCX/plugin/torch/flagcx`（**宿主 checkout**，不是镜像内那份）。
- 装置自证：两臂的 `_C…so` / `libflagcx.so` sha256 前 16 位不同（B `098f3227…`/`5af25cc9…`，
  A `a6a4a8e0…`/`8196a720…`），且每次运行打印实际加载路径。
- 缺陷归属的另一条硬事实：镜像声明的主干 `4e0e0cbc` **就是**本方向分支的 `merge-base`
  （`git merge-base origin/dev-1.0 HEAD` = `4e0e0cb…`），`dev-1.0` 至今未收到本方向 17 个提交中的
  任何一个 ⇒ **以该镜像为基础，缺陷必然稳定复现**；issue 里"第 4 次判定与前 3 次逐行一致"
  不是缺陷稳定，而是**镜像里的 FlagCX 从未变过**。

## 7. 待办（本轮之后）

见 §7.1–§7.4，按归属分。**带 ⚠️ 的是需要用户裁定的**。

### 7.1 本方向

| # | 项 | 出口标准 |
|---|---|---|
| 1 | ~~`env.baseline.yaml` 尚未生成~~（**已完成 2026-10-16**） | 已在 npu1-27 上 `export --require-clean` 双腿 rc=0（前置补齐后）；两份基线已入仓 |
| 2 | ~~npu1-27 的机制前置（compose v2 + pyyaml）~~（**已完成 2026-10-16**） | 用户级补齐：compose **v2.29.7**（官方 sha256 校验通过）+ pyyaml 6.0.2；未用 sudo。⚠️ 属**本机状态、不入仓**，换机器/账号要按 [`../910C/env/README.md`](../910C/env/README.md) §2 重做 |
| 3 | P800 / MLU590 / PPU 三芯片的 run-env 目录未建 | 各一份 `env/`，`docker compose config` 通过 + `export/check` rc=0 |
| 4 | 910C 推理腿 env（`910C/env/infer/`）尚未经真机起容器验证 | `up.sh` 起容器（本机 `--strict` 见 §5.1）+ 容器内 `serve_standard.sh` 得 `SERVE_STANDARD_PASS` |
| 5 | 把「逐 die 选卡 → 写 `.env`」做成一条命令 | 一个脚本产出 `DC_DIE_A/DC_DIE_B`（现在要人工从探测输出里抄） |
| 6 | 本方向原型代码是否需要自己的 package/固化形式 | 先判：镜像内是否需要它（当前不需要，按路径调用即可）；若某阶段需要，走「晋升」写进镜像配方 |

### 7.2 镜像侧（需镜像维护者/总组裁定）

| # | 项 | 说明 |
|---|---|---|
| 7 | **FlagCX 修复的晋升** | 本方向已在 `FlagRT/FlagCX` 的 `kistich/ascend-flagcx-adapt` 上给出干净、已推送的修复（`b576fa6a`，相对镜像 pin 的 `4e0e0cb` 领先 17 个提交）与两项验收证据。**镜像接收修复**（新版本 `v4/` 或给 v3 打补丁版）后才可动 `repro_status`。在收到之前，**本方向不得单方面声称 A-2 已解除** |
| 8 | FlagCX 两处**可编译性**缺陷是否随修复上游 | ① `plugin/torch/flagcx/src/backend_flagcx.cpp` 无条件 `#include <ATen/cuda/CUDAContext.h>`（由本方向 NVIDIA 线提交引入，纯昇腾栈编译失败）；② `makefiles/ascend.mk` 无条件链接 DAG 专用 `-lcust_opapi`（镜像无 `custom_opp` ⇒ 链接失败）。两处已在本方向分支修好，建议随修复一起上游 |
| 10 | ⭐ **`image.id` 的跨机口径**（本轮新暴露） | `env.baseline.yaml` 的 `image.id` 抄录自镜像基线（= 制造机 npu1-11 的 id）⇒ **除制造机外 `check` 恒 FAIL、`up.sh --strict` 恒拒绝启动**（`image_list.md` 只登记了「跨机 id 不同」，未写这条后果）。可选处置：① 基线允许记多个等价 id；② 分发统一走 `docker load` 离线包（content-derived id，跨机一致）；③ 明确「`--strict` 只在制造机可满足」并给出别机的等价判据 | **总组 / 镜像侧**（本方不擅改基线或 `baseline.py`） |
| 9 | `dev/images/TODO.md` 第 2 项（原生 DDP 验证脚本） | 本方向已补出（`ddp_native_test.py`，双卡自判定，已真机通过），可由总组据此关闭该项 |

### 7.3 总组文档侧（待修复口径）

| # | 项 | 说明 |
|---|---|---|
| 11 | `dev/issues/flagcx-c10d-collectives/README.md` 的现状段 | 现写「未修复，2026-10-11 第 4 次判定与前 3 次逐行一致」，未指出"镜像里的 FlagCX 从未变过"这一真实原因，也未记修复所在。本方向已按 CASE-TEMPLATE 给出更新稿与证据 |
| 12 | `使用与维护指南 §6` 的退役名单 | 本方向迁完后应从"仍在沿用旧写法"名单里移出 |
| 13 | `基座发版说明` 里"训练腿 demo 仍装配三项（shmem 占位、PYTHONPATH 前置、`VLLM_FL_FLAGOS_BLACKLIST`）"的表述 | 与 `ascend-train-comm/v3/demo/docker-compose.yml` 的实际内容不一致（该 demo 现只设 `HCCL_WHITELIST_DISABLE`，注释已写明三项不再需要）。属总组文档内部不一致，本方向不擅改，仅登记 |
| 14 | `image_list.md` 的 A-2 / 已知限制表述 | 随第 7 项一起更新 |

### 7.4 外部依赖（不受本方向控制）

| # | 项 | 状态 |
|---|---|---|
| 15 | PPU 平台侧弱引用缺陷上报厂商 | 最小复现 + 原生现场已备（本层侧已规避），需渠道 |
| 16 | MLU590 CNCL RDMA 归因实测 | 「范围外」有硬依据，「归因」无证据 ⇒ 待实测；MLU590 与 PPU 当前本侧网络不可达 |
| 17 | npu1-27 docker 存储层校验和损坏（`docker save` 不可用） | 已报修（`dev/images/TODO.md` 第 7 项），运行不受影响 |
