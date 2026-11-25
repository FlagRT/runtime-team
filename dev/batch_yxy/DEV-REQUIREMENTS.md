# 开发要求：机器、镜像与注意事项

> **事实基准**：runtime-team `dev-1.0@aaee365`（2026-11-25 核对）。规范原文在该分支的
> `dev/ENV-SPEC.md`、`dev/images/README.md`、`dev/images/USAGE_MATRIX.md`、
> `dev/images/RELEASE_INDEX.yaml`、`dev/images/CASE-TEMPLATE.md`、`dev/issues/README.md`、
> `dev/stack.lock.kunlunxin.v1.yaml`。**本分支不含这些文件**，引用时请带分支与提交，不要写相对链接。
>
> **范围**：batch_yxy（FlagScheduler）在各平台真机上开发与验收时要遵守的要求。
> 各平台的数据、排查记录与进度仍在 [`platforms/`](platforms/)，本文只写"要求"；
> 与 [`AGENTS-DEV.md`](AGENTS-DEV.md) 重叠的纪律以那里为原文，这里只做汇总与补充。

## 1. 什么机器、用什么镜像

现行环境以发布声明为准（入口 `dev/images/README.md`，两个锚点是使用矩阵与发布索引），不能根据目录是否存在或编号大小推断。

| 平台 | 机器 | 镜像 / 运行环境 | 公共线登记状态 | 说明 |
| --- | --- | --- | --- | --- |
| 昆仑芯 P800 · 推理 | P800-A；VM-20-169（见 1.1） | `…:tele4`；插件修复如何落地未定（见 1.1） | 镜像已登记；**现行环境为 null** | 本方向现行基座；运行环境尚未声明 |
| 昆仑芯 P800 · 旧栈 | P800-A | `chenyunlong/vllm-kunlun-runtime:0.11.0` | 未登记（非锁定镜像） | 只供历史数据溯源；新开发不用，结果不作验收依据 |
| 昇腾 910C | 910C 机 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | `ascend-infer-vllm/v1`；现行环境 `env/v2` | 本方向用自己的单卡开发容器（历史口径） |
| NVIDIA A6000 | GPU 机 | 无镜像：宿主 Conda `vllm` / `vllm_add` | 未登记 | 历史主开发机；与容器红线的不一致见 1.3 |

### 1.1 昆仑芯 P800（推理腿）

**机器**

- **P800-A**：内网 10.20.20.4（`VM-20-4-ubuntu`），SSH `xianyiyuan@81.70.153.29 -p 26008`；配额卡 **宿主 2、3、7**，空闲时可用。驱动与平台事故见 [`platforms/kunlunxin-p800/README.md`](platforms/kunlunxin-p800/README.md)，当前能否实测以同目录 `progress.md` 为准。
- **VM-20-169**：内网 10.20.20.169（`VM-20-169-ubuntu`）。2026-11-25 巡检：2 × AMD EPYC 9K84（192 核 / 384 线程）、1.5 TiB 内存、8 × P800 OAM（每卡 96 GiB）、宿主驱动 5.0.21.47，XPU 0–3 在 NUMA 0、4–7 在 NUMA 1，8 卡状态读数正常。模型在 `/data2/flagperf-kzhang519/models/Qwen3-Embedding-0.6B-modelscope`（可读、完整）。**不具备**：账号不在 `docker` 组，无法启动容器。**未确认**：机器代号、配额卡、tele4 镜像是否已在本机加载。

**镜像**（公共线登记名 `kunlun-infer-vllm/v1`；登记式基线——上游发布镜像，无仓内配方，不可重建）

- ref：`harbor.baai.ac.cn/flagrelease-public/kunlunxin001-gems5.0.0-treenone-triton3.0.0-cx0.13.0-plugin0.2.0-vllm0.20.2-cp310-pt29-x64-v5.0.21.43:tele4`
- image id 与 digest（同值）：`sha256:e180a0dd4267ebdfbce4ea5434710870a6ea7e96118457538cc02442907d54b1`
- 镜像内固化：vllm 0.20.2+empty、torch 2.9.0+cu129、vllm-plugin-FL 0.2.0+g38e7dbc（editable，`/workspace/vllm-plugin-FL`）、xmlir 1.0.0.1、triton 3.5.0、flag_gems 5.0.0、flagcx 0.13.0、transformers 4.57.6
- tag 只是可读名称。核对两个镜像是否同一制品，比较 image id 或 digest，不比较 tag。
- 训练腿镜像 `kunlun-operator-runtime` 不是本方向用的；公共线上两个昆仑芯 GEMM 缺陷 case 立在它上面。

**插件修复如何落地（未定；以下两种做法都未在本方向的机器上实测）**

镜像自带的 vllm-plugin-FL `0.2.0+g38e7dbc` 有 pooling-dispatch 缺陷，任何 embedding / pooling 任务必崩——公共缺陷 case `vllm-plugin-fl-pooling-dispatch` 对镜像 `kunlun-infer-vllm/v1` 的表态是"仍命中"。修复 `039fb46`（含修复本体 `0c833a4`）合入的是 `FlagRT/vllm-plugin-FL` 的 `dev-1.0`，现行发版镜像尚未重打包。

- **平台 README 与 `BOOTSTRAP.md` 的现行配方**：把 `FlagRT/vllm-plugin-FL@039fb46` 挂到 `/workspace/vllm-plugin-FL`，覆盖镜像内的副本。
- **这条配方很可能不成立**（2026-11-25 静态核对）：该检出的 `VENDOR_DEVICE_MAP` 与 `vllm_fl/dispatch/backends/vendor/` 里都没有 kunlunxin，`dev-1.0` 的全部提交也不涉及昆仑芯。按公共环境 `env/v1` 的说明，镜像自带的副本含 kunlunxin；被不含 kunlunxin 的副本遮蔽后，import 链会断在 `Vendor kunlunxin not found`。
- **公共线两个缺陷 case 的做法**：保留镜像自带的插件，在容器内对 `/workspace/vllm-plugin-FL/vllm_fl/worker/worker.py` 的那一行判断就地打补丁。补丁随容器删除而消失，每次重打；它是验证手段，不是环境的永久状态。
- 规范口径：缺陷"已修"必须写明形态——挂载覆盖，还是镜像内固化；两种形态互不折算，各自只对自己那一版运行环境成立。
- 镜像或插件升版后，核对新打包的 vllm-plugin-FL 是否含该修复。

### 1.2 昇腾 910C

- 镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`，与公共线登记的 `ascend-infer-vllm/v1` 是同一个 tag（digest `sha256:5cf8a2b6db8b06eb1bc7fc7d191d667aebf2b197351bdba13f776918c11ec7a7`）。公共线现行环境是 `env/v2`，用 `up.sh --strict` 启动。
- 本方向的历史口径：单卡开发容器 `flagos-batch-yxy-dev-910c`，由 `docker-compose.yml` 叠加 `../compose.base.yml` 启动，或用等价的 `scripts/start_container.sh`（`BATCH_DAVINCI_ID` 选宿主卡，容器内恒为 0 号设备）。
- `compose.base.yml` 在 `ENV-SPEC` §0 已列为 LEGACY，不在现行机制内。出结论的运行应改用公共线现行环境，或在报告里写明用的是本方向的历史容器。

### 1.3 NVIDIA A6000

- 无镜像：宿主 Conda 环境 `vllm`（0.21.0）与 `vllm_add`（Python 3.12.10 / torch 2.13.0+cu130）；4 × RTX A6000，本方向用 GPU3。
- 根 README 的红线是"所有安装与实验都在容器内进行"。宿主 Conda 是拆仓前的历史口径，沿用前需要确认以哪个为准。

### 1.4 旧栈（仅溯源）

`chenyunlong/vllm-kunlun-runtime:0.11.0`（vllm 0.11.0 + vllm-kunlun 0.11.0，torch 2.5.1+cu118）。`platforms/kunlunxin-p800/` 里 2026-11-10 至 11-17 的数据与两个厂商缺陷都出自它。它不是锁定镜像；按 `dev/stack.lock.kunlunxin.v1.yaml` 的规则（"结论性验证只在锁定镜像内做；日常调试容器结果不作为验收依据"），这些数据只能作为探索性数据。

## 2. 启动前后要核对什么

**统一入口**是 `dev/lib/up.sh <env-dir> --strict`：核对基线 → `docker compose up -d` → 安装环境层包 → 严格模式装后复核。需要 `dev-1.0` 的检出（含 `dev/lib` 与 `dev/images`）。

| 核对项 | 不一致时 |
| --- | --- |
| 镜像 id（本地镜像与登记值） | FAIL |
| 挂载来源（宿主目录与 `host_path`） | FAIL |
| 挂载仓 HEAD（与登记 commit） | FAIL |
| origin 远端不同、工作区有未提交改动、提交未推送 | WARN（不阻断） |
| 层包版本（仅容器运行时核对） | FAIL |

- 核对不一致时退出码为 1，`--strict` 拒绝启动；用法或基线文件错误时退出码为 2。
- `strict` 不覆盖镜像专用全检、完整模型功能、精度和性能，这些要分别验收。交付时应保证挂载仓干净且已推送。
- `up.sh` 默认以公共仓根为 `WORKSPACE_ROOT`（团队约定各仓收拢在 `runtime-team` 检出目录内）。[`BOOTSTRAP.md`](BOOTSTRAP.md) 的布局是 `~/fgw` 下并列，且只稀疏取出 `dev/batch_yxy`。要用统一入口，需要一个含 `dev/lib` 的 `dev-1.0` 检出，并按 `up.sh` 的实现用 `FLAGRT_WORKSPACE_ROOT` 指向 `~/fgw`（未实测）。

**本方向 P800 的 `docker run` 配方**（见平台 README；配方本身未经实测，见 1.1）没有工具核对，启动前逐项手工核对：

1. `docker image inspect <ref> --format '{{.Id}}'` 与 1.1 的 image id 逐字一致。
2. 容器内 `vllm_fl` 从镜像自带的 `/workspace/vllm-plugin-FL` 导入，没有被别的副本遮蔽，例如：
   `/root/miniconda/envs/python310_torch29_cuda/bin/python3 -c 'import vllm_fl; print(vllm_fl.__file__)'`
3. 若按公共 case 的做法打补丁：补丁后 `worker.py` 的那行判断含 `AsyncModelRunnerOutput`，且只改了这一行。
4. 收尾：按容器名删除自己的容器，`xpu-smi` 确认卡上没有残留进程。

## 3. 设备与共享机器纪律

设备模型按芯片实测，不要跨芯片照抄：

| | 昇腾 910C | 昆仑芯 P800 |
| --- | --- | --- |
| 带卡容器独占 | **是** | 否 |
| 开工前 | `docker ps` 是阻塞条件：有他人带卡容器就停手、约时间 | `docker ps` 只是礼貌；`xpu-smi` 挑 0% 利用率 / 0MiB 显存的卡 |
| 选卡 | `ASCEND_RT_VISIBLE_DEVICES` | 设备授予 `/dev/xpu<N>`，容器内即 0 号卡；若设 `CUDA_VISIBLE_DEVICES`，注意它按授予顺序重编号 |
| 并发 | DrvMng 并发容器上限约 3 | 多卡取同一 NUMA 组的连续空闲卡（0–3 / 4–7） |

- 设备授予要带 `/dev/xpuctrl`（缺它 `torch.cuda.device_count()` 为 0）；不需要 `--privileged`，不需要挂宿主驱动目录。
- **必不设** `XPU_EVENT_KL3_ENABLE`：设置后单卡单进程即挂死，自建环境也不得加回。
- P800-A 的配额卡是 2、3、7；公共线 `env/v1` 与缺陷 case 的默认避让卡是 1、2（分配挂死前科）。两者在 2 号卡上有出入，用 2 号前以 `xpu-smi` 实测为准。其他机器的配额卡未确认前，不要按 2/3/7 推断。
- NVIDIA：本方向用 GPU3，GPU0/1/2 为他人占用，用前 `nvidia-smi` 确认。
- 共同：**绝不**停止、重启或 `exec` 进入他人容器；用完把自己的收干净；不做任何需要 root 的驱动级操作。

## 4. 已知的环境坑（P800 推理腿）

1. 容器 `--network none` 时，`GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1` **必须用 shell export**。脚本里的 `os.environ.setdefault` 对 vllm spawn 的子进程不生效，会看到 `Unable to find address for: eth0` 而不是目标报错。
2. 模型目录带 sentence-transformers 根配置时，vllm 配置加载会崩。用剔除 `config_sentence_transformers.json` / `modules.json` / `configuration.json` 的符号链接视图（做法见 `BOOTSTRAP.md` 第三节），并把权重源目录按宿主同路径挂载，链接才能在容器内解析。
3. **不要把仓根挂到 `/workspace`，也不要用别的 vllm-plugin-FL 副本遮蔽镜像自带的 `/workspace/vllm-plugin-FL`。**镜像的 `/workspace` 自带含 kunlunxin 的源码树，被不含 kunlunxin 的副本遮蔽后，import 链断在 `Vendor kunlunxin not found`。
4. 容器内 `python` 落在 miniconda base（没有 torch / pytest），负载一律用 `/root/miniconda/envs/python310_torch29_cuda/bin/python3`；镜像 ENTRYPOINT 是 `/bin/bash`，命令要给 `-c`。
5. 设备 API 走 `torch.cuda` 命名空间重写。公共缺陷 case `vllm-plugin-fl-pooling-dispatch` 的崩溃正是在 `is_cuda_alike()` 为真的平台上触发的。
6. 现行栈上同一句话重复 `embed()` 的余弦只有 0.92–0.94（公共缺陷 case `vllm-plugin-fl-pooling-nondeterminism`，根因未知）。数值等价的判据不能按 cos ≈ 1 设计；映射正确性与数值一致性分别验收。
7. 旧栈的规避手段（`VLLM_ENABLE_V1_MULTIPROCESSING=0` 进程内引擎、`XPU_VISIBLE_DEVICES` 选卡、单次调用 ≤ 16384 token、关 prefix caching）只对旧栈成立，在现行栈上未验证。
8. P800-A 发生过两起驱动全局卡死，触发源是 bge 系模型加载崩溃。bge 系模型一律先小规模试探。
9. 新机器接入前确认三件事：账号在 `docker` 组、`sudo` 可用、数据盘上有自己的可写目录。VM-20-169 上实测免 root 的 singularity 不可用（非 setuid 安装，内核又限制普通用户创建用户命名空间）。

## 5. 结论与证据的要求

- **分别验收**：运行环境检查、组件验证、完整模型验证、精度、性能各下各的结论，不互相代替。
- **只在锁定镜像内下结论**，日常调试容器里的结果不作为验收依据。
- **结论范围等于证据范围**：没执行的项目、没测的版本如实标注，不虚构。
- **缺陷按坐标表态**：坐标 = 镜像血统 + 运行环境修订。每个坐标取三态之一——仍命中 / 已修（写明形态）/ **未在该版测过**。最后一态必须显式写出，沉默会被读成"不命中"。

当前需要写明的表态：

- 本方向的两个厂商缺陷（旧栈上已复现）：在现行腿上**未在该版测过**。
- 公共线的两个昆仑芯 GEMM 缺陷 case（编译、数值，立在训练镜像上）：推理腿这一坐标**未在该版测过**。
- 公共缺陷 `vllm-plugin-fl-pooling-dispatch`：公共线已在 `FlagRT/vllm-plugin-FL` 的 `dev-1.0` 修复（`039fb46`），并用容器内补丁在 P800 上验证为 PASS；现行发版镜像仍是"仍命中"。本方向尚未在自己机器上复验（`progress.md`「恢复后清单」第 1 步），所用环境也尚未声明；没有实测、没有声明之前，不能表态为"在该版已修"。

## 6. 记录与交付

沿用 `AGENTS-DEV.md`「结果记录纪律」（数据进 `platforms/<平台>/data/`，排查进 `investigation/`，进度回写 `progress.md`）。在此基础上对齐公共规范：

- **完整日志不进仓**：留在执行机器的 `/tmp/<样例名>-<时间戳>/`，文档里注明机器名与绝对路径；仓内只放 `summary.json` 与关键截选，每份截选在文件头注明来源日志与截取范围。
- **脚本自判定**：给出判定行（`overall: PASS/FAIL`）与退出码，执行者不需要人工判读日志。兼作复现与修复验收的脚本用三态：`PASS` / 0 = 缺陷不存在，`FAIL` / 1 = 缺陷复现，`ERROR` / 2 = 无法判定。
- **功能验收做成 case 四件套**：`run.sh`、`env/`、`scripts/`、`REPORT.md`、`evidence/summary.json`。`run.sh` 一条命令完成选卡核对 → `up.sh --strict` → 容器内执行 → 写结果 → 收尾，失败路径也要清理。
- **基线之外的依赖要声明**：模型权重、数据集、所需卡数、目标机器可达性，写在 `run.sh` 头部与报告的前置条件表里。
- **四个常踩的坑**：管道接 `tee` 后要启用 `pipefail` 或立即保存 `PIPESTATUS[0]`；宿主残留的 `build.done` 不能证明本次安装已就绪；`compose down` 需要启动时同样的变量，清理后核对容器与显存；wheel 保留合法的原始文件名。

## 7. 协作与合入

- 个人改动用 `<名字>/<功能>` 分支（本分支符合）；共享开发线只 merge，不 rebase。
- **合入五项门槛**须同时满足：使用的镜像符合现行声明；容器及分支命名符合规范；负责的接口章节完成并获至少一名下游确认；自己负责的 demo 已运行通过；没有引入未经统一管理的新运行环境分叉。对照本方向：镜像已登记，运行环境尚未声明，P800 新基座验收未完成。
- 公共运行环境的变更，要先说明适用芯片、场景、组件变化及影响，并与总组负责人确认。各方向只消费基座、不自建；诉求写进本方向在 `dev-1.0` 上的 `dev/batch_yxy/status.md`。
- 红线：不改宿主机器配置与驱动。

## 8. 待确认事项

- VM-20-169 的机器代号、配额卡，以及 tele4 镜像是否已在本机加载；账号的 docker 权限。
- 在容器内实测插件修复的落地形态：平台 README 的挂载配方是否可行。若不可行，改用公共 case 的容器内补丁，并据此修订平台 README、`BOOTSTRAP.md` 与本文。形态定下来之后，再决定是把它登记为运行环境，还是等镜像重打包。
- `dev-1.0` 上本方向的看板与 `status.md` 仍是 2026-11-04 之前的状态（"待启动""独立仓地址待方向报备"），而 `dev/repos.yaml` 已于 2026-11-14 收录 FlagScheduler。
- NVIDIA 侧的宿主 Conda 与"实验在容器内进行"的红线，以哪个为准。

## 出处

- 《镜像、运行环境与交付 case 使用指南》（刘鑫；事实基准 `dev-1.0@8aa89fa`，2026-11-21 核对）
- 规范原文（`dev-1.0@aaee365`）：`dev/ENV-SPEC.md`、`dev/images/README.md`、`dev/images/CASE-TEMPLATE.md`、`dev/issues/README.md`、`dev/stack.lock.kunlunxin.v1.yaml`、`dev/images/kunlun-infer-vllm/v1/`、`dev/issues/vllm-plugin-fl-pooling-dispatch/`、`dev/issues/vllm-plugin-fl-pooling-nondeterminism/`
- 本分支：`AGENTS-DEV.md`、`BOOTSTRAP.md`、`platforms/kunlunxin-p800/README.md`、`platforms/nvidia-a6000/README.md`、`platforms/ascend-910c/README.md`、`docker-compose.yml`、`scripts/start_container.sh`
