# 镜像索引

> 所有归档基座镜像的一览。详细 pin 清单以各 `<name>/v<N>/lock.yaml` 为准；本表只做索引。
> "用途 / 使用约束" 见消费方文档（`dev/stack.lock.910c.v1.yaml`、`docs/` 阶段目标文档），本表不涉及。
> 本文档里"官方"必须带主体名，三个主体（**华为昇腾官方** / **BAAI·FlagTree 官方** / **BAAI 内部（非发布物）**）的定义见 `README.md` §4.0。
> 按芯片分两大节归档，各节各自独立成谱系，不作跨芯片版本对照；每节内先给版本线说明，再给依赖/血缘对照表，最后是各系列的详细索引表。

---

# 昇腾 910C

## 版本线：FlagTree ascend3.5

> **BAAI·FlagTree 官方指南**：**FlagTree ascend 用户手册** <https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ascend>（分 ascend3.5 / ascend3.2 两条线，各自的 CANN / torch / triton / 基座镜像 / FlagTree 分支见手册）。

本目录归档的三个系列整套栈均落在该手册的 **ascend3.5** 线：

| 轴 | 本组镜像 | FlagTree ascend3.5 | FlagTree ascend3.2 |
|---|---|---|---|
| CANN | 9.0.0 | 9.0.0 | 8.5.0 |
| Python | 3.11 | 3.11 | 3.11 |
| torch (910C) | 2.10.0 | 2.10.0 | 2.6.0 |
| triton | 3.5.0 | 3.5.x | 3.2.x |
| vLLM | 0.20.2 | 0.20.2 | — |

- `triton_ascend 3.2.1`（下表各系列都列了）是**华为昇腾 Triton 插件** `triton-ascend` 的版本号，与 `triton` 是两个独立包。它的 wheel 从 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（即 ascend3.5 / vLLM 0.20.2 镜像）拷出，`import` 时自报 `3.2.0`（已知版本串差异）。**它不随 `triton` 或 FlagTree 线号推进，不代表 ascend3.2 线。**
- **未按 BAAI·FlagTree 官方手册的构建路径（v1）**：手册要求用 FlagTree 3.5 线基座镜像 + checkout `triton_v3.5.x` + `FLAGTREE_BACKEND=ascend` 自编译。v1 不含 FlagTree 构建物 —— 推理腿直接用**华为昇腾官方** `vllm-ascend`；训练腿用 **BAAI 内部** CANN 9.0.0 底座 + pip 装上游 `triton==3.5.0` + **华为昇腾官方** `triton-ascend==3.2.1`（wheel）+ FlagOS 组件（Torch-FL / FlagGems / FlagCX）。版本纪元对齐 ascend3.5，构建方式自成一路。v2 起才真正按手册路径构建（见下方"v2 候选血统"）。
- 全组通用 dev 底座（`VERSIONS.md` §1 记为 FlagTree 分支 `triton_v3.2.x`）是另一套东西，不用于原型结论性验证。**FlagTree fork 分支本原型不需要变更**（原型镜像不含 FlagTree 构建物）；日后若要在 ascend3.5 镜像里引入 FlagTree 构建物，其 fork 须先从 `triton_v3.2.x` 切到 `triton_v3.5.x`（手册规定 3.5 线配该分支），由编译层 / FlagTree 维护者决定。

### v2 候选血统：真正按手册构建路径（2026-09-12，见 `*/v2/`）

上面几条是 **v1**（现网生效版本）的口径：不按手册构建路径，triton 是 pip 装的
wheel 组合。`dev/images/*/v2/` 是**新血统**，第一次真正按 BAAI·FlagTree 官方
ascend3.5 手册的路径构建：BAAI·FlagTree 官方预构建基座镜像（取代 BAAI 内部手搭
底座）+ checkout `triton_v3.5.x` 上游 pinned commit + `FLAGTREE_BACKEND=ascend`
现场编译（取代 pip 装 triton wheel + 华为 triton-ascend wheel 的组合）。
**实测 `triton.__version__ == 3.5.1`**（手册版本线表格暗示 3.5.0，按实测记录）。
FlagCX 层改用公开 commit，不再有 `gaps` 段。详见
`ascend-operator-runtime/v2/lock.yaml` + `ascend-train-comm/v2/lock.yaml`。
**候选状态**：未进入 `dev/stack.lock.910c.v1.yaml` 的 `lock:`（现网仍锁 v1），
是否切换由总组另行裁定。

### v3 候选血统：Route A 默认 + 训练/推理统一血统（2026-09-22 起草，PHASE 2 ROUND 1-3 实机验证，见 `*/v3/`）

`dev/images/*/v3/` 在 v2 基础上迭代，目标是把设备后端默认路径从"torch_npu 与
torch_fl 共装但从未真正被用于计算"（v2 的实测结论）切换成"torch_npu 是唯一
默认路径，torch_fl 保留但降级为需要显式反向操作才能激活的 opt-in 组件"（**任务书
称之为 Route A**），同时让训练腿通信从"flagos 适配"（Torch-FL 的 "flagos"
ProcessGroup 包装 FlagCX）切到"HCCL 适配"（torch_npu 原生 + FlagCX 自己注册的
"flagcx" c10d 后端），对齐
`dev/device-context/910C/distributed_training/scripts/train_qwen_1_5b_npu.py`
的实测通路（2481 步双卡 DDP 训练闭环，zero torch_fl 依赖）；并新增 vLLM 推理
插件层，使同一血统的镜像既能跑训练也能跑 vLLM 推理服务（训练+推理统一，不再是
两条腿两条血统）。FlagGems/FlagTree triton 编译链原样保留（可插拔，供后续可选
启用，不默认激活）。

**⚠️ 阶段状态（PHASE 2 ROUND 3 已完成，2026-09-23；结论：训练腿"缺 sync"假说
证实但真实 loss/吞吐仍待拍板，推理腿发现新的、与之前担心的问题无关的
STOP CONDITION）**：

- **round 1（2026-09-22）**：vLLM 推理插件选用华为昇腾官方 `vllm-ascend`。
  精确 pin commit 核实无误（`367b8e62da799870a7476ce34f5f7658589a8aad`），但
  **`ascend-operator-runtime:v3` 设计 tag 构建失败（STOP CONDITION）**：
  vllm-ascend 依赖的 `triton-ascend==3.2.1` 不在公开 PyPI 索引，纯
  `pip install .` 无法解析。未产出镜像实体，vLLM 推理能力完全未验证。用一个
  跳过该失败层的**诊断镜像**验证了 Route A 默认生效、torch_fl guard
  fail-loud、FlagCX `all_reduce`/P2P `send`/`recv` 通信正确（均真机确认）。
- **round 2（2026-09-23，本次）**：重新核查 BAAI·FlagTree 官方 wiki 发现
  ascend3.5 线自己的"Run Qwen vLLM benchmark"一节走的是 `vllm-plugin-FL`
  （`github.com/flagos-ai/vllm-plugin-FL`），不是 vllm-ascend——核实其
  `pyproject.toml` 核心依赖不含 triton-ascend，不会撞上 round 1 那堵墙。精确
  pin `release/0.2` 分支 tip `8b059122e32b9ac47b9820a9c7b1bb95077481a4`（`git
  ls-remote` 实测取得）。**`docker build --network=host` 设计 tag 构建
  成功**（image id `9ad551058f2f`，不再需要诊断变体），`ascend-train-comm:v3`
  同步在新父镜像上重建成功（image id `43f3e2f70b4c`）。FlagGems pin 未改动，
  按已有 commit 构建/装配层面完全通过。发现本机另有一个
  `FlagRT/vllm-plugin-FL` 组织私有 fork 带未上游化的 ascend 专属正确性
  patch，按任务书"纯公开血统"要求未采用，记录在案。真机 Step 5 本轮会话未能
  执行（机器带卡容器并发数持续超限），记为 PENDING。详见
  `ascend-operator-runtime/v3/REBUILD.md`。
- **round 3（2026-09-23，本机卡资源释放后继续）**：
  - **训练腿（`ascend-train-comm:v3`）**：round 2 私有 fork 提出的"flagcx
    backend 异步返回，缺 `torch.npu.synchronize()`"假说，用真机 2 卡直接测试
    **证实成立**——`broadcast`/`all_gather` 无 sync 时复现原 STOP
    CONDITION，加 sync 后两个原语两个 rank 全部转为 PASS。**原 STOP
    CONDITION 结论更新为：是 2026-09-22 诊断脚本遗漏同步导致的误报，不是
    FlagCX 本体数据损坏**。但追加测试证实：真正的修复点必须落在 FlagCX/
    vllm_fl 自己的 c10d ProcessGroup 实现里（私有 fork 实际打的位置），
    不能靠训练脚本调用方用 Python 猴子补丁绕过（DDP 内部走 C++ 层校验，绕不
    过去）——**因此仍未产出真实 loss/吞吐数据**，卡在"是否吸收私有 fork 的
    sync 修复"这个需要总组拍板的决定上。详见
    `ascend-train-comm/v3/REBUILD.md`「ROUND 3」。
  - **推理腿（`ascend-operator-runtime:v3`）**：coexistence 检查（torch_npu
    Route A + flagcx + triton + vllm/vllm_fl 同进程）真机 **PASSED**，
    上面提到的两个私有 fork PrivateUse1 类型提升 bug 均**未在本血统触发**
    ——但真实推理烟雾测试发现了一个全新的、与那两个 bug 无关的
    **STOP CONDITION**：FlagGems 的 Triton-Ascend grid-stride-loop kernel
    代码路径（RoPE cos/sin cache 计算触发）间接 `import shmem`，这个模块
    既未装在镜像里，也不是公开 PyPI 包，`Dockerfile.repro` 从未提及——
    vLLM engine core 初始化失败，**未产出任何真实 `.encode()`/`generate()`
    输出**。round 1/2 的静态自检与训练侧验证从未触发这条代码路径，是本轮
    真实跑一次模型前向计算才第一次暴露。按任务书要求未尝试绕过（候选绕过
    方式均属"自行决定"，留待项目负责人裁定三个方向），详见
    `ascend-operator-runtime/v3/REBUILD.md`「ROUND 2/3 真机验证结果」。
- 退出期 SIGABRT 已知问题**确认复现**于 v3 默认路径（不导入 torch_fl，round 1
  发现，round 3 再次复现于 sync 测试的 teardown 阶段），推翻了"可能因不装
  torch_fl 而不复现"的推论——根因需重新定位（不是 STOP CONDITION，因为不影响
  已打印的正确结果）。
- 两条血统的 `repro_status` 维持 `🟡 partial-repro`（标准 🟢/🟡/🔴 里的
  🟡：部分功能真机验证通过，已知缺口明确记录）——训练腿从"broadcast/
  all_gather 数值错误、原因未知"进展到"确认是测试遗漏 sync，但修复需拍板"，
  推理腿从"构建成功、真机验证 PENDING"进展到"coexistence 通过，但推理本身
  卡在新发现的 FlagGems/shmem 依赖缺口"，见下方各系列表格与 `lock.yaml`。

**候选状态**：未进入 `dev/stack.lock.910c.v2.yaml` 的 `lock:`/`candidates:`
（out of scope，本次任务未改该文件），是否登记候选、是否切换由总组另行裁定
——鉴于训练腿真实 loss/吞吐仍待"是否吸收私有 fork 修复"拍板、推理腿仍卡在
FlagGems/shmem 依赖缺口，**不建议在这两项完成前登记为候选**。

## 依赖与血缘一览（昇腾 910C）

> 只覆盖当前有意义对比的代际（v1 现网生效、v2/v3 候选）；更早的归档细节见各自
> `lock.yaml`/`ARCHIVE.md`。commit 权威来源同各 `lock.yaml`，与实际部署状态如有出入以
> `lock.yaml`/`pins.*.yaml` 为准（本表不参与 `dev/lib/verify_env.py` 的校验，只做人读索引）。

**依赖 commit 对照**（同一行代表同一个库，跨代际对比版本是否推进）：

| 依赖库 | v1（现网生效） | v2（候选） | v3（候选） |
|---|---|---|---|
| FlagTree（提供 triton） | 不适用——v1 不含 FlagTree 构建物，triton 是 pip 装的 3.5.0 wheel | `triton_v3.5.x@15ec1a6cbc8d51f597f46459a500e96f3812c58f` | 同 v2，未变 |
| FlagGems | `@f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4` | 同 v1 | 同 v1/v2 |
| Torch-FL | `@af50297463d59ca4bb3aca59f51724afb5f6723a` | `@162582d678e40f133924a4d3b5b6df1cb8154dc7`（升级） | 同 v2 |
| FlagCX | `@55eb2ffff6988ae1db5e6ecb325472aecc93d238` + 2 owner 私有 patch（`bc84ea26…`/`524654689…`） | `@4e0e0cbcbf721169ca82348080f8353aebfe2c31`（换成公开主干，无私有 patch） | 同 v2，未变 |
| triton-ascend（华为官方插件，独立于上面几个仓库） | `3.2.1`（wheel，从 vllm-ascend 镜像拷出） | 不适用（v2 起改走 FlagTree 自编译 triton，不再用这个华为插件） | 不适用 |
| vllm-plugin-FL | 不适用 | 不适用 | `release/0.2@8b059122e32b9ac47b9820a9c7b1bb95077481a4` |

**血缘（父镜像）**：

| 镜像 | 父镜像 |
|---|---|
| `ascend-operator-runtime` v1 | BAAI 内部手搭底座 `pytorch-plugin-fl:manual-20260807-ascend-dev` |
| `ascend-operator-runtime` v2 | BAAI·FlagTree 官方预构建基座 `flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2` |
| `ascend-operator-runtime` v3 | 同 v2 基座，未变 |
| `ascend-operator-runtime-comm` v1 | ← `ascend-operator-runtime` v1 |
| `ascend-operator-runtime-comm` v2 | ← `ascend-operator-runtime` v2 |
| `ascend-operator-runtime-comm` v3 | ← `ascend-operator-runtime` v3 |
| `ascend-infer-vllm` v1 | 无父镜像——华为昇腾官方独立发布，非本组构建 |

## ascend-operator-runtime 系列（昇腾工具链 + FlagOS 组件底座）

| 层 | tag / 标识 | 血统 | 关键内置版本 | repro_status | 归档目录 |
|---|---|---|---|---|---|
| 基座 | `harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev` `@sha256:a36a3022…` | ubuntu 22.04 | CANN 9.0.0（**华为昇腾官方** toolkit 版本）· Python 3.12.13 · 昇腾工具链全套 | 🟢 已归档（镜像本体为 **BAAI 内部**手工构建底座，非华为发布物；BAAI harbor 可拉 + 本机 + 离线包，digest 锁定） | `ascend-operator-runtime/v1/ARCHIVE.md` |
| 基座变体 | `flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev-hostnet` | 同上（同 IMAGE ID，hostnet 变体） | 同上 | 🟢 同上 | 同上 |
| 运行时 | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64` (id `d948410966b0`) | ← 上面的基座 | Python 3.11.15 · torch 2.10.0+cpu · triton 3.5.0 · triton_ascend 3.2.1 · MPICH 4.1.3 · Torch-FL 0.1.0 `@af50297` · FlagGems `@f7ae8e6b` | 🟢 functional-repro | `ascend-operator-runtime/v1/` |
| 基座（**v2 候选血统**） | `harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2` | ubuntu 22.04 | CANN 9.0.0 · Python 3.11.15 · torch 2.10.0+cpu · torch_npu 2.10.0 · vLLM 0.20.2 | 🟢 BAAI·FlagTree 官方（公开 harbor + 官方离线包镜像双通道，本机已存在） | `ascend-operator-runtime/v2/lock.yaml` |
| 运行时（**v2 候选血统**） | `flagrt/ascend-operator-runtime:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-arm64` | ← 上面的 v2 基座 | Python 3.11.15 · torch 2.10.0+cpu(torch_npu 2.10.0 未卸载,共装) · **triton 3.5.1**(FlagTree 上游 `triton_v3.5.x`@`15ec1a6c` 现场编译,非 wheel) · 系统 mpich 4.0-3(apt) · Torch-FL 0.1.0 `@162582d` · FlagGems `@f7ae8e6b` | 🟢 functional-repro(真机 2 卡动态验证通过) | `ascend-operator-runtime/v2/` |
| 运行时（**v3 候选血统，partial-repro，round 3**） | `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`（id `9ad551058f2f`，round 2 构建成功，真实产出，不再是诊断变体；round 1 遗留的 `...-arm64-DIAGNOSTIC-novllm` id `4bab61434602` 已被取代，未删除但不建议再用） | ← 上面的 v2 基座（未变） | Python 3.11.15 · torch 2.10.0+cpu · **torch_npu 为默认生效设备后端（Route A，autoload，PrivateUse1="npu"，真机确认）** · triton 3.5.1(实测确认,与 v2 相同 commit) · Torch-FL `@162582d`(装但不默认 import,opt-in,guard fail-loud 真机确认) · FlagGems `@f7ae8e6b`(装但不默认激活,round 2 未改动此 pin;**round 3 发现其 Triton-Ascend grid-stride-loop kernel 路径缺 `shmem` 依赖，见下方 repro_status**) · **vllm-plugin-FL**（`github.com/flagos-ai/vllm-plugin-FL` release/0.2 `@8b059122e`，round 2 取代构建失败的 vllm-ascend，**构建成功**，`vllm_fl_installed: true`；round 3 真机确认 coexistence PASSED，真实推理因 FlagGems/shmem 缺口 FAILED，见 REBUILD.md） | 🟡 partial-repro(round 3：coexistence 真机通过；真实 `.encode()`/`generate()` 因 FlagGems 缺 `shmem` 模块的新 STOP CONDITION 未产出，与最初担心的两个 PrivateUse1 类型提升 bug 无关，见 `lock.yaml:known_issues`) | `ascend-operator-runtime/v3/` |

## ascend-operator-runtime-comm 系列

| tag | 血统 | 关键内置版本（在父层之上） | repro_status | 归档目录 |
|---|---|---|---|---|
| `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` (id `3b9e08f231d0`) | ← `ascend-operator-runtime:0.2.0` | FlagCX 0.13.0 `@55eb2ff` +2 patch(`bc84ea26…`/`524654689…`) · ENV `FLAGCX_TORCH_BACKEND=flagos` / `TORCH_DEVICE_BACKEND_AUTOLOAD=0` | 🟢 functional-repro（flagcx `.so` 逐字节一致） | `ascend-train-comm/v1/` |
| `flagrt/ascend-operator-runtime-comm:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`（**v2 候选血统**） | ← `ascend-operator-runtime:1.0.0-flagtree3.5-…` | FlagCX 0.13.0 `@4e0e0cb`(FlagRT/FlagCX 组织仓公开主干 tip,**无 owner 私有 commit/patch，无 gaps**) · 同上 ENV | 🟢 functional-repro（真机 2 卡 all_reduce/all_gather/p2p/async_all_reduce 40/40 通过；已知问题见 `lock.yaml:known_issues`） | `ascend-train-comm/v2/` |
| `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`（**v3 候选血统，partial-repro，round 3**，实际构建/验证用的是正式非诊断父镜像） | ← `ascend-operator-runtime:2.0.0-flagtree3.5-routeA-…` | FlagCX 0.13.0 `@4e0e0cb`（与 v2 **相同 commit**，未升级）· **消费路径改为 HCCL 适配**：ENV 删除 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` / `FLAGCX_TORCH_BACKEND=flagos`，`--device` 校验脚本改用 `torch_npu` + `dist.init_process_group(backend="flagcx")`，对齐 `train_qwen_1_5b_npu.py`（消费路径选择本身真机确认正确：`all_reduce`/P2P `send`/`recv` 结果正确）· `broadcast`/`all_gather` 原 STOP CONDITION **round 3 真机证实是诊断脚本遗漏 `torch.npu.synchronize()` 导致的误报**（加 sync 后两个原语两个 rank 全部 PASS），但真实 DDP loss/吞吐仍未产出——修复点须落在 FlagCX/vllm_fl 的 c10d 实现里，不能靠训练脚本调用方猴子补丁绕过（DDP 走 C++ 层校验），是否吸收私有 fork 的 sync 修复待总组拍板 | 🟡 partial-repro(round 3：sync 假说已证实，但真实 loss/吞吐待"是否吸收私有修复"拍板，见 `lock.yaml:known_issues` 与 `REBUILD.md`) | `ascend-train-comm/v3/` |

## ascend-infer-vllm 系列

| tag | 血统 | 关键内置版本 | repro_status | 归档目录 |
|---|---|---|---|---|
| `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` `@sha256:5cf8a2b6…` | **华为昇腾官方**镜像 | vLLM 0.20.2 · torch_npu · CANN · vllm_ascend | 🟢 华为昇腾官方（公共 registry，digest 锁定，已验证可拉） | `ascend-infer-vllm/v1/` |

---

# 昆仑芯 P800

## 版本线：FlagTree xpu3.6

> **BAAI·FlagTree 官方指南**：**FlagTree xpu 用户手册**
> <https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-xpu>（线号
> xpu3.6，无 ascend 手册那样的多线分叉；手册当前展示的最新 flagtree 版本是
> `0.7.0rc3+xpu3.6`）。

本条目显式 pin 在 `flagtree 0.6.1+xpu3.6`（不跟随手册当前最新版本）——
memory、device-context 两个子方向已各自验证过该具体组合可用，理由与实测
证据见 `kunlun-operator-runtime/v1/lock.yaml`。

## 依赖与血缘一览（昆仑芯 P800）

**依赖 commit 对照**：

| 依赖库 | v1（推荐默认底座） |
|---|---|
| FlagTree（提供 triton） | `flagtree 0.6.1+xpu3.6`（triton 3.6.0，pip 预编译 wheel，非源码构建） |
| FlagGems | `@73c5aff1`（editable） |
| FlagCX | `@3b43bdb2`（editable） |
| vllm-plugin-FL | editable，version 0.1.0，随基座镜像内置，本组未单独记录 commit pin |

**血缘（父镜像）**：

| 镜像 | 父镜像 |
|---|---|
| `kunlun-operator-runtime` v1 | BAAI·FlagTree 官方预构建基座 `flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base` |

## kunlun-operator-runtime 系列（昆仑芯工具链 + FlagGems 底座）

| 层 | tag / 标识 | 血统 | 关键内置版本 | repro_status | 归档目录 |
|---|---|---|---|---|---|
| 基座 | `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base` `@sha256:ea6d797a…` | ubuntu 22.04 | Python 3.10.18 · torch 2.9.0+cu129（CUDA 兼容层）· FlagGems `@73c5aff1`（editable，与本层运行时 pin 重合）· FlagCX `@3b43bdb2`（editable）· `vllm==0.13.0` · `transformers==4.57.1` · `vllm-plugin-fl`（editable） | 🟢 BAAI·FlagTree 官方（公开 harbor，本机已存在，digest 锁定） | `kunlun-operator-runtime/v1/lock.yaml` |
| 运行时（**推荐默认底座**） | `flagrt/kunlun-operator-runtime:1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64` (id `466ee5c61793`) | ← 上面的基座 | Python 3.10.18 · torch 2.9.0+cu129 · **flagtree 0.6.1+xpu3.6**（triton 3.6.0，pip 预编译 wheel，非源码构建）· FlagGems 4.2.1.rc.0 `@73c5aff1` | 🟢 functional-repro（训练 `TRAIN_LEG_PASS 6/6`、推理 `/v1/embeddings` 请求成功，各有一条使用前提；GEMM 编译崩溃等 4 项已知限制，见 `lock.yaml:known_issues`） | `kunlun-operator-runtime/v1/` |

---

# 当前生效版本

| 芯片 | 系列 | 生效 tag | 目录 |
|---|---|---|---|
| 昇腾 910C | ascend-operator-runtime | `flagrt/ascend-operator-runtime:0.2.0-…` | `ascend-operator-runtime/v1/` |
| 昇腾 910C | ascend-operator-runtime-comm | `flagrt/ascend-operator-runtime-comm:0.1.3-…` | `ascend-train-comm/v1/` |
| 昇腾 910C | ascend-infer-vllm | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | `ascend-infer-vllm/v1/` |
| 昆仑芯 P800 | kunlun-operator-runtime | `flagrt/kunlun-operator-runtime:1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64` | `kunlun-operator-runtime/v1/` |

> `ascend-operator-runtime/v2/`、`ascend-train-comm/v2/`、`*/v3/` 均为候选新血统，**未生效**
> （`dev/stack.lock.910c.v1.yaml`/`v2.yaml` 均未改，昇腾现网仍是上表的 v1），详见各自
> "版本线"小节。
