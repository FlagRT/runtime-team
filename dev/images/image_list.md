# 镜像索引

> 所有归档基座镜像的一览。详细 pin 清单以各 `<name>/v<N>/lock.yaml` 为准；本表只做索引，
> **过程性叙述（尝试了什么、round 之间为什么切换、STOP CONDITION 怎么发现的）统一记录在
> `PROCESS.md`**。"用途 / 使用约束"见消费方文档（`dev/stack.lock.910c.*.yaml` 现行版本、
> `docs/` 阶段目标文档），具体以消费方文档为准。基座设计的黄金准则与整体方案见
> `docs/运行时基座建设-黄金准则与方案存档.v1.md`（本表的"层级视图"就是该方案 §3
> 四层模型的具体实例）。
>
> 本文档里"官方"必须带主体名，三个主体（**华为昇腾官方** / **BAAI·FlagTree 官方** /
> **BAAI 内部（非发布物）**）的定义见 `README.md` §4.0。
>
> 按芯片分两大节归档，各节各自独立成谱系，版本对照仅在同芯片内进行。每节先给版本线速览，
> 再给依赖/血缘对照表，最后是各系列的**版本索引表 + 层级视图**——版本索引表一个版本
> 一行、按序向下堆叠，新增版本直接在表尾追加一行；层级视图是同一份数据的另一种呈现，
> 按 L1~L4 展示这个版本"由哪些层组成"，让人仅凭层级视图就能看懂血统构成。

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

- `triton_ascend 3.2.1`（下表各系列都列了）是**华为昇腾官方** Triton 插件 `triton-ascend` 的版本号，与 `triton` 是两个独立包，wheel 从 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` 拷出，import 自报 `3.2.0`。**它不随 `triton` 或 FlagTree 线号推进，不代表 ascend3.2 线。**
- **v1 未按 BAAI·FlagTree 官方手册的构建路径**：推理腿直接用**华为昇腾官方** `vllm-ascend`；训练腿用 **BAAI 内部** CANN 9.0.0 底座 + pip 装上游 triton + 华为昇腾官方 triton-ascend wheel + FlagOS 组件。版本纪元对齐 ascend3.5，构建方式自成一路。v2 起才真正按手册路径构建，详见 `PROCESS.md`。
- 全组通用 dev 底座（`VERSIONS.md` §1 记为 FlagTree 分支 `triton_v3.2.x`）是另一套东西，不用于原型结论性验证；日后若要在 ascend3.5 镜像里引入 FlagTree 构建物，其 fork 须先从 `triton_v3.2.x` 切到 `triton_v3.5.x`。

### 候选血统一览

| 版本线 | 状态 | 核心变化（一句话） | 过程详情 |
|---|---|---|---|
| v1 | 现网生效 | 自建底座 + pip wheel 组合，未按官方手册路径 | 本节上方 |
| v2 | 候选，未生效 | 切到 BAAI·FlagTree 官方手册路径构建；FlagCX 去除私有 patch | `PROCESS.md`「v1 → v2」 |
| v3 | 候选，🟡 partial-repro，两项决策已拍板落地（round 4） | Route A 设为默认设备后端；新增 vLLM 推理插件层（vllm-plugin-FL），训练+推理统一血统；round 4 加声明式覆盖层补丁并重建 | `PROCESS.md`「v3」/「round 4」 |

## 依赖与血缘一览（昇腾 910C）

> 只覆盖当前有意义对比的代际（v1 现网生效、v2/v3 候选）；更早的归档细节见各自
> `lock.yaml`/`ARCHIVE.md`。commit 权威来源同各 `lock.yaml`，与实际部署状态如有出入以
> `lock.yaml`/`pins.*.yaml` 为准（校验逻辑以 `dev/lib/verify_env.py` 为准，本表仅供人工索引阅读）。

**依赖 commit 对照**（同一行代表同一个库，跨代际对比版本是否推进）：

| 依赖库 | v1（现网生效） | v2（候选） | v3（候选） |
|---|---|---|---|
| FlagTree（提供 triton） | 不适用——v1 不含 FlagTree 构建物，triton 是 pip 装的 3.5.0 wheel | `triton_v3.5.x@15ec1a6cbc8d51f597f46459a500e96f3812c58f` | 同 v2，未变 |
| FlagGems | `@f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4` | 同 v1 | 同 v1/v2 |
| Torch-FL | `@af50297463d59ca4bb3aca59f51724afb5f6723a` | `@162582d678e40f133924a4d3b5b6df1cb8154dc7`（升级） | 同 v2 |
| FlagCX | `@55eb2ffff6988ae1db5e6ecb325472aecc93d238` + 2 owner 私有 patch | `@4e0e0cbcbf721169ca82348080f8353aebfe2c31`（换成公开主干，无私有 patch） | 同 v2，未变 |
| triton-ascend（华为官方插件，独立仓库） | `3.2.1`（wheel，从 vllm-ascend 镜像拷出） | 不适用（v2 起改走 FlagTree 自编译 triton） | 不适用 |
| vllm-plugin-FL | 不适用 | 不适用 | `release/0.2@8b059122e32b9ac47b9820a9c7b1bb95077481a4` |

**血缘（父镜像）**：

| 镜像 | 父镜像 |
|---|---|
| `ascend-operator-runtime` v1 | BAAI 内部手搭底座 `pytorch-plugin-fl:manual-20260807-ascend-dev` |
| `ascend-operator-runtime` v2 | BAAI·FlagTree 官方预构建基座 `flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2` |
| `ascend-operator-runtime` v3 | 同 v2 基座，未变 |
| `ascend-operator-runtime-comm` v1/v2/v3 | ← 各自对应版本的 `ascend-operator-runtime` |
| `ascend-infer-vllm` v1 | 无父镜像——华为昇腾官方独立发布，非本组构建 |

## ascend-operator-runtime 系列（昇腾工具链 + FlagOS 组件底座）

### 版本索引

| 版本 | tag（id） | repro_status | 归档目录 |
|---|---|---|---|
| v1 | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64`（`d948410966b0`） | 🟢 functional-repro | `ascend-operator-runtime/v1/` |
| v2 | `flagrt/ascend-operator-runtime:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-arm64` | 🟢 functional-repro（真机 2 卡验证通过） | `ascend-operator-runtime/v2/` |
| v3 | `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`（`be30a952c2eb`，round 4 重建含覆盖层补丁；round 2 为 9ad551058f2f） | 🟡 partial-repro | `ascend-operator-runtime/v3/` |

> 新增版本直接在此表追加一行；`(下一个版本占位行)`

### 层级视图

#### v1

```
L1 芯片底座    BAAI 内部手搭 CANN 底座（manual-20260807-ascend-dev）
              CANN 9.0.0（华为昇腾官方 toolkit）· ubuntu 22.04 · Python 3.11.15
L2 算子编译    triton 3.5.0（pip 官方 wheel，非 FlagTree 构建）
              triton-ascend 3.2.1（华为昇腾官方 wheel，从 vllm-ascend 镜像拷出）
              FlagGems 0.0.0+f7ae8e6b @f7ae8e6b（custom）
L3 Runtime    Torch-FL 0.1.0 @af50297（custom，默认路径）· MPICH 4.1.3
L4 挂载/装配  无 volume 挂载机制（早于 docker-compose+pins 机制）；
              无 vLLM 推理插件层（v1 推理腿走独立的 ascend-infer-vllm 系列）
```

#### v2

```
L1 芯片底座    FlagTree ascend3.5 官方预构建基座（取代 v1 的 BAAI 内部手搭底座）
              CANN 9.0.0 · Python 3.11.15 · torch_npu 2.10.0（共装，未卸载，未激活为默认）
L2 算子编译    FlagTree triton_v3.5.x@15ec1a6c 源码编译 → triton 3.5.1（official）
              FlagGems 0.0.0+f7ae8e6b @f7ae8e6b（custom，同 v1）
L3 Runtime    Torch-FL 0.1.0 @162582d（custom，升级自 v1；仍是默认路径）
              系统 mpich 4.0-3（apt）
L4 挂载/装配  无 volume 挂载机制（早于 docker-compose+pins 机制）；
              无 vLLM 推理插件层
```

#### v3

```
L1 芯片底座    同 v2 基座，未变
              torch_npu 2.10.0 为默认生效设备后端（Route A，PrivateUse1="npu"，真机确认）
L2 算子编译    FlagTree triton_v3.5.x@15ec1a6c（同 v2，triton 3.5.1，official）
              FlagGems @f7ae8e6b（同 v1/v2，装但不默认激活；⚠️ Triton-Ascend
              grid-stride-loop kernel 路径缺 shmem 模块，STOP CONDITION，见 lock.yaml）
L3 Runtime    Torch-FL @162582d（opt-in，需显式反向操作激活，guard fail-loud 真机确认）
              vllm 0.20.2（父镜像自带裸 vLLM）
              vllm-plugin-FL release/0.2 @8b059122e + 覆盖层补丁（round 2 新增；round 4 起叠加 assets/patches/ 声明式补丁，见 lock.yaml overlay_patches）
L4 挂载/装配  docker-compose.routeA.yml：${WORKSPACE_ROOT}:/workspace ·
              /usr/local/Ascend/driver 直通 · 16×davinci 设备 · shm_size 512g ·
              ipc: host
```

挂载点分类（`pins.routeA.yaml`，见 `docs/运行时基座建设-黄金准则与方案存档.v1.md` §3
GR3）：`FlagGems` / `Torch-FL` / `FlagCX` = **静态挂载点 strict**（精确 commit）；
`vllm-plugin-FL` = **动态挂载点 loose**（仅分支）。当前四者均在构建期固化进镜像，
尚未实现运行时挂载切换，属已知缺口（见黄金准则文档 §6）。

## ascend-operator-runtime-comm 系列（在上表基础上叠加通信层）

### 版本索引

| 版本 | tag | repro_status | 归档目录 |
|---|---|---|---|
| v1 | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64`（`3b9e08f231d0`） | 🟢 functional-repro | `ascend-train-comm/v1/` |
| v2 | `flagrt/ascend-operator-runtime-comm:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64` | 🟢 functional-repro（真机 2 卡 40/40 通过） | `ascend-train-comm/v2/` |
| v3 | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`（`7028028bb62c`，round 4 级联重建；round 3 为 43f3e2f70b4c） | 🟡 partial-repro | `ascend-train-comm/v3/` |

### 层级视图（只列相对 operator-runtime 同版本新增的 L3/L4；L1/L2 见上一节）

#### v1

```
L3 Runtime（通信）  FlagCX 0.13.0 @55eb2ff + 2 owner 私有 patch（bc84ea26… / 524654689…）
L4 挂载/装配        ENV FLAGCX_TORCH_BACKEND=flagos · TORCH_DEVICE_BACKEND_AUTOLOAD=0
                    （消费路径：flagos 适配）
```

#### v2

```
L3 Runtime（通信）  FlagCX 0.13.0 @4e0e0cb（FlagRT/FlagCX 组织仓公开主干 tip，
                    无 owner 私有 commit/patch，无 gaps）
L4 挂载/装配        同 v1 ENV（消费路径未变，仍是 flagos 适配）
```

#### v3

```
L3 Runtime（通信）  FlagCX 0.13.0 @4e0e0cb（同 v2 相同 commit，未升级）
                    ⚠️ 消费方需自行 torch.npu.synchronize()，否则 broadcast/all_gather
                    返回错误结果——是否吸收私有 fork 的 sync 修复待总组拍板
L4 挂载/装配        消费路径改为 HCCL 适配：删除 FLAGCX_TORCH_BACKEND/AUTOLOAD ENV，
                    改用 torch_npu 原生 + dist.init_process_group(backend="flagcx")，
                    对齐 train_qwen_1_5b_npu.py
```

## ascend-infer-vllm 系列

### 版本索引

| 版本 | tag | repro_status | 归档目录 |
|---|---|---|---|
| v1 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（`@sha256:5cf8a2b6…`） | 🟢 华为昇腾官方（digest 锁定，已验证可拉） | `ascend-infer-vllm/v1/` |

### 层级视图

#### v1

```
L1 芯片底座    华为昇腾官方一体镜像（CANN + torch_npu + vLLM 打包，非本组分层自建）
L2~L3 算子/Runtime  vllm_ascend（华为昇腾官方插件）· vLLM 0.20.2（官方）
L4 挂载/装配  不适用——官方一体镜像，未纳入本组 docker-compose+pins 挂载机制
```

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

### 版本索引

| 版本 | tag（id） | repro_status | 归档目录 |
|---|---|---|---|
| v1 | `flagrt/kunlun-operator-runtime:1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`（`466ee5c61793`） | 🟢 functional-repro | `kunlun-operator-runtime/v1/` |

### 层级视图

#### v1

```
L1 芯片底座    FlagTree xpu3.6 官方预构建基座 `@sha256:ea6d797a…`
              Python 3.10.18 · torch 2.9.0+cu129（CUDA 兼容层）
L2 算子编译    flagtree 0.6.1+xpu3.6（triton 3.6.0，pip 预编译 wheel，非源码构建）
              FlagGems 4.2.1.rc.0 @73c5aff1（editable）
L3 Runtime    FlagCX @3b43bdb2（editable）· vllm==0.13.0 · transformers==4.57.1 ·
              vllm-plugin-fl（editable，未单独记录 commit pin）
L4 挂载/装配  未接入 docker-compose+pins 机制（无独立 docker-compose 文件）
```

---

# 当前生效版本

| 芯片 | 系列 | 生效 tag | 目录 |
|---|---|---|---|
| 昇腾 910C | ascend-operator-runtime | `flagrt/ascend-operator-runtime:0.2.0-…` | `ascend-operator-runtime/v1/` |
| 昇腾 910C | ascend-operator-runtime-comm | `flagrt/ascend-operator-runtime-comm:0.1.3-…` | `ascend-train-comm/v1/` |
| 昇腾 910C | ascend-infer-vllm | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | `ascend-infer-vllm/v1/` |
| 昆仑芯 P800 | kunlun-operator-runtime | `flagrt/kunlun-operator-runtime:1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64` | `kunlun-operator-runtime/v1/` |

> `ascend-operator-runtime/v2/`、`ascend-train-comm/v2/`、`*/v3/` 均为候选新血统，**未生效**
> （`dev/stack.lock.910c.v1.yaml`/`v2.yaml` 均未改，昇腾现网仍是上表的 v1），详见 `PROCESS.md`。
