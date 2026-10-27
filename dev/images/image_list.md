# 镜像索引

> 本文件只做索引：有哪些镜像、各自状态、归档目录在哪。新增版本在对应表尾追加一行。
>
> | 想查 | 看哪里 |
> |---|---|
> | 版本事实（commit、资源 sha256） | v3：各 `v<N>/image.baseline.yaml`（`python3 dev/lib/baseline.py list` 一次列全）；v1/v2 与未接入基线机制的版本：各 `v<N>/lock.yaml` |
> | 层级构成（L1~L4 由哪些层组成） | 各 `v<N>/lock.yaml` 的 `layers:` 字段 |
> | 过程叙述（尝试了什么、round 之间为什么切换、STOP CONDITION 怎么发现） | [`PROCESS.md`](PROCESS.md) |
> | 怎么用、已知限制 | 各 `v<N>/README.md` 与其现行运行环境目录（910C 两腿为 `env/` 系列，其余为 `demo/`） |
> | 用途 / 使用约束 | 消费方文档：`dev/stack.lock.910c.*.yaml` 现行版本、`docs/` 阶段目标文档 |
> | 目录规范、入档门槛、"官方"的三个主体定义 | [`README.md`](README.md)（§2/§5/§4.0） |
> | 分层设计依据 | [`docs/运行时基座建设-黄金准则与方案存档.v1.md`](../../docs/运行时基座建设-黄金准则与方案存档.v1.md) |

---

# 昇腾 910C

## 版本线：FlagTree ascend3.5

**BAAI·FlagTree 官方指南**：FlagTree ascend 用户手册 <https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ascend>
（分 ascend3.5 / ascend3.2 两条线）。本目录归档的三个系列整套栈均落在 **ascend3.5** 线。其中
`ascend-infer-vllm`（现行推理腿）是华为昇腾官方镜像：版本纪元对齐 ascend3.5
（CANN 9.0.0 / Python 3.11 / torch 2.10.0 / vLLM 0.20.2），但**不是 FlagTree 构建物**。

| 轴 | 本组镜像 | ascend3.5 | ascend3.2 |
|---|---|---|---|
| CANN | 9.0.0 | 9.0.0 | 8.5.0 |
| Python | 3.11 | 3.11 | 3.11 |
| torch (910C) | 2.10.0 | 2.10.0 | 2.6.0 |
| triton | 3.5.0 | 3.5.x | 3.2.x |
| vLLM | 0.20.2 | 0.20.2 | — |

两个反复引起误读的点：

- `triton_ascend 3.2.1` 是**华为昇腾官方** Triton 插件 `triton-ascend` 的版本号，与 `triton` 是两个独立包
  （wheel 从 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` 拷出，import 自报 `3.2.0`）。它不随 `triton` 或
  FlagTree 线号推进，**不代表 ascend3.2 线**。
- 全组通用 dev 底座（`VERSIONS.md` §2）是另一套东西，不用于原型结论性验证。日后若要在 ascend3.5
  镜像里引入 FlagTree 构建物，其 fork 须先从 `triton_v3.2.x` 切到 `triton_v3.5.x`。

## 版本索引

> v1 未按官方手册路径构建（推理腿直接用华为昇腾官方 `vllm-ascend`，训练腿为 BAAI 内部底座 + pip
> wheel 组合），v2 起按手册路径构建；详见 `PROCESS.md`。

**`ascend-operator-runtime` 系列**（昇腾工具链 + FlagOS 组件底座）

| 版本 | tag（image id） | repro_status | 目录 |
|---|---|---|---|
| v1 | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64`（`d948410966b0`） | 🟢 functional-repro | `ascend-operator-runtime/v1/` |
| v2 | `flagrt/ascend-operator-runtime:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-arm64` | 🟢 functional-repro（真机 2 卡） | `ascend-operator-runtime/v2/` |
| v3 | `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`（npu1-11 `62bcaae3f839`） | 🟡 partial-repro | [`ascend-operator-runtime/v3/`](ascend-operator-runtime/v3/README.md) |

**`ascend-operator-runtime-comm` 系列**（在上表基础上叠加 FlagCX 通信层）

| 版本 | tag（image id） | repro_status | 目录 |
|---|---|---|---|
| v1 | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64`（`3b9e08f231d0`） | 🟢 functional-repro | `ascend-train-comm/v1/` |
| v2 | `flagrt/ascend-operator-runtime-comm:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64` | 🟢 functional-repro（真机 2 卡 40/40） | `ascend-train-comm/v2/` |
| v3 | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`（npu1-11 `e13a15d0d02b`） | 🟡 partial-repro | [`ascend-train-comm/v3/`](ascend-train-comm/v3/README.md) |

> v3 是**训练腿发布物**（🟡；已知限制 A-2~A-4，其中 A-2 按运行环境版本分述：现行 env/v1 挂载覆盖含修复的 FlagCX 后不命中），现行运行环境
> [`ascend-train-comm/v3/env/v1/`](ascend-train-comm/v3/env/v1/docker-compose.yml)（demo/ 为冻结的第 0 版验收记录）；两条 v3 的 🟡 成因与升 🟢
> 条件见各自 README 的「已知限制」，待修缺陷的 case 包见 [`../issues/`](../issues/README.md)。
> 跨机 image id 不同属预期（内容同源）：npu1-27 侧为 `be30a952c2eb` / `7028028bb62c`，见各 `lock.yaml`。

**`ascend-infer-vllm` 系列**（华为昇腾官方一体镜像，非本组构建）

| 版本 | tag | repro_status | 目录 |
|---|---|---|---|
| v1 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（`@sha256:5cf8a2b6…`） | 🟢 digest 锁定、已归档、已验证可拉；登记式基线 + env/v1 现行运行环境（ADR-0007/0008） | `ascend-infer-vllm/v1/` |

> 该系列为**现行推理腿**（华为昇腾官方镜像，digest 锁定，🟢），公开 registry 按 digest 拉取；
> 不携带已知限制 A-1~A-4 任何一条。起服务纪律（`unset VLLM_PLUGINS`、经
> `dev/device-context/prototype/scripts/serve_standard.sh`）见 [`../stack.lock.910c.v4.yaml`](../stack.lock.910c.v4.yaml)
> 的 `lock.infer` 与 `per_leg.infer`。`triton_ascend 3.2.1` wheel 即取自该镜像。

---

# 昆仑芯 P800

## 版本线：FlagTree xpu3.6

**BAAI·FlagTree 官方指南**：FlagTree xpu 用户手册 <https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-xpu>
（线号 xpu3.6，无多线分叉）。本条目显式 pin 在 `flagtree 0.6.1+xpu3.6`，不跟随手册当前最新版本——
memory、device-context 两个子方向已各自验证该组合可用，理由与实测证据见该版本 `lock.yaml`。

## 版本索引

**`kunlun-operator-runtime` 系列**（昆仑芯工具链 + FlagGems 底座）

| 版本 | tag（image id） | repro_status | 目录 |
|---|---|---|---|
| v1 | `flagrt/kunlun-operator-runtime:1.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`（`466ee5c61793`） | 🟡 配方在手，未实机验证 | `kunlun-operator-runtime/v1/` |
| v2 | `flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`（`cac5ddae5f62`） | 🟢 functional-repro（实机 docker build 重建 + 带卡自判定 `overall: PASS`） | `kunlun-operator-runtime/v2/` |

> v1 2026-10-11 由 🟢 降为 🟡：归档镜像经查实是 `docker commit` 产物，不是其
> `Dockerfile.repro` 的构建产物，🟢 的定义（README §5）不成立；镜像本身可用，其运行时
> 验证结论（771 passed、训练 6/6、推理 PASSED）仍然成立。证据与影响见
> `kunlun-operator-runtime/v1/lock.yaml` 的 `known_issues: v1-image-is-docker-commit-not-build`。
> v2 用修正后的配方走正规 `docker build` 重建，组件集合与 pin 与 v1 完全相同，**从零重建用 v2**。
> v1 未接入环境基线机制（无 `image.baseline.yaml` 与 demo 运行环境目录）；v2 已接入
> （`kunlun-operator-runtime/v2/image.baseline.yaml` + `kunlun-operator-runtime/v2/demo/`）。

---

# 寒武纪 MLU590

## 版本线：FlagOS cambricon-neuware4.4.3（无 FlagTree 线）

寒武纪在 **BAAI·FlagTree 官方手册中无条目**（无 cambricon 用户手册、无 flagtree 镜像线），
本芯片走 **FlagOS 官方运行时线**（`harbor.baai.ac.cn/flagos-runtime`，上游配方 `flagos-ai/build-infra`）。
镜像内 `triton 3.2.0+mlu1.7.2` 是**厂商 triton-mlu 变体**，不是 FlagTree 构建物 —— 勿按昇腾/昆仑芯的
flagtree 线号理解。

**档位由宿主驱动决定**（本芯片独有的选型第一判据）：FlagOS 寒武纪镜像按 NEUWARE 档位分线，
各档官方标注宿主驱动前置 —— `cambricon-neuware4.4.3` → 6.2.15；`cambricon-neuware4.7.2` → 6.5.48。
现行锁定取 4.4.3 档，依据与实测见 `../stack.lock.cambricon.v1.yaml` 的 `device_registration.selection`。

## 版本索引

**`cambricon-mlu-runtime` 系列**（FlagOS 官方寒武纪运行时基座，登记式基线）

| 版本 | tag（image id） | repro_status | 目录 |
|---|---|---|---|
| v1 | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`（`e887e0ca4333`） | 🟢 FlagOS 官方镜像（公开 Harbor，匿名可按 digest 拉取） | `cambricon-mlu-runtime/v1/` |

> 本系列为登记式基线（ADR-0007）：镜像整体为上游官方产物，无仓内 `Dockerfile.repro`、不可重建；
> 版本主锚 = Harbor digest `sha256:e55b420e…`，辅锚 = pip 版本号与镜像 Label 的上游配方 commit
> `827dc613…`。运行时镜像**不含 vLLM**，推理服务化用下方 `cambricon-infer-vllm` 系列（推理腿发布物）。

**`cambricon-infer-vllm` 系列**（FlagOS 官方寒武纪推理应用镜像，推理腿发布物，登记式基线）

| 版本 | tag（image id） | repro_status | 目录 |
|---|---|---|---|
| v1 | `harbor.baai.ac.cn/flagos-app/vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2`（`4f298aa57807`） | 🟢 FlagOS 官方镜像（公开 Harbor，匿名可按 digest 拉取；2026-10-24 serve_standard 口径验收 PASS） | `cambricon-infer-vllm/v1/` |

> 登记式基线（ADR-0007）：版本主锚 = Harbor digest `sha256:76a673a8…`；vllm 0.20.2+flagos +
> vllm-plugin-FL 0.2.2（寒武纪线必需）+ neuware4.4.3 栈。服务起停走 serve_standard.sh
> （DC_BACKEND=cambricon，embed 形态；READY_BUDGET 建议 ≥300s）。

---

---

# 平头哥 PPU

## 版本线：FlagTree ppu3.6

**BAAI·FlagTree 官方 PPU 线**（ppu3.6：PPU SDK v2.1.0 / Python 3.12 / torch 2.10.0 / vLLM 0.19.0+cu130）。
本系列两版：**v1 = 官方基座登记式**（未改一字节）；**v2 = 在 v1 之上加一层本组派生**（把服务化前置
`triton` 固化进镜像）。与昇腾/昆仑芯的差别：本系列的血统锚在上游 Harbor digest（基座）与
**单 wheel sha256**（派生层），没有 FlagTree git commit 式构建。

## 版本索引

**`thead-ppu-runtime` 系列**（FlagTree 官方 PPU 基座 + 本组派生层）

| 版本 | tag（image id） | repro_status | 目录 |
|---|---|---|---|
| v1 | `flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-base`（`@sha256:50a906e6…`） | 🟢 FlagTree 官方镜像（Harbor digest 锁定，登记式基线 ADR-0007）；`superseded_by: v2`（**非废弃**：仍是 v2 的基座层，可按 digest 拉取） | `thead-ppu-runtime/v1/` |
| v2 | `flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64`（`1ba7ead15378`） | 🟢 functional-repro（实机 `docker build` 6 轮，**末轮产物即交付镜像**；全新容器 `TRITON_PPU_SMOKE_PASS` + `SERVE_STANDARD_PASS`；离线包三重核验） | [`thead-ppu-runtime/v2/`](thead-ppu-runtime/v2/README.md) |

> v2 解决的是一件会让整条腿不可用的事：官方基座**不含 `triton`**，而 vLLM 起服务前门控
> `find_spec("flash_attn") and find_spec("triton")` ⇒ 服务化腿必然起不来；此前靠手工 `docker cp`
> 从共享目录补三件、**容器重建即丢**。v2 用共享盘既有 flagtree wheel（sha256 `b57daff7…`，
> 一个 wheel 同时提供 `triton` + `flagtree`）经 `pip install --no-index --no-deps` 固化成一层，
> 做到「拉起镜像即可服务化」。派生层**只做加法**：不升级/降级基座任何包、不装本组运行时。
>
> 分发：本机构建、**未 push**（`RepoDigests` 为空）⇒ 走离线包（[`thead-ppu-runtime/v2/ARCHIVE.md`](thead-ppu-runtime/v2/ARCHIVE.md)）
> + 仓内配方重建，与训练腿镜像同口径。跨机 image id 不同属预期，版本锚 = `lock.yaml` 的 `image_id` + wheel sha256。
> 离线包实测：17.15 GiB · sha256 `6ab0dfb9…` · `pigz -t` rc=0 · `docker load` 回读 image id 与发布物逐字相同。
>
> 流程状态（2026-10-27）：v2 已入档，**生效切换待总组裁定**（候选登记在
> [`../stack.lock.thead-ppu.v1.yaml`](../stack.lock.thead-ppu.v1.yaml) 的 `candidates:`；`lock:` 仍指 v1）。
>
> ⚠️ **两个会误判的版本标签**（2026-10-27 实测）：① `pip show triton` **无输出** —— `triton` 是
> flagtree wheel 内部的一段，没有独立 `dist-info`，用 `pip show`/`pip freeze` 判"装没装"会判成没装；
> ② `flagtree.__version__` 自报 **3.6.0**（跟随 triton 线号），而 pip 元数据是 **0.7.0+ppu.gite223f42c**。
> 版本锚一律用 pip 元数据 + wheel sha256，不用 `__version__`。

# 当前生效版本

> 2026-10-14 发版后口径。锁定声明见 [`../stack.lock.910c.v4.yaml`](../stack.lock.910c.v4.yaml)
> 与 [`../stack.lock.kunlunxin.v1.yaml`](../stack.lock.kunlunxin.v1.yaml)；平头哥 PPU 见
> [`../stack.lock.thead-ppu.v1.yaml`](../stack.lock.thead-ppu.v1.yaml)；寒武纪 MLU590 见
> [`../stack.lock.cambricon.v1.yaml`](../stack.lock.cambricon.v1.yaml)；限制清单、规避配置与
> 删除条件见 [`../../docs/基座发版说明-20261014.md`](../../docs/基座发版说明-20261014.md)。

| 芯片 | 系列 | 生效 tag | 目录 |
|---|---|---|---|
| 昇腾 910C | ascend-operator-runtime | `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`（**基座镜像：内部构建层，不对外发布**） | `ascend-operator-runtime/v3/` |
| 昇腾 910C（训练腿） | ascend-operator-runtime-comm | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`（**训练腿发布物**，🟡 带 4 条已知限制） | `ascend-train-comm/v3/` |
| 昇腾 910C（推理腿） | ascend-infer-vllm | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（`@sha256:5cf8a2b6…`，华为昇腾官方镜像，🟢，不带已知限制） | `ascend-infer-vllm/v1/` |
| 昆仑芯 P800 | kunlun-operator-runtime | `flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`（🟢） | `kunlun-operator-runtime/v2/` |
| 平头哥 PPU | thead-ppu-runtime | **生效版本 = v1**（`flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-base`，`@sha256:50a906e6…`，BAAI·FlagTree 官方镜像，🟢 登记式基线）—— 与 `stack.lock.thead-ppu.v1.yaml` 的 `lock:` 一致；**候选 v2**（`1ba7ead1…`，🟢 派生层，已入档、离线包已产出）待总组裁定；device-context 方向运行容器已在用 v2 | `thead-ppu-runtime/v1/`（生效）+ [`v2/`](thead-ppu-runtime/v2/README.md)（候选） |
| 寒武纪 MLU590 | cambricon-mlu-runtime | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`（`@sha256:e55b420e…`，FlagOS 官方镜像，🟢 登记式基座；不含 vLLM） | `cambricon-mlu-runtime/v1/` |
| 寒武纪 MLU590（推理腿） | cambricon-infer-vllm | `harbor.baai.ac.cn/flagos-app/vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2`（`@sha256:76a673a8…`，FlagOS 官方镜像，🟢 登记式基线） | `cambricon-infer-vllm/v1/` |

> 昇腾 910C **两腿分栈**：训练腿 = `ascend-train-comm` v3 完整镜像，推理腿 = 华为昇腾官方
> `vllm-ascend`；使用方按腿选镜像，结论性验证须注明所在腿。三条自建血统 + 官方推理镜像均
> 已接入基线机制（推理腿为登记式，ADR-0007）。
> 三条自建血统 v1/v2/v3 并存归档，v1/v2 供历史结论追溯。`kunlun-operator-runtime v1` 已
> `superseded_by: v2`，**不得用于从零重建**（其归档镜像是 `docker commit` 产物，见该目录
> `lock.yaml` known_issues）。
> 分发方式：组内镜像（训练腿）无 registry 要求，走离线包与仓内配方重建；推理腿为公开
> registry 官方镜像，按 digest 拉取。

<!-- changelog: 2026-10-14 —— 本文件对齐 dev/stack.lock.910c.v4.yaml 口径（依据指挥台
<!-- changelog: 2026-10-27 —— 平头哥 PPU 新增 v2（在官方基座之上加一层派生，把服务化前置 triton 固化进镜像；
     官方基座不含 triton ⇒ vLLM 起服务必挂、此前靠手工 docker cp 且容器重建即丢）。新增「平头哥 PPU」一级节与
     版本索引表；v1 加 superseded_by: v2（非废弃，仍是 v2 基座层）；「当前生效版本」表按 stack.lock 口径写 v1 生效、
     v2 为候选（candidates 已登记，生效切换待总组裁定）。⚠️ 该「v2 已在用」仅指 device-context 方向运行容器。 -->
<!-- changelog: 2026-10-15 —— 新增平头哥 PPU 行：thead-ppu-runtime v1（FlagTree 官方基座，登记式基线 ADR-0007 同款，Harbor digest 锚），锁定声明 dev/stack.lock.thead-ppu.v1.yaml。 -->
     ADR-0004「推理腿切回官方 vllm-ascend」，取代 ADR-0003 §3.5）。修订点：
     ① 版本线节补明 ascend-infer-vllm 为版本纪元对齐 ascend3.5、非 FlagTree 构建物；
     ② ascend-infer-vllm 系列加表注，标为现行推理腿（官方镜像 / digest 锁定 / 🟢 / 不带 A 条目）；
     ③ 「当前生效版本」表新增昇腾 910C 推理腿行；ascend-operator-runtime-comm 行与
        ascend-train-comm 版本索引表注统一改为「训练腿发布物（🟡 带 4 条已知限制）」
        —— 单一发布物（推理+训练超集）的旧口径作废，对外发布物按腿两件；
     ④ 表下说明段删除「两腿收敛到同一基座 / 官方推理镜像降为历史血统」整句，改为两腿分栈现状；
        锁定声明指针 v3 → v4（v3 已标 LEGACY）；分发方式按腿分述。
     过程痕迹只留在本注释段，渲染后不可见。 -->

<!-- changelog: 2026-10-15 —— 新增寒武纪 MLU590 行与同名一级节：cambricon-mlu-runtime v1
（FlagOS 官方运行时基座，登记式基线 ADR-0007 同款，Harbor digest 锚），锁定声明
dev/stack.lock.cambricon.v1.yaml. -->

<!-- changelog: 2026-10-23
ADR-0008 落地：910C 两腿现行运行环境由 demo/ 改记 env/v1（demo/ 冻结为第 0 版验收记录）；
ascend-train-comm v3 行同步修正已知限制计数口径（A-1 已退出，A-2 按运行环境版本分述）。
-->
