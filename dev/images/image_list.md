# 镜像索引

> 本文件只做索引：有哪些镜像、各自状态、归档目录在哪。新增版本在对应表尾追加一行。
>
> | 想查 | 看哪里 |
> |---|---|
> | 版本事实（commit、资源 sha256） | v3：各 `v<N>/image.baseline.yaml`（`python3 dev/lib/baseline.py list` 一次列全）；v1/v2 与未接入基线机制的版本：各 `v<N>/lock.yaml` |
> | 层级构成（L1~L4 由哪些层组成） | 各 `v<N>/lock.yaml` 的 `layers:` 字段 |
> | 过程叙述（尝试了什么、round 之间为什么切换、STOP CONDITION 怎么发现） | [`PROCESS.md`](PROCESS.md) |
> | 怎么用、已知限制 | 各 `v<N>/README.md` 与其 demo 运行环境 |
> | 用途 / 使用约束 | 消费方文档：`dev/stack.lock.910c.*.yaml` 现行版本、`docs/` 阶段目标文档 |
> | 目录规范、入档门槛、"官方"的三个主体定义 | [`README.md`](README.md)（§2/§5/§4.0） |
> | 分层设计依据 | [`docs/运行时基座建设-黄金准则与方案存档.v1.md`](../../docs/运行时基座建设-黄金准则与方案存档.v1.md) |

---

# 昇腾 910C

## 版本线：FlagTree ascend3.5

**BAAI·FlagTree 官方指南**：FlagTree ascend 用户手册 <https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ascend>
（分 ascend3.5 / ascend3.2 两条线）。本目录归档的三个系列整套栈均落在 **ascend3.5** 线：

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

> v3 是**对外唯一发布物**（基座超集，推理 + 训练），demo 运行环境
> [`ascend-train-comm/v3/demo/`](ascend-train-comm/v3/demo/docker-compose.yml)；两条 v3 的 🟡 成因与升 🟢
> 条件见各自 README 的「已知限制」，待修缺陷的 case 包见 [`../issues/`](../issues/README.md)。
> 跨机 image id 不同属预期（内容同源）：npu1-27 侧为 `be30a952c2eb` / `7028028bb62c`，见各 `lock.yaml`。

**`ascend-infer-vllm` 系列**（华为昇腾官方一体镜像，非本组构建）

| 版本 | tag | repro_status | 目录 |
|---|---|---|---|
| v1 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（`@sha256:5cf8a2b6…`） | 🟢 digest 锁定、已归档、已验证可拉 | `ascend-infer-vllm/v1/` |

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

# 当前生效版本

> 2026-10-14 发版后口径。锁定声明见 [`../stack.lock.910c.v3.yaml`](../stack.lock.910c.v3.yaml)
> 与 [`../stack.lock.kunlunxin.v1.yaml`](../stack.lock.kunlunxin.v1.yaml)；限制清单、规避配置与
> 删除条件见 [`../../docs/基座发版说明-20261014.md`](../../docs/基座发版说明-20261014.md)。

| 芯片 | 系列 | 生效 tag | 目录 |
|---|---|---|---|
| 昇腾 910C | ascend-operator-runtime | `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`（**基座镜像：内部构建层，不对外发布**） | `ascend-operator-runtime/v3/` |
| 昇腾 910C | ascend-operator-runtime-comm | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`（**唯一对外发布物** = 推理+训练超集，🟡 带已知限制） | `ascend-train-comm/v3/` |
| 昆仑芯 P800 | kunlun-operator-runtime | `flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`（🟢） | `kunlun-operator-runtime/v2/` |

> 昇腾 910C **训练腿与推理腿收敛到同一基座**（v3 完整镜像，使用方无需按场景选镜像）；原独立的华为昇腾官方推理镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（`ascend-infer-vllm/v1/`）自 v3 起降为历史血统，不再作为原型锁定基座的推理腿。
> 旧血统 v1/v2 保留供历史结论追溯。`kunlun-operator-runtime v1` 已 `superseded_by: v2`，
> **不得用于从零重建**（其归档镜像是 `docker commit` 产物，见该目录 `lock.yaml` known_issues）。
> 本次发版未向任何 registry 推送：组内镜像无 registry 要求，分发走离线包与仓内配方重建。
