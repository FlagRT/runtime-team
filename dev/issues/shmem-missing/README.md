# shmem 缺失：Triton tle 顶层无条件 `import shmem`

## 问题与现状

- **一句话**：FlagTree triton（ascend 线）的 `triton/experimental/tle` 在模块顶层无条件
  `import shmem`，而 `shmem` 不在基座镜像内，导致**任何含 for 循环的 Triton kernel 编译即失败**，
  包括 vLLM 框架自带的 kernel；与用哪个算子、是否加载模型无关。
- **归属方向**：算子编译方向。需求文本（现象、根因、三级请求、证据链）：
  [`决策备忘录 附录 B`](../../../docs/v3-决策备忘录与shmem跨组需求-20261008.md)。
- **状态（2026-10-14）**：**挂起，不跟进**。root cause 仍在，case 包与复现判据继续有效，归属不变
  （算子编译方向，组外），但不排期跟催；**不作为现行发版限制条目**。
  - **现行两条腿的标准路径均不触发**：训练腿不编译 Triton kernel（训练脚本只 import
    torch / torch_npu / transformers / datasets / DDP）；推理腿走华为昇腾官方 `vllm-ascend`
    （见 [`dev/stack.lock.910c.v4.yaml`](../../stack.lock.910c.v4.yaml) `lock.infer`），其 triton 为华为
    `triton-ascend`，不含本缺陷路径。
  - **残留影响**：该缺陷路径仍在 **现行训练腿发布物镜像**内（其 triton 为 FlagTree triton）——
    凡在该镜像内编译含 for 循环的 Triton kernel（例如启用 FlagGems 接管算子）仍会命中。
  - **现行运行环境里没有针对本缺陷的规避装配**：训练腿发布物的第 0 版运行环境
    [`demo/docker-compose.yml`](../../images/ascend-train-comm/v3/demo/docker-compose.yml) 不挂载占位模块、
    不前置 `PYTHONPATH`、不设 `VLLM_FL_FLAGOS_BLACKLIST`（训练腿标准路径不编译 Triton kernel，
    这三项对训练腿无作用）；第 1 版 [`env/v1/`](../../images/ascend-train-comm/v3/env/v1/docker-compose.yml)
    同样不含。占位模块资产 [`shmem_stub.py`](../../images/ascend-operator-runtime/v3/assets/shmem_stub.py)
    仍在仓内，供需要时自行挂入；该规避只在 FL 推理栈语境下成立，且**不彻底**：占位模块的函数被调用
    即报错，只适用于不触发分布式共享内存的场景。
  - **`VLLM_FL_FLAGOS_BLACKLIST=cos,sin` 不是现行规避配置**：`vllm-plugin-FL` 在 910C 已挂起。该配置只
    作用于 FlagGems 接管的 torch 算子，管不到 vLLM 核心 kernel
    （`vllm/v1/worker/block_table.py::_compute_slot_mapping_kernel` 同样命中），
    因此在 FL 推理栈上下游无法用配置完全规避。
- **调用链**（路径相对镜像内 `.../site-packages/triton/`，2026-10-11 在 image id `e13a15d0d02b` 内逐行核对）：
  `spec/ascend/compiler/code_generator.py:1177 visit_For` → 同函数 `:1197`
  `importlib.import_module("triton.experimental.tle")`（对任何 for 循环无条件执行）→
  `experimental/tle/__init__.py:1 from . import backends` →
  `experimental/tle/backends.py:12 _build_ascend_impl` 第 13 行 `from ...ascend.communication import ...` →
  `experimental/tle/language/dsa/ascend/communication.py:6 import shmem as ash` → `ModuleNotFoundError`。
  复现证据（报错落在 kernel 的 `for` 语句上）：
  [`assets/evidence/minirepro-20261011.txt`](assets/evidence/minirepro-20261011.txt)；
  `CompilationError` 以 `from None` 重抛，上述中间帧不出现在 traceback 中，故逐段结论来自源码核对。
- **影响**：vLLM 推理 engine 初始化即 `CompilationError`，且只有真实模型前向才触发——静态自检与构建全绿，
  对下游是隐蔽缺口。机制分析见
  [`ascend-operator-runtime/v3/REBUILD.md`](../../images/ascend-operator-runtime/v3/REBUILD.md)
  「ROUND 4 真机验证——推理腿」。

### 版本集合（分版表态）

坐标 = **镜像血统 + 运行环境修订**（第 0 版 = 冻结的 `demo/`，第 1 版起 = `env/vN`）。状态取三态之一：
在该版仍命中 / 在该版已修（写明形态）/ 未在该版测过。**「未在该版测过」必须显式写出**——缺陷不随
运行环境换版自动消失，沉默不等于不命中。

| 坐标 | 状态 | 依据 |
|---|---|---|
| `ascend-train-comm v3` · 第 0 版（[`demo/`](../../images/ascend-train-comm/v3/demo/docker-compose.yml)） | **在该版仍命中** | 2026-10-11 npu1-11：最小复现 `overall: FAIL`（退出码 1）。该版装配不含占位模块与 `PYTHONPATH` 前置 ⇒ 缺陷路径按缺省即命中 |
| `ascend-train-comm v3` · 第 1 版（[`env/v1/`](../../images/ascend-train-comm/v3/env/v1/docker-compose.yml)） | **未在该版测过** | 无本缺陷在该版的实测。该版与第 0 版的声明差异是 FlagCX 以 `pip-editable` 挂载覆盖，不触及 triton 编译路径，但**未实测即不表态为不命中** |
| `ascend-operator-runtime v3`（历史血统，推理腿已切华为官方 `vllm-ascend`）· 第 0 版 | **在该版仍命中** | 2026-10-10 全栈 9 步：撤除规避后 `vllm_encode_real_output: FAIL`；带规避 9/9 PASS |

## 复现

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | **现行训练腿发布物**：`flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`（npu1-11 image id `e13a15d0d02b`）—— [`dev/stack.lock.910c.v4.yaml`](../../stack.lock.910c.v4.yaml) `lock.train`，血统目录 [`dev/images/ascend-train-comm/v3`](../../images/ascend-train-comm/v3/README.md) | 按 [`ascend-train-comm/v3/REBUILD.md`](../../images/ascend-train-comm/v3/REBUILD.md) 构建，或从其 `ARCHIVE.md` 所列离线包 `docker load`；image id 由 `up.sh --strict` 按镜像基线核对 |
| 运行环境 | **训练腿** demo 运行环境 [`dev/images/ascend-train-comm/v3/demo/`](../../images/ascend-train-comm/v3/demo/docker-compose.yml) | 随仓。该镜像全部 repo 为固化资产，宿主检出不影响运行 |
| 卡 | 1 × 910C | 启动前 `docker ps` 确认本机无其他带卡容器（昇腾驱动对带卡容器独占） |
| 模型与数据 | **不需要**（最小复现不加载模型、不联网） | — |
| 模型目录变量 | `FLAGRT_MODELS_DIR` 指向一个实体目录 | demo 把它挂为 `/workspace/models:ro`；缺省值 `<仓根>/models` 在本机是软链，容器内无法解析，故启动时须显式给值（npu1-11：`/home/xliu969/models`） |

### 单入口命令（容器内，撤除 shmem 规避）

```bash
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0 flagrt-v3-demo-910c \
  bash -c 'PYTHONPATH=${PYTHONPATH#/opt/flagrt-stub:} python3 dev/issues/shmem-missing/assets/minirepro_shmem.py'
```

容器启动与收尾见 [`NOTES-step-by-step.md`](NOTES-step-by-step.md)。
`PYTHONPATH=${PYTHONPATH#/opt/flagrt-stub:}` 的作用是去掉占位模块目录，保证运行在未规避的环境里；
第 0 版 demo 的装配本就不含占位模块，该前缀剥离在这个环境里是空操作，命令照抄即可。
要跑带规避的对照组（预期 `overall: PASS`、退出码 0），须自行把
[`shmem_stub.py`](../../images/ascend-operator-runtime/v3/assets/shmem_stub.py) 以模块名 `shmem`
挂入容器、并把该目录前置到 `PYTHONPATH`——2026-10-11 的对照组读数即取自这样的装配。

### 预期结果（问题存在时，退出码 1）

```
triton.compiler.errors.CompilationError: at 5:4:
...
    for i in range(0, BLOCK, SUB):
    ^
ModuleNotFoundError("No module named 'shmem'")

=== VERDICT (minirepro_shmem, 1 card) ===
  repro: CONFIRMED - for-loop kernel compilation hit ModuleNotFoundError: No module named 'shmem'
overall: FAIL
```

### 完整栈路线（次要证据，不在本目录复制脚本）

9 步推理自判定脚本
[`ascend-operator-runtime/v3/assets/v3_step5_validate.py`](../../images/ascend-operator-runtime/v3/assets/v3_step5_validate.py)
在真实 vLLM 推理路径上命中同一缺陷（撤除两项规避配置后 `vllm_encode_real_output: FAIL`），命令与预期见
[`ascend-operator-runtime/v3/README.md`](../../images/ascend-operator-runtime/v3/README.md)「复现 shmem 缺失」。
该路线需 Qwen3-Embedding-0.6B 权重，耗时数分钟；最小复现足以定位缺陷，完整栈用于证明影响面。

## 修复验收

**判据**：在**不挂载 shmem 占位模块**（`PYTHONPATH` 不含 `/opt/flagrt-stub`）的环境下，上述单入口命令
退出码 0、判定行 `overall: PASS`。等价于：含 for 循环的 Triton kernel 编译通过且数值正确。

| 脚本 | 类别 | 命令 | 通过判据 | 最近一次结果与证据 |
|---|---|---|---|---|
| [`assets/minirepro_shmem.py`](assets/minirepro_shmem.py) | 复现 / 验证 | 见「单入口命令」 | 退出码 0 且 `overall: PASS` | 2026-10-11 npu1-11 demo 环境：撤除规避 `overall: FAIL`（退出码 1，缺陷复现）；带规避对照组 `overall: PASS`（退出码 0）。证据：[`assets/evidence/minirepro-20261011.txt`](assets/evidence/minirepro-20261011.txt) |
| `v3_step5_validate.py`（引用，不复制） | 验证 | 见「完整栈路线」 | 退出码 0，9 个 `[STEP]` 全 OK，且不设 `VLLM_FL_FLAGOS_BLACKLIST` | 2026-10-10 带规避 9/9 PASS；撤除规避按预期失败。证据：[`infer-demo-20261010.txt`](../../images/ascend-operator-runtime/v3/assets/evidence/infer-demo-20261010.txt) |

修复到货后，v3 demo 中的占位模块挂载、`PYTHONPATH` 前置与 `VLLM_FL_FLAGOS_BLACKLIST` 三项一并删除，
基座镜像 `repro_status` 由 🟡 升 🟢（[`ascend-operator-runtime/v3/README.md`](../../images/ascend-operator-runtime/v3/README.md)）。

### 复现脚本的两个坑（已在脚本内处理，自行改写时注意）

1. **Triton 编译缓存会掩盖缺陷**：缺省缓存目录 `~/.triton/cache` 在容器内跨运行累积（镜像本身不含编译缓存），同一容器内先跑过带 stub 的成功编译后，再撤除 stub 重跑会
   直接命中已编译产物，不再走 `visit_For`，判定假 PASS（实测遇到过）。脚本每次运行自建全新 `TRITON_CACHE_DIR`。
2. **输入输出不能同一张量**：原地读改写的 for 循环 kernel 在本后端得到错误数值（与本缺陷无关），会把
   判定污染成 `overall: ERROR`。脚本读入 `x`、写出 `y`。

<!-- changelog:
2026-10-22 —— 依据指挥台 ADR-0008《运行环境目录版本化（env/vN）与 demo 冻结》修订：
  ① 新增「版本集合」节（口径参照 Debian 的 Found/Fixed 为版本集合而非单值，坐标 = 血统 + 运行环境
     修订）：ascend-train-comm v3 第 0 版仍命中；第 1 版 env/v1 **未在该版测过**——仓内没有本缺陷
     针对 env/v1 的任何实测，env/v1 与第 0 版的声明差异只是 FlagCX 挂载覆盖、不触及 triton 编译路径，
     但未实测一律不表态为不命中；历史血统 ascend-operator-runtime v3 第 0 版仍命中。
  ② 更正一处已过期的装配描述：原文记「训练腿 demo 内有占位模块挂载与 PYTHONPATH 前置两项遗留装配、
     可选清理（T-025）」。实际 T-025 已于 2026-10-14（738da2b）执行，第 0 版 demo 的 compose 里
     这三项（占位模块挂载、PYTHONPATH 前置、VLLM_FL_FLAGOS_BLACKLIST）均已不存在，env/v1 也不含；
     据此改写该条，并在「复现」里补明：该环境下命令中的 PYTHONPATH 前缀剥离是空操作，带规避对照组
     须自行挂入占位模块（2026-10-11 的对照组读数取自当时含该装配的环境）。
     本条只校正装配事实，未改动复现命令本身，也未重跑验证（本批无 docker）。
  未改动：NOTES-step-by-step.md 与 assets/（其中指向 demo 的命令按 ADR-0008 决定 4 保持有效）。
-->

