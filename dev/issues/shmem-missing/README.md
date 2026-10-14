# shmem 缺失：Triton tle 顶层无条件 `import shmem`

## 问题与现状

- **一句话**：FlagTree triton（ascend 线）的 `triton/experimental/tle` 在模块顶层无条件
  `import shmem`，而 `shmem` 不在基座镜像内，导致**任何含 for 循环的 Triton kernel 编译即失败**，
  包括 vLLM 框架自带的 kernel；与用哪个算子、是否加载模型无关。
- **归属方向**：算子编译方向。需求文本（现象、根因、三级请求、证据链）：
  [`决策备忘录 附录 B`](../../../docs/v3-决策备忘录与shmem跨组需求-20261008.md)。
- **状态（2026-10-14）**：未修复，root cause 仍在，case 包继续有效、继续跟催。
  - **适用面**：v3 完整镜像（训练腿）+ FL 追赶线（vllm-plugin-FL 推理栈）。910C 现行推理腿为华为昇腾官方
    `vllm-ascend`（见 [`dev/stack.lock.910c.v4.yaml`](../../stack.lock.910c.v4.yaml) `lock.infer`），
    不含本 case 的触发路径，本 case 不阻塞现行推理腿。
  - **训练腿临时规避**（两项，均写在
    [`demo/docker-compose.yml`](../../images/ascend-train-comm/v3/demo/docker-compose.yml)）：shmem 占位模块
    [`shmem_stub.py`](../../images/ascend-operator-runtime/v3/assets/shmem_stub.py) 以模块名 `shmem` 挂载，
    并前置于 `PYTHONPATH`。**规避不彻底**：占位模块的函数被调用即报错，只适用于不触发分布式共享内存的场景；
    两项撤除须实测验证（T-025）。
  - **FL 栈上的结论（追赶线）**：`VLLM_FL_FLAGOS_BLACKLIST=cos,sin` 不是现行规避配置。它只作用于 FlagGems
    接管的 torch 算子，管不到 vLLM 核心 kernel（`vllm/v1/worker/block_table.py::_compute_slot_mapping_kernel`
    同样命中），因此在 FL 推理栈上下游无法用配置完全规避；该栈在 v3 镜像上同时带 shmem 占位模块与该配置时推理 1 卡 9/9 PASS。
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

## 复现

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`（npu1-11 image id `e13a15d0d02b`） | 按 [`ascend-train-comm/v3/REBUILD.md`](../../images/ascend-train-comm/v3/REBUILD.md) 构建，或从其 `ARCHIVE.md` 所列离线包 `docker load`；image id 由 `up.sh --strict` 按镜像基线核对 |
| 运行环境 | [`dev/images/ascend-train-comm/v3/demo/`](../../images/ascend-train-comm/v3/demo/docker-compose.yml) | 随仓。v3 全部 repo 为固化资产，宿主检出不影响运行 |
| 卡 | 1 × 910C | 启动前 `docker ps` 确认本机无其他带卡容器（昇腾驱动对带卡容器独占） |
| 模型与数据 | **不需要**（最小复现不加载模型、不联网） | — |
| 模型目录变量 | `FLAGRT_MODELS_DIR` 指向一个实体目录 | demo 把它挂为 `/workspace/models:ro`；缺省值 `<仓根>/models` 在本机是软链，容器内无法解析，故启动时须显式给值（npu1-11：`/home/xliu969/models`） |

### 单入口命令（容器内，撤除 shmem 规避）

```bash
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0 flagrt-v3-demo-910c \
  bash -c 'PYTHONPATH=${PYTHONPATH#/opt/flagrt-stub:} python3 dev/issues/shmem-missing/assets/minirepro_shmem.py'
```

容器启动与收尾见 [`NOTES-step-by-step.md`](NOTES-step-by-step.md)。
`PYTHONPATH=${PYTHONPATH#/opt/flagrt-stub:}` 的作用是去掉占位模块目录，即还原未规避的原始环境；
去掉这一段即为对照组（带规避，预期 `overall: PASS`、退出码 0）。

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
