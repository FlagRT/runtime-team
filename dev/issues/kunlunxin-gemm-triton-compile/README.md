# 昆仑芯 GEMM：flagtree/triton 对 FlagGems GEMM kernel 的编译路径不可用

## 问题与现状

- **一句话**：昆仑芯 P800 上，FlagGems 的 GEMM 类算子（`mm`/`bmm`/`addmm`）在
  flagtree `0.6.1+xpu3.6`（triton 3.6.0）下**编译阶段即失败，无一个形状可用**；
  与 FlagGems pin 到哪个 commit 无关，与是否加载模型无关。
- **归属方向**：算子编译方向（flagtree / triton xpu 后端）。
- **状态（2026-10-12）**：未修复，**无可用 workaround**。官方发布栈用 triton 3.0.0 线可绕开，
  但那是另一条版本线、不在本 pin 的验证范围内，换线等于重做一轮全量验证。
- **两种失败形态是同一个缺陷**（2026-10-12 复核定论，此前未定论）。形态由 **shape × dtype**
  决定，落在同一条编译流水线（`ttir → ttxir → llir → elf → xpubin`）的两个不同阶段：

  | 形态 | 失败阶段 | 代码位置 | 为什么长这样 |
  |---|---|---|---|
  | **原生 abort** | `llir` | 镜像内 `<site-packages>/triton/backends/xpu/compiler.py:486` 的 `pm.run(mod, 'make_llir')` | 该行**没有 try/except**，LLVM 原生断言不可捕获 → `abort()`。断言原文：`llvm/lib/IR/Instructions.cpp:758 llvm::CallInst::init` 的 `"Calling a function with bad signature!"` |
  | **`OutOfResources: uni_sram`** | `ttxir` | 同文件 `:362-367` 的 `pm.run(mod, 'make_ttxir')` | 该 `except` 把**任何**异常一律包成 `OutOfResources(0, 0, f"uni_sram {e}")` |

  早一步在 `ttxir` 就失败的形状走不到 `llir`，于是拿到被包装的异常；过了 `ttxir` 的才撞上
  LLVM 断言、原生死掉。
- **⚠️ `OutOfResources: uni_sram` 这个报错会骗人**：消息里的 `Required: 0, Hardware limit: 0`
  是构造函数里**写死的零**，并非真的 uni_sram 不足；`e` 的原文只是 `PassManager::run failed`。
  照字面去「调小 `BLOCK` / `num_stages`」是条死路。这是本 case 要上报的**第二个**独立问题。
- **触发路径**：autotuner → libentry → `flag_gems/runtime/backend/_kunlunxin/ops/mm.py:168`
  → triton JIT compile。48 帧完整栈与其同 v1 记录的逐行比对见
  [`../../images/kunlun-operator-runtime/v2/assets/evidence/gemm-failure-signature-recheck-20261012.txt`](../../images/kunlun-operator-runtime/v2/assets/evidence/gemm-failure-signature-recheck-20261012.txt)。
- **影响**：需要 GEMM 重的算子路径的消费方（阶段 3 推理链路）必须绕开 FlagGems 的 GEMM。
  同根因家族的第二类受影响算子是 `index_select`（vLLM rotary embedding 路径，
  `VLLM_FL_PREFER=vendor` 可绕），见
  [`../../images/kunlun-operator-runtime/v1/lock.yaml`](../../images/kunlun-operator-runtime/v1/lock.yaml)
  `known_issues: flagos-index-select-invalid-device-function`。
  镜像侧结论与取舍见
  [`../../images/kunlun-operator-runtime/v2/lock.yaml`](../../images/kunlun-operator-runtime/v2/lock.yaml)
  `known_issues: gemm-same-defect-two-failure-stages` 与 v1 的 `gemm-compile-sigabrt`。
- **上游版本声明滞后的佐证**（非本组实测）：FlagGems 仓 `backends.yaml` 仍声明
  `triton==3.0.0+a48aedef`（xpu3.0 时代遗留，与 xpu3.6 线不符），说明 xpu3.6 线未对 GEMM
  路径做回归测试。另 FlagGems 自带的 `test_accuracy_mm` 定义上方就带着注释
  `# TODO: failed at (1, 1, 2)` —— 上游已知小形状有问题。

## 复现

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | **昆仑芯发布物 `kunlun-operator-runtime v2`**：`flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`（本机 image id `sha256:cac5ddae5f62…`）—— [`dev/stack.lock.kunlunxin.v1.yaml`](../../stack.lock.kunlunxin.v1.yaml) `lock.runtime`，血统目录 [`dev/images/kunlun-operator-runtime/v2`](../../images/kunlun-operator-runtime/v2/README.md) | 按 [`kunlun-operator-runtime/v2/REBUILD.md`](../../images/kunlun-operator-runtime/v2/REBUILD.md) 从公开源头重建（~15 分钟），或用 [`其 ARCHIVE.md`](../../images/kunlun-operator-runtime/v2/ARCHIVE.md) 的离线包 `docker import`；image id 由 `up.sh --strict` 按镜像基线机器核对 |
| 运行环境 | [`kunlun-operator-runtime/v2/demo/`](../../images/kunlun-operator-runtime/v2/demo/docker-compose.yml) | 随仓。v2 全部 repo 为固化资产，宿主检出不影响运行 |
| 卡 | 1 × P800 XPU | **P800 没有「带卡容器独占」**（与昇腾相反）：`docker ps` 只是共享机礼貌，用 `xpu-smi` 挑 0% 利用率 / 0MiB 显存的卡，跑负载时以 `CUDA_VISIBLE_DEVICES=<该物理卡号>` 自律。设备模型四条实测结论见 demo 的 `docker-compose.yml` 文件头 |
| 模型与数据 | **不需要**（最小复现不加载模型、不联网） | — |
| 模型目录变量 | `FLAGRT_MODELS_DIR` 指向一个实体目录 | demo 把它挂为 `/workspace/models:ro`；缺省值 `<仓根>/models` 在本机是软链，容器内无法解析，故启动时须显式给值 |
| 代码仓 | 无（本环境 `mounted_repos: {}`） | — |

### 单入口命令

```bash
FLAGRT_MODELS_DIR=<任一实体目录> dev/lib/up.sh dev/images/kunlun-operator-runtime/v2/demo --strict
docker exec flagrt-kunlun-v2-demo-p800 \
  env CUDA_VISIBLE_DEVICES=<xpu-smi 挑中的空闲卡号> \
  /root/miniconda/envs/python310_torch29_cuda/bin/python3 \
  /workspace/dev/issues/kunlunxin-gemm-triton-compile/assets/minirepro_gemm.py
docker compose -f dev/images/kunlun-operator-runtime/v2/demo/docker-compose.yml down
```

容器启动与收尾的逐步操作见 [`NOTES-step-by-step.md`](NOTES-step-by-step.md)。
脚本覆盖 4 个形状 × 3 个 dtype = 12 组：前三个形状取自 FlagGems 自带
`tests/test_blas_ops.py::test_accuracy_mm` 的 `MNK_SHAPES`，第四个 64×64 是镜像验证脚本
用的那组 —— **两类形状都要跑，否则只会看到一种形态**。

### 预期结果（问题存在时，退出码 1）

```
--- 逐组形态 ---
M-N-K           dtype     rc  form
1-1-32          float16   1   native-abort@make_llir
1-1-32          float32   1   native-abort@make_llir
1-1-32          bfloat16  0   OutOfResources:uni_sram@make_ttxir
15-160-1024     float16   0   OutOfResources:uni_sram@make_ttxir
...
64-64-64        bfloat16  0   OutOfResources:uni_sram@make_ttxir

=== VERDICT (minirepro_gemm) ===
  组数: 12  原生 abort: 2  OutOfResources: 10  编译并算对: 0
  repro: CONFIRMED - 2 组在 make_llir 阶段原生 abort（LLVM 断言 "Calling a function with bad signature!"）
  repro: CONFIRMED - 10 组在 ttxir 阶段失败，被 compiler.py:367 误标为 OutOfResources: uni_sram（Required/Hardware limit 是硬编码零，并非资源不足）
overall: FAIL
```

## 修复验收

**判据**：同一条单入口命令退出码 **0**、判定行 **`overall: PASS`** —— 12 组全部编译通过
**且数值在容差内**（fp16 1e-2 / bf16 5e-2 / fp32 1e-3，与 CPU float32 参考比最大绝对偏差）。
编译通过但数值错记 `wrong-numerics`，**不算修复**。

| 脚本 | 类别 | 命令 | 通过判据 | 最近一次结果与证据 |
|---|---|---|---|---|
| [`assets/minirepro_gemm.py`](assets/minirepro_gemm.py) | 复现 / 验证 | 见「单入口命令」 | 退出码 0 且 `overall: PASS` | 2026-10-12 本机卡 7：`overall: FAIL`（退出码 1，缺陷复现），12 组全失败、2 abort / 10 OutOfResources。证据：[`assets/evidence/minirepro-20261012.txt`](assets/evidence/minirepro-20261012.txt) |

**报缺陷时请用 `(1,1,32)` + float32 这一组做最小复现**，不要用 64×64：前者给出原生 abort
加 LLVM 断言原文，信息量最大；后者只会给出被误标的 `OutOfResources`，把排查引向
「调小 block size / num_stages」这条死路。

本 case 实际包含**三件可独立处理的事**，按优先级：

1. **GEMM kernel 编译失败**（主缺陷）。验收判据同上。
2. **`compiler.py:362-367` 的错误误标**：ttxir 阶段的任意异常被一律包成 `OutOfResources`，
   且 `Required`/`Hardware limit` 硬编码为零，掩盖真实失败原因。修好它不解决 GEMM，但能让
   所有后续排查少走弯路。验收：该路径报出的异常类型与消息反映真实失败原因。
3. **`compiler.py:486` 无错误处理**：`pm.run(mod, 'make_llir')` 外没有 `try/except`，
   LLVM 断言直接 `abort()` 掉整个进程，调用方无法降级、无法把该算子加黑名单。这也是本
   case 的复现脚本**必须为每个形状起子进程**的原因（否则判定脚本会被一起带走，连判定行
   都打不出来）。验收：编译失败以可捕获的 python 异常抛出，调用方可降级。

修复到货后：本目录脚本判定转 `overall: PASS`；
[`kunlun-operator-runtime/v2/lock.yaml`](../../images/kunlun-operator-runtime/v2/lock.yaml)
的 `known_issues: gemm-same-defect-two-failure-stages` 与 v1 的 `gemm-compile-sigabrt` 一并关闭；
`v2/assets/verify_kunlun_runtime.py` 的 GEMM 探测由 `KNOWN` 行改为计入判定。

### 复现脚本的两个坑（已在脚本内处理，自行改写时注意）

1. **必须起子进程**：`(1,1,32)` + fp16/fp32 在 `make_llir` 原生 `abort()`，在同一进程里跑会
   把判定脚本一起带走。脚本每个形状一个子进程，父进程只收结果。
2. **pytest 不加 `-s` 会让 abort 形态的输出整段消失**：进程在 pytest 写出报告前就死了，
   被捕获的 stdout/stderr 随之丢弃，落盘是 **0 字节 + rc=1**，看起来像「什么都没发生」。
   实测同一用例 `-s` 三次各 880 字节含断言、`-q` 三次 0 字节。用 FlagGems 自带 pytest 复跑时
   必须 `pytest -x -s`；另外设 `PYTHONFAULTHANDLER=1` 反而会压掉 faulthandler 的 Python 栈
   dump（只剩 LLVM 断言行）。本目录脚本不走 pytest，不受此影响。
