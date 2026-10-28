# 旧 XPU SDK：SDNN GEMM / Epilogue Lowering 兼容修复

这是独立的、限定 SDK 版本的实验兼容补丁。当前仓库新版 XPU Pipeline 保持原样；本目录不安装包、不覆盖 SDK 文件、不替代新版原生 ReLU 融合 Pass。cherry-pick 本提交后，按下面命令显式启用副本修复。

## 问题与修复

| 补丁 | 失败路径 | 修复位置/方式 |
|---|---|---|
| `01-sdnn-index-intrinsics.patch` | SDNN 转换的 `llvm.core_id` / `llvm.load_param` i32 intrinsic translation 不匹配 | `make_llir` 中 `llvm.to_module` 前，转为显式同名 intrinsic |
| `02-sdnn-scalar-relu-legacy.patch` | GEMM/Bias 后 `maximum(x, +0)` 的 scalar-C SDNN maximum 在 LLVM Lowering 失败 | XPU3 Bufferization 前物化片上零 Tensor，使用既有 Tensor maximum Lowering；保留单 Kernel |

02 依赖 01；`epilogue` profile 自动按顺序应用两者。02 仅覆盖单函数、静态二维 FP32、唯一可定位正零标量常量，其他形式不改写。新增片上缓冲和 fill 存在资源/性能成本；本补丁没有性能提升或新算法贡献主张。

## 精确适用版本

实测：P800 OAM / XPU3；Torch `2.9.0+cu129`，Triton 模块 `3.6.0`，FlagTree 发行包 `0.6.1+xpu3.6`；驱动 `5.0.21.47`。镜像：

```text
flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608
image ID sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975
```

原 SDK compiler SHA256 必须为：
`7133710c0153c52dfe17822eb03cfd026a19c7c861b293dd3c9e818b6a64abef`。

配套 `libtriton.so` 的历史实测 SHA256 为：
`28942c20acad0edde742553da22936974602cabc26b132ba23be30f6bbe87f5a`。
脚本自动校验 compiler hash；接收方仍需核对配套二进制和实际设备，不可仅按包版本号认定相同环境。

生成副本统一使用 LF，其 hash：

- GEMM：`6091230312bd05ff97a29bc10c09146c57818ea1c5f668958332b59b3a008f1d`
- Epilogue：`dbb2852ac6a9632c6d4d47b8aa958edad44066400489a89003698bab09394099`

早期实验副本使用 CRLF/混合换行，原始 hash 分别为 `e4d54a054e290d1c8889d890a52adfb70336093116cab26b410f6715300a24b0`、`18dfebd7a6dbd911c33d1874529b94120eae4db7bb315571ab5a801fdb391d5a`；本次仅规范换行，代码内容一致并重新上机验证。

## 自包含复现

在已有 SDK 容器中，用该 SDK 对应 Python 执行。工作目录为本目录；输出目录应为尚不存在的独立路径。选择实际空闲的 XPU 逻辑设备，以下 0 仅为示例。

```bash
export CUDA_VISIBLE_DEVICES=0
export FLAGTREE_BACKEND=xpu

# 原始 GEMM+Bias+ReLU：该旧 SDK 下预期失败、退出 1。
timeout -k 5 90 python -B reproduce.py --mode original --epi 3 --output /tmp/sdnn-original

# 仅 intrinsic 修复：GEMM 预期通过、退出 0。
timeout -k 5 90 python -B reproduce.py --mode gemm --epi 0 --output /tmp/sdnn-gemm

# 仅 intrinsic 修复不足以解决 scalar ReLU：预期失败、退出 1。
timeout -k 5 90 python -B reproduce.py --mode gemm --epi 3 --output /tmp/sdnn-before-relu-fix

# 两项修复：GEMM+Bias+ReLU 预期通过、退出 0。
timeout -k 5 90 python -B reproduce.py --mode epilogue --epi 3 --output /tmp/sdnn-epilogue
```

`reproduce.py` 在独立进程中加载 compiler 副本，只替换 `make_ttxir`/`make_llir`；工具和 bitcode 仍从原 SDK 查找。缓存与 IR 写入该次输出目录。源程序是单个融合 Triton Kernel，无 native fallback；输出 JSON、输入/实际/FP64 参考 NPZ 和编译产物均保存在输出目录。

测试 M/K/N：128/128/128、77/131/193、160/96/80；Tile 64/64/32；各使用 random、zero、constant、mixed_scale 四类输入（seed 20261019），每次成功运行共 12 项。

精度条件：全输出有限；逐元素 `abs_error <= 1e-4 + 1e-4 * abs(reference)`；relative L2 `<= 1e-4`。所有输出经真实设备同步后检查。mixed_scale 并不代表覆盖全部 FP32 极限值；NaN/Inf、负零位级语义、其他 dtype 和大规模资源边界尚未覆盖。

2026-10-19 发布前实测：original/epi3 失败；gemm/epi3 失败；gemm/epi0 `FINAL passed SAMPLES 12`；epilogue/epi3 `FINAL passed SAMPLES 12`；安装 compiler hash 保持原样。10 月 17 日更广的历史实验还有 27 组配置/108 次数值检查及 8 项 IR guard 检查；后者是历史证据，不是本自包含脚本全部重跑的范围。

## 只生成副本及回退

```bash
SDK_COMPILER=$(python -c 'import triton.backends.xpu.compiler as c; print(c.__file__)')
python prepare.py --source "$SDK_COMPILER" --profile epilogue --output /tmp/xpu-compiler-copy.py
python -m unittest discover -s . -p test_prepare.py
```

生成器对原文件 SHA256、patch hunk 的全部上下文和输出 SHA256 校验；拒绝未知 SDK、原地覆盖、覆盖已有输出文件。对新版 SDK 应先复现并检查原生 Pass，不强行套用旧补丁。

回退使用一个新进程的 `--mode original` 即可；修复只在显式进程内加载，安装目录从未改变。输出目录和修复副本可保留审计，不必改动系统环境或删除缓存。

## TTXIR 动态形状 / BF16 诊断实验

`diagnostic` profile 仅用于定位后续 XPU 实验，不是已验收的修复 profile：

```bash
python prepare.py --source "$SDK_COMPILER" \
  --profile diagnostic --output /tmp/xpu-compiler-copy-diagnostic.py

# 在容器内对 FlagGems 的最小 GEMM 测试运行诊断 profile
python run_pytest.py --profile diagnostic \
  --output /tmp/xpu-compiler-copy-diagnostic.py \
  tests/test_blas_ops.py::test_accuracy_mm
```

该 profile 在 intrinsic 兼容补丁基础上增加 `03-sdnn-ttxir-diagnostics.patch`：

- `XPU_SDNN_PRESERVE_TTXIR_ERRORS=1` 时保留 `make_ttxir` 原始异常，避免把动态形状或
  bufferization 类型错误继续包装成 `OutOfResources: uni_sram`；
- `XPU_SDNN_SKIP_EARLY_COMBINE=1` 时跳过第一次 `TritonSDNNCombineBefore`，用于单因素
  Pass 变体对照；默认值为 `0`，不改变原 Pipeline。

这两个开关不能单独证明动态形状或 BF16 已修复。真实 P800 运行需要在与
`BASELINE` SHA、`libtriton.so`、FlagTree/FlagGems 版本一致的容器中，对照记录
`default`、`XPU_SDNN_PRESERVE_TTXIR_ERRORS=1` 和 `XPU_SDNN_SKIP_EARLY_COMBINE=1` 的
TTIR/TTXIR、失败 Pass、数值结果和退出状态。当前仓库不包含 TritonSDNN/StructuredToMemref
实现源码；如果 Pass 变体仍失败，应由 SDK/厂商提供对应源码或重建的共享 Pass archive，
再实现真正的动态类型合同和 BF16 element-type 修复。

### 已验证的 FlagGems 侧兼容补丁

在 P800 标准镜像中，`04-flaggems-mm-block-pointer.patch` 将 Kunlunxin
`mm_kernel` 的普通指针加法改为 Triton block pointer，并使用边界检查读取/写回。
配合以下运行参数：

```bash
export PYTHONPATH=/env/FlagGems/src
export KLX_USE_AUTOTUNE=0
export XMLIR_MATMUL_FAST_MODE=1
```

已在独立实验容器中验证 FlagGems `test_accuracy_mm` 的 18 组测试：

```text
18 passed, 30 warnings
```

验证命令由 `run_acceptance.sh` 固化。脚本会先备份容器内 FlagGems 的 `mm.py`，
应用补丁，运行测试，并在退出时恢复原文件。该结果证明当前验收矩阵通过，
但 `XMLIR_MATMUL_FAST_MODE=1` 的严格 FP64 参考误差仍需按模型实际精度标准单独评估。

### MM/BMM/ADDMM 扩展验收

为覆盖验收方提出的 `mm/bmm/addmm` 矩阵算子集合，新增：

- `05-flaggems-bmm-block-pointer.patch`：将 Kunlunxin BMM 的 A/B/O 访问改为 block pointer，
  并在写回前转换为输出 dtype；在 `KLX_USE_AUTOTUNE=0` 下使用稳定的单一 tile 配置，
  避免 XPU3 autotuner 对大形状返回零耗时。
- `05-flaggems-addmm-block-pointer.patch`：将 Kunlunxin ADDMM 的 A/B/C 访问改为
  block pointer，将 bias 以 FP32 传入并按输出指针 dtype 写回；兼容模式固定
  `BLOCK_SIZE_M/N/K=32`，避免 XPU3 autotuner 对大形状返回零耗时。
- `06-flaggems-addmm-composition.patch`：ADDMM 使用 FP32 输入的 FlagGems MM，
  在 FP32 中完成 alpha/beta 与 bias 组合，最后转换回输入 dtype。这样保留 FlagGems
  路径并避免当前 SDK 对 ADDMM masked bias load 的类型合同错误；该路径不是 CPU/native fallback。

三项测试由 `run_blas_all_acceptance.sh` 固化；直接 ADDMM block-pointer 路径由
`run_blas_acceptance.sh` 单独固化。旧版直接 ADDMM 变体曾在 bias load 上报告
`element types do not match`/`other matches ptr type`，并在大形状触发 autotuner 零耗时。
当前 `05-flaggems-addmm-block-pointer.patch` 已加入 FP32 bias、固定 tile 和默认
Pipeline 验证，独立 ADDMM 72 组及三算子 99 组均已通过。

## 集成边界

本目录是旧 SDK 的文本级 IR bridge 兼容包，不是同版本 C++ SDNN 编译器的重建修复。正式进入新版 compiler 默认路径前，维护方需核对源代码/预编译对象版本、已有 intrinsic 和 ReLU Pass 的处理并单独回归。接收方如需在真实模型中应用，可参考 `reproduce.py` 的进程内加载方式，但须对自身输入合同补测；仅 cherry-pick 并不会自动修复所有已安装 Python 进程。

本提交只含本目录中的补丁、生成器、Kernel、复现脚本、主机单测与说明。不包含模型、凭据、实验容器管理、其他开发分支代码、历史日志或大体积二进制。
