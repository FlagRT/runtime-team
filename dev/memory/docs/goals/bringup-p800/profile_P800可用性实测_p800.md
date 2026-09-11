# 昆仑芯 P800 路线 A 可用性实测报告（2026-08-21）

> 执行：xliu969（Hermes 协助）｜ 设备：P800（8× OAM KL3，96GB/卡）｜ 容器：flagos-fl-dev-p800
> 目标：验证 FlagOS 官方栈（路线 A：厂商 torch + FlagGems + FlagCX + vllm-plugin-FL + vLLM）在昆仑芯 P800 上能否端到端跑通推理。
> 原则：只测不修。所有失败保留完整 traceback，未改任何上游代码。自行变通项单独标注。

---

## D. 一句话判定（先给结论）

> **昆仑芯 P800 上路线 A 当前处于「需变通可用」档** —— 官方发布镜像（triton 3.0.0 栈）上 FlagGems 算子层与 vLLM 0.13 + vllm-plugin-FL 端到端推理均实测可用（Qwen3-4B 正确出文本）；但本机 dev 容器（flagtree 0.6.1+xpu3.6 / triton 3.6.0）存在 triton 版本偏差导致 FlagGems GEMM 编译崩溃，且 MoE 模型（Qwen3.6-35B-A3B）触发 vendor 算子 causal_conv1d_update 硬失败，均需对齐官方发布栈或等待厂商修复。

---

## A. 分层可用性表

| 层 | 判定 | 证据 | 变通项 |
|---|---|---|---|
| **设备层**（厂商 torch 2.9.0+cu129） | ✅ 通过 | cuda.is_available=True、8 卡可见、4096² matmul 0.021s（6.45 TFLOPS FP32）、elementwise/relu 正常、D2H 1.70 GB/s / H2D 7.32 GB/s（见 S2.1） | 无 |
| **FlagGems**（kunlunxin 后端） | ⚠️ 部分通过 | **dev 容器（triton 3.6.0）**：rmsnorm/layernorm/fused_add_rms_norm/softmax/softmax_backward/rotary 全过（776 passed），**mm/bmm/addmm 三个 GEMM 测试全部 SIGABRT 崩溃**（triton xpu compiler make_llir 段，见 S2.2）；**官方发布镜像（triton 3.0.0）**：mm/bmm/rmsnorm/softmax 冒烟全过（精度 1.75e-4 内） | 变通项①：dev 容器将 flagtree 对齐官方声明（backends.yaml: triton==3.0.0+a48aedef / flagtree==0.5.1+xpu3.0）或直接用官方 flagrelease 镜像（triton 3.0.0）。不做则 GEMM 系算子编译即崩 |
| **FlagCX**（klx 适配器） | ✅ 通过 | FLAGCX_ADAPTOR=klx 下 torchrun 2 卡 allreduce：正确性 PASS（值精确求和），带宽 128MB→27.8 GB/s、512MB→28.3、1GB→28.3（单向；2x 口径 ~56.6 GB/s，PCIe Gen5 x16 饱和），数据校验 PASS（见 S2.3） | 无（注意：torch.cuda.Event 计时在该平台返回 0，须用墙钟+同步测量） |
| **vllm-plugin-FL**（kunlunxin vendor） | ✅ 通过 | VLLM_PLUGINS=fl 插件激活（PlatformFL，device_type=cuda）；platform 自动识别 kunlunxin（torch_xmlir 存在）；kunlunxin.yaml prefer=flagos、strict=false、op 顺序 flagos→vendor→reference；KunlunxinBackend 可导入；推理时确认 `Op 'attention_backend' using 'vendor.kunlunxin'`（见 S2.4/S3） | 无 |
| **端到端推理** | ✅ 通过（Dense 模型）/ ❌ 失败（MoE 模型） | **Qwen3-4B offline**：加载 84.5s（权重 1.47s/7.56GiB + init 71.5s），KV 缓存 504,000 tokens/69.22GiB，CUDA graph 正常捕获（**无需 enforce_eager**），3 请求 192 tokens 1.99s = 96.5 tok/s，中英文输出正确；**vllm serve**：chat completion 2.10s/64 tokens（含思考），预热后 0.68s；**Qwen3.6-35B-A3B（MoE）**：默认模式 graph capture 阶段 `xpudnn::causal_conv1d_update failed ret=1` 硬失败；enforce_eager 模式跑完但**输出乱码**（attention 输入格式不匹配警告 seq_len(5)<num_heads(16)）（见 S3） | 变通项②：MoE 模型 enforce_eager=True 可避免 graph capture 崩溃，但输出不可用 → 该模型在当前栈上实际不可用 |

**关键结论**：路线 A 的完整构成（厂商 torch + flag_gems + flagcx + vllm-plugin-FL + vllm 0.13）在 P800 上**官方发布镜像内可端到端跑通**；推理主链路（attention）走 vendor/kunlunxin（vllm 原生 PagedAttention 的昆仑实现），rms_norm/silu_and_mul/rotary/fused_moe 注册为 flagos OOT 算子。前轮调研"vllm-plugin-FL 的 kunlunxin impl 走 vllm 原生 PagedAttention、不依赖厂商算子库"已被实测确认（attention_backend 即 KunlunxinAttentionBackend）。

---

## B. 完整环境基线快照（可复现）

### B.1 宿主
```
主机        VM-0-2-ubuntu, Ubuntu 24.04.2 LTS, kernel 6.8.0-87-generic
驱动        KLRM 5.0.21.47 (build 515.58, f46d26c3..., 2026-04-24)
XPU-RT     10.2 (xpu-smi 报告)
设备        8× P800 OAM, KL3, 98304 MiB HBM/卡, PCIe Gen5 x16, ECC on
模块        kunlun + kunlun_peermem + ib_uverbs
/proc/kunlun/dev4/dma_excp_mask = 0 (mask dma len zero 0)
容器        flagos-fl-dev-p800 (host 网络, shm 512g, 全卡, /workspace=/data2/xliu969/code/runtime-team, /data1,/data ro)
镜像        flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608
```

### B.2 容器软件栈（flagos-fl-dev-p800 / venv python310_torch29_cuda）
```
Python      3.10.18 (conda)
torch       2.9.0+cu129 (cuda 12.9; COMMIT_SHA e9e3db62; USE_CUDA=ON, USE_NCCL=1, USE_XCCL=OFF, USE_XPU=OFF; GCC 13.3; cuDNN 9.10.2; 昆仑芯 CUDA 兼容版 xpytorch)
torch.xpu   存在 (True)
torch_xray  2.0.4 (XPU 运行时)
xmlir       1.0.0.1
xtorch_ops  0.1.2640+9bc36a18 (厂商融合算子库)
torch_plugin 0.1.0 (仅 runtime 初始化 .so, 无 torch_fl 显存池)
flag_gems   4.2.1rc0 (editable /env/FlagGems @73c5aff1 = 官方发布 commit)
flagcx      0.10.0 (editable /env/FlagCX/plugin/torch, klx 适配器)
flagtree    0.6.1+xpu3.6 (triton 3.6.0)   ← 与官方声明 triton==3.0.0+a48aedef 不一致
vllm        0.13.0 (昆仑芯版)
vllm-plugin-fl 0.1.0 (editable /env/xvllm-plugin-FL, 入口 vllm.platform_plugins: fl=vllm_fl:register)
libcudart   存在: libcudart.so.12.9.1.kunlun, 来源 = triton/backends/xpu/xpu3/so + torch_xmlir (xblas/xre) 自带, 非 NVIDIA 包
XCCL/BKCL   torch_xmlir/xccl/so/libbkcl.so (厂商通信库, FlagCX klx 底层)
```

### B.3 官方对照栈（flagrelease 发布镜像，zhenghaojia 容器实测）
```
triton 3.0.0 + flag_gems 4.2.1.rc.0 + torch 2.9.0+cu129 + vllm 0.13.0 + plugin 0.1.0 + flagcx 0.10.0
镜像: harbor.baai.ac.cn/flagrelease-public/qwen3.6-35b-a3b-nomtp-kunlunxin-gems_4.2.1rc0-vllm_0.13-plugin_0.1-cx_0.10.0-python_3.10.18-x86_64-driver_515.58:2604161518
```

### B.4 环境变量（路线 A 运行口径）
```
VLLM_PLUGINS=fl  VLLM_FL_PLATFORM=kunlunxin  VLLM_FL_PREFER=flagos|vendor
USE_FLAGGEMS=1  GEMS_VENDOR=kunlunxin  KLX_USE_AUTOTUNE=0  (CUDA_VISIBLE_DEVICES 选卡)
```

---

## S0–S3 执行明细

### S0 环境基线
命令与原始输出见上文 B.1/B.2（全部保留在本次会话记录）。要点：
- torch 为 USE_CUDA=ON 的兼容编译（device_name=cuda），torch.xpu 亦存在 → 双通道
- libcudart.so.12 是昆仑改版（.kunlun 后缀），随 triton-xpu 后端与 torch_xmlir 分发
- dma_excp_mask 仍为 0（昨日记录中的 flagos kernel 异常隐患，今日未触发于被测路径，GEMM 崩溃实为编译期而非 kernel 期）

### S1 官方安装流程
1. **官方技能存在**：flagos-ai/skills（main 分支）含 install-stack-flagos 技能（vLLM→FlagTree→FlagGems→FlagCX→vllm-plugin-FL 安装顺序，Gate：vLLM 与 vllm-plugin-FL 必须成功）。
2. **vendor-mappings.md 无 kunlunxin 条目**：Make Flags 表与 FLAGCX_ADAPTOR 表均无昆仑芯（nvidia/ascend/iluvatar/metax/mthreads/amd/enflame）。**官方技能未覆盖昆仑芯**。但 FlagCX 源码 _build_config.py 实际支持 klx（-DUSE_KUNLUNXIN_ADAPTOR，且 xpu-smi 自动检测→klx），插件侧 kunlunxin.yaml 平台配置齐全 → 官方技能文档滞后于代码。
3. **组件分发形态**（本机实测）：
   - vLLM 0.13.0：pip wheel（镜像自带昆仑版）
   - FlagTree：FlagOS PyPI 预编译 wheel（镜像自带 flagtree 0.6.1+xpu3.6；官方 backends.yaml 声明 0.5.1+xpu3.0 —— 见坑）
   - FlagGems：源码编译安装（editable @73c5aff1）；无预编译 wheel（本机由源码装，编译无报错）
   - FlagCX：两阶段（C++ make + torch plugin），本机 editable 已装，无重新编译
   - vllm-plugin-FL：源码 editable（/env/xvllm-plugin-FL 来自官方 tarball）
4. **官方容器镜像存在**：harbor.baai.ac.cn 下 flagrelease-public 有 kunlunxin 发布镜像（见 B.3；另有 kunlunxin001-gems5.0.0-…-vllm0.20.2 与新线镜像）。**优先用官方发布镜像即可获得可用栈**（triton 3.0.0）。

### S2 分层验证
- **S2.1 设备层**：通过。matmul 6.45 TFLOPS(FP32)、D2H 1.70/H2D 7.32 GB/s。带宽偏低疑为 xpytorch 兼容层 DMA 路径/测量首传开销，不影响可用性判定。
- **S2.2 FlagGems**：pytest 子集（--mode quick）：
  - 通过（776 用例）：test_accuracy_rmsnorm(1) / layernorm(2) / fused_add_rms_norm(1) / softmax(2) / softmax_backward(2) / apply_rotary_pos_emb(768)
  - 崩溃（SIGABRT, 进程级）：test_accuracy_mm / bmm / addmm —— `triton/backends/xpu/compiler.py:486 make_llir`（编译期 LLIR 生成崩溃），栈：autotuner→libentry→flag_gems mm.py:168
  - 归因：dev 容器 flagtree/triton 0.6.1+xpu3.6 与官方声明（triton 3.0.0+a48aedef）不符；**官方发布镜像（triton 3.0.0）同版本 flag_gems mm/bmm/rmsnorm/softmax 冒烟全过**（误差 ≤1.75e-4）→ GEMM 崩溃为环境版本偏差，非 FlagGems 代码缺陷
- **S2.3 FlagCX**：通过。2 卡 allreduce 正确性 PASS；带宽 128MB/512MB/1GB 均 ~28.3 GB/s 单向（2x 口径 56.6，PCIe Gen5 x16 上限内）；数据校验 PASS（真实通信确认）。注意点：该平台 torch.cuda.Event 计时恒为 0，必须用 torch.cuda.synchronize + 墙钟。
- **S2.4 vllm-plugin-FL**：通过。插件加载/激活日志明确；platform 自动识别 kunlunxin；kunlunxin.yaml（prefer=flagos, strict=false）；推理日志确认 attention_backend=vendor.kunlunxin（KunlunxinAttentionBackend）、rms_norm/silu_and_mul/rotary/fused_moe 注册为 flagos OOT 算子。

### S3 端到端
- **Qwen3-4B offline（卡1，ModelScope 拉取）**：✅
  - 权重加载 1.47s（7.56 GiB），引擎 init（profile+KV+graph capture）71.5s，合计 84.5s
  - KV cache 504,000 tokens / 69.22 GiB（gpu_mem_util 默认 0.9，加载后 used 93.80GiB）
  - CUDA graph 捕获正常（mixed prefill-decode PIECEWISE + decode FULL 各 51 个 size，18s）→ **无需 enforce_eager**
  - 生成 3 请求 192 tokens / 1.99s = **96.5 tok/s**（批处理），中英文输出正确（含知识问答）
- **Qwen3-4B serve（端口 8001）**：✅ chat completion 2.10s/64 tokens（含 think 段）；预热后 0.68s；/v1/models 正常
- **Qwen3.6-35B-A3B MoE（卡2，复用机内已有模型 /data1/zhenghaojia/models，arch=Qwen3_5MoeForConditionalGeneration 即插件注册的 qwen3_5_moe）**：❌
  - 默认模式：加载 OK（65.5 GiB/8.3s，attention 走 vendor.kunlunxin），**CUDA graph 捕获 dummy run 时崩溃**：`ValueError: Check 0 == ret failed, left operand=0, xpudnn::causal_conv1d_update failed, ret=1`（vendor/kunlunxin impl → xtorch_ops.causal_conv1d_update → xpudnn 返回 1）
  - enforce_eager=True 变通：可跑完（192 tokens/7.97s=24.1 tok/s）但**输出乱码**，且日志出现 `Input tensor shape suggests potential format mismatch: seq_len (5) < num_heads (16)`（vendor attention 输入格式问题）→ 实际不可用
  - 佐证：该机其他同事对同一模型的 vLLM 尝试也失败（但卡在 TP=8>可见卡数，未到算子层）；本失败为首次到达 causal_conv1d 算子层的新数据
  - 注：causal_conv1d 失败点在 vendor 算子（xtorch_ops/xpudnn），与 triton 版本无关，官方发布栈大概率同样失败（未在发布容器复测，避免干扰他人任务）

### S4 昇腾对照
**无法严格对照，注明原因**：① vllm 版本不同——昇腾侧 vllm 0.20.2 + torch 2.10.0+cpu + triton_ascend，P800 侧 vllm 0.13.0 + torch 2.9.0+cu129，不满足"同 vllm 版本"；② 910c 机器不在本会话可达范围（本机即 P800），无法现场同 prompt 复测；③ 昇腾侧记录中无常规 tokens/s 与 TTFT 数据（被长序列 prefill P0 阻塞）。仅做定性并列：

| 指标 | 昇腾 910c（记录） | 昆仑芯 P800（本次实测） |
|---|---|---|
| 模型 | Qwen3-4B 单卡 | Qwen3-4B 单卡 |
| 加载 | 31.89GiB / 24.6s | 93.80GiB(含KV池) / 84.5s（含 graph 71.5s） |
| KV 预分配 | 170,224 tokens (24.3GiB, 76%) | 504,000 tokens (69.22GiB) |
| 首次 attention/预热 | 437s | graph capture 18s（预热 ~0.4s 级） |
| 长序列 prefill | 2048×4 并发 22min 未完成（P0） | 昨日 Qwen2.5-1.5B：2048×4 并发 2.5s @3292 tok/s |
| 端到端生成 | 无稳定数字（P0 阻塞） | 96.5 tok/s（3 请求批处理） |

---

## C. 阻塞项清单

| # | 阻塞项 | 归属 | 现象/证据 | 已知 workaround |
|---|---|---|---|---|
| 1 | dev 容器 FlagGems GEMM 编译崩溃（SIGABRT @ triton xpu make_llir） | **我方配置**（环境版本偏差） | flagtree 0.6.1+xpu3.6（triton 3.6.0）≠ 官方 backends.yaml 声明 triton==3.0.0+a48aedef；官方发布镜像（triton 3.0.0）同 commit flag_gems 全过 | 对齐 triton 3.0.0（装 flagtree 0.5.1+xpu3.0）或直接用官方 flagrelease 镜像；不影响 vendor 路径推理（Dense 模型实测可用） |
| 2 | MoE 模型（Qwen3.6-35B-A3B）causal_conv1d_update 硬失败 | **厂商 SDK**（xtorch_ops/xpudnn 算子 ret=1） | vendor/kunlunxin impl → xtorch_ops.causal_conv1d_update → xpudnn 返回 1；eager 模式可跑但乱码（attention 格式警告） | 无可靠 workaround（enforce_eager 仅避免崩溃，输出不可用）；需厂商修 xpudnn causal_conv1d 或等 FLA 路径完善；属昆仑芯 FLA/混合注意力支持缺口 |
| 3 | 官方 install-stack-flagos 技能 vendor-mappings.md 无 kunlunxin | **FlagOS 组件**（文档滞后） | Make Flags / FLAGCX_ADAPTOR 表无 klx；源码实际支持（_build_config.py ADAPTOR_MAP 含 klx，xpu-smi 自动检测） | 按源码实际支持手工执行：make USE_KUNLUNXIN=1（Phase 1 未在本会话复验，本机为预装）+ FLAGCX_ADAPTOR=klx 装 plugin/torch |
| 4 | torch.cuda.Event 计时恒为 0 | 厂商 SDK（xpytorch 兼容层） | allreduce 带宽首测事件计时 0ms → ZeroDivisionError | 用 torch.cuda.synchronize + 墙钟测（本次已用） |
| 5 | 插件 0.1.0 裸 import flag_gems.runtime.backend.device | FlagOS 组件（已修，记录在案） | flag_gems>5.0.2 移除该模块；插件已有 try/except fallback（当前 /env 版本已含） | 无（现版本已含 fallback） |
| 6 | torch_plugin 0.1.0 无 torch_fl 显存池 | 厂商 SDK | flagos 显存池（FLAGOS_USE_CACHING_ALLOCATOR/memory_stats）在 P800 缺失 | 无（memory 子方向 V2 前置缺口，与本次推理可用性无关） |
| 7 | /proc/kunlun/dev4/dma_excp_mask=0 | 厂商 SDK（驱动层） | 昨日记录：flagos 路径 profile_run 曾触发 KL3 kernel 异常 status 719，triton 警告指向该 mask | 宿主 sudo `echo 1 > /proc/kunlun/dev4/dma_excp_mask`（未在本会话复测；今日 GEMM 崩溃为编译期，与此无关） |

**阻塞项归属统计**：厂商 SDK 3 项（#2/#4/#6，#7 驱动层）、FlagOS 组件 2 项（#3 文档、#5 已修）、我方配置 1 项（#1）。

---

## 附：原始证据位置
- 探针脚本：`dev/memory/probes/routeA_s2_1_device.py`、`routeA_s2_3_allreduce.py`、`routeA_s3_offline.py`、`routeA_s3_serve_client.sh`
- 原始日志：`dev/memory/docs/routeA-p800-20260821-logs/`（FlagGems 全量/rotary、S3 offline 4B、S3 MoE 默认/eager）
- 模型：`/workspace/models/Qwen3-4B`（ModelScope，gitignore）
- 参考：昨日记录 `dev/memory/docs/P800适配-执行记录-20260820.md`

## 复现命令
```bash
# 容器内
source /root/miniconda/bin/activate python310_torch29_cuda
# S2.2 FlagGems 子集（4 类算子）
cd /env/FlagGems/tests && GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0 \
  python -m pytest -q --mode quick test_blas_ops.py::test_accuracy_mm \
  test_norm_ops.py::test_accuracy_rmsnorm test_reduction_ops.py::test_accuracy_softmax \
  test_special_ops.py::test_apply_rotary_pos_emb
# S2.3 FlagCX 2 卡 allreduce
CUDA_VISIBLE_DEVICES=1,2 FLAGCX_ADAPTOR=klx torchrun --nproc-per-node=2 \
  /workspace/dev/memory/probes/routeA_s2_3_allreduce.py
# S3 offline（Qwen3-4B）
CUDA_VISIBLE_DEVICES=1 VLLM_PLUGINS=fl VLLM_FL_PLATFORM=kunlunxin "VLLM_FL_PREFER=flagos|vendor" \
  USE_FLAGGEMS=1 GEMS_VENDOR=kunlunxin KLX_USE_AUTOTUNE=0 \
  python -u /workspace/dev/memory/probes/routeA_s3_offline.py
# S3 serve
vllm serve /workspace/models/Qwen3-4B --served-model-name Qwen3-4B --port 8001
bash /workspace/dev/memory/probes/routeA_s3_serve_client.sh 8001 Qwen3-4B
```
