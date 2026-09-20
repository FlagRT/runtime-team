# 910C / P800 两实例 · 验证配置与依据（参考手册）

> 版本：v1.0（2026-09-20）｜ 负责人：Kistich（hliu553）
> **用途**：给后续接入者（第三家芯片 / 框架方向 / 验收方）一份"我们到底是怎么跑的、为什么这么配"的依据链。
> **原则**：每条配置都附**为什么这么定**与**证据指针**；凡未在本仓库存档的，明确标注「未记录/待补」，不臆造。
> 上级看板：`../README.md`（原型）｜ 实例看板：`../../910C/README.md`、`../../P800/README.md`

---

## 1. 一页速查（两实例配置对照）

| 维度 | 910C（第一实例，已完成） | P800（第二实例，阶段 0–4 完成） |
|---|---|---|
| 芯片 | 昇腾 910C（4 NPU / 8 chip，HBM 64 GB，CANN 9.0.0） | 昆仑芯 P800（8 卡，96 GB/卡） |
| **设备 API 命名空间** | `npu`（`torch_npu`） | **`cuda`**（XPytorch + `torch_xray` 符号重写；`torch.xpu` **不可用**） |
| **选卡变量** | `ASCEND_RT_VISIBLE_DEVICES` | **`CUDA_VISIBLE_DEVICES`** |
| 训练腿镜像 | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`（38.3 GB，本机已有） |
| 训练腿容器 | `flagos-proto-train-910c` | `hliu553-device-context-p800` |
| 训练腿 Python | `/usr/local/python3.11.15/bin/python3` | conda env `python310_torch29_cuda`（Python 3.10.18） |
| 推理腿镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（digest `sha256:5cf8a2b6…`） | 与训练腿同容器 |
| **训练框架** | `transformers`（`AutoModelForCausalLM`）+ `AdamW` + `torch.distributed` | **同一个脚本** `proto_train_leg.py` |
| **训练设备后端** | `runtime.use("flagos")`（= torch_fl）；通信后端 `flagos` | `runtime.use("kunlun")`；通信后端 `cpu:gloo,cuda:flagcx` |
| **推理框架（服务化）** | vLLM 0.20.2（**vllm-ascend 官方镜像自带**） | vLLM 0.13.0 + **vllm-plugin-FL**（`VLLM_FL_PLATFORM=kunlunxin`） |
| 模型 | `Qwen/Qwen3-Embedding-0.6B`（验收模型） | 同（共享缓存 snapshot 路径） |
| 训练超参 | 50 步 / batch 4 / seq 128 / world_size 2 | 同（同一脚本默认口径） |
| 推理服务参数 | `--runner pooling --convert embed --port 8100` | 同上 + `--max-model-len 4096 --gpu-memory-utilization 0.25 --enforce-eager` |
| 训练结果 | **2117.4 tok/s**（两卡合计）、24.18 s、6/6 PASS | **3482.2 tok/s**、14.7 s、6/6 PASS |
| 推理（单卡前向） | 66.32 句/s、avg 45.24 ms、10/10 PASS | 53.12 句/s、p50 56.17 ms、13/13 PASS |
| 推理（服务化） | **108.35 句/s**、p50 27.4 ms、10/10 PASS | 30.70 句/s、p50 96.4 ms、10/10 PASS |
| 错误注入 → 恢复闭环 | `ascend` 与 `flagos` **双后端**均 PASS | KL3 设 / 不设**两组均 PASS 且逐字节一致** |

---

## 2. 逐项配置与依据

### 2.1 镜像

| 实例 | 配置 | 为什么 | 依据 |
|---|---|---|---|
| 910C 训练腿 | `flagrt/ascend-operator-runtime-comm:0.1.3-…-flagcx0.13.0g55eb2ffp2-arm64` | 全组**统一基座**（总组裁定，各方向只消费不自建） | `dev/stack.lock.910c.v2.yaml` `lock.train`；`release.stage=candidate`（torch_fl 例外线） |
| 910C 推理腿 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | 华为昇腾官方推理镜像，vLLM 与 triton-ascend 配套 | 同上 `lock.infer`；digest 已记录 |
| P800 | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608` | 本机**已有**（38.3 GB）→ 官方手册的 59.9 GB `pull` + 32 GB `load` 全部跳过；FlagGems 源码已在容器内 `/env/FlagGems` | `P800/README.md` §2 认知 6 |

> **纪律**：结论性验证只在锁定镜像内做；P800 目前**尚未进入** `stack.lock`（第二实例镜像未入锁，诉求已在 `STATUS.md` 登记）。

### 2.2 模型

| 项 | 值 | 为什么 | 依据 |
|---|---|---|---|
| 验收模型 | `Qwen/Qwen3-Embedding-0.6B` | 全组 9 月统一验收模型（向量检索任务，便于跨芯片比对语义区分度） | `stack.lock.910c.v2.yaml: acceptance_model` |
| 910C 路径 | `/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B` | — | `prototype/runtime/proto/*.json` 的 `model` 字段 |
| P800 路径 | `/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…` | **必须用 `snapshots/<hash>`**：传仓库根目录会报 `Unrecognized model … Should have a model_type key`（根目录只有 `blobs/`、`refs/`、`snapshots/`） | `P800/README.md` §2 认知 5；`proto_infer_leg.py` 的 `resolve_model()` 已自动解析 |
| 其他模型 | 910C 服务化健康检查另用 `Qwen3-4B`（生成类，非 embedding） | 生成任务与 embedding 任务分开验证 | `910C/distributed_inference/inference/start_vllm_serve_910c.sh`；`results_p3/serve_health_result.json` |

### 2.3 训练形态与参数

| 项 | 值 | 为什么 | 依据 |
|---|---|---|---|
| 框架 | `transformers` + `AdamW` + `torch.distributed` | **刻意用最小形态**：设备方向要验证的是"设备层能力是否成立"，最小形态才能把**设备因素单独暴露**（单变量原则）。反例：P800 的 KL3 挂死只有在纯 torch 形态下才定位到"三处自旋点都在厂商 `libxpucuda.so`" | `proto_train_leg.py` docstring："本方向训练腿（transformers 纯 torch）" |
| 规模 | 2 卡（world_size=2）| 设备方向只负责**最小正确规模（≤2 卡）**；8 卡→多机归分布式方向（避免各方向挤在同一规模重复验证） | 月度计划 §附录「规模递进」 |
| 超参 | 50 步 / batch 4 / seq 128 | 足以暴露收敛性问题 + 让通信反复触发（KL3 缺陷正是在反复通信中暴露） | `proto_train_leg.py`（`MAX_STEPS`/`BATCH`/`SEQ`）；实际值见两实例结果 JSON 的 `perf` |
| 学习率 | `LR` 环境变量，默认 `1e-5` | 小模型微调稳定区间 | `proto_train_leg.py`（**910C 与 P800 实际所用 LR 未在仓内逐次记录**，标注待补） |
| 910C 设备后端 | `flagos`（torch_fl） | **该镜像禁止 `torch_npu` 与 Torch-FL 共存**（自带校验脚本直接报错）；必须 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 且**先 `import torch_fl` 再 `import torch`**。属全组登记在案的**权宜例外**，不代表路线变更 | `stack.lock.910c.v2.yaml` `per_leg.train` + `device_registration.exception` |
| P800 设备后端 | `kunlun` | 昆仑芯走 `torch.cuda` 命名空间 | `prototype/runtime/backends/kunlun/backend.py` |
| P800 通信后端 | `cpu:gloo,cuda:flagcx` + `FLAGCX_ADAPTOR=klx` | **只有 `flagcx` 一条路可用**：`nccl` 挂死；`xccl` 未编译（`Distributed package doesn't have XCCL built in`）；`kccl` 无响应 | `P800/README.md` §2 认知 4 |
| 通信后端名差异 | 910C 注册为 **`flagos`**；P800 注册为 **`flagcx`**，且**必须显式 `import flagcx`** | 同一 FlagCX 在不同芯片上注册的后端名不同 —— 换芯片不只换设备命名空间，连通信后端名也变 | `P800/README.md` §2 认知 3 |

### 2.4 推理形态与参数

| 项 | 值 | 为什么 | 依据 |
|---|---|---|---|
| 形态 | **两种**：① 单卡前向（`transformers`）② 服务化（vLLM OpenAI 兼容） | 前向形态用于**快速暴露设备因素**；服务化形态用于对齐"推理服务"验收口径 | `proto_infer_leg.py` / `proto_infer_serve.py` |
| 服务参数 | `--runner pooling --convert embed --port 8100` | 该 vLLM 版本**没有 `--task` 参数**，embedding 服务用 `--runner pooling --convert embed`（两实例同口径，可直接比对） | 910C：`DC_STAGE_SUMMARY_20260909.md` §3.1；P800：实测 `--task embed` 报 `unrecognized arguments` |
| P800 追加参数 | `--max-model-len 4096 --gpu-memory-utilization 0.25 --enforce-eager` | `--enforce-eager` 关闭 CUDA Graph 以求稳定（**代价见 §4：吞吐偏低，不作性能结论**）；`max-model-len 4096` 与 910C 一致，便于用同一个"超长输入"判据 | `P800/docs/KUNLUN_P800_STAGE34_VERIFY_20260920.md` §1.3 |
| 910C 推理框架 | vLLM 0.20.2（vllm-ascend 官方镜像自带） | **显式禁用 vllm-plugin-FL**：设 `VLLM_PLUGINS=fl` 后 `current_platform.device_type` 为空 → `RuntimeError: Device string must not be empty`；根因是 **vllm-plugin-FL 没有 ascend 后端（"currently CUDA only"）** | `910C/distributed_inference/inference/start_vllm_serve_910c.sh` 注释「坑 A5」；`910C/distributed_inference/docs/DEVICE_CONTEXT_INFERENCE_PLAN_20260831.md:44` |
| P800 推理框架 | vLLM 0.13.0 + **vllm-plugin-FL** | 反过来：昆仑芯**需要**该插件提供 FL platform（`VLLM_FL_PLATFORM=kunlunxin`）；容器里已以 editable 方式装好（`/env/xvllm-plugin-FL`） | 实测：`vllm platform: <vllm_fl.platform.PlatformFL>` |

### 2.5 选卡与并发

| 实例 | 规则 | 依据 |
|---|---|---|
| 910C | **带卡容器并发上限 3**：超限 `acl.init()` 返回 500000、`torch.npu.device_count()=0`（设备"凭空消失"）；两条腿**串行**跑 | `stack.lock.910c.v2.yaml` `rules` 第 1 条（最高优先级） |
| P800 | 选卡用 `CUDA_VISIBLE_DEVICES`（**不是** XPU 侧变量）；共享机需先 `xpu-smi` 挑**空闲且连续**的卡并记录用卡 | `P800/README.md` §2 认知 2；曾因他租户占卡误判"flagcx 不可用" |

### 2.6 算子路径

| 实例 | 配置 | 为什么 | 依据 |
|---|---|---|---|
| 910C 训练腿 | 镜像内置 FlagCX + 算子库 | 锁定基座已含 | `stack.lock.910c.v2.yaml` `lock.train.notes` |
| P800 | `VLLM_FL_PREFER=vendor` + `USE_FLAGGEMS=0` + `GEMS_VENDOR=kunlunxin` + `KLX_USE_AUTOTUNE=0` | 取 **vendor 算子路径**以避开 FlagGems 路径及其 KL3 依赖，与本方向"不设 `XPU_EVENT_KL3_ENABLE`"的锁定口径一致 | `P800/docs/…STAGE34_VERIFY_20260920.md` §1.3 |

---

## 3. 关键决策的依据链（"为什么不是别的"）

| 决策 | 结论 | 完整依据 |
|---|---|---|
| 设备注册路线 | **Route A（厂商官方 torch 插件）为主线**；910C 训练腿的 flagos(torch_fl) 是**镜像约束下的例外** | `stack.lock.910c.v2.yaml` `device_registration`；`Fla/docs/01-overview/flagos_overview.md:40-56`（A 线=生产交付 / B 线=预研不承担交付） |
| 910C 推理为何不用 vllm-plugin-FL | 该插件**无 ascend 后端**，启用即破坏 platform 选择 | 见 §2.4「坑 A5」 |
| P800 推理为何用 vllm-plugin-FL | 昆仑芯需要它提供 FL platform；容器内已装 | §2.4 |
| P800 通信为何只有 flagcx | nccl 挂死 / xccl 未编译 / kccl 无响应 | §2.3 |
| 为何不设 `XPU_EVENT_KL3_ENABLE` | 该变量 + 设备侧集合通信 → 概率性永久挂死（实测 ≈89%），属**厂商运行时层缺陷**（已定位到函数级，3 处自旋点均在厂商 `libxpucuda.so`，偏移 `+0x94080`）。规避后 A/B 单变量实测有效；且**本方向两条腿均不依赖 FlagGems**，故该变量不属验证前置条件。⚠️ 但它同时是 FlagGems kunlunxin 后端的官方推荐变量（`tools/env.sh` / `backends.yaml` / CI `P800.yml` 三处均设 1）⇒ 是否可关须上游确认，**本方向不擅自改锁定口径** | `P800/docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md`；`P800/README.md` §3 |
| 为何用最简形态而非 Megatron-LM-FL | 设备层要单独暴露设备因素；Megatron 会引入并行/通信/显存调度，把设备问题与框架问题混在一起。**规模化验收载体**（Megatron-LM-FL 训练 / vllm-plugin-FL 推理）归 **framework-adapter 方向**；政策文档明确"两者对 torch_fl 零引用，全部构建在厂商 torch 插件上" | `dev/memory/docs/common/policy_设备层路线变更指南.md:53,63`；`dev/framework-adapter/README.md` |

---

## 4. 结果可比性说明（不要误读）

| 对比项 | 可比？ | 说明 |
|---|---|---|
| 训练吞吐 2117（910C）vs 3482（P800） | ✅ 同构可比 | 同脚本、同超参、同 world_size；差异来自芯片与通信栈。**不构成性能结论** |
| 向量维度 / 语义区分度 | ✅ 高可比 | 910C 前向 0.638 / 服务化 0.4123；P800 前向 0.6392 / 服务化 0.4102 —— **几乎一致，符合"同一模型+同一池化口径"的预期** |
| 推理吞吐 108（910C 服务化）vs 30.70（P800） | ⚠️ **不宜直接比** | 两边启动参数不同（P800 多了 `--enforce-eager`）、算子路径不同（vendor vs 官方栈）、且 P800 为共享机单卡。**只用于证明"服务化形态跑通且语义正确"** |
| 910C 前向 66.32 vs P800 53.12 句/s | ⚠️ 参考 | 同为前向形态但硬件/环境不同 |

---

## 5. 环境前提与坑（复用清单）

| # | 坑 | 适用 | 依据 |
|---|---|---|---|
| 1 | 带卡容器并发 ≤3；两腿串行 | 910C | `stack.lock` `rules` 第 1 条 |
| 2 | 训练腿 `AUTOLOAD=0` + 先 `import torch_fl` 再 `import torch` | 910C 训练腿 | `stack.lock` `per_leg.train` |
| 3 | 服务化推理**禁用** `VLLM_PLUGINS=fl`，并 `unset VLLM_PLUGINS` | 910C 推理腿 | §2.4 坑 A5 |
| 4 | `PYTHONPATH=/env/FlagGems/src` 是**硬前置**：`vllm_fl` import 时依赖 `flag_gems.runtime.backend.device.DeviceDetector`，而 site-packages 里的 `flag_gems` 安装不完整 → 缺该变量时 vLLM 直接报 `Failed to infer device type`（**看着像设备问题，其实是包问题**） | P800 推理服务化 | 本次实测（2026-09-20）；`vllm_fl/utils.py:9` |
| 5 | HF 缓存必须指向 `snapshots/<hash>` | 两实例（P800 强制） | §2.2 |
| 6 | 用卡前 `xpu-smi` 挑空闲卡；`kill -9` 才能清理挂死进程（SIGTERM 无效） | P800 | `P800/README.md` §3 |
| 7 | SSH 非标准端口 **26008**；需 `docker` 组与自有可写目录 | P800 | `P800/README.md` 环境差异表 |

---

## 6. 可复现命令

**910C（容器内）**

```bash
# 设备自检 / conformance
python3 runtime/smoke_runtime.py
python3 runtime/conformance/runner.py --backend ascend
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 runtime/conformance/runner.py --backend flagos
python3 runtime/conformance/runner.py --backend ascend --cases infer_cases

# 训练腿（2 卡）
DC_BACKEND=flagos torchrun --nproc_per_node=2 runtime/proto/proto_train_leg.py

# 推理腿（服务化：先起服务，再验证）
vllm serve <model> --runner pooling --convert embed --port 8100
python3 runtime/proto/proto_infer_serve.py --backend ascend
python3 runtime/proto/proto_infer_leg.py   --backend ascend

# 错误注入 → 恢复闭环（两后端）
python3 runtime/proto/proto_error_recovery_loop.py --backend ascend
python3 runtime/proto/proto_error_recovery_loop.py --backend flagos
```

**P800（容器内；`DEV` 换成本次挑中的空闲卡）**

```bash
# 设备自检 / conformance（同一条命令，只改 backend）
CUDA_VISIBLE_DEVICES=$DEV python3 runtime/smoke_runtime.py --backend kunlun
CUDA_VISIBLE_DEVICES=$DEV python3 runtime/conformance/runner.py --backend kunlun

# 训练腿（2 卡）
CUDA_VISIBLE_DEVICES=6,7 DC_BACKEND=kunlun FLAGCX_ADAPTOR=klx \
DC_ROOT=/workspace/prototype DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B \
DC_OUT_DIR=/workspace/out \
python3 -m torch.distributed.run --standalone --nproc_per_node=2 runtime/proto/proto_train_leg.py

# 推理腿：单卡前向
DEV=6 bash P800/probes/F_infer_leg.sh

# 推理腿：服务化（脚本内含启动 → 就绪 → 验证 → 停机 → 用卡复查）
DEV=6 bash P800/probes/F2_vllm_serve.sh

# 错误闭环：两设置对照
DEV=6 bash P800/probes/G_error_loop.sh
```

---

## 7. 未覆盖 / 待补（诚实标注）

| # | 项 | 说明 |
|---|---|---|
| 1 | 910C 训练腿 `transformers` 版本 | 镜像内**补装**，版本未在仓内逐次记录（指针：`dev/images/ascend-train-comm/v1/`） |
| 2 | 两实例实际使用的 `LR` | 脚本默认 `1e-5`，逐次取值未记录 |
| 3 | 8 卡 / 跨机规模 | 设备方向只负责 ≤2 卡最小规模；8 卡归分布式方向（P800 的 NUMA 拓扑已实测：XPU0-3 / XPU4-7） |
| 4 | Megatron-LM-FL / TransformerEngine-FL / verl-FL | **均未验证**（全 `dev/` 仅 2 处提及，属政策文档举例） |
| 5 | P800 镜像入锁 | 第二实例镜像尚未进 `stack.lock`；诉求已在 `STATUS.md` 登记 |
| 6 | 910C 遗留 4 项 | real 多卡多进程压测 / 芯片级错误真实触发 / 流优先级调度效果 / 训练侧完整 epoch 吞吐复测 |
| 7 | P800 厂商缺陷上报渠道 | 待定：直连昆仑芯 vs 经总组转达 |
