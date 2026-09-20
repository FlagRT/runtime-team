# 910C / P800 两实例 · 验证配置与依据（参考手册）

> 版本：v1.1（2026-09-20）｜ 负责人：Kistich（hliu553）
> **用途**：给后续接入者（第三家芯片 / 框架方向 / 验收方）一份"我们到底是怎么跑的、为什么这么配"的参考。
> **写法约定**：每条配置都给出 ①**为什么这么定**（设计意图）+ ②**实测依据**（现象 / 数字 / 命令 / 报错原文），
> 依据一律写成**可独立阅读的事实**，不写成"请看某文件的某章"。
> 凡本方向未取得证据的，明确标注「未记录 / 未验证」，不臆造。

---

## 1. 一页速查（两实例配置对照）

| 维度 | 910C（第一实例，已完成） | P800（第二实例，阶段 0–4 完成） |
|---|---|---|
| 芯片 | 昇腾 910C（4 NPU / 8 chip，HBM 64 GB，CANN 9.0.0） | 昆仑芯 P800（8 卡，96 GB/卡） |
| **设备 API 命名空间** | `npu`（`torch_npu`） | **`cuda`**（XPytorch + `torch_xray` 符号重写；`torch.xpu` 不可用） |
| **选卡变量** | `ASCEND_RT_VISIBLE_DEVICES` | **`CUDA_VISIBLE_DEVICES`** |
| 训练腿镜像 | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`（38.3 GB，本机已有） |
| 训练腿容器 | `flagos-proto-train-910c` | `hliu553-device-context-p800` |
| 训练腿 Python | `/usr/local/python3.11.15/bin/python3` | conda env `python310_torch29_cuda`（Python 3.10.18） |
| 推理腿镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | 与训练腿同容器 |
| **训练框架** | `transformers`（`AutoModelForCausalLM`）+ `AdamW` + `torch.distributed` | 同一个脚本 |
| **训练设备后端 / 通信后端** | `runtime.use("flagos")`（torch_fl）／通信后端 `flagos` | `runtime.use("kunlun")`／通信后端 `cpu:gloo,cuda:flagcx` |
| **推理框架（服务化）** | vLLM 0.20.2（昇腾官方镜像自带） | vLLM 0.13.0 + vllm-plugin-FL（`VLLM_FL_PLATFORM=kunlunxin`） |
| 模型 | `Qwen/Qwen3-Embedding-0.6B` | 同一模型（走共享缓存快照路径） |
| 训练超参 | 50 步 / batch 4 / seq 128 / world_size 2 | 同上 |
| 推理服务参数 | `--runner pooling --convert embed --port 8100` | 同上 + `--max-model-len 4096 --gpu-memory-utilization 0.25 --enforce-eager` |
| 训练结果（2 卡） | **2117.4 tok/s**、24.18 s、6/6 PASS | **3482.2 tok/s**、14.7 s、6/6 PASS |
| 推理（单卡前向） | 66.32 句/s、平均 45.24 ms、10/10 PASS | 53.12 句/s、p50 56.17 ms、13/13 PASS |
| 推理（服务化） | **108.35 句/s**、p50 27.4 ms、10/10 PASS | 30.70 句/s、p50 96.4 ms、10/10 PASS |
| 错误注入 → 恢复闭环 | `ascend`、`flagos` 两个后端各自跑通 | 设 / 不设 `XPU_EVENT_KL3_ENABLE` 两组均跑通且结果一致 |

---

## 2. 逐项配置与依据

### 2.1 镜像

| 实例 | 配置 | 为什么这么定 | 实测依据 |
|---|---|---|---|
| 910C 训练腿 | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | 全组统一基座，由总组裁定；各方向只消费不自建。它是**唯一的结论性环境依据**——日常调试容器里跑出的结果不作为验收依据 | 镜像内已内置通信库（FlagCX 0.13.0）与算子库，`transformers` 需容器内补装；标注为 `release.stage=candidate`，属"torch_fl 例外线"，华为昇腾官方主线是 `torch_npu`。可凭仓内 `Dockerfile.repro` + assets 重建，并已用 `docker save` 落盘备份 |
| 910C 推理腿 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | 华为昇腾官方推理镜像，vLLM 0.20.2 与该镜像自带的 triton-ascend 插件成套发布，避免自建组合 | 已验证可从上游拉取，digest 为 `sha256:5cf8a2b6db8b06eb1bc7fc7d191d667aebf2b197351bdba13f776918c11ec7a7` |
| P800 | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608` | 该镜像**在本机镜像库里已经存在**（38.3 GB），无需联网重新获取 | 昆仑芯官方手册要求先 `pull` 59.9 GB 镜像、再 `load` 32 GB 产出包；实测这两步**全部跳过**——镜像在位，且 FlagGems 源码也已放在容器内 `/env/FlagGems`，接入当天零下载 |

> **纪律**：P800 目前尚未进入全组锁定基座（第二实例镜像未入锁），诉求已按流程登记待总组裁定。

### 2.2 模型

| 项 | 值 | 为什么这么定 | 实测依据 |
|---|---|---|---|
| 验收模型 | `Qwen/Qwen3-Embedding-0.6B` | 全组 9 月统一验收模型：向量检索任务，输出是可量化的向量，便于跨芯片比对"语义区分度"这类与硬件无关的指标 | 同一模型在两实例上给出近乎一致的语义区分度（服务化 0.4123 / 0.4102），说明该指标可作跨芯片一致性判据 |
| 910C 路径 | `/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B` | 该机数据盘本地目录 | 两条腿的运行结果 JSON 中 `model` 字段即此路径 |
| P800 路径 | `/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | **必须给到 `snapshots/<hash>` 这一层**：该目录是 HuggingFace 缓存根，只含 `blobs/`、`refs/`、`snapshots/` | 直接传缓存根目录会报 `Unrecognized model ... Should have a model_type key`；实测改为快照路径后加载正常。脚本已内置 `resolve_model()` 自动把 `models--xxx` 解析到 `snapshots/<hash>`，两种传法都能用 |
| 其他模型 | 910C 服务化健康检查另用 `Qwen3-4B` | 生成类任务与 embedding 任务分开验证，避免用同一条判据混测两种能力 | 生成侧健康检查输出 68.4 tok/s（256 tokens）、`ready_wait 55 s`；embedding 侧另用 0.6B 模型，两者判据不混用 |

### 2.3 训练形态与参数

| 项 | 值 | 为什么这么定 | 实测依据 |
|---|---|---|---|
| 框架 | `transformers` + `AdamW` + `torch.distributed` | **刻意用最小形态**。设备方向要回答的是"设备层能力是否成立"，只有最小形态才能把设备因素单独暴露出来——框架越厚，设备问题越容易被框架行为掩盖 | 反面例证很实在：P800 的挂死问题是在纯 torch 形态下才定位到函数级（三处自旋点全部位于厂商 `libxpucuda.so`，偏移 `+0x94080`）。若一开始就跑重型框架，设备问题与框架问题会混在一起 |
| 规模 | 2 卡（world_size = 2） | 设备方向只负责**最小正确规模（≤2 卡）**；8 卡到多机属分布式方向，避免各方向挤在同一规模重复验证 | 两实例都是 2 卡 DDP，且三项集合通信对照（all_reduce / all_gather / P2P）在两实例上都全对 |
| 超参 | 50 步 / batch 4 / seq 128 | 步数要足够多，让集合通信**反复触发**——挂死类缺陷往往在第 n 次通信才出现，短跑测不出来 | P800 实测挂死点游走在第 3–4 次、第 41–50 次、第 101–120 次通信之间；同一次实验里单次 2.38 GB 大通信 0.336 s 能过，但反复调用会非确定性挂住。50 步是能稳定覆盖该区间的档位 |
| 学习率 | 环境变量 `LR`，默认 `1e-5` | 0.6B 小模型微调的稳定区间 | 两实例各 50 步 loss 均单调下降无 NaN：910C `15.4497 → 11.15`，P800 `15.4488 → 11.1481`。**⚠️ 两次实际使用的 LR 取值未记录**（脚本默认 1e-5，运行时是否覆盖未留痕）——后续跑请顺手记环境变量 |
| 910C 设备后端 | `flagos`（即 torch_fl） | 该锁定镜像**禁止 `torch_npu` 与 Torch-FL 运行时共存**，因此训练腿只能走 torch_fl；这是全组登记在案的权宜例外，不代表路线变更 | 镜像自带校验脚本在两者共存时直接报错；可用的调用方式是 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 且**先 `import torch_fl` 再 `import torch`**。我们的原型通过 `runtime/backends/flagos` 接入 |
| P800 设备后端 | `kunlun` | 昆仑芯的设备 API 走 `torch.cuda` 命名空间，因此后端按 `cuda` 命名空间实现 | 四条独立证据：① `torch.xpu.is_available()` 返回 False（`AssertionError: Torch not compiled with XPU enabled`）；② `torch.cuda.device_count()` 返回 8；③ 编译标志为 `USE_XPU=OFF`；④ 官方 xpu3.6 单测的 `--device` 默认值就是 `'cuda'`。机制为 XPytorch 兼容层 + `torch_xray` 符号重写 |
| P800 通信后端 | `cpu:gloo,cuda:flagcx`，另设 `FLAGCX_ADAPTOR=klx` | 实测**只有 flagcx 这一条路可用**，其余三种后端都不可用 | `nccl` 挂死；`xccl` 未编译，报 `Distributed package doesn't have XCCL built in`；`kccl` 无响应。可用路径要求**显式 `import flagcx`**（否则 `cuda:flagcx` 未注册），CPU 侧用 gloo |
| 通信后端名差异 | 910C 注册为 `flagos`；P800 注册为 `flagcx` | 同一份 FlagCX 在不同芯片上注册的后端名不同——换芯片不只换设备命名空间，连集合通信后端名也要换 | 910C 训练腿镜像里 flagcx 注册名为 `flagos`；P800 上必须用 `flagcx` 且先 `import flagcx`。这一条是接入时最容易踩的坑之一，已列入接入检查清单 |

### 2.4 推理形态与参数

| 项 | 值 | 为什么这么定 | 实测依据 |
|---|---|---|---|
| 形态 | 两种：① 单卡前向（`transformers`）② 服务化（vLLM OpenAI 兼容接口） | 前向形态跑得快、变量最少，用于快速暴露设备因素；服务化形态对齐"推理服务"验收口径（吞吐、时延、超长输入防御） | 两实例都做了这两种形态，且两形态的向量维度/范数/区分度一致（同一模型同一池化口径） |
| 服务参数 | `vllm serve <model> --runner pooling --convert embed --port 8100` | 该版本 vLLM **没有 `--task` 参数**，embedding 服务必须用 `--runner pooling --convert embed`；两实例统一该口径才能横向比对 | 先按 `--task embed` 启动，报 `vllm: error: unrecognized arguments: --task embed`；改查参数表确认可用值是 `--runner {auto,draft,generate,pooling}` 与 `--convert {auto,classify,embed,none,reward}` |
| P800 追加参数 | `--max-model-len 4096 --gpu-memory-utilization 0.25 --enforce-eager` | `--enforce-eager` 关闭图捕获以换稳定（**代价见 §4，不作性能结论**）；`max-model-len 4096` 与 910C 保持一致，好让"超长输入"用同一条判据 | 超长输入 6001 tokens > 4096 时服务返回 HTTP 400 并给出 `This model's maximum context length is 4096 tokens`；该错误体经统一分级得到 L2_PARAM / raise，与 910C 判据完全一致。服务就绪耗时 25 s，停机后卡显存归零 |
| 910C 推理框架 | vLLM 0.20.2（官方镜像自带），**显式禁用 vllm-plugin-FL** | 开启该插件会破坏昇腾侧的 platform 选择，服务根本起不来；根因是**这个插件没有 ascend 后端**（其文档自述 currently CUDA only） | 设 `VLLM_PLUGINS=fl` 后 `current_platform.device_type` 变为空，抛 `RuntimeError: Device string must not be empty`。因此启动脚本里写死 `unset VLLM_PLUGINS`——这条纪律来自实测踩坑，不能删 |
| P800 推理框架 | vLLM 0.13.0 + **vllm-plugin-FL**（容器内以 editable 方式安装） | 与昇腾相反：昆仑芯**需要**这个插件提供 FL platform 才能被 vLLM 识别 | 加载后 `from vllm.platforms import current_platform` 得到 `<vllm_fl.platform.PlatformFL>`，插件日志显示 `Platform plugin fl is activated`、`OpManager initialized: 9 ops with 29 implementations`。⚠️ 同一插件在两个芯片上"一个禁用、一个必需"，这条对第三家芯片选型很关键 |

### 2.5 选卡与并发

| 实例 | 规则 | 为什么这么定 | 实测依据 |
|---|---|---|---|
| 910C | 带卡容器**并发上限 3**；训练腿与推理腿**串行**执行，跑完一条停掉再起下一条 | 超限后设备会"凭空消失"，且报错信息指向驱动/日志，很容易误判为镜像或代码问题 | 4 个并发容器时容器内 `acl.init()` 返回 500000，并伴随 `DrvMngGetConsoleLogLevel failed(ret=4)`、`path string is NULL`，表现为 `torch.npu.device_count()=0`；降到 3 个后立即恢复。已作为最高优先级规则置顶 |
| P800 | 选卡用 `CUDA_VISIBLE_DEVICES`；共享机上先 `xpu-smi` 挑空闲卡并记录用卡 | 该机是共享机（曾达 25 人在线 / 22 个容器），被他人占用的卡会导致设备侧报错，看起来像框架缺陷 | 实测 `CUDA_VISIBLE_DEVICES=2` → `device_count()=1`；`=2,5` → `2`。反面教训：曾把首次失败判为"flagcx 在 P800 上适配缺陷"，换到空闲的卡 6/7 后三类通信全通过——真因是卡被其他租户占用（观察到他人在卡 1 上跑 166→502 MiB）。该"缺陷"结论已撤销 |

### 2.6 算子路径

| 实例 | 配置 | 为什么这么定 | 实测依据 |
|---|---|---|---|
| 910C 训练腿 | 镜像内置 FlagCX 与算子库 | 锁定基座已包含，无需额外配置 | 训练腿 50 步全程无算子类报错，通信三类对照全对 |
| P800 | `VLLM_FL_PREFER=vendor` + `USE_FLAGGEMS=0` + `GEMS_VENDOR=kunlunxin` + `KLX_USE_AUTOTUNE=0` | 取 **vendor 算子路径**，避开 FlagGems 路径，从而不与已知厂商缺陷（见 §3）产生依赖，与本方向"不设 `XPU_EVENT_KL3_ENABLE`"的口径保持一致 | 该组合下服务正常起来并跑完 10/10；另实测一个陷阱——即便不用 FlagGems 算子，插件在 import 阶段仍依赖 `flag_gems.runtime.backend.device.DeviceDetector`，而 site-packages 里那份 flag_gems 安装不完整（`pip show` 看得见包，但该子模块导不进来），缺 `PYTHONPATH=/env/FlagGems/src` 时 vLLM 直接报 `Failed to infer device type` 退出 |

---

## 3. 关键决策的依据链（"为什么不是别的"）

| 决策 | 结论 | 支撑它的实测事实 |
|---|---|---|
| 设备注册路线 | **Route A：各芯片厂商官方 torch 插件为主线**；910C 训练腿的 torch_fl 是镜像约束下的例外 | 上层框架（vllm / sglang / Megatron / TransformerEngine / verl / FlagScale）**对 torch_fl 零引用**，各芯片设备层现状都是厂商插件；Megatron-LM-FL 等框架的多芯片适配也全部构建在厂商 torch 插件之上，自建等价实现为人月级。全组已把 Route A 定为主线，torch_fl 降为预研支线、不承担交付 |
| 910C 推理为何不用 vllm-plugin-FL | 该插件**无 ascend 后端**，一旦启用即破坏 platform 选择 | 启用后 `current_platform.device_type` 为空 → `RuntimeError: Device string must not be empty`；插件自述 currently CUDA only。故 910C 推理腿脚本固定 `unset VLLM_PLUGINS` |
| P800 推理为何用 vllm-plugin-FL | 昆仑芯**需要**它提供 FL platform，否则 vLLM 认不出设备 | 加载后 platform 为 `<vllm_fl.platform.PlatformFL>`；不加载则 vLLM 报 `Failed to infer device type`（另需 `PYTHONPATH` 见 §2.6） |
| P800 通信为何只有 flagcx | 其余三种后端在本环境都不可用 | `nccl` 挂死；`xccl` 报 `Distributed package doesn't have XCCL built in`；`kccl` 无响应。唯一可用路径需 `import flagcx` + `init_process_group("cpu:gloo,cuda:flagcx")` + `FLAGCX_ADAPTOR=klx` |
| 为何不设 `XPU_EVENT_KL3_ENABLE` | 该变量与设备侧集合通信组合会**概率性永久挂死**（厂商运行时层缺陷）；规避手段经单变量对照验证有效 | 26 次单变量探针运行：两个要素（`XPU_EVENT_KL3_ENABLE=1` ＋ 存在设备侧集合通信）缺一不挂；该组合 18 次运行 16 次挂死（≈89%）。挂死点游走于第 3–120 次通信之间，与数据量、形状、reduce op、用卡对、同步间隔**均无关**。用带真值校验的探针复测（不设变量时 `2^120` 精确匹配、`rel_err = 0.000e+00`）确认"通过是真通过"。A/B 对照：同一脚本同一用卡，不设 → 退出码 0；设 1 → 只到 `[step 0]` 即挂死、退出码 124。⚠️ 该变量同时是 FlagGems 昆仑芯后端的官方推荐变量，故**是否可关须上游确认，本方向不擅自改锁定口径** |
| 为何用最简形态而非 Megatron-LM-FL | 设备层需要单独暴露设备因素；规模化验收载体（Megatron-LM-FL 训练 / vllm-plugin-FL 推理）归框架适配方向 | 全仓检索：Megatron 仅作为"上层框架"被政策文档提及两次，**无任何验证脚本或证据**；而 P800 的关键缺陷正是在纯 torch 形态下才定位到函数级（三处自旋点位于厂商 `libxpucuda.so`） |

---

## 4. 结果可比性说明（不要误读）

| 对比项 | 可比？ | 说明 |
|---|---|---|
| 训练吞吐 2117.4（910C）vs 3482.2（P800）tok/s | ✅ 同构可比 | 同脚本、同超参、同 world_size、同为两卡合计；差异来自芯片与通信栈。**仅作形态验证，不构成性能结论** |
| 语义区分度（服务化 0.4123 / 0.4102；前向 0.638 / 0.6392） | ✅ 高可比 | 同一模型、同一池化口径（last-token 池化 + query 指令前缀）；两实例近乎一致，正是"设备抽象不改变数值语义"的体现 |
| 推理吞吐 108.35（910C）vs 30.70（P800）句/s | ⚠️ **不宜直接比** | 启动参数不同、算子路径不同（vendor vs 官方栈）、P800 为共享机单卡。**P800 侧已做三层拆解**（同卡、同模型、同批文本 3 句 × 5 轮）：单卡前向 53.12 句/s（p50 56.17 ms）→ vLLM offline 引擎 35.63（p50 84.14 ms）→ 服务化 HTTP 30.70（p50 96.4 ms）。⇒ **约 70% 的差距来自 vLLM 引擎执行路径、约 30% 来自 HTTP 与服务端**；`--enforce-eager` **不是**瓶颈——去掉它反而降到 22.83 句/s（p50 132.3 ms）。插件本身的贡献**未能单独隔离**（关掉插件设备就认不出，无法做 A/B） |
| 单卡前向 66.32（910C）vs 53.12（P800）句/s | ⚠️ 参考 | 同为前向形态但硬件与环境不同 |

---

## 5. 环境前提与坑（复用清单）

| # | 坑 | 适用 | 实测现象 / 触发条件 |
|---|---|---|---|
| 1 | 带卡容器并发 ≤3，两条腿串行 | 910C | 4 个并发时 `acl.init()` 返回 500000、`torch.npu.device_count()=0`，设备"凭空消失"；降到 3 个即恢复 |
| 2 | 训练腿 `AUTOLOAD=0` 且**先 `import torch_fl` 再 `import torch`** | 910C 训练腿 | 该镜像禁止 `torch_npu` 与 Torch-FL 共存，自带校验脚本直接报错 |
| 3 | 服务化推理 `unset VLLM_PLUGINS` | 910C 推理腿 | 设 `VLLM_PLUGINS=fl` → `current_platform.device_type` 为空 → `RuntimeError: Device string must not be empty` |
| 4 | `PYTHONPATH=/env/FlagGems/src` 是**硬前置** | P800 推理服务化 | 缺该变量时 `vllm_fl` 导入 `flag_gems.runtime.backend.device` 失败，vLLM 报 `Failed to infer device type` 退出——**看着像设备问题，其实是 Python 包问题** |
| 5 | HF 缓存路径必须到 `snapshots/<hash>` | 两实例（P800 强制） | 传缓存根目录报 `Unrecognized model ... Should have a model_type key` |
| 6 | 用卡前 `xpu-smi` 挑空闲卡；挂死进程必须 `kill -9` | P800 | `timeout` 发出的 SIGTERM **无法中断**这类挂死进程（挂死点持 GIL 自旋、信号被推迟），曾观察到进程存活 73 分钟；清理后要复查 `xpu-smi` 确认卡释放 |
| 7 | SSH 非标准端口 **26008**；需 `docker` 组 + 自有可写数据目录 | P800 | `~/.ssh/config` 缺 `Port` 行时表现为连接被拒，易误判为网络/VPN 问题 |

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

# 推理腿：服务化（脚本内含 启动 → 就绪等待 → 验证 → 停机 → 用卡复查）
DEV=6 bash P800/probes/F2_vllm_serve.sh

# 错误闭环：设 / 不设 XPU_EVENT_KL3_ENABLE 两设置对照
DEV=6 bash P800/probes/G_error_loop.sh
```

---

## 7. 未覆盖 / 待补（诚实标注）

| # | 项 | 说明 |
|---|---|---|
| 1 | 910C 训练腿的 `transformers` 版本 | 该镜像是"容器内补装 transformers"，版本当次未留痕；下次跑请记录 |
| 2 | 两实例实际使用的 `LR` | 脚本默认 `1e-5`，实际是否被环境变量覆盖未留痕；下次跑请记录 |
| 3 | 8 卡 / 跨机规模 | 设备方向只负责 ≤2 卡最小规模。P800 的 NUMA 拓扑已实测（XPU0-3 属 NUMA0、XPU4-7 属 NUMA1；组内走 XL 私有链路、跨组走 SYS） |
| 4 | Megatron-LM-FL / TransformerEngine-FL / verl-FL | **本方向均未验证**；全仓检索仅见政策文档提及，无脚本与证据 |
| 5 | P800 镜像入锁 | 第二实例镜像尚未进全组锁定基座，诉求已登记待总组裁定 |
| 6 | 910C 遗留 4 项 | real 多卡多进程压测 / 芯片级错误真实触发 / 流优先级调度效果 / 训练侧完整 epoch 吞吐复测 |
| 7 | P800 厂商缺陷上报渠道 | 待定：直连昆仑芯支持，或经总组转达 |
