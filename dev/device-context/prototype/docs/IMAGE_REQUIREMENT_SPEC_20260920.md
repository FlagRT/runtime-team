# 设备方向 · 镜像需求说明书

> 版本：v1.0（2026-09-20）｜ 负责人：Kistich（hliu553）｜ 提交对象：总组（基座裁定）
> **用途**：把两实例（910C / P800）的实测结论，翻译成"我们对基座镜像的需求"，供总组裁定；
> 同时列出**上游可查验的官方镜像文档**，便于逐条比对。
> **写法约定**：需求逐条给"实测依据"（现象/数字/报错原文），不写成"见某文件某节"。

---

## 1. 先给可查验的上游官方文档（比较基准）

| # | 文档 | 位置 | 给什么 |
|---|---|---|---|
| 1 | **FlagTree 多后端支持表**（官方 README） | `https://github.com/flagos-ai/FlagTree` | 每个后端基于哪条 Triton 版本线、分属哪个 protected 分支；官方原话：**"避免环境兼容问题的最佳实践是使用 User Manual 里推荐的镜像"** |
| 2 | **User manual for xpu**（KLX 昆仑芯） | `https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-xpu` | P800 推荐镜像、容器启动参数、驱动版本、flagtree wheel 版本；**并明确"测试前需 `export XPU_EVENT_KL3_ENABLE=1`"** |
| 3 | **User manual for ascend**（华为昇腾） | `https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ascend` | 910B/910C 推荐镜像（含 vLLM 版）、CANN 安装、Qwen vLLM benchmark 步骤 |
| 4 | **vllm-plugin-FL 官方文档** | `https://community.qtorque.io/flagos-ai/vllm-plugin-FL` | 各芯片要配的 flagtree wheel（如 `flagtree==0.6.1rc1+ascend3.5`、`...=0.6.1+metax3.6`）；`TRITON_ALL_BLOCKS_PARALLEL=1`；**昇腾需 eager 执行**；`USE_FLAGGEMS=0` 走原生 CUDA 算子 |
| 5 | 华为昇腾官方推理镜像 | `quay.io/ascend/vllm-ascend` | 厂商官方 vLLM 移植版（910C 推理腿所用） |

**官方多后端版本线（决定"同一镜像不能跨芯片"）**：main 分支（Triton 3.6）覆盖 NVIDIA / AMD / 燧原 / 天数 / 海光 / 摩尔线程 / 达摩院 / 辉羲 / 沐曦 / 曦望 / **KLX 昆仑芯** / **T-Head 平头哥** / 进迭时空 / 清微；`triton_v3.5.x` 是**华为昇腾**；`triton_v3.2.x` 是**寒武纪**。

---

## 2. 我们两实例镜像 vs 官方推荐镜像（差异对照）

| 实例 | 我们实际用的 | 官方手册/厂商推荐 | 差异与原因 |
|---|---|---|---|
| **910C 训练腿** | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64`（组内） | `harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2`（19.2 GB） | **版本线一致**（CANN 9.0.0 / py3.11 / torch 2.10 / vLLM 0.20.2），但血统不同：官方镜像**不含 FlagCX**（我们训练腿的集合通信依赖它），且官方镜像带 vLLM（训练腿用不上） |
| **910C 推理腿** | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（**华为昇腾官方**） | 同上 FlagTree 镜像（含 vllm0.20.2） | vLLM 版本同为 0.20.2；我们用华为官方移植版（自带 ascend 平台，可直接 `vllm serve`） |
| **P800** | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`（38.3 GB，digest `sha256:cd53efa4…`） | `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-**base**`（59.9 GB pull / 32 GB load） | **同系列不同变体**：官方是 `-base`，我们用 `flaggems-main-dev`（含 FlagGems main 分支 dev 版）。本机还存有 `...-base-ssh` 变体（94.5 GB）。**官方推荐的是 `-base`** |

> 结论：**两个实例都没有用官方手册推荐的那个镜像**。910C 是"模型组为通信而自建"，P800 是"用了同系列的 FlagGems dev 变体"。
> ⇒ 这一点在向总组提需求时必须先讲清楚，否则会出现"你们怎么不用官方镜像"的质疑。

---

## 3. 基于实测结论的镜像需求（逐条带依据）

### 3.1 硬性需求（不满足则跑不通）

| # | 需求 | 实测依据 |
|---|---|---|
| **H1** | 设备可见 + 厂商 torch 栈可导入 | 这是所有工作的前提。昆仑芯侧四条独立证据表明设备 API 走 `torch.cuda` 命名空间：`torch.xpu.is_available()` 报 `AssertionError: Torch not compiled with XPU enabled`；`torch.cuda.device_count()` 返回 8；编译标志 `USE_XPU=OFF`；官方单测 `--device` 默认值即 `'cuda'`。**若镜像里设备不可见，smoke 直接跑不了** |
| **H2** | 训练腿：镜像内必须有**可用的集合通信路径** | P800 实测四种后端只有一种可用：`nccl` 挂死；`xccl` 未编译（报 `Distributed package doesn't have XCCL built in`）；`kccl` 无响应；唯一可用是 `flagcx`（需显式 `import flagcx` + `init_process_group("cpu:gloo,cuda:flagcx")` + `FLAGCX_ADAPTOR=klx`）。⇒ **镜像不带 FlagCX 或等价通信库，训练腿无法验证** |
| **H3** | 推理腿：镜像内必须有**让 vLLM 认出设备的路径** | P800 容器里是社区 vLLM（`vllm/platforms/` 只有 cpu/cuda/rocm/tpu/xpu，**无 kunlun**），`current_platform` 初始为 `UnspecifiedPlatform`；靠 vllm-plugin-FL 提供 `PlatformFL` 后才可用。⇒ 若镜像既无厂商 vLLM 移植版、又无平台插件，推理腿无法服务化 |
| **H4** | 依赖可完整导入（避免"包在但子模块导不进来"） | P800 推理服务化的硬前置：`PYTHONPATH=/env/FlagGems/src`。原因：`vllm_fl` import 时依赖 `flag_gems.runtime.backend.device.DeviceDetector`，而 site-packages 里那份 `flag_gems`（`pip show` 显示 5.3.4.post1.dev12）**该子模块不可导入**，缺该变量时 vLLM 直接报 `Failed to infer device type` 退出 —— **看着像设备问题，其实是 Python 包问题** |
| **H5** | 容器启动参数与宿主机驱动匹配 | 官方 xpu 手册要求带 `--device=/dev/xpu0..7`、`--device=/dev/xpuctrl`、`--device /dev/fuse`、`--privileged`、`--shm-size=256g`、`--ulimit memlock=-1` 等；宿主机驱动版本需匹配（官方记 host `Driver 5.0.21.47`、容器内 `515.58`）。**设备节点漏挂会表现为"卡不可见"** |

### 3.2 期望需求（影响可信度与效率）

| # | 需求 | 实测依据 |
|---|---|---|
| **E1** | **环境变量口径必须全组统一裁定**（当前存在冲突） | 官方 xpu 手册明确要求"**测试前需 `export XPU_EVENT_KL3_ENABLE=1`**"；但我们实测：该变量 **＋ 设备侧集合通信** 组合下，18 次运行 16 次**永久挂死（≈89 %）**，挂死点游走于第 3–120 次通信之间，与数据量/形状/reduce op/用卡对/同步间隔均无关；三处自旋点全部位于厂商 `libxpucuda.so`（偏移 `+0x94080`）。A/B 单变量对照：不设 → 退出码 0；设 1 → 只到 `[step 0]` 即挂死、退出码 124。**⇒ 官方手册要求与真实多卡训练实测相悖，这是必须上报厂商并由总组统一口径的点**，不能各方向自行决定 |
| **E2** | 镜像应自带**完整的 FlagGems 安装**（或给出正确 `PYTHONPATH`） | 见 H4；当前需要人工指定 `PYTHONPATH=/env/FlagGems/src` 才能起来 |
| **E3** | python 环境应可直接激活（conda/venv 路径明确） | P800 容器默认 `python` 无 torch；实际环境在 conda env `python310_torch29_cuda`（py3.10.18 / torch 2.9.0+cu129 / transformers 4.57.1 / vLLM 0.13.0）。**不知道路径时第一步就卡住** |
| **E4** | 镜像有 digest 且可归档 | 910C 推理腿有 digest（`sha256:5cf8a2b6…`）、P800 有 digest（`sha256:cd53efa4…`）；910C 训练腿无 registry、靠 `docker save` + 重建配方。**有 digest 是"结论可背书"的前提** |
| **E5** | 官方推荐镜像与"我们需要的组件"最好合一 | 官方 910C 镜像不含 FlagCX（训练腿必需），导致我们只能用组内自建镜像。若官方镜像能内置 FlagCX（或提供可复现的叠加上层），就不必维护两套血统 |

### 3.3 可协商（不影响本方向验收）

| # | 项 | 说明 |
|---|---|---|
| N1 | vLLM 版本 | 三实例分别为 0.20.2（910C）/ 0.13.0（P800 容器）/ 0.11.0（昆仑芯官方 vllm-kunlun）。**跨实例性能数字不要混用**：实测同一模型在 P800 上"引擎路径"本身比裸 transformers 慢约 50 %（p50 56.17 → 84.14 ms），HTTP 再叠约 15 %（→ 96.4 ms），这部分是形态与栈的差异，不是芯片差异 |
| N2 | 算子库 | 本方向不需要（不实现算子）。P800 上我们刻意取 vendor 路径（`VLLM_FL_PREFER=vendor` + `USE_FLAGGEMS=0`）以避开 FlagGems 路径与 E1 的变量依赖 |
| N3 | 镜像大小 / 是否含 ssh 等便利组件 | 例如 P800 同系列有 `-base`（32 GB load）与 `-base-ssh`（94.5 GB）两种变体，按需选 |

---

## 4. 请总组裁定的三件事

| # | 事项 | 我们的诉求 | 依据 |
|---|---|---|---|
| 1 | **P800 镜像入锁** | 把 `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`（digest `sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975`）或其官方 `-base` 对应版本纳入基座 | P800 现已完成阶段 0–4（conformance 13+6、训练腿 6/6、推理腿 13/13 与服务化 10/10、错误闭环两设置一致），但**结论建立在一个未入锁的镜像上**；910C 的同类结论有锁定基座背书，P800 没有 |
| 2 | **`XPU_EVENT_KL3_ENABLE` 口径冲突** | 请总组明确"锁定口径是设还是不设"，并支持向昆仑芯上报该冲突（官方手册要求设 1，实测该组合 89 % 挂死） | 见 E1；这同时是厂商缺陷上报的核心证据，需权威渠道 |
| 3 | **910C 训练腿镜像血统** | 官方推荐镜像不含 FlagCX，而训练腿必需；请裁定：① 官方镜像 + 叠加 FlagCX（需可复现配方），还是 ② 维持组内自建镜像并补齐重建配方 | 见 E5 与 §2；当前训练腿镜像无 registry，靠 `docker save` + `docker.repro` 保证可复现 |

---

## 5. 附：第三家（寒武纪）接入时的镜像检查顺序

```text
① 官方文档：FlagTree「User manual for cambricon」是否存在（FlagTree 里寒武纪属于 triton_v3.2.x 分支）
② 厂商侧：寒武纪是否提供官方 PyTorch/推理镜像（torch_mlu + NeuWare/CNRT）
③ 起容器后先跑两条判据
     python -c "import torch, torch_mlu; print(torch.mlu.device_count())"
     python -c "from vllm.platforms import current_platform; print(current_platform)"
④ 训练腿通信：探测可用后端（厂商 CCL / flagcx），FlagCX 官方已含 cncl adaptor
⑤ 归档（dev/images/<name>/v1/ + lock.yaml + ARCHIVE.md）→ 验证（smoke / conformance 13+6 / 两条腿）
   → 把 §4 式的四项材料提给总组入锁
```

**必须提前确认的分叉点**：寒武纪的 vLLM 支持形态 —— 是"厂商移植版 vLLM"（像昇腾 `vllm-ascend`，可直接 `vllm serve`）
还是"社区 vLLM + 平台插件"（像昆仑芯需要 vllm-plugin-FL，且要额外注意 H4 那类依赖完整性）。
