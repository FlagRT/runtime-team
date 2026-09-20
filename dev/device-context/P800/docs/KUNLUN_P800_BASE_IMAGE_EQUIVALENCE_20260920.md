# 官方 `-base` 镜像 · 等价性验证报告（P800 第二实例）

> 日期：2026-09-20 ｜ 负责人：Kistich（hliu553）｜主机：43.180.254.67 ｜用卡：XPU6 / XPU6,7
> 目的：把 P800 现有全部验证结论，在**上游官方推荐镜像**上原样重跑一遍，判断
> ① 结论是否可复现 ② 已知厂商缺陷是「镜像相关」还是「厂商运行时/驱动固有」
> ③ 是否可以把镜像切到官方推荐的那一个（即「官方镜像能不能直接承载我们的验证」）

---

## 0. 结论摘要

| # | 结论 | 判定 |
|---|---|---|
| **1** | **全部结论在官方 `-base` 镜像上可复现**：conformance 13+6、smoke 42/0、训练腿、推理腿前向、服务化 —— 逐项通过，且多数 `detail` 字符串**逐字相同** | ✅ 等价 |
| **2** | **KL3 概率性挂死与镜像无关**：官方 `-base` 上 A 组（KL3=1 + 集合通信）**3/3 挂死**、B 组（不设）**2/2 通过**，自旋特征与现用镜像一致 | ✅ 一致重现 |
| **3** | **官方 `-base` 开箱不含 `triton`**：vLLM 服务化路径直接不可用（`vllm_fl → flag_gems → triton` 断链）。须按官方手册 1.2 节装 `flagtree===0.7.0rc3+xpu3.6`（3.3 GB）后才有 `triton 3.6.0` | ⚠️ 补齐前提 |
| **4** | 补 `flagtree` 后，`-base` 与现用 `flaggems-main-dev` 的软件栈**版本完全对齐**（triton 3.6.0 / torch 2.9.0+cu129 / vLLM 0.13.0 / transformers 4.57.1） | ✅ 可互换 |

**对镜像入锁的含义**：`-base` **可以**作为 P800 的入锁镜像候选，但它不是"开箱可用"的，
必须在镜像配方里显式包含"按手册装 flagtree"这一步；否则推理腿的服务化形态不可复现。

---

## 1. 两个被对照的镜像

| | 现用（所有既有结论的来源） | 官方推荐 / 本次对照 |
|---|---|---|
| 镜像 | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608` | `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base` |
| RepoDigest | 无（本地导入） | `sha256:ea6d797a7d44ef97d7c0c0ed492f69c8ed2e024c927b2bfb5eef53e498e4eb34` |
| 大小 | 磁盘占用 **107 GB**（镜像层 `38 316 770 646` B = 38.3 GB） | 磁盘占用 **94.2 GB**（镜像层 `33 817 700 693` B = **33.8 GB**） |
| 标注 | — | `maintainer: huangyun <huangyun07@kunlunxin.com>`，`description: xvllm_ubuntu2204_torch29 环境` |
| 官方手册定位 | — | FlagTree wiki「User manual for xpu」1.1 节明确推荐这一只 |

容器启动参数（本次为对照而**刻意与现用容器对齐**，只换镜像这一个变量）：
`--device=/dev/xpu0..7 --device=/dev/xpuctrl --device=/dev/fuse --shm-size=64g`，
挂载 `/data2/hliu553 → /workspace`、`/data1/dinghaisong/hf_cache → /hf_cache`，非 privileged、bridge 网络。

> 说明：官方手册给出的启动参数更严格（`--privileged --net=host --shm-size=256g
> --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined`）。
> 本次用精简参数**仍然全部跑通**，说明这些参数不是本方向验证路径的必要条件；
> 但作为入锁镜像的**推荐启动参数**仍应照手册给全。

---

## 2. 逐项等价性对照

### 2.1 统一运行时 conformance（同一套用例、同一后端 `kunlun`）

| 用例集 | 现用镜像 | 官方 `-base` | 判定 |
|---|---|---|---|
| 设备上下文与多流 13 例 | `CONFORMANCE_PASS 13/13` | `CONFORMANCE_PASS 13/13` | **逐用例一致** ✅ |
| 推理 6 例 | `CONFORMANCE_PASS 6/6` | `CONFORMANCE_PASS 6/6` | **逐用例一致** ✅ |

逐用例状态比对结果：**13 例与 6 例均无任何一项状态不同**。

### 2.2 真实后端通用自检（smoke）

| 项 | 现用镜像 | 官方 `-base` |
|---|---|---|
| 结果 | 42 通过 / 0 失败 | **42 通过 / 0 失败** |

自检覆盖：`device_count`、`memory_stats` 口径、统一 Stream/Event、`Stream.context`、
`Event.record + wait_host` 有界返回、`probe_device`、`recover_device` 统一契约、
`translate_error` 类型与后端名回填、能力声明与 `supports()` 自洽。

### 2.3 训练腿（2 卡微调，50 步 / b4 / s128）

| 项 | 现用镜像 | 官方 `-base` |
|---|---|---|
| 结论 | `TRAIN_LEG_PASS 6/6` | **`TRAIN_LEG_PASS 6/6`** |
| loss 曲线 | 15.4488 → 11.1481 | **15.4488 → 11.1481（逐位相同）** |
| 6 项检查 | 全通过 | 全通过（状态一致） |
| 吞吐 | 3482.2 tok/s | 3633.0 tok/s |

**关于吞吐**：首次在 `-base` 上跑得 2574 tok/s，与现用差 −26%，初看像镜像差异；
**交替复测后否定该判断**（同一时段、同卡、同脚本依次跑）：

| 次序 | 容器 | 吞吐 |
|---|---|---|
| 1 | `-base` | 3589.4 tok/s |
| 2 | 现用 | 3541.3 tok/s |
| 3 | `-base` | 3633.0 tok/s |

三次落在 3541–3633 区间（极差 2.6%），**属共享机负载波动**。⇒ **训练腿无镜像性能差异**，
首次的 2574 是环境噪声，不可作为结论。

### 2.4 推理腿（单卡前向）

| 项 | 现用镜像 | 官方 `-base` |
|---|---|---|
| 结论 | `INFER_LEG_PASS 13/13` | **`INFER_LEG_PASS 13/13`** |
| 向量维度 | 1024 | **1024** |
| 语义区分度 | 0.6392 | **0.6392** |
| 吞吐 / 时延 | 53.12 句/s，p50 56.17 ms，p90 57.83 ms | 50.92 句/s，p50 56.93 ms，p90 65.39 ms |
| 错误分级 | L2_PARAM → raise（`graded_by=message_hint`） | 同 |
| 如实跳过项 | `vendor_code_map` | `vendor_code_map`（同） |

**14 项检查的 `detail` 字符串逐字相同（14/14）** —— 包括 `dim=1024`、首条范数 1.0、
跨流可见性、D2H 采样、长驻 20 轮无 NaN/Inf 等。

### 2.5 推理腿（vLLM 服务化，`--runner pooling --convert embed`）

| 项 | 现用镜像 | 官方 `-base`（补 `flagtree` 后） |
|---|---|---|
| 结论 | `SERVE_LEG_PASS 10/10` | **`SERVE_LEG_PASS 10/10`** |
| 维度 / 范数 | 1024 / 1.000000 | 1024 / 1.000000 |
| 语义区分度 | 0.4102 | **0.4102** |
| 吞吐 / p50 | 30.70 句/s，p50 96.4 ms | 31.37 句/s，p50 94.1 ms |
| 超长输入 | HTTP 400 → 统一分级 L2_PARAM/raise + 业务继续 | 同 |
| 服务与设备上下文共存 | 同卡跨流计算 = 3.0 | 同 |

**10 项检查的 `detail` 有 9 项逐字相同**，唯一不同项是 `throughput`（数字本身）。

---

## 3. 关键发现

### 3.1 ⚠️ 官方 `-base` 开箱不含 `triton`，服务化路径不可用

在 `-base` 上原样跑服务化脚本，服务在 90 s 内起不来，vLLM 侧报：

```
RuntimeError: Failed to infer device type, please set the environment variable
`VLLM_LOGGING_LEVEL=DEBUG` to turn on verbose logging to help debug the issue.
```

打开 `VLLM_LOGGING_LEVEL=DEBUG` 后调用链完整体现为：

```
vllm_fl/__init__.py:6            → from vllm_fl.utils import get_op_config
vllm_fl/utils.py:8               → import flag_gems
flag_gems/__init__.py:6          → from flag_gems import testing
flag_gems/testing/__init__.py:3  → from flag_gems import runtime
flag_gems/runtime/__init__.py:3  →
flag_gems/runtime/configloader.py:4 → import triton
→ ModuleNotFoundError: No module named 'triton'
```

实测两个官方变体的 `triton` 状态：

| 镜像变体 | site-packages 有 `triton` | `import triton` |
|---|---|---|
| `...:202608-base` | ❌ 无目录、pip 无记录 | `ModuleNotFoundError` |
| `...:202608-base-ssh` | ❌ 同上 | `ModuleNotFoundError` |
| `flaggems-main-dev:202608`（现用） | ✅ 有 | `triton 3.6.0` |

**补齐方式（来自官方手册 1.2 节「Source-free Installation」原文）**：

```bash
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # Repeat the cmd until fully uninstalled
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
python3.10 -m pip install flagtree===0.7.0rc3+xpu3.6 $RES
```

本次实测：`resource.flagos.net` 可达（HTTP 200）；wheel 3.3 GB，约 2 分 24 秒装完；
装后 `triton 3.6.0`、`import flag_gems` / `import vllm_fl` 均 OK —— **与现用镜像的 triton 版本号完全一致**。

⇒ 这解释了现用镜像的来历：`flaggems-main-dev` = `-base` + `flagtree`(含 triton) + FlagGems + vllm-plugin-FL 的**预装变体**。

⇒ 也修正了一条既有认知：**「`-base` 开箱即可跑 vLLM」不成立**；
它还解释了为什么现用镜像上必须 `PYTHONPATH=/env/FlagGems/src`
（`flag_gems` 的 Python 包只在 `/env/FlagGems/src`，site-packages 里装得不完整）
—— 两个坑叠加起来，都表现为同一句 `Failed to infer device type`。

### 3.2 ✅ KL3 概率性挂死与镜像无关（对厂商上报价值最高的一条）

同一探针（`dc_probe_verify.py`，带 `2^120` 真值校验）、同脚本、同一时段、同卡（XPU6,7）：

| 组 | 变量 | 官方 `-base` 结果 | 现用镜像历史结果 |
|---|---|---|---|
| A ×3 | `XPU_EVENT_KL3_ENABLE=1` + 设备侧集合通信 | **3/3 挂死**（A3 完成到 `rep=40`） | 16/18 挂死（≈89%），挂死点游走第 3–120 次通信 |
| B ×2 | 不设该变量 | **2/2 完成**，`VERIFY value=1.329228e+36 expected=2^120=1.329228e+36 rel_err=0.000e+00 exact=True` | 完成，退出码 0，真值精确 |

挂死现场特征（两镜像一致）：进程状态 `Rsl`、`utime` 累积至 ~9700（自旋）、
卡 6/7 **利用率 100% 而显存仅 366 MiB**、`timeout` 的 SIGTERM **无法中断**（须 `kill -9`）。

⇒ **该缺陷由厂商运行时/驱动层引起，与本方向所用镜像、容器参数无关。**
这条把「镜像相关」这个可能性彻底排除了，上报昆仑芯时不会被反问"是不是你们镜像的问题"。

### 3.3 结论：镜像可互换

在补 `flagtree` 的前提下，`-base` 与现用镜像给出的**结论完全等价**（多数 `detail` 逐字相同，
数值差异仅为吞吐数字，且已证明属共享机噪声）。故：

- 所有既有 P800 结论**不需要重新推翻**；
- 若总组倾向用官方镜像入锁，**证据链已经具备**（本报告的 13+6 / 42 / 两条腿 / KL3 对照全部取自 `-base`）。

---

## 4. 对镜像入锁的建议（相较此前版本的更新）

| 项 | 此前建议 | 本次更新后的建议 |
|---|---|---|
| 入锁镜像候选 | 现用 `flaggems-main-dev`（未入锁） | **官方 `-base`（有 digest，`sha256:ea6d797a…`）** —— 血统清晰（`maintainer: huangyun@kunlunxin.com`）、镜像层小 4.5 GB（33.8 GB vs 38.3 GB；磁盘占用 94.2 GB vs 107 GB） |
| 配方要求 | 未明确 | 必须显式包含：① 按手册装 `flagtree===0.7.0rc3+xpu3.6`；② `PYTHONPATH=/env/FlagGems/src`；③ 推理腿算子路径取 vendor（`VLLM_FL_PREFER=vendor` + `USE_FLAGGEMS=0`） |
| 环境变量口径 | 请求裁定 KL3 变量冲突 | **冲突的证据更强了**：官方手册要求测试前 `export XPU_EVENT_KL3_ENABLE=1`，而该设置在我们这里（两个镜像上均）导致 89%–100% 挂死 |
| 启动参数 | 未明确 | 按官方手册给全（`--privileged --net=host --shm-size=256g --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined`）；本次精简参数虽跑通，但不应作为推荐值下发 |

---

## 5. 原始证据索引

| 文件（均在 `P800/probes/`） | 内容 |
|---|---|
| `I_base_conformance_13_20260920.json` / `I_base_conf13_20260920.log` | `-base` 上 13 例结果与完整输出（13/13） |
| `I_base_conformance_infer6_20260920.json` / `I_base_conf6_20260920.log` | `-base` 上推理 6 例结果与输出（6/6） |
| `I_base_smoke_20260920.log` | `-base` 上通用自检（42/0） |
| `I_base_train_leg_result_rank{0,1}_20260920.json` | `-base` 上训练腿结果（loss 15.4488→11.1481） |
| `I_ref_train_leg_result_rank0_20260920.json` | 同批次现用镜像训练腿结果（交替对照用） |
| `I_base_infer_leg_result_20260920.json` / `I_base_infer_leg_20260920.log` | `-base` 上推理腿前向（13/13） |
| `I_base_serve_result_20260920.json` / `I_base_serve_20260920.log` | `-base` 上服务化（10/10，含 vLLM 启动日志） |
| `I_base_kl3_ab_20260920.log` + `kl3_A{1,2,3}_KL3on.log` + `kl3_B{1,2}_KL3unset.log` | KL3 等价性对照完整记录（含挂死现场 CPU 时间与卡利用率） |
| `H_kl3_equivalence.sh` | KL3 对照脚本（后台轮询 + `kill -9`，因 `timeout` 无法中断自旋进程） |

复现命令（在 `-base` 容器内）：

```bash
# 一次性补齐（按官方手册 1.2 节）
source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda
python3 -m pip install "flagtree===0.7.0rc3+xpu3.6" \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple

export PYTHONPATH=/env/FlagGems/src CUDA_VISIBLE_DEVICES=6,7 FLAGCX_ADAPTOR=klx
# conformance 与自检
cd /workspace/prototype/runtime/conformance && python3 runner.py --backend kunlun
python3 runner.py --backend kunlun --cases infer_cases
cd /workspace/prototype && python3 runtime/smoke_runtime.py --backend kunlun
# 训练腿
cd /workspace/prototype/runtime/proto
DC_BACKEND=kunlun DC_ROOT=/workspace/prototype \
DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
DC_OUT_DIR=/workspace/out_base MAX_STEPS=50 BATCH=4 SEQ=128 \
python3 -m torch.distributed.run --standalone --nproc_per_node=2 proto_train_leg.py
# 推理腿（前向 / 服务化）与 KL3 对照
DEV=6 OUT=/workspace/out_base TAG=_base bash P800/probes/F_infer_leg.sh
DEV=6 OUT=/workspace/out_base TAG=_base bash P800/probes/F2_vllm_serve.sh
bash P800/probes/H_kl3_equivalence.sh
```

---

## 6. 未覆盖 / 待办

| # | 项 | 说明 |
|---|---|---|
| 1 | `-base` 的 P800 镜像**归档** | 官方镜像有 digest，按归档要求还需 `dev/images/<name>/v<N>/` 下补 `lock.yaml` + `ARCHIVE.md` |
| 2 | 现用 `flaggems-main-dev` 的血统说明 | 无 digest、靠 `docker save`；若切换为 `-base`，需说明该变体的重建配方（= `-base` + `flagtree` + FlagGems + vllm-plugin-FL） |
| 3 | 容器启动参数的规范化 | 本次用精简参数跑通；入锁时是否要求 `--privileged`/`--net=host`/`shm 256g` 需总组口径 |
| 4 | KL3 变量口径冲突 | 证据已足（两镜像均复现），仍待总组与昆仑芯侧裁定 |
| 5 | 与其他芯片的横向可比性 | 910C 用厂商官方栈、P800 用社区 vLLM + FL 插件，性能数字**不可直接横比**；本报告不含跨芯片性能结论 |
