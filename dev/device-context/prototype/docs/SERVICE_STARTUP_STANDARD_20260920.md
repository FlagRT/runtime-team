# 组内服务启动标准（下游服务复用指南）

> 版本：v1.1 ｜ 日期：2026-09-20 ｜ 制定：device-context 子方向（Kistich）
> **读者**：运行时层各子方向（显存 / 分布式 / 监控 / 精度 / 算子 / 调度 / 框架适配）、
> 以及任何**需要在国产芯片上把推理服务跑起来**的方向与验收方。
> **效力**：本文档是**启动流程的组内标准**——各方向**统一走这一套脚本与参数**，
> 不要各自维护一份。与《运行时层接口约定》的分工见 §1。
> **配套资产**：`prototype/scripts/serve_standard.sh`（唯一入口）
>
> **v1.1 变更（2026-09-20 晚，910C 真机补跑时发现并修的三处缺口）**
> 1. **服务入口就绪**：`vllm` 不在 `PATH` 时自动激活厂商 python 环境
>    （P800 的 vLLM 在 conda 环境内；不激活直接报 `nohup: failed to run command 'vllm': No such file or directory`）
> 2. **功能冒烟两形态补齐**：生成形态（ascend）此前只验就绪、无冒烟，现补 `/v1/completions`
>    冒烟，并把冒烟纳入 verdict（`PASS` = ready=1 **且** smoke=1）
> 3. **用卡快照容器内降级**：`npu-smi` / `xpu-smi` 是宿主工具，容器内没有 →
>    退回 torch 侧查询（`torch.npu` / `torch.cuda` 的 `mem_get_info`），日志不再为空
>
> **两实例真机验证（同一脚本版本）**：910C `SERVE_STANDARD_PASS`（就绪 30 s、生成冒烟 8 tokens）；
> P800 `SERVE_STANDARD_PASS`（就绪 25 s、维度=1024 范数=1.000000）——见 §5。

---

## 1. 为什么单独成文（与《接口约定》的分工）

本方向已有三份易混文档，职责如下——**新增的启动标准不复用其中任何一份的正文**：

| 文档 | 性质 | 回答的问题 | 读者 | 变更流程 |
|---|---|---|---|---|
| `INTERFACE_CONTRACT_DC_20260908.md`（**接口约定**） | **规范 / 效力文件** | "我承诺什么**接口语义**" | 对接人 | 走变更流程：更新文档 → 知会全部下游 → conformance 回归 |
| `REFERENCE_TWO_INSTANCES_CONFIG_20260920.md`（**两实例配置手册**） | **参考 / 实测记录** | "**我们当时**怎么跑的、为什么这么配" | 后续接入者、验收方 | 随实测更新，无流程 |
| `NEW_CHIP_ONBOARDING_MANUAL_20260920.md`（**接入手册**） | **操作手册** | "**新芯片**怎么接进来" | 芯片接入工程师 | 随接入实践更新 |
| **本文档（启动标准）** | **规范 / 操作** | "**你们**必须怎么起服务" | 各服务消费方 | 见 §9 |

**为什么启动标准不塞进接口约定**：
接口约定管的是 **API 语义承诺**（`use()` / `set_device()` / `translate_error()` 的语义、插件规范、两条纪律），
它的每次修改都要走"知会全部下游 + conformance 回归"的流程。
而启动脚本与参数会随实例增加、镜像变化而**更频繁地演进**（本批就已经因为 P800 新增了一批环境变量）。
把两者混在一份文件里，会让"接口变更评审"被启动参数变更污染，也会让下游搞不清
"哪些是我必须遵守的语义契约、哪些只是当前推荐的启动方式"。

⇒ 因此：**接口约定加一条指针（§2 已落地），正文留在本文档。**

---

## 2. 唯一入口

```bash
# P800（容器内）
DC_BACKEND=kunlun DEV=6 \
MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
bash prototype/scripts/serve_standard.sh

# 910C（容器内）
DC_BACKEND=ascend MODEL=/mnt/raid/hliu553/models/Qwen3-4B SERVED_NAME=qwen3-4b \
bash prototype/scripts/serve_standard.sh
```

**跨芯片只改 `DC_BACKEND` 一个变量**；两芯片的环境差异（选卡变量、平台插件、`PYTHONPATH`）
全部收敛在脚本内部——**下游不需要知道，也不应该各自实现**。

### 2.1 参数表（全部环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `DC_BACKEND` | `ascend` | `ascend` \| `kunlun` |
| `MODEL` | 按后端给默认值 | 模型路径。⚠️ **必须给到 `snapshots/<hash>`**（给 HF 缓存根目录会报 `Unrecognized model ... Should have a model_type key`） |
| `SERVED_NAME` | `qwen3-4b` / `qwen3-embedding-0.6b` | 服务暴露的模型名 |
| `PORT` | `8100` | 服务端口 |
| `DEV` | 全部可见卡 | 选卡。**ascend 走 `ASCEND_RT_VISIBLE_DEVICES`、kunlun 走 `CUDA_VISIBLE_DEVICES`** ——由脚本按后端选择，不要自己 export |
| `TP` | `1` | tensor parallel |
| `MAX_MODEL_LEN` | `4096` | 与两实例保持一致，便于"超长输入"用同一判据比对 |
| `GPU_MEM_UTIL` | 空 | 空则不传。共享机上建议给（P800 实测用 `0.25`） |
| `EAGER` | `1` | 加 `--enforce-eager`（关图捕获）。**P800 实测去掉它反而更慢**（p50 96.4 → 132.3 ms），故默认保留 |
| `DC_OUT_DIR` | `/tmp/dc_serve` | 日志与 pid 目录 |
| `DC_CONDA_ENV` | `python310_torch29_cuda` | 仅当 `vllm` 不在 `PATH` 时使用：脚本按此名激活 conda 环境（P800 用；910C 镜像自带 vllm，不触发） |
| `HOST` | `127.0.0.1` | 绑定地址。**默认不对外暴露**；跨容器 / 跨机访问需显式 `HOST=0.0.0.0` |
| `STOP_AFTER` | `0` | `1` = 就绪并冒烟后立即停机（验证/CI 用）；`0` = 保持运行（供下游调用） |
| `READY_BUDGET` | `180` | 就绪等待上限（秒） |
| `ERROR_TRANSLATION` / `MONITOR` | `0` | 集成层开关，**当前仅 910C 可用**，见 §6 |
| `EXTRA_ARGS` | 空 | 追加的 vllm 参数（**仅在标准参数不满足需求时使用，见 §8**） |

### 2.2 统一的服务参数口径

```
vllm serve <MODEL> --served-model-name <NAME> --host <HOST> --port <PORT>
     --max-model-len 4096 --tensor-parallel-size 1
     [embedding 形态] --runner pooling --convert embed
     [--gpu-memory-utilization] [--enforce-eager]
```

**为什么是这几个参数**（每条都有实测依据）：

| 参数 | 依据 |
|---|---|
| `--runner pooling --convert embed` | **该版本 vLLM 没有 `--task` 参数**。按 `--task embed` 启动会报 `vllm: error: unrecognized arguments: --task embed`；查参数表确认可用值为 `--runner {auto,draft,generate,pooling}` 与 `--convert {auto,classify,embed,none,reward}`。两实例**统一这个口径**才能横向比对 |
| `--max-model-len 4096` | 与两实例保持一致，使"超长输入防御"可用同一判据：6001 tokens > 4096 → HTTP 400 → 统一分级 **L2_PARAM / raise** 且业务继续 |
| `--enforce-eager` | P800 实测单变量对照：**去掉它更慢**（22.83 句/s、p50 132.3 ms vs 30.70 句/s、p50 96.4 ms） |
| `--tensor-parallel-size` | 默认 1；910C 侧曾用 `TP=2/4` 做过数值等价对照（**须用 greedy，temperature=1.0 下必然发散**） |

---

## 3. 标准流程（脚本已实现，下游不要重写）

```
① 服务入口就绪           ← vllm 不在 PATH 时按 DC_CONDA_ENV 激活厂商 python 环境
                           （找不到则**报错退出**，不静默失败；见 §4 纪律 5）
② 按 DC_BACKEND 设环境   ← 两芯片差异全在这里
③ 清理残留进程           ← 含 EngineCore 子进程，见 §4 硬纪律 3
④ 打印用卡现状           ← npu-smi / xpu-smi；容器内没有则降级为 torch 侧查询，见 §4 纪律 6
⑤ 启动 vllm serve        ← 统一参数口径
⑥ 就绪轮询               ← 判据：GET /v1/models 返回 200（另加进程存活检查）
⑦ 功能冒烟（两形态都有）  ← embedding 形态 POST /v1/embeddings；生成形态 POST /v1/completions
⑧ 收尾                   ← STOP_AFTER=1 则停机 + 复查卡释放；否则打印 base_url / model / pid
```

**就绪判据为什么用 `/v1/models` 而不是日志关键字**：日志关键字会随 vLLM 版本变化，而 HTTP 200 是**服务真的可用**的充分判据。

**为什么既要"就绪"又要"冒烟"**：就绪只证明端口通了、引擎起来了；冒烟才证明**服务真的能算**。
只看 200 会掩盖"起来了但算不出"的情况——因此 `STOP_AFTER=1` 的 verdict 要求 **ready=1 且 smoke=1**。

---

## 4. 六条硬纪律（违反会出真问题）

| # | 纪律 | 实测依据 |
|---|---|---|
| 1 | **`ascend` 必须禁用 `vllm-plugin-FL`；`kunlun` 必须启用它** | 同一插件在两芯片上**可用性相反**：昇腾侧设 `VLLM_PLUGINS=fl` 后 `current_platform.device_type` 变空 → `RuntimeError: Device string must not be empty`（该插件**无 ascend 后端**）；昆仑芯侧**没有它 vLLM 就认不出设备**（社区 vLLM 的 `vllm/platforms/` 无 kunlun）。脚本已按后端分别处理 |
| 2 | **`kunlun` 必须 `PYTHONPATH=/env/FlagGems/src`** | site-packages 里那份 `flag_gems` 子模块不完整（`pip show` 看得见包，但 `runtime.backend.device` 导不进来）→ 缺它报 `Failed to infer device type`（**看着像设备问题，其实是 Python 包问题**） |
| 3 | **停机必须连 `EngineCore` 子进程一起清理** | 只杀主进程后 `VLLM::EngineCore` 会残留并持续占卡：实测卡 6 仍被占 **73850 MiB / 96 GiB**，下一次启动报 `Free memory on device (23.85/96.0 GiB) ... less than desired GPU memory utilization`。脚本用 `kill -9` + 变量拼接（避免 `pkill` 命中自身） |
| 4 | **`MODEL` 必须给到 `snapshots/<hash>`** | 传 HF 缓存根目录报 `Unrecognized model ... Should have a model_type key`（根目录只有 `blobs/`、`refs/`、`snapshots/`） |
| 5 | **服务入口要就绪：`vllm` 必须在 `PATH`** | P800 的 vLLM 装在 conda 环境 `python310_torch29_cuda` 内，**不激活就直接** `nohup: failed to run command 'vllm': No such file or directory`（启动即退出，日志里只有这一行，很容易被误判为"服务起不来"）。脚本已自动激活（`DC_CONDA_ENV` 可覆盖）；910C 镜像自带 `/usr/local/python3.11.15/bin/vllm`，不触发 |
| 6 | **容器内没有 `npu-smi` / `xpu-smi`**（它们是宿主工具） | 910C 的 vllm-ascend 镜像实测 `npu-smi: command not found`。脚本降级为 torch 侧查询：`[torch.npu:0] free=60.91GiB / total=61.27GiB`。**要看整机 16 卡全貌请到宿主执行 `npu-smi info`** |

### 4.1 三个已知行为（不是缺陷，但会被问）

| 现象 | 说明 |
|---|---|
| 日志里出现 `Killed  nohup vllm serve ...` | 这是 `cleanup()` 的 `kill -9` 生效时 bash 的作业通知，**属正常**，说明停机清理确实执行了 |
| 反复"起服务→外部杀→再起"会在容器内累积 `<defunct>` 僵尸进程 | 僵尸的父进程是容器 PID 1（`sleep infinity`，不回收子进程）。实测僵尸**不占卡、不占内存**，仅 PID 未回收（`ppid=1`、状态 `Z`）。**用 `STOP_AFTER=1` 走脚本自身停机路径不会产生僵尸**（脚本会回收自己的子进程）；若长时间反复外部 `kill`，可 `docker restart <容器>` 清理 |
| 宿主/其他容器连不上服务 | 默认 `HOST=127.0.0.1`（只在容器内可达）。**跨容器或跨机访问须 `HOST=0.0.0.0`**，并自行确认网络与访问控制策略 |

---

## 5. 验证记录

**同一脚本版本（`serve_standard.sh` v1.1）在两实例真机各跑一遍**：

| 实例 | 启动命令 | 结果 |
|---|---|---|
| **910C** | `DC_BACKEND=ascend DEV=0 STOP_AFTER=1` | **`SERVE_STANDARD_PASS (ready=1 smoke=1)`**：服务就绪 **30 s**；生成冒烟 **8 tokens**（`1+1=` → `'2 is a basic arithmetic fact, but'`）；用卡快照 **free=60.91GiB / total=61.27GiB，停机前后一致**（确认释放）；旧容器内**无残留 vllm 进程**、宿主 `ss` 显示 **8100 端口已释放**。证据：`910C/probes/L_serve_standard_910c_20260920.log` |
| **P800** | `DC_BACKEND=kunlun DEV=6 PORT=8200 GPU_MEM_UTIL=0.25 STOP_AFTER=1` | **`SERVE_STANDARD_PASS (ready=1 smoke=1)`**：服务就绪 **25 s**；embedding 冒烟 **维度=1024 范数=1.000000**；停机后**无残留进程**且卡 6 释放至 **0 MiB**。证据：`P800/probes/L_serve_standard_p800_v2_20260920.log` |

**910C 侧补跑的历史**（如实记录，供追溯）：
1. 首轮（脚本 v1.0）：`SERVE_STANDARD_PASS`，就绪 **45 s** —— 但发现**生成形态没有功能冒烟**（与 embedding 形态强度不对等）⇒ 补冒烟并把冒烟纳入 verdict；
2. 次轮（脚本 v1.1 前半）：`SERVE_STANDARD_PASS (ready=1 smoke=1)`，就绪 **30 s**，冒烟 8 tokens ⇒ 通过；
3. **同期在 P800 复跑时暴露新缺口**：脚本未激活 conda 环境 → `nohup: failed to run command 'vllm'` ⇒ 补"服务入口就绪"步骤；
4. 末轮（**最终脚本**，两实例并行）：910C 就绪 **30 s**、P800 就绪 **25 s**，均 `SERVE_STANDARD_PASS (ready=1 smoke=1)` —— 上表即此轮。

> 结论：**910C 侧统一脚本已真机验证通过**，不再是待补项。旧脚本 `start_vllm_serve_910c.sh`
> **仍保留**（它含 D10/D11 集成，见 §6），但**下游新需求请走统一脚本**。

**回归影响**：既有两套脚本（`910C/.../start_vllm_serve_910c.sh`、`P800/probes/F2_vllm_serve.sh`）
**均保留不删**（F2 仍是本方向的多镜像对照探针；910C 脚本含 D10/D11 集成）。
本标准的适用对象是**下游各方向的新启动需求**。

---

## 6. 集成层现状（D10/D11）与待上提项

910C 版旧脚本额外挂了两项本方向的集成资产：

| 集成 | 内容 | 当前范围 |
|---|---|---|
| **D10 错误码翻译** | serve 以 `python -c "import inject_error_translation; inject_error_translation.serve()"` 包装器启动，未捕获异常先过 `translate_error` 再走原逻辑（**不碰 vLLM 源码**） | **仅 910C 可用**（资产在 `910C/distributed_inference/inference/`） |
| **D11 设备状态监控** | 并行启动 `device_state_monitor.py`，周期探活 + 四态查询，状态变化打印事件并落 JSON | 同上 |

统一脚本**保留这两个开关**（`ERROR_TRANSLATION=1` / `MONITOR=1`），但**未上提前在 P800 上不可用**。

**待上提项（列为下一步）**：按"原型与芯片无关"原则，这两个资产应上提到 `prototype/` 侧
（它们本身不含厂商专有逻辑：`inject_error_translation` 走统一错误对象、
`device_state_monitor` 走统一 `device_state`），上提后两芯片均可启用。
**上提前不要在下游文档里承诺该项可用。**

---

## 7. 与"自建脚本"的关系

**标准要求：不要各自维护启动脚本。** 具体地：

1. **不要再抄一份 `my_start_vllm.sh`**——抄的那一刻就与标准分叉了（本标准的由来正是这件事）。
2. **需要新能力时**（新增参数、新增后端、集成层），走 §9 的变更流程改**同一个脚本**，
   而不是另起一份。
3. **确需自建时**（例如某方向要跑与本标准差异很大的形态），必须：
   - 在脚本头部写明"与《组内服务启动标准》的差异清单"；
   - 把差异回报本方向（Kistich），由本方向判断是并入标准还是保持分化。
4. `EXTRA_ARGS` 只用于**临时调试**，不得写进长期运行的启动命令。

---

## 8. 基座层面新发现的同步机制

**约定：本方向在接入/验证中暴露的基座层问题（环境约束、能力边界、厂商缺陷、镜像问题），
一律先落到 `dev/device-context/STATUS.md`**（"基座与约束"或"阻塞与需要协调的事项"两节），
由**总组统一收拢裁定**，再并入统一基座 `dev/stack.lock.910c.v2.yaml`。
本方向**只消费不自建**，不擅自修改锁定镜像口径与公共资产。

**已按此机制登记的事项（截至 2026-09-20）**：

| # | 事项 | 登记位置 | 状态 |
|---|---|---|---|
| 1 | **P800 镜像入锁**（建议以官方 `-base` 为准，配方须含 `flagtree` 补齐步骤） | STATUS「阻塞与需要协调的事项」 | 待总组裁定 |
| 2 | **`XPU_EVENT_KL3_ENABLE` 口径冲突**（官方手册要求设 1；实测设 1 时 18 次运行 16 次挂死；**已在两个镜像上一致重现** ⇒ 与镜像无关，归属厂商运行时层） | 同上 | 待总组与芯片厂商裁定 |
| 3 | **910C 训练腿镜像血统**（官方推荐镜像不含训练腿必需的 FlagCX） | 同上 | 待总组裁定 |
| 4 | **训练腿镜像未发布到 registry**（临时机器绑定资产，其他机器需取 `docker save` 包） | 同上 | 待总组/镜像 owner 推进 |
| 5 | **容器启动参数口径**（官方手册要求 `--privileged`/`--net=host`/`shm 256g`；实测精简参数亦可跑通，但不应作为下发值） | 同上 | 待总组裁定 |
| 6 | **同一个 FL 插件跨芯片可用性相反**（`vllm-plugin-FL` 昆仑芯必需、昇腾必须禁用） | 基座与约束 | 已登记，供第三家接入前评估 |
| 7 | **官方 `-base` 镜像开箱缺 `triton`**（`vllm_fl → flag_gems → triton` 断链，须装 `flagtree` 补齐） | 基座与约束 | 已登记 |
| 8 | **P800 厂商缺陷：流优先级不可用**（裸调触发 PyTorch `INTERNAL ASSERT`，上游上报非法优先级区间） | 基座与约束 | 已登记（本层已主动拦截不透传） |
| 9 | **镜像"服务入口"与"用卡工具"口径不一致**（P800 镜像的 `vllm` 不在默认 `PATH`，须激活 conda 环境 `python310_torch29_cuda`；910C 镜像**不含 `npu-smi`**，它是宿主工具） | 基座与约束 | 已登记，供下游与第三家接入前评估（本标准已内部消化） |

> 使用本标准时若遇到**上表未覆盖**的基座问题，请同样落进 `STATUS.md`
> 并知会本方向，不要各自在本地打补丁。

---

## 9. 变更流程

```
新需求/新发现 → 改 prototype/scripts/serve_standard.sh（唯一入口）
             → 本文档 §2 参数表与 §3 流程同步更新
             → 在两实例真机上各跑一次（STOP_AFTER=1 验证模式）
             → 知会已接入的下游方向
```

**版本节奏**：本标准的版本随组件版本走（当前对应组件 `runtime-v0.2.0`）。
破坏性变更（参数改名、默认值改变、流程步骤变化）**提前一周知会**。
