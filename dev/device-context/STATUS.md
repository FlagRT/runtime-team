# device-context · 当前状态

更新：2026-09-22（第 3 家寒武纪 MLU590 环境普查完成；**新增阻塞项：root 权限开通**）
　　**同日追加（09-22 下午）：镜像血统对齐核查 —— 发现 FlagOS 官方镜像体系（含寒武纪）；
寒武纪镜像来源结论更正 + 档位定档 `neuware4.4.3`（驱动同 6.2.x 线，镜像阻塞已解除）；
910C 官方对应物已登记进基座草稿 `candidates:`；P800 KL3 缺陷获官方印证**
　　**同日追加（09-22 傍晚）：第 3 家寒武纪「接入」阶段启动 —— 后端落地（代码层）完成 +
新增《离线契约自检》工具（35/0）；真机验证待 `docker` 组权限**
　｜上次例行更新 2026-09-20 ｜ 负责人：Kistich（hliu553）｜ **更新节奏：每周三**

> 本文件按全组约定维护：**各子方向 STATUS.md 是总组收拢诉求与裁定基座调整的依据**。
> 结论性环境依据见 `dev/stack.lock.910c.v2.yaml`（总组定稿，位于 **`dev-1.0` 分支**；本方向只消费不自建）。

## 更新机制（本方向约定）

| 项 | 约定 |
|---|---|
| **更新节奏** | **每周三**评估并更新本文件 |
| **评估范围** | 本方向职责：**设备上下文（设备抽象与执行上下文）+ 多流 Stream**，以及相关的错误码翻译与状态恢复 |
| **无新增内容时** | **不动**本方向的 `dev/stack.lock.910c.yaml` 工作草稿；一切以总组定稿的 `dev/stack.lock.910c.v2.yaml`（dev-1.0）为准 |
| **有新增内容时** | ① 先写入本方向工作草稿 `dev/stack.lock.910c.yaml`（留在 `kistich/device-context` 分支，**不直接 PR**）；② 同步进本文件的「基座与约束」或「阻塞与需要协调的事项」；③ 由总组收拢裁定后并入 v1 |
| **触发更新的典型情形** | 实测到新的环境约束（如新的并发/权限/镜像限制）、新的能力边界（如某后端新增或不支持某能力）、新的坑或缺陷、对基座的新诉求 |

> 依据：总组在 `stack.lock.910c.v2.yaml` 中约定「统一基座的确定/调整/发布由总组裁定，
> 依据各子方向 STATUS.md 暴露的问题与意见；各方向只消费不自建」。

## 当前阶段

对应《运行时层原型验证-战略目标-910C》§5「设备抽象与执行上下文」及运行时层模块拆分表子方向 1。
本月目标：交付统一运行时原型，并基于该原型完成训练腿 / 推理腿的设备侧验收。

当前状态：**统一 API 与 Backend 插件机制已完成**，并在**两个芯片实例**上取得实证 ——
910C（第一实例）两条腿全闭环；**P800（第二实例，昆仑芯）阶段 0–5 全部完成**
（接入 → 训练腿 → **推理腿 13/13** → **推理腿服务化 10/10** → **错误闭环两设置对照 PASS** → **镜像等价性验证** → **阶段 5 收敛三件套**），
并已在**上游官方推荐镜像 `-base`** 上完成**等价性验证**（全部结论复现、KL3 缺陷一致重现）。
阶段 5 交付：《新芯片接入手册》（8 步流程 + 验收清单 13 项）、接口约定修订建议 **6 条**、原型 release **`runtime-v0.2.0`**。
**多流 Stream 16 项验收基线已在 P800 上逐项比对完成**（14 通过 / 1 如实标注不支持 / 1 不适用；探针 8/8 与 910C 逐项一致）。
**《组内服务启动标准》已发布**（下游服务复用指南 + 唯一入口 `prototype/scripts/serve_standard.sh`）——
把此前 910C / P800 **各自维护的两套启动脚本收敛为一套**，跨芯片只改 `DC_BACKEND`，
并明确各方向不再自建启动脚本；**两实例真机均已实测 `SERVE_STANDARD_PASS`**
（910C 就绪 30 s + 生成冒烟 8 tokens；P800 就绪 25 s + 维度 1024 范数 1.000000）。
**两实例已在当前代码上完成全量复核（2026-09-22）**：smoke 51/0（ascend）与 42/0（kunlun）、
conformance 13+6 双侧全绿、语义基线 8/8 双侧、推理腿 14/14（ascend，含自证字段）、
训练腿 6/6（loss 与历史逐位吻合）、错误闭环双侧 5/0/0 —— 复核缺口 G1/G2 已关闭，
证据 `910C/probes/recheck_*_20260922.json`（7 份）+ `P800/probes/recheck_*_20260922.json`（4 份）。
**第 3 家（寒武纪 MLU590）已进入「接入」阶段**：
① **环境普查 ✅**（两台测试机各 **8 × MLU590-M9** / 96 GB 卡 / 11T 盘挂 `/srv`；**验收模型已在共享 HF 缓存**，无需下载）；
② **镜像定档 ✅** `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`（实测可匿名拉取）；
③ **后端落地（代码层）✅** —— 新增 `prototype/runtime/backends/cambricon/`（**13 抽象 + `build()` +
`supports()` 如实声明 + `known_issues()`**；`name="cambricon"` / `device_type="mlu"`），
并新增**《离线契约自检》工具**（`prototype/scripts/backend_offline_check.py`，**35 通过 / 0 失败**）
且回写进《新芯片接入手册》**§4.4**；本机 `smoke_runtime.py` **28/0** ⇒ 新增后端未造成回归。
④ **真机验证 ⛔ 待 `docker` 组权限**（`/srv` 不可写、不在 `docker` 组）——
**镜像与代码侧均已不阻塞**，剩余步骤（conformance 13+6 → 多流 16 项 → 训练腿 → 推理腿 → 错误闭环）
按 [`MLU590/docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md`](MLU590/docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md) §5 的 A1–A10 执行。

> ⚠️ **口径边界（必读）**：上表 ③ 的"完成"指**代码层完成**。
> 本实例**尚未在任何寒武纪设备上跑过一次** —— `torch.mlu` 的真实 API 形态、conformance 是否通过、
> 两条腿能否跑通，**全部必须到容器内实测才能下结论**；离线自检**只证明实现逻辑与契约形态**。

未完成项：组件下游反馈收集、全组联合 demo 合稿、**第 3 家真机验证（待 `docker` 组权限）**。

## 已有量化结果（实跑）

### 910C 实例（第一实例，已完成）

| 项 | 结果 |
|---|---|
| 组件自检 | 37/37（无 NPU 环境时昇腾项自动 SKIP） |
| conformance 昇腾后端 | 13/13 + 推理 6/6 |
| conformance FlagOS 后端 | 13/13（锁定训练镜像） |
| 训练腿 2 卡分布式微调 | 两 rank 均 6/6：loss 15.4497 → 11.15（50 步）、2117 tok/s、通信三类对照（all_reduce / all_gather / P2P）全对 |
| 推理腿单卡服务化 | 10/10：维度 1024、范数 1.0、语义区分度 0.4123、108 句/s（p50 27.4ms）、超长输入 → L2_PARAM/raise 且业务继续 |
| 推理腿单卡前向（对照） | 10/10：区分度 0.638、66–79 句/s、无 NaN |
| 错误注入 → 恢复闭环 | 推理腿 5 闭环 / 0 失败；训练腿 4 闭环 / 1 跳过（该后端无有界同步，如实标注）/ 0 失败 |
| **统一启动脚本**（组内服务启动标准 v1.1，09-20） | **`SERVE_STANDARD_PASS (ready=1 smoke=1)`**：服务就绪 **30 s**；生成冒烟 **8 tokens**（`1+1=` → `'2 is a basic arithmetic fact, but'`）；用卡快照 `free=60.91GiB / total=61.27GiB`（停机前后一致）；宿主侧 8100 端口已释放、容器内无残留 `vllm` 进程 |

以上结果**均在锁定镜像内取得**（`stack.lock.910c.v2.yaml` 的两腿镜像），非个人调试容器。

### P800 实例（第二实例，昆仑芯；阶段 0–4 已完成 + 镜像等价性已验证）

| 项 | 结果 |
|---|---|
| 接入成本（首次实测） | 环境普查 → 五域基线 → `kunlun` backend 落地 → conformance 全绿，**单日完成**（对比月度计划目标：单个新芯片 ≤5 人天） |
| 组件自检 | **42 通过 / 0 失败** |
| conformance 昆仑芯后端 | **13/13**（e1/e2×2/e3/f1/r/s1–s4/t1–t3） |
| conformance 推理 6 例 | **6/6**（i1–i6） |
| 训练腿 2 卡分布式微调 | 两 rank 均 **6/6**：loss **15.4488 → 11.1481**（50 步）、**3482.2 tok/s**、14.7 s |
| 通信三类对照 | **3/3 全对**：all_reduce 3.0/3.0、all_gather [0.0, 1.0]、P2P 一致 |
| 五域基线实测 | 设备抽象 ✅ 8 卡 / 96 GiB；多流 ✅ 跨流 Event 依赖正确、4096² matmul 15.859 ms；算子 ✅ FlagGems add max diff = 0.0；错误 ⚠️ 厂商码不透出；状态恢复仅 probe 级 |
| 接入规范检验 | 实现 **13 个抽象方法**（设备 4 / 多流 7 / 错误翻译 1 / 状态恢复 1）+ `build()` 登记 + `supports()` 如实声明，**未新增任何框架改造**（注册表已预留 `kunlun`） |
| 上游厂商缺陷（已上报） | KL3 事件同步概率性挂死：**16/18 ≈ 89%**（本轮基线 4/4）；根因定位到**函数级**（3 处自旋点，均在厂商 `libxpucuda.so`）；最小复现 **约 10 秒** |
| 规避验证（A/B 单变量） | 同一脚本同一用卡，唯一变量 `XPU_EVENT_KL3_ENABLE`：**不设 → 退出码 0**；**设 1 → 退出码 124（超时）** |
| **阶段 3 · 推理腿（09-20）** | **`INFER_LEG_PASS 13/13`**（1 项如实跳过）：维度 **1024**（与 910C 一致）、语义区分度 **0.6392**（910C 0.638）、**53.12 句/s**、p50 **56.17 ms**、真实参数异常 → **L2_PARAM/raise（confident）** 且业务继续 |
| **阶段 3 补 · vLLM 服务化（09-20）** | **`SERVE_LEG_PASS 10/10`**：维度 **1024**、区分度 **0.4102**（910C 0.4123）、**30.70 句/s**、p50 **96.4 ms**、超长输入（6001 tokens > 4096）→ HTTP 400 → **L2_PARAM/raise** + 业务继续；服务与设备上下文同卡共存不冲突（跨流计算 = 3.0）。⚠️ 硬前置：`PYTHONPATH=/env/FlagGems/src`（否则 vllm-plugin-FL 报 `Failed to infer device type`）|
| **阶段 4 · 错误闭环两设置对照（09-20）** | 两组均 **`ERROR_RECOVERY_LOOP_PASS`（闭环 5 / 跳过 0 / 失败 0）**，且**逐字节一致**（除时间戳）⇒ **关闭 `XPU_EVENT_KL3_ENABLE` 不损失错误诊断能力**；同时说明该厂商缺陷**不影响单进程设备上下文路径** |
| **框架缺陷第 4 例（09-20，已修）** | 错误对象**跨模块类不相等**（`conformance/errors.py` 被 importlib 动态加载为独立模块，其 `ErrorCategory` 是 IntEnum）→ `FlagosError.disposition` 取 `DISPOSITION[cat]` **KeyError**。修在**框架层**（新增 `coerce_category` / `normalize_error`，并在 `translate_via_backend` 加归一化兜底）+ `kunlun.translate_error` 显式归一 |
| **框架缺陷第 5 例（09-22，已修）** | **后端侧错误对象回填不对称**：`kunlun.translate_error` 回填 `backend=self.name`（9-20 修复时引入），而 ascend/flagos 未回填（`FlagosError.backend` 是文档化字段，直调后端方法时为 None）；同一复跑还暴露 smoke 判据不公平（给声明 error_map 的后端注入**无厂商码**消息却断言必须 code_map）。修：ascend/flagos 补回填 + 判据改两条诚实断言（无码不伪称 code_map / 含码样例必走 code_map，样例由后端自带 `SAMPLE_CODED_ERROR`）。修复后 ascend smoke **51/0**、P800 **42/0**，conformance 13+6 双侧回归通过 |
| **镜像等价性验证（09-20）** | 在**上游官方推荐镜像** `harbor.baai.ac.cn/...:202608-base`（digest `sha256:ea6d797a…`，33.8 GB）上重跑全套：conformance **13+6 逐用例一致**、smoke **42/0**、训练腿 **6/6（loss 逐位相同 15.4488→11.1481）**、推理腿 **13/13**（`detail` **14/14 逐字相同**）、服务化 **10/10**（区分度 0.4102 一致）、**KL3 挂死一致重现（A 组 3/3 挂死、B 组 2/2 通过且 `2^120` 真值精密匹配）** ⇒ **两镜像结论等价**；**KL3 缺陷与镜像无关**，归属厂商运行时/驱动层 |
| ⚠️ 官方镜像的补齐前提（09-20） | 官方 `-base`（及 `-base-ssh`）**开箱不含 `triton`** → `vllm_fl → flag_gems → triton` 断链，服务化报 `Failed to infer device type`。须按官方手册 1.2 节 `python3.10 -m pip install flagtree===0.7.0rc3+xpu3.6 --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple`（实测源 HTTP 200、wheel 3.3 GB、约 2.5 分钟），装后 `triton 3.6.0` 与现用变体**版本号一致** |
| **多流 Stream 16 项基线（09-20）** | **14 项通过 / 1 项如实标注不支持 / 1 项不适用**：探针 8 项 **`STREAM_SEMANTICS_PASS 8/8`**（与 910C **逐项一致**）；**S-7 图捕获首次实测 `GRAPH_CAPTURE_PASS 5/5`**（据此为 `kunlun` 补上 `graph_capture` 能力声明）；S-16 补测 **2000 流无限制**；唯一差异 **S-12 流优先级不支持**（上游上报非法优先级区间 → 触发 PyTorch INTERNAL ASSERT；本层主动拦截不透传、不声明该能力） |
| **统一启动脚本**（组内服务启动标准 v1.1，09-20） | **`SERVE_STANDARD_PASS (ready=1 smoke=1)`**：服务就绪 **25 s**；embedding 冒烟 **维度 1024 / 范数 1.000000**；停机后**无残留进程**、卡 6 释放至 **0 MiB**。⚠️ 需先激活 conda 环境（vLLM 不在默认 `PATH`，脚本按 `DC_CONDA_ENV=python310_torch29_cuda` 自动处理） |

**诚实标注**：P800 训练腿证据在 `XPU_EVENT_KL3_ENABLE` **未设置**下取得；
该变量开启时本环境概率性挂死（厂商缺陷），**不能代表开启时的行为**。

诚实标注：
- 两 rank 末值 loss 存在约 4e-3 差异（11.1497 / 11.1541），源于集合通信浮点累加顺序，属固有特性；
- `ERR99999` 偶发进程终止复现 1 次、后续 4 次未复现 → **观察项，不作结论**。

## 基座与约束（本方向实测）

- **设备注册路线**：遵循 v1 的 Route A 原则；训练腿因锁定镜像约束本月走 flagos（torch_fl），
  为 v1 中登记的**权宜例外**，不代表路线变更。
- **训练腿镜像禁止 torch_npu 与 Torch-FL 共存**（自带校验脚本直接报错），
  必须 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 且先 `import torch_fl` 再 `import torch`。
- **带卡容器并发上限 3**（已写入 v1 规则置顶）：超限 `acl.init()` 返回 500000、设备不可见。
- **同一个 FL 插件在两家芯片上可用性相反**（第三家选型须逐个确认，不能类推）：
  `vllm-plugin-FL` 在**昆仑芯必需**（社区 vLLM 的 `vllm/platforms/` 无 kunlun，靠它提供 FL platform
  —— 不加载则 vLLM 报 `Failed to infer device type`）；在**昇腾必须禁用**
  （设 `VLLM_PLUGINS=fl` 后 `current_platform.device_type` 变空 → `RuntimeError: Device string must not be empty`，
  该插件自述 currently CUDA only）。
- **官方 `-base` 镜像开箱不含 `triton`**：`vllm_fl → flag_gems → triton` 断链，推理服务化报
  `Failed to infer device type`（**看着像设备问题，其实是包问题**）。须按官方手册 1.2 节
  `python3.10 -m pip install flagtree===0.7.0rc3+xpu3.6`（实测源 HTTP 200、wheel 3.3 GB、约 2.5 分钟）。
  ⇒ 若以该镜像入锁，**配方须显式包含此步骤**，否则第三家接入者会卡在同一位置。
- **P800 流优先级不可用（上游）**：裸调 `Stream.priority_range()` 触发 PyTorch 自身
  `INTERNAL ASSERT FAILED at c10/cuda/CUDAStream.h:188`（XPytorch 上报非法优先级区间）。
  本层已**主动拦截、绝不透传**（避免进程级 abort），`kunlun` 后端**不声明** `stream_priority` 能力（如实）。
- **镜像的"服务入口"与"用卡工具"口径不一致**（09-20 启动标准补跑时实测，两条均为镜像特征、非缺陷）：
  ① **P800 镜像的 `vllm` 不在默认 `PATH`**——它在 conda 环境 `python310_torch29_cuda` 内，不激活就
  `nohup: failed to run command 'vllm': No such file or directory`（启动即退出，极易误判为"服务起不来"）；
  ② **910C 镜像不含 `npu-smi`**（`npu-smi: command not found`，它是宿主工具），容器内查卡需退回 torch 侧
  （`torch.npu.mem_get_info`）。
  ⇒ 本方向已在统一启动脚本内消化（自动激活环境 / 快照降级）；**建议总组在基座层明确"镜像须自带可用服务入口"**，
  否则第三家接入者会各自踩一遍。
- 本轮仓库整理与验证**未新增或升级公共依赖**；容器内按需补装 `transformers`（v1 已登记）。
- **⭐ 基座层新发现（2026-09-22 追加）：FlagOS 官方镜像体系**（`flagos-ai/build-infra` 构建，
  registry 前缀 `flagos-base` / `flagos-runtime` / `flagos-dev` / `flagos-app`，当前版本 **2.2.0**）。
  **覆盖 14 个后端、含寒武纪**（FlagTree 手册覆盖不到的一家）；`configs.yaml` 是唯一 source of truth，
  文档站由它自动生成。**对我们的意义**：
  ① **910C 有官方对应物**：`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c:2.2.0`
   （digest `sha256:1048d622c928e86dd004ddb58b8b88602d91dc3c15458fda0262ca091e3ffb35`，5.4 GiB，
   官方标**宿主驱动前置 26.0.rc1**）—— py3.11 / torch 2.10.0+cpu / torch-npu 2.10.0 /
   **flagtree 0.7.0rc2+ascend3.5** / triton 3.5.0(+triton_ascend 3.2.1) / flag_gems 5.4.0-rc2.post3 /
   CANN 9.0.0（含 **910C Ops**），与我们锁定栈（CANN 9.0 / py3.11 / torch 2.10 / triton 3.5 / ascend3.5）
   **逐项一致**，只是来源不同（我们 = 组内 `flagrt` 镜像 + 华为 `vllm-ascend`）。
   ✅ **已在工作草稿 `dev/stack.lock.910c.yaml` 的 `candidates.train.official` 登记**（与既有组内自建候选
   `internal_v2` 并列，**仅登记、不生效、不切换**），并出专项对照
   [`910C/docs/OFFICIAL_RUNTIME_COUNTERPART_20260922.md`](910C/docs/OFFICIAL_RUNTIME_COUNTERPART_20260922.md)。
   ⚠️ **两处缺口须注意**：① 该镜像**设备后端为 `npu`（torch_npu，即 Route A）**
   ⇒ 若切它，训练腿的 `flagos`（torch_fl）**权宜例外可一并取消**（V2 里 10 月 TODO 随之关闭）；
   ② 但**官方 runtime 镜像不含 FlagCX**（`flagcx-ascend` 镜像线最后 push 2026-02-02，陈旧不可用）
   ⇒ **训练腿缺口未解、不可直接切换**，与「官方推荐镜像不含 FlagCX」是同一个已登记诉求。
   **是否切换由总组裁定，本方向不自行切换**；
   ② **每个 `base|runtime/<backend>.md` 都写明「Host driver」前置**，这是选镜像的**第一判据**（见下条）；
  ③ 建议将本体系列为《镜像选择与确定指南》的**最高优先级来源**（各方向已在按该口径对齐）。
- **⭐ 寒武纪档位受宿主驱动硬约束（2026-09-22 实测 + 官方口径）**：寒武纪在 FlagOS 官方体系里有**两档**，
  **不是按 torch 版本选，而是按宿主驱动选**：

  | 档位 | py | torch / torch-mlu / triton | 官方标注宿主驱动前置 |
  |---|---|---|---|
  | `flagos-runtime-cambricon-neuware4.4.3:2.2.0` | 3.10 | 2.7.1+cpu / 1.29.2 / 3.2.0+mlu1.7.2 | **6.2.15** |
  | `flagos-runtime-cambricon-neuware4.7.2:2.2.0` | 3.12 | 2.11.0+cpu / 1.33.1 / 3.4.0+mlu2.1.1 | **6.5.48** |

  两台测试机实测驱动 **v6.2.29** ⇒ 落 **6.2.x 线**，**只能用 4.4.3 档**；
  4.7.2 档需宿主驱动升至 **6.5.48**（会动宿主、影响他人）。
  **⇒ 请总组/管理员裁定走哪条**：A 直接用 4.4.3（同驱动线，风险最低）；B 升驱动到 6.5.48 后用 4.7.2。
  ⚠️ 6.2.29 能否跑标 6.2.15 的 4.4.3 **未实测**（同 minor 属推断），须起容器实测定论。
- **⭐ P800 存在两条血统，不要方向侧自行切换（2026-09-22）**：我们现用 FlagTree 线的
  `flagtree-xpu3.6-…:202608-base`（= FlagTree 手册给 P800 的唯一镜像，与类脑指向同一条 xpu3.6 线，
  **镜像本身无需调整**）；FlagOS 官方线则是 `flagos-runtime-kunlunxin-xre5.37.1:2.2.0`
  （flagtree 0.7.0rc2+xpu3.6，但**底层 SDK 换代到 XRE 5.37.1**，前置要求宿主驱动 **5.37.1**，
  而我们实测宿主为 **5.0.21.47**）。**⇒ 建议总组明确 P800 走哪条**，方向侧不自行切换。
- **⭐ 对昆仑芯 KL3 缺陷的官方印证（2026-09-22）**：FlagOS 官方 `configs.yaml` 的昆仑芯 vLLM 应用层原文——
  `# XPU_EVENT_KL3_ENABLE deliberately NOT set: it is the P1 fake-hang trigger (device timeout) on this XRE stack`
  —— 即**官方明确不设该变量**并称其为「P1 假挂死触发器」。这与本方向独立定位的结论一致，
  **可作为上报时「上游已承认该触发器」的旁证**。
- **⚠️ 一处待核对差异（不臆断）**：类脑 P800 记录 `Triton 3.5.0`，我们 P800 基线实测 `triton 3.6.0`
  （两者 `flagtree` 均为 `0.6.1+xpu3.6`）。未查明原因，只影响算子编译路径，**不影响本方向结论**。

## 下一步（拟在下次周会前推进）

**P800（第二实例）收敛 —— 本周剩余 + 下周**

1. ✅ **阶段 3 推理腿（2026-09-20 完成）**：单卡前向 `INFER_LEG_PASS 13/13` ——
   维度 **1024**（对齐 910C）、语义区分度 **0.6392**、**53.12 句/s**、p50 **56.17 ms**、
   真实参数异常 → **L2_PARAM/raise 且业务继续**。证据：`P800/probes/F_infer_leg_*`。
   - ✅ **vLLM 服务化形态亦已完成**（同日）：`SERVE_LEG_PASS 10/10` —— 区分度 0.4102、30.70 句/s、
     p50 96.4 ms、超长输入 → L2_PARAM/raise + 业务继续。证据：`P800/probes/F2_*`。
     ⚠️ 两条接入手册级环境要点：① `PYTHONPATH=/env/FlagGems/src` 是**硬前置**（site-packages 的
     `flag_gems` 安装不完整，缺 `runtime.backend.device`）；② 算子路径取 vendor
     （`VLLM_FL_PREFER=vendor` + `USE_FLAGGEMS=0`）以避开 FlagGems 路径与其 KL3 依赖。
2. ✅ **阶段 4 错误闭环（2026-09-20 完成）**：两设置对照（唯一变量 `XPU_EVENT_KL3_ENABLE`），
   两组均 **闭环 5 / 跳过 0 / 失败 0** 且**逐字节一致** ⇒ **关闭该变量不损失错误诊断能力**；
   并说明该厂商缺陷**不影响单进程设备上下文路径**。证据：`P800/probes/error_recovery_loop_kunlun_KL3{off,on}.json`。
2b. ✅ **官方 `-base` 镜像等价性验证（2026-09-20 完成）**：全套结论在官方推荐镜像上复现
   （conformance 13+6 逐用例一致、smoke 42/0、训练腿 loss 逐位相同、推理腿 `detail` 14/14 逐字相同、
   服务化 10/10、**KL3 挂死一致重现**）。⚠️ 期间查明 `-base` 开箱缺 `triton`，补齐命令已实测走通。
   证据与完整对照见 `P800/docs/KUNLUN_P800_BASE_IMAGE_EQUIVALENCE_20260920.md`。
3. ✅ **阶段 5 收敛（2026-09-20 完成，三件套）**：
   ① **《新芯片接入手册》** —— 4 条厂商栈判别路径 / 13 个抽象方法清单 / 8 步接入流程 /
   **可勾选验收清单 13 项** / 9 条跨芯片坑 / 厂商问题上报模板；
   ② **接口约定修订建议 6 条**（P800 是现行约定的首次非昇腾检验）：`device_type`/`vendor` 分离、
   `device_state` 入契约、`.native` 逃生舱约束、**错误对象跨模块类归一**（已修在框架层）、
   有界同步降级契约（前瞻性）、`known_issues()` 入契约；
   ③ **原型 release `runtime-v0.2.0`**（第二实例接入版）—— 第二实例接入 + 4 个框架修复 + 脚本后端无关化，
   **未改任何已有接口签名**（接口版本仍为 v0.1 原型期）。
2c. ✅ **多流 Stream 16 项验收基线逐项比对（2026-09-20 完成）**：P800 侧 **14 项通过 / 1 项如实标注不支持
   （S-12 流优先级，上游缺陷）/ 1 项不适用**；探针 8 项 `STREAM_SEMANTICS_PASS 8/8`（与 910C 逐项一致）；
   **S-7 图捕获首次实测 `GRAPH_CAPTURE_PASS 5/5`**（据此为 `kunlun` 补上 `graph_capture` 能力声明）；
   S-16 补测 2000 流无限制。探针已后端无关化并上提为**跨后端共用资产**
   （`prototype/probes/probe_stream_semantics_full.py`）。
   ⚠️ 过程中有一处**自我纠错**：S-7 首测判为"不支持"实为探针用法错误（捕获区内做了同步，违反 CUDA Graph 契约），
   修正后 5/5 通过——已撤回错误判断。本批内容按约定**随 10 月提交**。
4. **厂商缺陷上报**：待确认渠道后提交（主提交昆仑芯 XPytorch/XRE，抄送 FlagCX，
   知会 FlagGems/FlagTree）。

> 本日另发现并修复**第 4 个"只有非昇腾实例才暴露"的框架缺陷**（错误对象跨模块类不相等 → `disposition` KeyError），
> 已修在框架层（对既有后端无回归），详见 `P800/docs/KUNLUN_P800_STAGE34_VERIFY_20260920.md` §3。

**910C（第一实例）遗留**

5. **组件 v0.1.0 下发与反馈**：等下游接入，按周迭代 `v0.1.x`。
6. ~~组织架构合入 dev-1.0~~ → ✅ **已完成**（**PR #19**，2026-09-17 合入）：
   **P800 第二实例接入 + 按芯片重组目录（一份原型 + 两个芯片实例）**。
   合入时按约定排除了本方向基座工作草稿（`dev/stack.lock.910c.yaml`），草稿仍留特性分支；
   并借本次同步对齐了基座引用 v1 → **v2**。
7. **通信接口约定**：与分布式方向确认 flagcx 路线（`prototype/docs/DESIGN_DIST_COMM_20260908.md`）。

## 阻塞与需要协调的事项
### 🟠 第 3 家（寒武纪 MLU590）环境开通（2026-09-22 实测登记）

两台测试机（`Mlu-1` = 10.1.1.21、`Mlu-2` = 10.1.1.22）已可 SSH 免密登录，硬件与环境均具备
（各 **8 × MLU590-M9**，96 GB/卡；驱动 v6.2.29 / 固件 v1.5.0；11T 盘挂 `/srv`；
`cnmon` CNVMON v6.2.29 可用），但**接入动作被 root 权限阻塞**：

| # | 事项 | 实测依据 | 是硬需求吗 |
|---|---|---|---|
| 1 | **建 `/srv/hliu553` 并 chown 给 `hliu553`** | `/srv` 属主 `root:root 755`；实测 `mkdir: cannot create directory '/srv/hliu553': Permission denied`；虽在 `sudo` 组但 `sudo -n` 不可用（需密码） | ✅ 硬需求（否则数据只能放 `/home`，仅 208G/220G 且两机不共享） |
| 2 | **把 `hliu553` 加入 `docker` 组** | 现有成员 `gpfs, liangfan1, daizijian, huangxiang, qiyiyan, leihuhu`；我们 `id -nG` = `hliu553 sudo`；`docker.sock` 属 `root:docker` | ✅ 硬需求（本方向验证流程全部在带卡容器内） |
| ~~3~~ | ~~镜像获取~~ → ✅ **2026-09-22 已解除，不再阻塞**（见下方更正段） | — | 🟢 **已解除** |

> **⚠️ 2026-09-22 更正：原第 3 项「寒武纪镜像只能走官方渠道、需申请凭据或 tarball」不成立。**
> 实测 + 官方核对结论：
> ① FlagTree 确实**没有**寒武纪手册（wiki 26 页无 cambricon，17 个 `User-manual-for-*` 无 cambricon）
> —— 此条仍成立；但「因此镜像只能走寒武纪渠道」是**错误的推论**；
> ② **FlagOS 官方在 BAAI Harbor 上已有寒武纪三代镜像 + FlagGems 周测镜像**
> （`flagos-base` / `flagos-runtime` / `flagos-app` 共 12 个仓 + `flaggems/cambricon-flaggems-test-mlu590-m9de-*` 2 个仓）；
> ③ **实测可匿名拉取**（Docker Registry v2 匿名 token 取 manifest 成功，digest 已取得；
> `harbor.baai.ac.cn/api/v2.0/projects` 匿名可读、相关项目 `public=true`）；
> ⇒ **不需要寒武纪私仓凭据**（`docker.cambricon.com` 等三处私仓不再是硬前置）。

> **✅ 档位已定（2026-09-22 本方向决策，无需他人动作）**：
> 走 **`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`**
> （digest `sha256:e55b420ee98e0fdef6c18a27b633d67b988ecef81a52a0fef5a0c6636c91d5c2`，2.5 GiB，
> py3.10 / torch 2.7.1 / torch-mlu 1.29.2 / triton 3.2.0+mlu1.7.2）。
> 理由：其官方标注驱动前置 **6.2.15** 与我们实测 **v6.2.29** 同属 **6.2.x 线**，起容器风险最低、可立即解除阻塞；
> `neuware4.7.2`（官方标驱动 **6.5.48**）需动两台共享机宿主驱动、影响他人，且非当前瓶颈。
> **代价如实标注**：4.4.3 为旧档（与 4.7.2 差两代）——本方向对 torch-mlu 小版本不敏感，故可接受。
>
> **📋 驱动升级（→ 6.5.48 走 4.7.2 档）列为「上报预案」，非当前诉求**：
> **不凭版本号要求升级，只凭证据要求升级**。仅当原型接入 / 设备上下文验证出现
> 「疑与旧档强相关、且已排除我方五域」的不可解问题时启动；四条门槛（可复现现象 / 已排除我方五域 /
> 已排除环境与用法 / 能指出与档位的关系）与上报模板（问题 / 原因分列，含影响面与不升级后果）
> 见 `MLU590/docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` §0.2。
> 本方向承诺：**先把"能否不升宿主驱动、只在容器内换 4.7.2 用户态栈"的验证做在前面**，
> 确认必须动宿主驱动才对外提出。**纪律**：设备上下文结论必须标注取得时所处档位（同 P800 的条件标注要求）。
>
> **⚠️ 仍未验证（不臆断）**：① 6.2.29 能否跑标 6.2.15 的 4.4.3（同 minor 属推断）；
> ② 两台测试机能否出网拉 `harbor.baai.ac.cn`（本次只做了接口级验证，**未在带卡机 `docker pull`**）；
> ③ 应用镜像内 vLLM 是厂商移植版还是社区版 + 插件。
>
> 完整证据链见 [`prototype/docs/IMAGE_LINEAGE_ALIGNMENT_20260922.md`](prototype/docs/IMAGE_LINEAGE_ALIGNMENT_20260922.md)，
> 寒武纪专项更正与定档见 [`MLU590/docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md`](MLU590/docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md) §0。

**顺带澄清一条（无需任何人动作）**：docker 的镜像数据**本来就在 11T 盘上**——
`/var/lib/docker` 是 **→ `/srv/var/lib/docker` 的符号链接**（`readlink -f` 实证），
`/proc/mounts` 中 overlay2 的 `lowerdir/upperdir` 全部位于 `/srv/var/lib/docker/overlay2/`。
⇒ **不需要改 `daemon.json` 的 `data-root`**（改动需重启 docker，风险大于收益）。

请求 root 在两台各执行一次：

```bash
sudo mkdir -p /srv/hliu553 && sudo chown -R hliu553:hliu553 /srv/hliu553 && sudo chmod 750 /srv/hliu553
sudo usermod -aG docker hliu553        # 执行后需重新登录 SSH 生效
```


- **🔴 P800 镜像入锁（新增诉求，2026-09-20）**：P800 阶段 0–4 结论目前建立在**未入锁**镜像上
  （`flaggems-main-dev:202608`，无归档、未入锁），纪律上缺锁定基座背书。
  **本方向建议以官方 `-base` 为准**：`harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base`
  （digest `sha256:ea6d797a7d44ef97d7c0c0ed492f69c8ed2e024c927b2bfb5eef53e498e4eb34`，33.8 GB，血统 `maintainer: huangyun@kunlunxin.com`）。
  理由：① 该镜像上**已重跑出全套等价证据**（conformance 13+6 逐用例一致、smoke 42/0、两条腿 PASS、KL3 对照一致）；
  ② 有 digest、镜像层小 4.5 GB（33.8 GB vs 38.3 GB；磁盘占用 94.2 GB vs 107 GB）。
  ⚠️ **配方必须写明补齐步骤**：官方 `-base` 开箱**不含 `triton`**，须按官方手册 1.2 节
  `python3.10 -m pip install flagtree===0.7.0rc3+xpu3.6 --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple`，
  否则推理腿服务化不可复现（报 `Failed to infer device type`）。
  另请一并裁定：容器启动参数是否要求按官方手册给全（`--privileged --net=host --shm-size=256g --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined`）——
  本次用精简参数（非 privileged / bridge / shm 64g）仍全部跑通，但不应作为推荐值下发。
- **训练腿镜像未发布到 registry**（v1 标注 repro_status 🟡、临时机器绑定资产）：
  当前仅 npu1-27 可用，其他机器需向镜像 owner 取 `docker save` 包 → **请总组/镜像 owner 推进发布**，
  否则其他机器无法按锁定基座复现。
- **训练腿 torch_fl 例外的退出口径**：v1 记有 TODO「10 月起评估训练腿切回 Route A 的成本」，
  请总组给出时间表与责任方（涉及镜像是否需出带 torch_npu 的版本）。
- **通信接口约定**待分布式方向回复（启动方式 / flagcx 接口形态 / 对照用例归属 / 训练镜像是否换版）。
- **昆仑芯 P800：按接入规范新建「第二个芯片实例」（2026-09-14）**：阻塞已全部解除
  （连接 26008 / `docker` 组 / `/data2/hliu553`），容器 `hliu553-device-context-p800` 运行中。
  - **定位（重要）**：P800 是统一运行时原型的**第二个接入实例** —— 迁移的是
    《运行时层接口约定》的**规范与方法**，**910C 的实现与结论不迁移**（不照搬代码）。
    9 月完成原型接入与训推验证，11 月为整个运行时层的最终交付。
  - 本方向职责＝**原型搭建 + 接入的头和框架**：接入 → 训推验证暴露问题 → 属五域的**先简单修复**
    → 完成后把原型 **release 给运行时层其他子方向**做验证与迭代循环。
  - 算力/内存/CPU 充足；镜像 `flaggems-main-dev:202608` **本机已有** →
    跳过手册的 59.9 GB pull 与 32 GB load；FlagGems 源码已在容器内 `/env/FlagGems`（无需联网）。
  - **关键认知：昆仑芯设备 API 走 `torch.cuda`，`torch.xpu` 不可用**
    （编译标志 `USE_XPU=OFF` 实测；选卡变量为 `CUDA_VISIBLE_DEVICES`）→ 已写入接入规范建议。
  - **五域基线**：设备抽象 ✅ / 多流 Stream-Event ✅（**跨流 Event 依赖实测正确**）/
    FlagGems 算子 ✅（`add` max diff = 0.0）；错误 ⚠️（厂商码不可得）；状态恢复仅 probe 级。
  - **✅ 阶段 1（接入）已完成**：`backends/kunlun/` 落地；
     **conformance 13/13 + 推理 6/6**；`smoke_runtime.py` **42 通过 / 0 失败**；
     证据见 `prototype/runtime/conformance/conformance_runtime_kunlun*.json` 与
     `P800/probes/smoke_kunlun_20260914.txt`。
  - **接入过程暴露并已修 3 个「非昇腾实例才能暴露」的框架/判据缺陷**（与昆仑芯本身无关）；
    **2026-09-20 阶段 3 又暴露并修复第 4 个**（错误对象跨模块类不相等 → `disposition` KeyError，
    详见阶段 3/4 验证报告 §3）：
     ① `registry` 注册日志急切求值 `info()` → 缺厂商依赖时**中断整个 discover()**；
     ② conformance `f1` 硬要求厂商错误码，超出其自称的「类别/位置/根因」三投影契约；
     ③ `smoke_runtime.py` 只覆盖昇腾 → 新增「真实后端通用自检（后端无关）」。
     三条均已修复并回归（含向后兼容实测），详见接入方案 §7.5。
  - **【需对外提交】2 项**：① 流优先级 `torch.cuda.Stream.priority_range()` 稳定触发
    PyTorch `INTERNAL ASSERT FAILED at c10/cuda/CUDAStream.h:188`；② 厂商错误码不透出到 Python 异常。
  - **✅ 阶段 2（训练腿）已完成（标注条件）**：卡 6,7，**两 rank TRAIN_LEG_PASS 6/6**、
    loss **15.4488 → 11.1481**（50 步、无 NaN）、**3482 tok/s**（两卡合计、14.7 s）、
    通信三类对照全对（`3.0/3.0`、`[0.0, 1.0]`、P2P 一致）。**A/B 单变量对照**：
    同一脚本 `XPU_EVENT_KL3_ENABLE` **未设 → 退出码 0**；**设为 1 → 只到 `[step 0]` 即挂死、退出码 124**。
    ⇒ **规避方案有效，且缺陷确实存在，两者互为证明**。
    **诚实标注**：本组证据在 `XPU_EVENT_KL3_ENABLE` **未设置**下取得，**不能代表开启该变量时的行为**。
    证据：`P800/probes/E_train_leg_result_rank{0,1}.json`、`E_train_ab.log`。
    与 910C 基线对照（同构可比）：loss 15.4497→11.15 / 2117 tok/s ⇢ **15.4488→11.1481 / 3482 tok/s**。
  - 阶段 2 过程记录（集合通信验证与卡点定位）：分布式后端探测结论
    —— `nccl` 挂死、`xccl` 未编译（`Distributed package doesn't have XCCL built in`）、`kccl` 无响应，
    **可用路径只有 `flagcx`**：`import flagcx` + `init_process_group("cpu:gloo,cuda:flagcx")` + `FLAGCX_ADAPTOR=klx`
    （与 xliu969 已验证的 Route A 一致）。
    **a) 又一条接入手册级别的坑**：同一个 FlagCX，在 910C 上注册的后端名是 `flagos`，
      **在 P800 上是 `flagcx`** —— 换芯片不只换设备命名空间，连集合通信后端名也变。
    **b) 卡 6,7 复测：通信三类对照全通过**
      （`all_reduce got=[1.0,3.0,5.0,7.0]` / `all_gather [0.0,1.0]` / `p2p OK`，EXIT_CODE=0）。
    **d) 训练腿已跑到梯度同步环节，卡点已定性**：用 `faulthandler` 定时 dump 线程栈，
      两 rank 精确定位到 `proto_train_leg.py:164`（梯度同步循环内）→ 反推出
      **进程组建立 / 模型加载 / 前向+反向 / 三类通信校验全部通过**。
      定点探针进一步给出关键反差：**单次 2.38 GB 大通信 0.336 s 通过**，
      但**反复调用集合通信会非确定性挂死**（同一操作分别在第 3–4 次、第 41–50 次挂住）。
    **e) 归属判定：不在我方五域**。判定探针**使用裸 `torch.distributed.all_reduce`**
      （仅 `import flagcx` 注册后端），**未经过我方 `RuntimeBackend` 任何一层**；
      而 910C 上同一套训练腿 50 步稳定跑通（2117 tok/s）
      ⇒ 不是 FlagCX 整体不可用，而是 **P800 这一份构建/适配的问题** → 列为候选对外提交项（§7.4 C3），
      定案前需补：换卡复现 + 向上游确认（日志含 `[[BKCL-1245] allow masking GC signal handlers via BKCL_GC_SIGNAL_MASK]`）。
    **g) ⭐ 根因已定位到函数级（第二轮深挖）**：自建带 `SYS_PTRACE` 的调试容器 + gdb，
      抓到**三处挂死点**，全部在厂商 `libcuda.so`（实为 `libxpucuda.so` 兼容层）与
      flagcx c10d 插件的交界处：
      ① `torch.cuda.synchronize()` → `cudaDeviceSynchronize` → 厂商 libcuda **用户态自旋**；
      ② `c10d::flagcxBackend::allreduce` → **`flagcxBackend::syncStream`** → `CUDAEvent::record`
         → `cudaEventRecordWithFlags` → 厂商 libcuda 自旋（含 `sched_yield`）；
      ③ 通信域**首次初始化**死锁：rank1 阻塞在 `bkcl::net_socket_all_gather` 的 `recv()`，
         rank0 卡在 `bkcl::kl3::init_device_param` → `xpu_free`。
    **h) 判别条件（⚠️ 已更正，单变量探针 26 次运行）**：**两要素**，缺一不挂 ——
      ① `XPU_EVENT_KL3_ENABLE=1`；② 存在设备侧集合通信（flagcx）。
      该组合 18 次运行 16 次挂死（**≈89%**；本轮基线 4/4 = 100%），挂死步数游走
      （rep 0/20/30/40/70/100），**与数据量/形状/reduce op/用卡对/同步间隔均无关**。
      **更正说明**：初版误把「设备同步」列为必要条件之一，依据是旧探针
      `V5_nosync` 报「通过」——但**该探针不自证**（只是没等就退出，从未确认 120 次通信是否完成）。
      用**带真值校验**的新探针复测：KL3=1 + 循环内完全不显式同步 → **仍 2/2 挂死**
      （挂在第 101–120 次 `all_reduce` **内部**）。
      机理：`dist.all_reduce` → `flagcxBackend::allreduce` → **`syncStream`** → `CUDAEvent::record`
      —— **flagcx 插件每次 all_reduce 内部自带一次设备事件记录**，故未显式同步照样挂。
      三处暴露点：P1 `all_reduce` 内部（根本）/ P2 显式 `torch.cuda.synchronize()` / P3 通信域首次初始化（复现率低）。
      真值校验旁证：不设 KL3 时 `value = 2^120` **精确匹配（rel_err = 0.000e+00）** ⇒ 「通过」是真通过。
    **j) 责任层判定（核对结论）**：**既不是算子层、也不是编译层**，应提交**厂商运行时/驱动层** ——
      ① 算子层（FlagGems）排除：`flag_gems` **未导入**、探针源码 0 引用、无 `.pth` 自动 enable；
      ② 编译层（FlagTree/triton）排除：`/root/.triton` mtime 仍为镜像构建时、近 2 小时无编译产物、
      日志无 triton/jit/compile 字样、挂死路径上用的是 BKCL 预编译内核；
      ③ **指向厂商层**：`XPU_EVENT_KL3_ENABLE` 在全栈中**只被 `libxpucuda.so` 读取（1 处）**，
      且位于 `CUDA_*` 运行时旋钮块中；三处自旋帧均在 `libxpucuda.so` 内。
    **k) ✅ 镜像无关性（2026-09-20 新增证据，进一步收窄责任面）**：在**上游官方推荐镜像**
      `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base`
      （digest `sha256:ea6d797a…`）上，用同一探针、同一脚本、同一时段、同卡（XPU6,7）重跑：
      **A 组（设 KL3=1 + 集合通信）3/3 挂死；B 组（不设）2/2 通过且 `2^120` 真值精密匹配。**
      挂死现场特征与现用镜像一致：进程状态 `Rsl`、`utime` 累积至 ~9700（自旋）、
      卡 **100% 利用率而显存仅 366 MiB**、`timeout` 的 SIGTERM 无法中断（须 `kill -9`）。
      ⇒ 该缺陷**与镜像变体、容器参数无关，由厂商运行时/驱动层引起**；
      上报时不再可能被反问"是不是你们镜像的问题"。
      证据：`P800/probes/I_base_kl3_ab_20260920.log` 及 `I_base_kl3_A{1,2,3}*` / `I_base_kl3_B{1,2}*`。
      **提交建议**：主提交昆仑芯 XPytorch/XRE（含函数级栈 + 偏移 `+0x94080` + 最小复现）；
      抄送 FlagCX（`syncStream` 是否必需）；知会 FlagGems/FlagTree（复核该变量在 xpu3.6 是否仍必要，
      其回归为单进程单卡、覆盖不到本缺陷）。详见
      [`P800/docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md`](P800/docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md)。
    **i) ⚠️ 规避手段有代价，不可擅改**：不设/设 `0` 该变量后 4/4 全通过，
      但它同时是 **FlagGems kunlunxin 后端的官方推荐变量**
      （`tools/env.sh`、`src/flag_gems/backends.yaml`、CI `P800.yml` 三处均设 1）
      ⇒ 是否可关须上游确认，**本方向不擅自改锁定镜像口径**。
    **f) 实操教训**：`timeout` 的 SIGTERM **无法中断**这类挂死进程（挂死点持 GIL 自旋、信号被推迟），
      清理必须 `kill -9` 并按 PID 强杀，再复查 `xpu-smi` 确认卡释放。
    **c) ⚠️ 一次需要更正的判断**：曾把首次失败（卡 0,1 报 KL3 内核异常 `status=299`；
      卡 0,2 超时挂死）判为「flagcx/BKCL 在 P800 上的适配缺陷」。**该判断不成立** ——
      同一份代码在**卡 6,7** 上三类通信全通过；真因指向**共享机上被其他租户占用的卡**
      （实验期间卡 1 被他人反复占用 166→502 MiB / 100%，卡 0/3/4 亦有他人负载）。
      ⇒ 拟对外提交的「flagcx 缺陷」一项**已撤销**；`dma_excp_mask` 开关**无需变更**。
      ⇒ **实操纪律**：共享机上用卡前先 `xpu-smi` 挑**连续且空闲**的卡，并在实验记录写明用卡。
  - 详见环境汇总 / 基线实测 / 接入方案三份文档（`P800/docs/KUNLUN_P800_*_20260914.md`）。
  - **基座级约束建议（待总组裁定）**：① 昆仑芯机器需 `docker` 权限 + 自有可写数据目录；
    ② **昆仑芯设备 API 为 `torch.cuda` 而非 `torch.xpu`**（跨方向通用）。
- **【已上报 · 需芯片厂商适配】昆仑芯 KL3 事件同步概率性挂死（2026-09-14）**
  - **根本原因**：厂商 CUDA 兼容运行时 `libxpucuda.so`（`libcuda.so.1` 实为符号链接）——
    在 `XPU_EVENT_KL3_ENABLE=1` 且存在设备侧集合通信时，设备事件同步原语
    （`cudaEventRecordWithFlags` / `cudaDeviceSynchronize`）会**概率性永久自旋**。
    **归属已确认在厂商层**（算子层 FlagGems、编译层 FlagTree/triton 均已用硬证据排除）。
  - **复现率 ≈89%**（18 次运行 16 次；本轮基线 4/4 = 100%）；最小复现约 10 秒内挂死；
    三处自旋点（`dist.all_reduce` 内部 `flagcxBackend::syncStream` / 显式 `synchronize` /
    通信域首次初始化）全部自旋在 `libxpucuda.so` 内（一次实测偏移 `+0x94080`）。
  - **⭐ 上游已承认该触发器（2026-09-22 新增官方印证，上报时务必引用）**：
    FlagOS 官方镜像构建仓 `flagos-ai/build-infra` 的 `configs.yaml`（其镜像体系唯一 source of truth）中，
    昆仑芯 vLLM 应用层原文为
    `# XPU_EVENT_KL3_ENABLE deliberately NOT set: it is the P1 fake-hang trigger (device timeout) on this XRE stack`。
    三层意义：① 官方**明确不设**该变量（`deliberately NOT set`）—— 与我方规避动作**逐字一致**，
    不是我们自创的权宜手段；② 官方为其**命名并定级**「**P1 fake-hang trigger（设备超时）**」，
    与我方实测现象特征（进程 `Rsl`、`utime` 自旋累积、卡 100% 利用而显存仅 366 MiB、SIGTERM 无法中断）吻合；
    ③ 官方注释同时说明**这不是某个镜像变体的偶发**，与本方向「镜像无关性」结论（官方 `-base` 上 3/3 一致重现）
    **相互独立地指向同一结论**。
    ⇒ 上报时**可直接指向一条上游内部口径冲突**：FlagGems `backends.yaml` 设 `XPU_EVENT_KL3_ENABLE: "1"`
    而 FlagOS 官方 `configs.yaml` 明确不设 —— 请示知 FlagGems/FlagTree 对齐。
    ⚠️ **不可外推**：官方注释的作用域是该注释所在栈（`xre5.37.1`），我们实测环境是 FlagTree xpu3.6 线
    ⇒ 它证明的是「上游承认该变量是挂死触发器」，**不是**「我们遇到的挂死已被修复」。
  - **需协调**：请总组确认**上报渠道**（直连昆仑芯支持，还是经总组转达）；
    并转达两个问题：① `XPU_EVENT_KL3_ENABLE` 的语义是什么（**镜像与厂商文档里零说明**）？
    ② 关闭后设备异常是否仍能上报到 Python 层（影响我方「错误捕获」职责）？
  - **本方向不停摆（已按职责边界处理）**：
    ① **关键解锁**：我方训练腿（transformers 纯 torch）与推理腿（vLLM / vllm-plugin-FL，
    源码中 `flag_gems` **0 处引用**）**均不依赖 FlagGems**，故 `XPU_EVENT_KL3_ENABLE`
    **不属于我方验证前置条件** —— 在**本方向验证运行**中不设它，属"去掉不需要的变量"，
    **不是修改 FlagGems 口径**（不改公共资产）。
    ② **训练腿分档**：单卡证据（不受阻）+ 多卡在"不设该变量"下取证据（**逐条标注条件**）；
    "开启该变量下的多卡结果"标为阻塞项，**不硬凑、不伪造**。
    ③ **标注已落地五处**：`kunlun` 后端 `info()["known_issues"]`（机器可读，其他子方向接入即可见）
    + `proto_train_leg.py` 开跑前告警（不设变量时不误报）+ 核对报告 + 本文件 + 接入手册提纲。
    ④ **flagcx 侧不可规避**（已查源码）：`syncStream`（`backend_flagcx.cpp:424`）为流序正确性所必需，
    14 处集合通信各调一次，无开关 —— 故规避杠杆只在厂商那个变量上。
    ⑤ **阶段 4 错误闭环**将「设 / 不设该变量」两种设置各跑一次做对照（该变量可能影响设备异常上报，
    而错误捕获属我方五域）。
  - **若厂商不修**：训练腿以「单卡证据 + 多卡标注条件证据 + 归属判定 + 最小复现」交付，
    登记为「已识别、需上游修复、不影响其余交付」，**不阻塞 release**。
  - 详见 `P800/docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md` 与进度报告 §2.6。
- 本方向**不自行更换基座**，上述诉求提交总组裁定。

---

## 基座草稿本周状态（每周三填写）

**2026-09-16（周三）**

> **基座版本对齐说明**：总组已于 **2026-09-12 发布 `stack.lock.910c.v2.yaml`**，
> 本方向本次同步时一并对齐（此前一直引用 v1）。**v2 相对 v1 仅新增 `candidates:` 段**
> （登记训练腿候选新血统镜像 `dev/images/*/v2/`），**`lock:` / `rules:` / `per_leg:` 与 v1 逐字节一致**
> —— 即未发生任何镜像切换，本方向既有结论不受影响。

- 本周是否有需提总组的基座更新：**无 910C 侧新增**（以 `stack.lock.910c.v2.yaml` 为准）
- 工作草稿 `dev/stack.lock.910c.yaml` 本周是否改动：**未改动**
- 本周新增内容**全部属昆仑芯 P800 侧**（非 910C），故不写入 910c 草稿。**建议总组裁定是否在 v1 之外
  单列「新芯片（昆仑芯）机器使用约束」** —— 目前累计 3 条：
  ① 需 `docker` 组权限 + 自有可写数据目录（`/data2/hliu553`）；
  ② **设备 API 为 `torch.cuda` 而非 `torch.xpu`**（`USE_XPU=OFF` 实测，跨方向通用）；
  ③ **共享机用卡前必须 `xpu-smi` 挑「连续且空闲」卡并记录用卡**（他租户负载会污染结论，本轮已实际踩到）。
- **另需总组知悉**：P800 上存在**厂商运行时缺陷**（KL3 事件同步概率性挂死 ≈89%），
  已定位到函数级并如实标注，我方以「本方向验证运行不设该变量」规避、不阻塞其余交付；
  **上报渠道待确认**（直连昆仑芯支持 or 经总组转达）。

**2026-09-22（周二，提前填写 —— 本次为镜像血统核查带出的基座变更）**

- 本周是否有需提总组的基座更新：**有 2 项（均为登记性质，不改变现网）**
  1. **910C 新增「FlagOS 官方对应物」候选血统**：`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c:2.2.0`
     （digest `sha256:1048d622…`，5.4 GiB，官方标宿主驱动前置 **26.0.rc1**）。
     与我们锁定栈**逐项一致**；**设备后端为 `npu`（Route A）**，若切它可取消训练腿的 `flagos` 权宜例外；
     ⚠️ **但官方 runtime 不含 FlagCX**（`flagcx-ascend` 镜像线最后 push 2026-02-02，陈旧）
     ⇒ **训练腿缺口未解、不可直接切换**。是否切换请总组裁定。
  2. **建议把「宿主驱动前置（Host driver）」吸收进 `dev/images/<name>/v<N>/lock.yaml`**：
     FlagOS 官方每份 `base|runtime/<backend>.md` 都明示该行，这正是我们入档材料目前缺的字段
     —— 本次若早有此字段，寒武纪档位就不会先按 torch 版本锚错（现在已按驱动修正）。
- 工作草稿 `dev/stack.lock.910c.yaml` 本周是否改动：**有改动**
  （新增 `candidates:` 段：`internal_v2` 组内自建候选 + `official` 官方对应物候选；
  **`lock:` / `rules:` / `per_leg:` 段未动** ⇒ 未发生任何镜像切换，既有结论不受影响）。
  按约定草稿**留在特性分支、不直接 PR**，由总组收拢裁定后并入 v1/v2。
- **另需总组知悉（本日新增）**：P800 的 KL3 缺陷拿到了**上游官方印证** ——
  FlagOS 官方 `configs.yaml` 明确不设 `XPU_EVENT_KL3_ENABLE` 并称之为
  「**P1 fake-hang trigger (device timeout)**」⇒ 与我方独立定位一致，
  且暴露出 **FlagGems `backends.yaml`（设 1）与 FlagOS 官方 `configs.yaml`（明确不设）的口径冲突**，
  上报时可直接指向该冲突（详见 §阻塞与需要协调的事项 · KL3 条）。
- **寒武纪（第 3 家）侧不需要总组动作**：镜像已从 FlagOS 官方 Harbor 取得并定档 `neuware4.4.3`
  （驱动同 6.2.x 线）；仅剩 `docker` 组权限一项硬阻塞待管理员开通。
