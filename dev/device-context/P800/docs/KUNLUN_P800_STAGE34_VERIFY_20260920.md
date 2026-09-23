# P800 阶段 3 / 阶段 4 验证报告

> 日期：2026-09-20 ｜ 负责人：Kistich（hliu553）｜ 实例：昆仑芯 P800（第二实例）
> 环境：容器 `hliu553-device-context-p800` ｜ conda `python310_torch29_cuda`（torch 2.9.0+cu129 / transformers 4.57.1 / vLLM 0.13.0）
> 用卡：**XPU 6**（单卡；共享机上按 `xpu-smi` 挑选空闲卡）｜ 模型：`/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…`

---

## 0. 结论速览

| 阶段 | 结论 | 关键数字 |
|---|---|---|
| **阶段 3 · 推理腿（单卡前向）** | ✅ **PASS 13/13**（1 项如实跳过） | 维度 **1024** ｜ 语义区分度 **0.6392** ｜ **53.12 句/s** ｜ p50 **56.17 ms** |
| **阶段 3 补 · 推理腿（vLLM 服务化）** | ✅ **PASS 10/10** | 维度 **1024** ｜ 语义区分度 **0.4102** ｜ **30.70 句/s** ｜ p50 **96.4 ms** ｜ 超长输入 → **L2_PARAM/raise** + 业务继续 |
| **阶段 4 · 错误闭环** | ✅ **PASS，两设置完全一致** | 闭环 **5 / 跳过 0 / 失败 0**；KL3 设与不设**逐字节一致**（除时间戳） |
| **过程产出** | 🐞 **修复第 4 个"只有非昇腾实例才暴露"的框架缺陷** | 错误类型跨模块类不相等 → `KeyError`（详见 §3） |
| **环境产出** | ⚠️ 两条**接入手册级**环境要点 | 见 §2.2（`PYTHONPATH=/env/FlagGems/src` 是硬前置） |

---

## 1. 阶段 3 · 推理腿（单卡前向 + 向量正确性 + 错误分级）

**判据来源**：对齐 910C 推理腿口径（维度 / 语义区分度 / 句每秒 / p50 时延 / 错误分级 + 业务继续）。

### 1.1 与 910C（第一实例）同构对照

| 项 | 910C（第一实例） | P800（第二实例） | 判读 |
|---|---|---|---|
| 向量维度 | 1024 | **1024** | ✅ 完全一致 |
| 语义区分度 | 0.638（单卡前向） | **0.6392** | ✅ 同量级 |
| 范数 / NaN | 1.0 / 无 | **1.0 / 无** | ✅ |
| 吞吐 | 66–79 句/s（单卡前向）· 108 句/s（服务化） | **53.12 句/s**（单卡前向） | 可比，详见 §1.2 |
| p50 时延 | 27.4 ms（服务化） | **56.17 ms** | 形态不同，不做直接结论 |
| 错误分级 | 507046 → L2/L3 全链路 | **真实参数异常 → L2_PARAM / raise（confident）** | ✅ |

### 1.2 全部检查项（13/13）

| 检查 | 结果 | 明细 |
|---|---|---|
| device_count | ✅ | 可见设备 1（`CUDA_VISIBLE_DEVICES=6` 限定） |
| stream | ✅ | `<Stream backend=kunlun native=Stream>` |
| memory_stats | ✅ | `{total_mb: 98304, used_mb: 32, free_mb: 98272}`（96 GiB） |
| probe | ✅ | 设备探活 |
| model_load | ✅ | transformers → `cuda:0` |
| vector_dim | ✅ | 1024 |
| vector_normalized | ✅ | 范数 `[1.0, 1.0, 1.0]` |
| vector_no_nan | ✅ | 无 NaN |
| semantic_order | ✅ | 同主题 0.7822 > 异主题 0.1429 |
| semantic_gap | ✅ | **区分度 0.6392**（判据 ≥0.10） |
| inject_ok | ✅ | 真实注入：`mat1 and mat2 shapes cannot be multiplied (4x4 and 3x3)` |
| translate | ✅ | **L2_PARAM → raise**（`graded_by=message_hint`, `confident=True`） |
| business_continue | ✅ | 错误后业务继续，校验值 128.0 |
| vendor_code_map | ⏭ 如实跳过 | 该后端不支持 `error_map`（无厂商错误码透出）——不计入失败 |

**性能**：batch 3 × 5 轮 → avg 56.48 ms / **p50 56.17 ms** / p90 57.83 ms / **53.12 句/s**。

> 说明：本阶段只做**单卡前向形态**。910C 的 108 句/s 来自 **vLLM 服务化形态**，两者不可直接比
> ⇒ 服务化形态待补（见 §4）。
>
> 关于"错误码如实跳过"：昆仑芯的厂商错误码不透出到 Python 层（已在接入阶段定性与上报），
> 故本后端只有 `message_hint` 分级路径。**不伪造、不补零**，如实跳过并说明 —— 这是接入规范的要求。

### 1.3 阶段 3 补 · vLLM 服务化形态（2026-09-20）

**形态**：`vllm serve <model> --runner pooling --convert embed --port 8100 --max-model-len 4096 --enforce-eager`
（启动口径与 910C 一致）；平台 = 容器内社区 **vLLM 0.13.0** + **vllm-plugin-FL**（`VLLM_FL_PLATFORM=kunlunxin`）。

**结果：`SERVE_LEG_PASS 10/10`**（一键脚本 `probes/F2_vllm_serve.sh`，含启动 → 就绪等待 → 验证 → 停机 → 用卡复查）

| 检查 | 结果 | 明细 |
|---|---|---|
| device_context | ✅ | 后端=kunlun 设备数=1 |
| stream_on_serving_device | ✅ | **服务同卡上跨流计算 = 3.0**（服务与设备上下文共存不冲突） |
| service_ready | ✅ | `/v1/models` 200，served model = `qwen3-embedding-0.6b`（服务就绪 t=25 s） |
| embedding_valid | ✅ | 维度 **1024**、有限、首条范数 1.000000 |
| semantic_order | ✅ | 同主题 0.6964 > 异主题最小 0.2862 |
| semantic_gap | ✅ | **区分度 0.4102**（判据 ≥0.10） |
| throughput | ✅ | **30.70 句/s**，p50 **96.4 ms**，max 104.5 ms（batch 3 × 5 轮） |
| error_injection | ✅ | 超长输入（6001 tokens > max-model-len 4096）→ **HTTP 400 拒绝** |
| error_grading_consistency | ✅ | 统一分级 **L2_PARAM / raise**（与 910C 判据一致） |
| business_continues | ✅ | 注入后仍可正常请求，维度一致 |

**与 910C（第一实例）对照**

| 项 | 910C | P800 | 判读 |
|---|---|---|---|
| 服务化形态 | vllm-ascend（`--runner pooling --convert embed`） | vLLM 0.13.0 + vllm-plugin-FL（同参数） | ✅ 同口径 |
| 向量维度 | 1024 | **1024** | ✅ |
| 语义区分度 | 0.4123 | **0.4102** | ✅ 几乎一致 |
| 吞吐 / p50 | 108 句/s / 27.4 ms | **30.70 句/s / 96.4 ms** | ⚠️ 偏低，原因如实标注（见下） |
| 超长输入错误闭环 | L2_PARAM/raise + 业务继续 | **同** | ✅ |

> 吞吐/时延偏低的原因（如实标注，**不据此下性能结论**）：本次以 `--enforce-eager` 启动（关闭 CUDA Graph）
> 且算子路径取 vendor（`VLLM_FL_PREFER=vendor` + `USE_FLAGGEMS=0`，为避开 FlagGems 路径与其 KL3 依赖），
> 同时该卡为共享机上的单卡。**性能调优不在本方向职责范围**，此处只用于验证"服务化形态可跑通且语义正确"。

**⚠️ 两个环境要点（接入手册级，本次实测得出）**

1. **`PYTHONPATH=/env/FlagGems/src` 是硬前置**：`vllm_fl` 在 import 时依赖
   `flag_gems.runtime.backend.device.DeviceDetector`，而 site-packages 里的 `flag_gems`
   安装不完整（`pip show` 有 5.3.4.post1.dev12，但 `runtime.backend.device` 不可导入）
   → 不设该变量时 vLLM 直接报 `Failed to infer device type` 退出。
2. **算子路径选 vendor**：`VLLM_FL_PREFER=vendor` + `USE_FLAGGEMS=0` + `GEMS_VENDOR=kunlunxin`
   + `KLX_USE_AUTOTUNE=0` —— 避开 FlagGems 路径，从而不与已知厂商缺陷（KL3）产生依赖，
   与本方向"不设 `XPU_EVENT_KL3_ENABLE`"的锁定口径一致。

### 1.4 吞吐差异拆解：服务化 30.70 vs 前向 53.12 句/s

**动机**：同一模型、同一张卡（XPU6）、同一批文本（3 句 × 5 轮），前向形态 53.12 句/s 而服务化只有
30.70 句/s。需要弄清差异来源，避免把"形态差异"直接读成"插件劣化"。

**三层对照 + 一个单变量**

| 档 | 形态 | 句/s | p50 | 相对上一档 |
|---|---|---|---|---|
| **A** | 原型单卡前向（`transformers` 直跑，**不经过 vLLM**） | 53.12 | 56.17 ms | — |
| **B** | **vLLM offline 引擎**（`from vllm import LLM`，同 conda env、同插件、**无 HTTP**） | **35.63** | **84.14 ms** | p50 **+49.8 %** |
| **C** | vLLM 服务化（HTTP `/v1/embeddings`，`--enforce-eager`） | 30.70 | 96.4 ms | p50 **+14.6 %** |
| **C2** | vLLM 服务化（**去掉** `--enforce-eager`，启用图捕获） | 22.83 | 132.3 ms | p50 **+37.3 %（更慢）** |

**结论（三条）**

1. **主因在"引擎执行路径"，不在 HTTP**：A→B 占 p50 总差距 40.2 ms 中的 **28.0 ms（约 70%）**，
   B→C 只占 **12.3 ms（约 30%）**。即 vLLM 引擎（请求预处理 → 调度 → forward → pooling → 后处理，
   叠加插件注入的 platform / worker / 模型类覆盖）比"transformers 直跑"慢约 50 %，
   HTTP 往返与序列化再叠加约 15 %。
2. **`--enforce-eager` 不是瓶颈**：关掉它（启用图捕获）反而更慢（22.83 句/s、p50 132.3 ms）
   ⇒ 图捕获在本栈上无收益，**保留 `--enforce-eager` 的选择是对的**。
3. **vllm-plugin-FL 的贡献未能单独隔离（诚实标注）**：该插件**无法关闭** —— 两个芯片上的实测都表明
   它是"设备被 vLLM 识别"的前提（关掉即 `Failed to infer device type`；昇腾侧则相反，它在那边没有
   ascend 后端，必须禁用）。因此"引擎路径慢 50 %"里**插件 dispatch 与 vLLM 自身调度各占多少，本实验无法分离**。

**要分离插件贡献需要另一条对照（未做）**：用昆仑芯官方 `vllm-kunlun` 镜像
（本机他人容器在用的 `chenyunlong/vllm-kunlun-runtime:0.11.0-xpytorch20260226`，配 `VLLM_PLUGINS=kunlun`）
跑同一批文本。约束：① 换镜像属**基座变更**（本方向只消费不自建，结论性验证须在锁定基座内）；
② 其 vLLM 为 0.11.0，与本容器 0.13.0 不同代；③ 属另一套插件，结果不能与现有数据混用。

**旁证（本轮新发现的坑）**：对照过程中发现 `vllm serve` 被杀主进程后，
**`VLLM::EngineCore` 子进程会残留并持续占卡** —— 实测卡 6 仍被占 **73850 MiB / 96 GiB**，
导致下一次启动报 `Free memory on device (23.85/96.0 GiB) ... less than desired GPU memory
utilization (0.25, 24.0 GiB)`。已在 `F2_vllm_serve.sh` 的停机逻辑中一并 `kill -9` 清理（与 910C 侧
「清理残留 EngineCore 子进程」属同类纪律）。

---

## 2. 阶段 4 · 错误注入 → 恢复闭环（**两设置对照**）

**设计**：同一脚本、同一用卡，**唯一变量 `XPU_EVENT_KL3_ENABLE`（设 / 不设）**。
理由：该变量是厂商事件同步开关，可能影响设备异常上报路径；而错误捕获属我方五域职责
⇒ 「关掉它是否损失诊断能力」这一差异本身就是产出。

**结果**：两组**逐字节一致**（除时间戳）—— `ERROR_RECOVERY_LOOP_PASS`，闭环 **5 / 跳过 0 / 失败 0**。

| 注入项 | 分级 | 处置 | 设备侧动作 | 业务继续 |
|---|---|---|---|---|
| shape_mismatch | **L2_PARAM** | raise | 上抛调用方（参数类，重试无意义） | ✅ |
| oom | **L1_RESOURCE** | retry | `retry_ok` | ✅ |
| stream_timeout | **L3_EXECUTION** | replay | `recover_probe=ok` + `replay_ok`（真实 TimeoutError，进程存活） | ✅ |
| l4_by_code | **L4_FATAL** | device_recovery | `recover_probe=ok` | ✅ |

（两设置的四条结果完全相同，故只列一份。）

### 2.1 本阶段回答的问题

> **「关闭 `XPU_EVENT_KL3_ENABLE` 是否损失错误诊断能力？」→ 不损失。**

依据：在四类代表性注入（参数 / 资源 / 执行 / 致命）上，**两者的分级、处置、恢复动作、业务继续完全一致**。
补充边界说明：错误闭环是**单进程单卡、不走集合通信**；而 KL3 缺陷的两要素是「`KL3=1` ＋ 设备侧集合通信」，
故本对照**不触及**该缺陷，两组都跑完属预期 —— 这同时说明 **该缺陷不影响单进程设备上下文路径**。

---

## 3. 🐞 过程中发现并修复的框架缺陷（第 4 例）

**现象**：阶段 3 首次运行时，模型/向量/性能全部通过，但第 5 节错误分级**崩溃**：

```text
File "/workspace/prototype/runtime/api/errors.py", line 63, in disposition
    return DISPOSITION[self.category]
KeyError: <ErrorCategory.L2_PARAM: 2>
```

**根因（跨模块类对象不相等）**：`kunlun.backend._load_errors()` 用 `importlib` 把
`conformance/errors.py` 加载为**独立模块**（`dc_conformance_errors`），其 `ErrorCategory` 是
**IntEnum**（L1=1..L4=4）；而 `runtime/api/errors.py` 的是字符串枚举。两者**取值相同但不是同一个类对象**
⇒ `DISPOSITION[cat]` 查表失败。

**为什么昇腾没暴露**：`ascend.backend._to_unified()` 自带 `_INT_TO_CATEGORY` 数值转换，
把 IntEnum 归一为统一枚举；`kunlun` 当时是**直接透传** `fe.category`。

**修复（两处，均为最小变更）**

1. `runtime/api/errors.py`：新增公开工具 `coerce_category(value)`（接受统一枚举 / 整数与 IntEnum / 名称字符串）
   + `normalize_error(fe)`；并在**框架入口 `translate_via_backend()` 加归一化兜底** ——
   任何后端返回异类对象都会被重建为 api 层 `FlagosError`（幂等）。
2. `runtime/backends/kunlun/backend.py`：`translate_error()` 改用 `coerce_category(...)` 显式归一。

**验证（本地无硬件）**：归一后 `L2_PARAM → raise` 可取；`normalize_error` 幂等（同一对象返回）；
`coerce_category` 对 IntEnum / int / 字符串 / 大小写 / 非法值的行为逐一核对；并**复现了修复前的 KeyError**。

**意义**：这是**第 4 个"只有非昇腾实例才暴露"的框架缺陷**（前 3 个见接入方案 §7.5）。
三处修复的共同特征：**框架里隐含了"只有昇腾后端"的假设** —— 这正是"新建第二个实例"的价值所在。
修在框架层而非只修 kunlun，意味着**寒武纪接入时不会重犯**。

---

## 4. 为跑通阶段 3/4 做的代码改动（后端无关化）

| 文件 | 改动 |
|---|---|
| `runtime/proto/proto_infer_leg.py` | **V2 后端无关化**：路径/模型走 `DC_ROOT`/`DC_MODEL`/`DC_OUT_DIR`/`DC_ROUNDS`；设备串取 `runtime.current().device_type`（原硬编码 `npu:0`）；同步走 `runtime.synchronize()`；**真实异常注入**替代伪造的 ACL 错误码字符串；补 p50/p90；新增 `resolve_model()` 自动解析 `models--xxx → snapshots/<hash>`；清理死代码 |
| `runtime/proto/proto_error_recovery_loop.py` | `--backend` 支持 `DC_BACKEND` 环境变量（与另两条腿一致） |
| `runtime/api/errors.py` | 见 §3 |
| `runtime/backends/kunlun/backend.py` | 见 §3 |

---

## 5. 待办

| # | 项 | 说明 |
|---|---|---|
| 1 | ~~阶段 3 补：vLLM 服务化形态~~ ✅ **已完成（2026-09-20，`SERVE_LEG_PASS 10/10`）** | 见 §1.3；启动方式与两条环境要点已固化进 `probes/F2_vllm_serve.sh` |
| 2 | **阶段 5 收敛** | ① 《新芯片接入手册》（含六条关键认知 + 已知缺陷表 + 验收清单）② 接口约定修订建议（现已有 4 条候选：`device_type/vendor` 分离、`device_state` 入契约、`.native` 逃生舱约束、**错误对象跨模块类归一**）③ 原型 release |
| 3 | 厂商缺陷上报渠道 | 待定：直连昆仑芯 vs 经总组转达 |
| 4 | 四类通信/KL3 相关 | 厂商缺陷本体仍待厂商修复（已定性到函数级）；本方向不阻塞 |

---

## 6. 证据索引（`probes/`）

| 文件 | 内容 |
|---|---|
| `F_infer_leg.sh` / `F_infer_leg_20260920.log` | 阶段 3 执行脚本与完整日志 |
| `F_infer_leg_result_20260920.json` | 阶段 3 结果（13/13，含 env 与 perf） |
| `F2_vllm_serve.sh` / `F2_vllm_serve_20260920.log` | **阶段 3 补（服务化）**一键脚本与完整日志（启动 → 就绪 → 验证 → 停机 → 用卡复查） |
| `F2_serve_result_20260920.json` | 服务化结果（`SERVE_LEG_PASS 10/10`，含 perf 与错误分级一致性） |
| `F2_vllm_server_boot_20260920.log` | vLLM 服务启动原始日志（FL 平台插件激活 → 路由注册 → 就绪） |
| `G_error_loop.sh` / `G_error_loop_20260920.log` | 阶段 4 两设置对照脚本与日志 |
| `error_recovery_loop_kunlun_KL3off.json` | 阶段 4 结果：**不设** KL3 |
| `error_recovery_loop_kunlun_KL3on.json` | 阶段 4 结果：**设** KL3（与上面除时间戳外完全一致） |
