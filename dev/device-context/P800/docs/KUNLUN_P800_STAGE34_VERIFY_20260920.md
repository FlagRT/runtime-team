# P800 阶段 3 / 阶段 4 验证报告

> 日期：2026-09-20 ｜ 负责人：Kistich（hliu553）｜ 实例：昆仑芯 P800（第二实例）
> 环境：容器 `hliu553-device-context-p800` ｜ conda `python310_torch29_cuda`（torch 2.9.0+cu129 / transformers 4.57.1 / vLLM 0.13.0）
> 用卡：**XPU 6**（单卡；共享机上按 `xpu-smi` 挑选空闲卡）｜ 模型：`/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…`

---

## 0. 结论速览

| 阶段 | 结论 | 关键数字 |
|---|---|---|
| **阶段 3 · 推理腿** | ✅ **PASS 13/13**（1 项如实跳过） | 维度 **1024** ｜ 语义区分度 **0.6392** ｜ **53.12 句/s** ｜ p50 **56.17 ms** |
| **阶段 4 · 错误闭环** | ✅ **PASS，两设置完全一致** | 闭环 **5 / 跳过 0 / 失败 0**；KL3 设与不设**逐字节一致**（除时间戳） |
| **过程产出** | 🐞 **修复第 4 个"只有非昇腾实例才暴露"的框架缺陷** | 错误类型跨模块类不相等 → `KeyError`（详见 §3） |

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
| 1 | **阶段 3 补：vLLM 服务化形态** | 对齐 910C 的 108 句/s / p50 27.4ms 口径；容器内已有 vLLM 0.13.0，需准备启动方式与 `proto_infer_serve.py` 的昆仑芯适配 |
| 2 | **阶段 5 收敛** | ① 《新芯片接入手册》（含六条关键认知 + 已知缺陷表 + 验收清单）② 接口约定修订建议（现已有 4 条候选：`device_type/vendor` 分离、`device_state` 入契约、`.native` 逃生舱约束、**错误对象跨模块类归一**）③ 原型 release |
| 3 | 厂商缺陷上报渠道 | 待定：直连昆仑芯 vs 经总组转达 |
| 4 | 四类通信/KL3 相关 | 厂商缺陷本体仍待厂商修复（已定性到函数级）；本方向不阻塞 |

---

## 6. 证据索引（`probes/`）

| 文件 | 内容 |
|---|---|
| `F_infer_leg.sh` / `F_infer_leg_20260920.log` | 阶段 3 执行脚本与完整日志 |
| `F_infer_leg_result_20260920.json` | 阶段 3 结果（13/13，含 env 与 perf） |
| `G_error_loop.sh` / `G_error_loop_20260920.log` | 阶段 4 两设置对照脚本与日志 |
| `error_recovery_loop_kunlun_KL3off.json` | 阶段 4 结果：**不设** KL3 |
| `error_recovery_loop_kunlun_KL3on.json` | 阶段 4 结果：**设** KL3（与上面除时间戳外完全一致） |
