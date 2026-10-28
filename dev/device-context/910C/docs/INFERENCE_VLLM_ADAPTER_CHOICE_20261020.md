# 昇腾 910C 推理的 vLLM 适配层选型：vllm-ascend（现行）vs vllm-plugin-FL

> 用途：与**算子组**对齐「为什么 910C 推理腿用华为官方 `vllm-ascend`、不用 `vllm-plugin-FL`」。
> 本文自包含：现象、报错原文、数字、命令都在正文里，不依赖跳转；引用处只给路径供复核。
> 日期：**2026-10-20** ｜ 实例：第 1 家（910C，`torch_npu`）｜ 未测与不可外推的部分见 §6。

---

## 0 一页结论

| 问题 | 结论 |
|---|---|
| 是「两个插件二选一」吗？ | ❌ **不是**。**设备栈决定插件**：910C 走厂商官方 torch 插件（`torch_npu`）⇒ vLLM 的设备适配层只能是 `vllm-ascend` |
| `vllm-plugin-FL` 在 910C 能用吗？ | ❌ **不能**。它**没有 ascend 后端**；只要 `VLLM_PLUGINS` 里含 `fl`，vLLM 起服务即报 `RuntimeError: Device string must not be empty`（`current_platform.device_type` 为空） |
| 有没有「两者性能谁快」的数据？ | ❌ **没有，而且不该有**。它起不来 ⇒ 不存在同口径性能对比。**任何「vllm-ascend 快 N 倍」的表述都不成立** |
| 那为什么镜像里还留着 `vllm-plugin-FL`？ | 只在**训练镜像**里作为推理栈**遗留组件**保留：默认路径不激活（opt-in），且打了 1 个声明式补丁。它**覆盖不到 c10d**，解决不了训练腿的 A-2 |
| 换成别的芯片还成立吗？ | ⚠️ **不成立**。昆仑芯 P800 恰好相反：那边 `vllm-plugin-fl` 是**默认推荐底座**（必设 `VLLM_FL_PREFER=vendor` 绕开 K-2）。结论是**按栈/按芯片**，不是对插件的普适评价 |

---

## 1 两者不是同类项（先对齐定位）

| 维度 | `vllm-ascend` | `vllm-plugin-FL` |
|---|---|---|
| 维护方 | **华为昇腾官方**（`vllm-project/vllm-ascend`） | FlagOS 侧（`flagos-ai/vllm-plugin-FL`） |
| 面向的设备栈 | **`torch_npu`**（厂商官方 torch 插件） | **`torch_fl`**（FlagOS 原生栈） |
| 在本方向的角色 | **910C 推理腿发布物**（现行） | **训练镜像内的遗留组件**，默认路径不激活 |
| 分发与可复现 | 公开 registry **按 digest 拉取**，`repro_status` 🟢 | 随训练镜像构建期 `git fetch` + 1 个声明式补丁（`release/0.2` tip `8b05912`） |
| 已知限制 | **无**（限本方向已测范围，见 §6） | 两条跨栈缺陷（§2.3），已打补丁 |

---

## 2 910C 上的实测证据

### 2.1 可用性（决定性证据）

**`vllm-ascend`（现行推理腿）**

| 判据 | 读数 |
|---|---|
| 起服务（组内唯一入口 `serve_standard.sh`） | `SERVE_STANDARD_PASS (ready=1 smoke=1)` |
| 服务化四形态（TP=1/2 × `EAGER`=1/0） | **全 PASS**（TP=2 实测 `world_size=2` + `backend=hccl`） |
| 数值等价（Qwen3-4B dense，`greedy`，`max_tokens=64`） | TP=2、TP=4 对 TP=1 **逐字一致 4/4**（text + token_ids） |
| 吞吐（同口径，小模型 4B / 小 batch） | TP=1 **98.2** tok/s · TP=2 72.3 · TP=4 42.1（TP 越大越低，通信开销 > 计算收益，**属预期**；此处只证数值等价，不作为扩展性结论） |
| NaN / 乱码 | 均无 |

**`vllm-plugin-FL`（在 910C 上）**

```text
# 设了 VLLM_PLUGINS=fl 起服务
RuntimeError: Device string must not be empty
```

- 根因：它是为 **`torch_fl` 栈**提供设备层的插件，在 `torch_npu` 栈上会**干扰 ascend platform 的选择**，导致 `current_platform.device_type` 为空。
- ⇒ 形成**硬纪律**：910C 推理腿启动前 `unset VLLM_PLUGINS`（写进发版说明，不是建议而是必做项）。

> ⚠️ 注意这不是「装错版本」「参数没给对」这类可调问题：插件里根本没有 ascend 设备实现，**配置层面无解**。

### 2.2 配方与分发（作为发布物的前提）

| 项 | 值 |
|---|---|
| 镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` |
| digest | `sha256:5cf8a2b6db8b06eb1bc7fc7d191d667aebf2b197351bdba13f776918c11ec7a7` |
| 获取 | 公开 registry 直接 `docker pull …@sha256:5cf8a2b6…`，**不需要**离线包传递 |
| 可复现性 | 🟢 digest 锁定；**被动继承的官方产物**，镜像不可重建是设计边界（非本组配方） |
| 携带的训练腿限制 | **不携带** A-2 / A-3 / A-4 任何一条 |

### 2.3 `vllm-plugin-FL` 在我们这里踩过的两个真缺陷（**建议算子组一并看**）

| # | 现象（可复现） | 修复 | 性质 |
|---|---|---|---|
| ① | ascend 后端 **`bool × int` 类型提升**错误 ⇒ token id 坍缩成 `1`，embedding 查表命中同一行 | 显式转 `int64`（`vllm_fl/dispatch/backends/vendor/ascend/impl/vocab_parallel_embedding.py`） | 跨栈缺陷，与设备无关的类型规则 |
| ② | Python communicator 的 `all_reduce` / `all_gather` **返回后没有同步** ⇒ 通信未完成即读，出现 NaN（round 3 真机交替对照证实：不加 sync 复现，加 sync 转 PASS） | `all_reduce` 后补 `torch.npu.synchronize()`；新增 `all_gather` 覆盖实现（逻辑同基类 + sync） | 与训练腿 A-2 同族（完成语义），但**只覆盖 Python 侧** |

- 补丁形态：`0001-fix-ascend-int64-mask-promote-flagcx-sync.patch`
  （`sha256 b97d0d8b2a22539a2d2887bc0345d9e229ea77cc4414c24e8e5fe6ce23b5f928`，来源 FlagRT fork commit `5d545c9`），
  构建期 `git apply` 后安装；**上游合入后整层删除**（上游优先，不切私有 fork）。
- ⚠️ **覆盖面边界（关键）**：该补丁只覆盖**推理侧 Python communicator**，**覆盖不到 c10d 路径**
  ⇒ 训练腿的集合通信完成语义缺陷（A-2）**它解决不了**，必须在 FlagCX 侧修（已修完并合入，见 §7）。

---

## 3 选型判据（按权重）

| # | 判据 | 为什么是这条 | 落点 |
|---|---|---|---|
| 1 | **栈匹配** | 910C 的设备栈是 `torch_npu`（2026-09-22 起三实例统一走厂商官方 torch 插件）。vLLM 的设备层必须与设备栈同源，否则连 platform 都判定不出来 | `vllm-ascend` ✅ / `vllm-plugin-FL` ❌ |
| 2 | **可复现分发** | 发布物必须能被别人按固定内容拉到（digest 可核对），否则「结论在哪台机器上成立」说不清 | 公开 registry + digest ✅ |
| 3 | **维护归属** | 设备层越贴近厂商越省事：算子/通信的实现细节在厂商手里 | 官方维护 ✅ |

> **代价（如实说）**：选官方插件 = **把设备层交给厂商**，我们不改它、也改不动它；遇到问题只能报。这也是**训练腿**仍保留 `torch_npu` + FlagCX 自维护路线的原因（训练侧要能自己修通信语义）。

---

## 4 与训练腿 / 其他芯片的边界

| 范围 | 现状 |
|---|---|
| **910C 训练腿** | `torch_npu` + **FlagCX（我方自维护）**；vLLM 不参与。镜像内 `vllm-plugin-FL` 仅保留、默认不激活（`Torch-FL` 同理，opt-in） |
| **路线 B（`torch_fl`）** | **2026-09-22 整体退出**：原型内已无活跃路径，`use("flagos")` 如实报 `BackendNotFound`（不是禁用，是断开）。历史资产保留 + 归档横幅，保住 A/B 选型的原始证据链 |
| **P800（昆仑芯）** | **相反结论**：`vllm-plugin-fl` 是默认推荐底座，必设 `VLLM_FL_PREFER=vendor` 绕开 K-2（`index_select` 走 FlagGems OOT 路径触发 `CUDA error: invalid device function`）。⚠️ **910C 的结论不能套到 P800** |

---

## 5 给算子组的三条（可直接落地）

1. **别把「装上 `vllm-plugin-FL`」等同于「昇腾上能用」**：它目前**没有 ascend 后端**。若计划在昇腾上走它，前置是补后端 + 修 platform 判定，否则只会得到 `Device string must not be empty`。
2. **§2.3 那两个缺陷是跨栈的、可复现的**（类型提升 / 通信完成语义），做 FlagOS 侧的 ascend 支持时必须一起修；补丁已按 upstream-first 整理，需要可直接取（sha256 见 §2.3）。
3. **`VLLM_PLUGINS` 这类全局开关会污染 platform 判定**：在可能同时装两个插件的环境里请**显式 `unset`**，不要依赖默认值——它决定的是「谁来当设备层」，不是「启用哪个优化」。

---

## 6 没测的 / 不能外推（先说清边界）

| # | 项 | 说明 |
|---|---|---|
| 1 | **未做两者同口径性能对比** | 前置条件不存在（`vllm-plugin-FL` 在 910C 上起不来）⇒ 不得表述为「某插件更快/更慢」 |
| 2 | 模型与场景覆盖窄 | 只测 **Qwen3-4B dense**、`greedy`、`max_tokens=64`、TP=1/2/4；**MoE、其他参数规模、高并发/长上下文未测** |
| 3 | 未测第三方对 `vllm-plugin-FL` 的昇腾改动 | P800 那份是**厂商镜像内预装**的版本（`f9d95f4…`），非我们构建；不能据此判断昇腾侧是否存在可用分支 |
| 4 | 性能数字仅作数值等价的附带读数 | 小模型小 batch 下 TP 越大吞吐越低是**预期**，不构成扩展性结论 |

---

## 7 证据索引（仓内路径，供复核）

| 结论 | 证据 |
|---|---|
| 起服务 PASS / 服务化四形态 / TP 数值等价与两个坑 | `dev/device-context/910C/distributed_inference/docs/INFERENCE_QWEN3_TP_COMPARE_20260901.md` |
| 推理腿镜像登记（digest / image id / 不携带 A 条） | `dev/images/ascend-infer-vllm/v1/README.md` · `.../image.baseline.yaml` |
| 硬纪律 `unset VLLM_PLUGINS` 与限制清单 | `docs/基座发版说明-20261014.md` §「推理腿必须设置」 |
| `vllm-plugin-FL` 补丁（内容 / sha256 / 来源 commit / 保质期） | `dev/images/ascend-operator-runtime/v3/assets/patches/vllm-plugin-FL/README.md` |
| 路线 B 退出与归档处置 | `dev/device-context/prototype/docs/ROUTE_B_ARCHIVED_20260922.md` |
| P800 上 `vllm-plugin-fl` 为默认底座 + K-2 | `docs/基座发版说明-20261014.md` §「昆仑芯 P800 · 已知限制」 |
| 训练腿 A-2（集合通信完成语义）与修复验收 | `dev/issues/flagcx-c10d-collectives/README.md` · `dev/device-context/910C/docs/FLAGCX_C10D_COMPLETION_20261016.md` |
