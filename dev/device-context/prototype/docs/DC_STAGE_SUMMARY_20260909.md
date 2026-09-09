# 设备上下文 · 阶段性总结（2026-09-09）

> **范围说明**：本文是**设备上下文（子方向 1）自身职责**的阶段性证据汇总，
> **不是**精度与性能方向的全链路验收报告汇总。
> 我们只对自己这段负责：设备抽象与执行上下文、多流 Stream、错误码翻译、状态恢复，
> 以及基于统一原型的训练腿 / 推理腿设备侧证据。
> 其余子方向（显存、调度、算子适配、分布式、监控、精度与性能）的证据由各自方向出具。

- 分支：`kistich/device-context`
- 统一基座：`dev/stack.lock.910c.yaml`
- 统一原型：`dev/device-context/prototype/runtime/`
- 验收模型：Qwen3-Embedding-0.6B

---

## 1. 我们交付了什么

| 交付项 | 内容 | 位置 |
|---|---|---|
| 统一基座配置 | 锁定两腿镜像、使用规则（含并发上限 3）、合入把关五条 | `dev/stack.lock.910c.yaml` |
| 统一运行时 API | 后端选择 / 设备 / 流与事件 / 错误翻译 / 状态恢复 | `prototype/runtime/` |
| Backend 插件机制 | 抽象基类 + 注册表 + 自动发现 | `prototype/runtime/backends/` |
| 昇腾后端（torch_npu） | 推理腿使用 | `prototype/runtime/backends/ascend/` |
| FlagOS 后端（torch_fl） | 训练腿使用；依镜像约束新建 | `prototype/runtime/backends/flagos/` |
| 统一 conformance | 13 例 + 推理 6 例，跨后端可跑 | `prototype/runtime/conformance/` |
| 接口约定文档 | API 承诺 + 插件接入规范 + 两条硬纪律 | `prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` |

**设计主张**：新增一家芯片 = 实现一个 backend + 跑通 conformance。
本轮 flagos 后端从零到 conformance 13/13，是对这条主张的一次实证。

---

## 2. 训练腿证据（2 卡分布式微调，锁定训练镜像）

脚本：`prototype/runtime/proto/proto_train_leg.py`
接入：`runtime.use("flagos")` + `set_device(local_rank)`，通信 `torch.distributed(backend="flagos")`，底层 `flagcx`

| 判据 | 结果 |
|---|---|
| 设备上下文 | rank0 → 卡0、rank1 → 卡1，各建统一流 |
| 通信 all_reduce | 实测 3.0 = 期望 3.0 |
| 通信 all_gather | 收集到 `[0.0, 1.0]` |
| 通信 P2P | 双 rank 收发内容一致 |
| **loss（50 步）** | **15.4497 → 11.15**，单调下降、无 NaN、无死锁 |
| 吞吐 | 2117 tok/s（两卡合计） |
| 判定 | 两 rank 均 **6/6 TRAIN_LEG_PASS** |

**诚实标注**：两 rank 末值 loss 存在约 4e-3 差异（11.1497 / 11.1541），
源于集合通信浮点累加顺序差异，属固有特性（与 TP 数值不等价结论同源）。

镜像自带 canary（`verify_flagcx_p2p.py`）两 rank 亦全部 passed。

---

## 3. 推理腿证据（单卡推理，锁定推理镜像）

### 3.1 服务化形态（验收标准 2 要求"推理服务"）

脚本：`prototype/runtime/proto/proto_infer_serve.py`
形态：vLLM OpenAI 兼容服务（`--runner pooling --convert embed`），经统一 API 接入设备

| 判据 | 结果 |
|---|---|
| 设备上下文 | 后端 ascend、绑定 device 0；**服务运行同卡上跨流计算 = 3.0** |
| 服务就绪 | `/v1/models` 200，served model 正确 |
| 向量正确性 | 维度 1024、无 NaN/Inf、范数 1.000000 |
| 语义可区分 | 同主题 0.6980 vs 异主题 0.2857，**区分度 0.4123** |
| 吞吐时延 | **108 句/s**，p50 27.4ms |
| 错误注入 | 超长输入 → HTTP 400 拒绝 |
| 分级一致性 | **L2_PARAM / raise**（参数类上抛，重试无意义） |
| 业务继续 | 注入后请求正常、维度一致 |
| 判定 | **10/10 SERVE_LEG_PASS** |

### 3.2 前向形态（对照）

脚本：`prototype/runtime/proto/proto_infer_leg.py` —— 10/10：
区分度 0.638、66–79 句/s、向量有限且归一化。
（与 3.1 的差异主要在任务形态与批处理，两者互为补充。）

---

## 4. 错误注入 → 恢复闭环（验收标准 3，设备侧职责）

脚本：`prototype/runtime/proto/proto_error_recovery_loop.py`（后端无关，换 `--backend` 即可）

| 腿 | 结果 |
|---|---|
| 推理腿 ascend | **5 闭环 / 0 跳过 / 0 失败**：L2_PARAM→raise、L1_RESOURCE→retry 成功、**真实流同步超时→L3_EXECUTION→重放**、L4→`recover_device` 成功，业务均继续 |
| 训练腿 flagos | **4 闭环 / 1 跳过 / 0 失败**：超时因该后端无有界同步能力，**如实跳过不伪造** |

**归因核查（重要）**：上一版记录的两条"发现"（超时=进程级致命、一次性大显存 OOM 拖死进程）
经受控对照实验**均被推翻**；真实的收获是两个接口缺陷的修复：

1. `recover_device` 跨后端返回类型不一致（ascend `bool` / flagos `dict`）→ 统一为 `dict`
2. `recovered` 语义不一致（底层只在 ISOLATED 才重建，导致"设备正常无需重建"被误报为"恢复失败"）
   → 统一为"设备当前可用" + `detail` 三态区分

详见 `docs/ERROR_RECOVERY_LOOP_20260909.md`。

---

## 5. 组件自检（跨后端一致性）

| 项 | ascend | flagos |
|---|---|---|
| conformance 13 例 | ✅ 13/13 | ✅ 13/13 |
| 推理 6 例 | ✅ 6/6 | —（该腿为训练镜像） |
| 冒烟自检 | ✅ 37/37 | — |
| 有界同步 | ✅ 支持 | ❌ 不支持（如实声明） |
| real 重建 | ✅ 支持（待压测后作默认） | ❌ 仅 probe |
| 流优先级 | ✅ 支持 | ❌ 不支持 |

能力声明经 `supports()` 暴露，供 conformance 与上层生成"未支持项报告"，**不伪造**。

---

## 6. 已知限制与缺口（如实列出）

| # | 缺口 | 说明 |
|---|---|---|
| 1 | 历史模型未在统一原型上复跑 | 历史 910C 双卡 DDP（Qwen2.5-1.5B）与 vLLM+TP（Qwen3-4B）均为旧代码路径（`runtime.use` 出现 0 次） |
| 2 | `ERR99999` 偶发终止 | 复现 1 次、后续 4 次未复现 → **观察项，不作结论** |
| 3 | flagos 后端无有界同步 | 超时类能力不可用，已如实声明 |
| 4 | `real` 模式重建未作默认 | 本地多进程联调已过，生产默认前需多卡多进程压力测试调优 |
| 5 | 接口版本仍为 v0.1 | 原型期允许破坏性变更（提前知会）；稳定承诺在 v1.0 |
| 6 | 与分布式方向的通信接口约定 | 已出备忘（`docs/DESIGN_DIST_COMM_20260908.md`），待对方回复 |

---

## 7. 复现方式

```bash
cd dev/device-context/prototype

# 组件自检（本地即可，无 NPU 时昇腾项自动 SKIP）
python3 runtime/smoke_runtime.py

# 昇腾后端 conformance（推理腿镜像）
python3 runtime/conformance/runner.py --backend ascend
python3 runtime/conformance/runner.py --backend ascend --cases infer_cases

# FlagOS 后端 conformance（训练腿镜像，需 AUTOLOAD=0 且先 import torch_fl）
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 runtime/conformance/runner.py --backend flagos

# 训练腿 2 卡微调
torchrun --nproc_per_node=2 runtime/proto/proto_train_leg.py

# 推理腿服务化（先起服务）
vllm serve <model> --runner pooling --convert embed --port 8100
python3 runtime/proto/proto_infer_serve.py --backend ascend

# 错误注入 → 恢复闭环
python3 runtime/proto/proto_error_recovery_loop.py --backend ascend
```

---

## 8. 一句话结论

设备上下文在 910C 上完成了一次完整闭环：**统一基座已锁定、统一 API 与两个后端已可切换、
训练腿与推理腿均基于该原型拿到实证、错误注入到恢复的链路在两条腿上均可闭环**；
剩余为规模与覆盖面的扩展（见 §6），不再是机制性阻塞。
