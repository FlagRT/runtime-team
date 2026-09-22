# 统一运行时组件 v0.2.0 · 发布说明

> **版本**：v0.2.0（**第二实例接入版**）｜ **发布日期**：2026-09-20 ｜ **维护**：device-context（子方向 1）
> **分发单元**：`dev/device-context/prototype/`（自包含，可整体拷出）
> **Git tag**：`runtime-v0.2.0`（上一版 `runtime-v0.1.0`，2026-09-09）
> **接口状态**：**接口版本仍为 v0.1（原型期）**——本版**未改任何已有接口签名**，符合接口约定 §4 对 v0.1.x 的承诺
> （"吸收 9 月下游反馈，不改已有接口签名"）。组件版本与接口版本不强行对齐，理由见 §7。

---

## 1. 本版一句话

**统一运行时原型完成了第二个芯片实例（昆仑芯 P800）的接入与全链路验证，
并在过程中修掉 4 个"只有非昇腾实例才会暴露"的框架缺陷。**

**主张不变**：新增一家芯片 = 实现一个 backend + 跑通 conformance。
本版把这个主张从"一家实证"推进到**两家实证**，且第二家**单日完成接入**。

---

## 2. 本次交付（相对 v0.1.0 的增量）

### 2.1 新增后端

| 交付项 | 说明 |
|---|---|
| **`kunlun` 后端** | 昆仑芯 P800；`name="kunlun"` / `device_type="cuda"`；实现全部 13 个抽象方法；`known_issues()` 结构化声明 12 字段 |

新增一家芯片**未改动任何框架代码结构**——注册表已预留后端名，这正是插件化设计的预期效果。

### 2.2 框架修复（4 个缺陷，全部"只有非昇腾实例才会暴露"）

| # | 缺陷 | 现象 | 修法 |
|---|---|---|---|
| 1 | `registry.discover()` 的 `logger.debug(...)` **急切求值** | 未装某后端依赖时，日志参数的求值本身抛异常 | 改 DEBUG 守卫 + try/except |
| 2 | conformance `case_f1_error_translation` 硬编码依赖错误码映射 | 无 `error_map` 能力的后端（如 kunlun，厂商错误码不透出到 Python 层）无法如实跳过 | 按 `supports("error_map")` 分支，生成 stub-skip |
| 3 | `RuntimeBackend` 缺"已知问题"声明位 | 厂商缺陷只能写在文档/聊天记录里，下游接入者重复踩坑 | 新增可选方法 `known_issues()`（默认 `[]`，**零回归**） |
| 4 | **错误对象跨模块类不相等** → `disposition` 取 `KeyError` | `conformance/errors.py` 被 `importlib` 动态加载为独立模块，其 `ErrorCategory` 是 `IntEnum`，与 `api` 层枚举"取值相同但类不同"→ `api/errors.py` 的 `DISPOSITION[self.category]` 抛 `KeyError: ErrorCategory.L2_PARAM` | 新增 `coerce_category()` / `normalize_error()`，并在框架入口 `translate_via_backend()` 加归一化兜底（**修在框架层，第 3 家芯片不会重犯**）；`kunlun.translate_error` 显式归一 |

第 4 条是**真实崩溃**：P800 推理腿首次运行时第 5 节直接崩在 `api/errors.py`。
按昇腾后端自带 `_INT_TO_CATEGORY` 转换，所以昇腾上从不暴露——这是典型的"单实例测不出"缺陷。
本地做了无硬件验证（归一后 `disposition` 可取、幂等成立、各类输入行为正确，**并复现了修复前的 `KeyError`**）。

### 2.3 两条腿脚本后端无关化

| 脚本 | 变化 |
|---|---|
| `proto_train_leg.py` | 后端/通信后端/路径/模型/超参全部由 `DC_*` 环境变量驱动（默认值保持 910C 原口径）；新增 `_preflight_env_check()`，命中后端已声明的已知缺陷环境条件时**开跑前告警** |
| `proto_infer_leg.py` | **V2**：路径与模型走环境变量；设备串取 `runtime.current().device_type`（不再写死 `npu:0`）；同步走 `runtime.synchronize()`；**用真实异常注入替代原来的伪造错误码字符串**；厂商码用例按 `supports("error_map")` 分支；补 p50/p90 时延；`resolve_model()` 自动解析 `models--xxx → snapshots/<hash>` |
| `proto_error_recovery_loop.py` | 补 `DC_BACKEND`（与另两条腿一致） |
| `smoke_runtime.py` | 新增第 `[6]` 节「真实后端通用自检（后端无关）」+ `--backend` 参数 |

⇒ 同一份脚本在两个芯片实例上运行，**改动只有 `DC_BACKEND` 一行**。

### 2.4 验证资产

- conformance 结果并列归档：`ascend` / `kunlun` 两后端 13 例 + 推理 6 例（本版发布时在册的后端；
  第 3 家 `cambricon` 于同日稍后接入，见 RELEASE_NOTES 后续版本与各芯片目录）
- P800 探针与原始证据：`P800/probes/`（含真值校验探针、KL3 剂量-反应探针、一键脚本）
- **官方 `-base` 镜像等价性验证**：全套结论在官方推荐镜像上复现，并查明该镜像**开箱不含 `triton`** 的完整依赖断链

### 2.5 新增文档

| 文档 | 用途 |
|---|---|
`docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md` ｜ **《新芯片接入手册》**：4 条判别路径 · 13 抽象清单 · 8 步流程 · 可勾选验收清单 · 9 条跨芯片坑 · 厂商上报模板
`docs/INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md` ｜ **接口约定修订建议 6 条**（含 1 条已修代码、1 条前瞻性条款并如实标注）
`docs/REFERENCE_TWO_INSTANCES_CONFIG_20260920.md` ｜ 两实例验证配置与依据（镜像 / 模型 / 训推框架 / 参数逐项**依据链** + 可复现命令）
`docs/IMAGE_SELECTION_GUIDE_20260920.md` ｜ 镜像选择与确定指南（需求画像 / 来源优先级 / 入档入锁两道门槛）
`docs/IMAGE_REQUIREMENT_SPEC_20260920.md` ｜ 镜像需求说明书（提交总组）

---

## 3. 接入方式

```python
import sys; sys.path.insert(0, "<path-to>/prototype")
import runtime

runtime.use("kunlun")          # 或 "ascend" / "cambricon"；切换芯片只改这一行
runtime.set_device(0)
s = runtime.create_stream()
```

**P800（昆仑芯）环境前置**（实测硬前置，缺任一条推理腿服务化会报 `Failed to infer device type`）：

```bash
export PYTHONPATH=/env/FlagGems/src       # site-packages 里的 flag_gems 子模块不完整
export FLAGCX_ADAPTOR=klx                 # 集合通信唯一可用路径
# 若使用官方 -base 镜像，须先补齐 triton：
#   pip install flagtree===0.7.0rc3+xpu3.6 \
#     --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
```

- `prototype/` 目录**自包含**：conformance 依赖的 errors / recovery / device_state / npu_events 已内置，整体拷出即可用。
- 详细 API 与两条硬纪律见 `prototype/README.md`；接入流程见《新芯片接入手册》。

---

## 4. 三条纪律（新增第 3 条）

1. **跨流传递内存必须 `record_stream`** —— 否则缓冲可能被提前回收 → 数据竞争。
2. **错误隔离是分层的** —— API 级错误只影响该次调用、可流级重试；芯片级故障影响该设备全部流，
   流级重试无效，必须走 `recover_device`。处置一律按 `FlagosError.disposition`，**禁止按错误消息字符串判断**。
3. **【新增】错误对象必须归一为 api 层枚举** —— 后端 `translate_error()` 返回的 `category`
   必须是 `api/errors.py` 的 `ErrorCategory` 实例，**不得使用其他模块自行定义的、取值相同的枚举或整数**
   （否则字典查找会失败，本版第 4 个缺陷即由此产生）。框架侧已加归一化兜底，但后端仍应显式归一。

**补充纪律（沿用 v0.1.0 实测结论）**：调用 `translate_error` 必须传**完整原始消息**，不得截断——
截断会丢失语义导致保守误判（参数错误被判成可重放）。

---

## 5. 验证结果（两实例实跑）

同一套 conformance 判据、同一份两条腿脚本：

| 项 | 910C（第一实例，昇腾） | P800（第二实例，昆仑芯） |
|---|---|---|
| 组件自检 | 昇腾真机 **37/37**（无 NPU 环境时昇腾项自动 SKIP） | **42 通过 / 0 失败** |
| conformance（13 例） | `ascend` 13/13 | **`kunlun` 13/13** |
| conformance（推理 6 例） | 6/6 | **6/6** |
| 训练腿 2 卡微调 | 两 rank 6/6：loss 15.4497→11.15、2117 tok/s | **两 rank 6/6**：loss 15.4488→11.1481、3482 tok/s |
| 推理腿（单卡前向） | 10/10：区分度 0.638、66–79 句/s | **13/13**：区分度 0.6392、53.12 句/s、p50 56.17 ms |
| 推理腿（服务化） | 10/10：区分度 0.4123、108 句/s、p50 27.4 ms | **10/10**：区分度 0.4102、30.70 句/s、p50 96.4 ms、超长输入 → L2_PARAM/raise |
| 错误注入 → 恢复闭环 | 推理腿 5 闭环 / 训练腿 4 闭环（1 项如实跳过） | 设/不设 `XPU_EVENT_KL3_ENABLE` 两组**各 5 闭环 / 0 失败**且逐字节一致 |
| **官方镜像等价性** | — | ✅ 全部结论在官方 `-base` 上复现（conformance 逐用例一致、推理腿 `detail` **14/14 逐字相同**） |

---

## 6. 已知限制（如实列出）

| # | 限制 |
|---|---|
| 1 | **P800 已知厂商缺陷**：KL3 事件同步概率性永久挂死（现用镜像 16/18 ≈89%；官方镜像 A 组 3/3），归属**厂商运行时/驱动层**（算子层、编译层、**镜像因素**均已硬证据排除）。已结构化声明在 `kunlun` 后端 `known_issues()` 中，并已上报待裁定 |
| 2 | **P800 镜像尚未入锁**：第二实例结论建立在一个未入锁的镜像上。建议以官方 `-base` 入锁（等价性验证已通过），配方须含 `flagtree` 补齐步骤；诉求已登记 |
| 3 | 非昇腾两家的**有界同步口径不同**（如 P800 需按厂商原语判断），超时类能力须如实声明 |
| 4 | `real` 模式设备重建可用但**未作生产默认**，需多卡多进程压力测试调优后再启用 |
| 5 | 流优先级：ascend 支持范围查询与并发正确性，**调度效果**需压测验证 |
| 6 | 多流 Stream **16 项验收基线**尚未对新后端逐项比对（P800 目前只验了跨流 Event 依赖等少数项）<br>**（发布后补充，2026-09-20）已完成**：P800 逐项比对结果 **14 通过 / 1 如实标注不支持 / 1 不适用**，探针 `STREAM_SEMANTICS_PASS 8/8` 与 910C 逐项一致；据此为 `kunlun` 补上 `graph_capture` 能力声明。本批内容随下一版纳入。 |
| 7 | 历史模型（Qwen2.5-1.5B / Qwen3-4B）尚未在统一原型上复跑 |
| 8 | `ERR99999` 偶发进程终止：复现 1 次、后续 4 次未复现，**观察项，不作结论** |
| 9 | 跨芯片性能数字**不可直接横向比对**（910C 用厂商官方栈、P800 用社区 vLLM + FL 插件，形态与栈均不同） |

---

## 7. 版本节奏与反馈

- **组件版本**：v0.1.0（2026-09-09）→ **v0.2.0（本版，第二实例接入）** → v0.3.x（20 模型反馈迭代）→ …
- **接口版本**：仍为 **v0.1 原型期**（允许破坏性变更、会提前一周知会），稳定承诺在 **v1.0（计划 2027.06）**。
  两者不强行对齐的理由：本版是"接入能力扩展 + 缺陷修复"，**没有改任何已有接口签名**，
  按接口约定 §4 属于 v0.1.x 的承诺范围；10 月计划中的 "v0.2" 指的是**接口版本**的迭代节点。
- **反馈渠道**：device-context 方向；请附**复现脚本 + 结果 json**，便于定位。
- **接入新芯片前**：请先读《新芯片接入手册》与接口约定 §2（Backend 插件接入规范）。
- **使用规则**（含**带卡容器并发上限 3**，超限会导致 `acl.init()` 返回 500000、设备"消失"）
  见 `dev/stack.lock.910c.v2.yaml`，**使用前请先读**。
