# 接口约定修订建议（设备上下文章节）

> 版本：**v1.1** ｜ 日期：2026-09-20（v1.0）/ **2026-09-22（v1.1 增补）** ｜ 提出方：device-context 子方向（Kistich）
> **对象**：`prototype/docs/INTERFACE_CONTRACT_DC_20260908.md`（现行 v0.1.0 定稿）
> **由来**：该约定是**昇腾单一实例**时期定稿的。
> 第 2 家（昆仑芯 P800）接入是对它的**首次非昇腾检验**（v1.0 的 6 条）；
> **第 3 家（寒武纪 MLU590）接入是第三次检验**，本轮又暴露 3 条（**第 7–9 条**，
> 含一条**在 910C 第一实例上长期存在的缺陷**）—— 明细见
> [`BACKEND_SYMMETRY_AUDIT_20260922.md`](BACKEND_SYMMETRY_AUDIT_20260922.md)。
> **写法**：每条给出 ①现状 ②问题（实测依据，可独立阅读）③建议条文（可直接抄进约定）④兼容性影响。
> 第 1–4、6–8 条有实测依据；第 5、9 条为**前瞻性/操作性条款**，已如实标注。


> ⚠️ **路线 B（torch_fl）历史档案 —— 已冻结，非当前口径**
> 本文记录的是**当时**的做法与结论。路线 B 已于 2026-09-22 整体退出：原型里的该后端已**删除**，
> 三个芯片实例（昇腾 910C / 昆仑芯 P800 / 寒武纪 MLU590）**当前一律走厂商官方 torch 插件路线**
> （`torch_npu` / `torch.cuda` 兼容层 XPytorch / `torch_mlu`）。
> 当前口径见 `dev/device-context/README.md` 与各芯片目录 `README.md`；取舍依据见
> `summary/DEVICE_ABSTRACTION_ROUTE_AB_SUMMARY_20260922.md`；归档索引见
> `dev/device-context/prototype/docs/ROUTE_B_ARCHIVED_20260922.md`。

---

## 汇总

| # | 修订项 | 类型 | 依据强度 | 当前是否已改代码 |
|---|---|---|---|---|
| 1 | `device_type` 与 `vendor` 分离 | **契约修订** | 强（三后端取值已不同） | 未改（建议先定契约） |
| 2 | `device_state` 纳入 Backend 契约 | **契约补漏** | 强（声明与实现不符） | ✅ **已修**（`kunlun` 补实现；另发现文档四态名有误） |
| 3 | `.native` 逃生舱约束 + `record_stream` 能力位 | **契约补漏** | 中（设计风险，未致故障） | 未改 |
| 4 | 错误对象**跨模块类归一** | **契约 + 已修** | 强（真实崩溃，已复现） | ✅ 已修在框架层 |
| 5 | 不支持有界同步时的声明与降级契约 | 前瞻性条款 | 弱（未触发） | 未改 |
| 6 | `known_issues()` 纳入契约 | **契约补漏** | 强（两家实例已用） | ✅ 已实现（可选方法） |
| **7** | **能力降级必须"整组一致"**（可观测字段不得只改其一） | **契约 + 已修** | **强（三家实例同款，已实测复现）** | ✅ 已修（`kunlun` / `cambricon`）+ 4 条防回归判据 |
| **8** | **声明的能力必须可调用**；`info()["supports"]` 键集合按能力全集呈现 | **契约补漏 + 已修** | **强（910C `flagos` 实测 AttributeError + 键名漂移）** | ✅ 已修（`flagos`）+ 2 条自检判据 |
| **9** | **验证资产的可达性与证据卫生**（自检须覆盖全部已接入后端；脚本不得硬编码单一厂商的错误码） | 操作性条款（建议进接入手册） | 中（已致 2 处假证据 + 1 处长期盲区） | ✅ 已做（自检扩到四家 + 显式 SKIP；清掉硬编码码） |
| **10** | **厂商扩展是"懒加载"：凡拼厂商专有字符串（设备串 / 集合通信后端名）之前，必须先经后端触碰一次设备** | **契约补漏（新增条款）** | **强（910C 实测两处，其一使 conformance 整轮 ABORT）** | ✅ 已修（`conformance/runner.py` warm-up、`proto_train_leg.py` 进程组初始化）+ 写入接入手册坑 10 |

---

## 1. `device_type` 与 `vendor` 分离

### 现状

`base.py:30` 要求实现者提供 `name` / `device_type` 两个类属性，`base.py:36` 声明 `device_type: str = ""`，
`base.py:146` 把它放进 `info()` 返回。三个后端当前取值：

| 后端 | 取值 | 源码行 | 该行注释 |
|---|---|---|---|
| `ascend` | `"npu"` | `ascend/backend.py:38` | — |
| `flagos` | `"flagos"` | `flagos/backend.py:97` | `# 设备串前缀：flagos:0` |
| `kunlun` | `"cuda"` | `kunlun/backend.py:113` | `# 见模块 docstring：设备串按命名空间，不按厂商` |

### 问题

**同一个字段被同时当作两个语义使用**——"设备串前缀"与"厂商标识"。三个取值已经把这两件事的矛盾暴露出来：

1. `"flagos"` **不是合法的设备串前缀**（`torch.device("flagos:0")` 在通用语义下无意义），它是**厂家/插件标识**；
2. `"cuda"` 是**合法的设备串前缀**，但它**不能标识厂商**——昆仑芯复用 `torch.cuda` 命名空间
   （编译标志 `USE_XPU=OFF`；`torch.xpu.is_available()` 抛 `AssertionError: Torch not compiled with XPU enabled`；
   `torch.cuda.device_count()` 返回 8），**将来接 NVIDIA 也是 `cuda`，两者会撞名**。
   `kunlun` 后端自己的注释已承认"设备串按命名空间，不按厂商"；
3. 上层若按 `f"{device_type}:{i}"` 拼设备串，在 `flagos` 上得不到有效设备串，在 `cuda` 上拿不到厂商标识。

### 建议条文

```
RuntimeBackend 须声明三个类属性：
  name        插件名（唯一注册名，如 "kunlun"）
  device_type 设备串前缀，取值必须能被 torch.device("<device_type>:0") 构造（如 "npu" / "cuda" / "mlu"）
  vendor      厂商标识（唯一且稳定，如 "ascend" / "kunlunxin" / "cambricon"）—— 新增，必填

约束：
  · 厂商判别一律用 vendor，禁止用 device_type 判别；
  · device_type 用于拼设备串；同命名空间可被多厂商复用（cuda → kunlunxin / nvidia），
    因此 (device_type, vendor) 才是完整标识；
  · info() 返回值新增 "vendor" 字段。
```

### 兼容性

新增字段，对既有调用**无破坏**；`ascend` / `flagos` 的 `device_type` 取值可保持现状
（`npu` 合法；`flagos` 虽然是插件名而非设备串前缀，但按本条应改为 `"flagos"` 之外的合法前缀，
或明确其设备串由后端自身解析、上层不得拼接）。建议在 v0.1.x 内以"新增 + 文档澄清"落地，v0.2 起强制。

---

## 2. `device_state` 纳入 Backend 契约

### 现状

| 位置 | 情况 |
|---|---|
| `runtime/__init__.py:49` | `device_state` **在 `__all__` 中对外暴露** |
| `runtime/__init__.py:101-102` | `def device_state(ordinal=0): return current().device_state(ordinal)` —— **直接转发到后端** |
| `backends/base.py` | **既无抽象声明、也无任何提及**（`def device_state(` 与字符串 `device_state` 均为 0 命中） |
| `ascend/backend.py` | ✅ 有实现 |
| `flagos/backend.py` | ❌ 无实现 |
| `kunlun/backend.py` | ❌ 无实现，**但 `_capabilities` 里声明了它** |

### 问题

在 **API 层暴露、契约层缺位**，导致三种不一致：

1. **调用即崩**：在 `flagos` / `kunlun` 上调用 `runtime.device_state()` 会直接 `AttributeError`
   —— 上层看到接口存在就会用，这属于"承诺了一个不存在的接口"；
2. **声明与实现不符（更严重）**：`kunlun/backend.py:130` 在 `_capabilities` 里写着
   `"device_state",  # 四态机`，于是 `supports("device_state")` **返回 True**，
   而实际调用会 `AttributeError`。这比"不支持"更糟——**"如实声明能力"是本层的基本纪律，这里恰好违反了它**；
3. 接口约定 §1.5 已把 `device_state(ordinal)` 写进**API 承诺**（"设备四态查询
   AVAILABLE / DEGRADED / ISOLATED / UNKNOWN"），但 §2 的 Backend 插件规范里没有它
   —— 新接入者按 §2 实现时会漏掉，事后才在上层调用时发现。

**附带发现（第四种不一致，本次核对时发现）**：接口约定 §1.5 的**四态名写错了**——
文档写 `AVAILABLE / DEGRADED / ISOLATED / **UNKNOWN**`，而实现（`conformance/device_state.py` 的
`DeviceState` 枚举）是 `AVAILABLE / DEGRADED / ISOLATED / **DESTROYED**`（"已销毁：优雅退出/资源回收完成"）。
`ascend.backend.device_state` 的 docstring 也写作 `DESTROYED`，即**文档与两处实现不一致，错在文档**。

### 建议条文

```
§1.5 状态恢复（API 承诺）：
  ① 四态名更正为 AVAILABLE / DEGRADED / ISOLATED / DESTROYED（与实现一致）；
  ② 补一句：该能力由后端实现；后端若不支持，必须在 supports("device_state") 中如实声明为 False，
     使调用方可预判（禁止声明为 True 却无实现）。

§2 Backend 接入规范：将 device_state 列入"必须实现或如实声明不支持的接口"：
  device_state(ordinal) -> DeviceState   设备四态之一（AVAILABLE / DEGRADED / ISOLATED / DESTROYED）
```

### 兼容性

对 `ascend` 无影响（已实现）。

**`kunlun` 已修（2026-09-20）——选了"补齐实现"而不是"删声明"**，理由与做法：

- **为什么补实现**：`device_state` 的实现是 `conformance/device_state.py` 里的**进程内状态机**
  （不依赖任何厂商原语，四态 + 转换事件 + 订阅者），`ascend` 就是这么复用的，
  `kunlun` 已有 `_load_errors()` 的同类加载模式 ⇒ 补齐成本极低，
  且能消除"上层在 P800 上调 `runtime.device_state()` 直接崩"这一真实问题（只删声明解决不了它）。
- **做法（含一个必须注意的坑）**：新增 `_load_device_state()`，**用标准 `import`（共享 `sys.modules`）**
  ——与 `ascend` 一致；**不能**照抄 `_load_errors()` 的 `importlib` 独立模块名加载方式。
  原因：`errors` 是无状态纯函数（两份无所谓），而 `device_state` 是**有状态单例**，
  两次加载会得到**两份状态机**（conformance 设的状态后端查不到、反之亦然）——
  那种"静默错"比 `AttributeError` 更难查。代码内已写明这一约束。
- **验证**：
  ① 本地无硬件——`backend._load_device_state() is conformance 侧 device_state` 为 **True**（单例性）；
  `supports("device_state")` 为 True 且 `hasattr` 成立；conformance 侧 `set_device_state(0, DEGRADED)`
  后 `backend.device_state(0)` 能查到 `DEGRADED`（状态可见性）。
  ② P800 真机——`runtime.device_state(0)` 返回 `DeviceState.AVAILABLE`；
  `info().capabilities` 含 `device_state` 且 `supports()` 与 `info()` 自洽；
  **smoke 42/0、conformance 13/13 + 6/6 无回归**。

`flagos` 仍无实现（`_capabilities` 中未声明该能力，故 `supports()` 为 False，属于**如实声明**，无需改动）。

---

## 3. `.native` 逃生舱约束 + `record_stream` 能力位

### 现状

| 位置 | 情况 |
|---|---|
| `api/stream.py:41` | `Stream.native` —— 直接把**原生流对象**交给上层 |
| `api/stream.py:101` | `Event.native` —— 同上 |
| `api/stream.py:72-80` | `record_stream(tensor)` 通过 `getattr(tensor, "record_stream", None)` **运行时探测**，探测不到就抛错 |
| `supports()` 能力键 | **没有 `record_stream` 键**（无法预判某后端是否支持） |

> 说明：`api/stream.py:16-19` 已把"跨流传递内存必须 `record_stream`"写成**纪律 1**，
> 本条不是要改这条纪律，而是要给这条纪律的**实现路径**加上契约约束。

### 问题

**设计风险，本次未致故障，但已被意识到**——设计上写"需要厂商特有操作时才用 `.native`"，
但**没有任何约束机制**：

1. 没有能力位、没有审计日志：下游随手一用，**可移植性就悄悄破了**，且事后难溯源；
2. `record_stream` 作为**纪律 1 的唯一实现路径**，却是靠 `getattr` 运行时探测的——
   上层无法提前判断某后端是否支持（910C 有、其他后端未逐一确认过），
   违反纪律的风险要等到运行时才暴露；
3. 三处 `native` 直接暴露原生对象后，上层代码会与厂商栈耦合，**违反"换芯片不改代码"的主张**。

### 建议条文

```
1. .native 定为"显式逃生舱"，使用须满足：
   · 后端的 info() 中记录本后端被取用 .native 的次数（审计），便于事后定位可移植性破坏点；
   · 文档层面：.native 只允许用于"统一 API 尚未覆盖且厂商特有能力"的场景，
     一旦上层使用，该处代码视为"绑定该厂商"，不得再声称跨芯片可移植。
2. record_stream 增加能力位：
   supports("record_stream") -> bool
   后端在 _capabilities 中如实声明；上层在跨流传递前先判断，不支持时走保守同步路径
   （而不是等到 getattr 失败才报错）。
```

### 兼容性

能力位为新增，无破坏；审计计数只增字段。建议 v0.2 起要求 `record_stream` 能力位必备。

---

## 4. 错误对象**跨模块类归一**（真实崩溃，已修在框架层）

### 现状

`api/errors.py` 定义 `ErrorCategory` 枚举与 `DISPOSITION` 表；
`FlagosError.disposition` 的实现按 `DISPOSITION[self.category]` 取值。
`api/errors.py` 的 `translate_via_backend()` 是统一入口，负责给后端翻译结果补后端名与处置信息。

### 问题（实测崩溃，P800 接入时暴露）

**现象**：P800 推理腿首次运行时第 5 节直接崩：

```
KeyError: ErrorCategory.L2_PARAM
  位于 api/errors.py:63  →  DISPOSITION[self.category]
```

**根因**：`kunlun/backend.py` 的 `_load_errors()` 用 `importlib` 把 `conformance/errors.py`
**动态加载为独立模块**，其 `ErrorCategory` 是一个 `IntEnum`，与 `api/errors.py` 的枚举
**"取值相同但类不同"**（`ConfCat.L2_PARAM == ErrorCategory.L2_PARAM` 在值上成立，
但作为字典键是两个不同的键对象）→ `DISPOSITION[cat]` 取不到 → `KeyError`。

**为什么昇腾没暴露**：`ascend` 后端自带 `_INT_TO_CATEGORY` 转换，把外部枚举转成 api 层枚举后才构造对象；
`kunlun` 直接透传外部对象，于是踩中。

**这是第 4 个"只有非昇腾实例才暴露"的框架缺陷**（前 3 个在接入阶段暴露）。

**已修（2026-09-20，修在框架层而非某个后端）**：

1. `api/errors.py` 新增 `coerce_category(value)`：把 `IntEnum` / `int` / `str`（如 `"L2_PARAM"` / `"l2_param"`）
   统一归一为 api 层 `ErrorCategory`；无法识别时返回 `None`。
2. 新增 `normalize_error(err)`：把"取值相同但类不同"的类 `FlagosError` 对象归一为真正的 `FlagosError`，**且幂等**（已是统一对象则原样返回）。
3. `translate_via_backend()` 加归一化兜底 —— **任何后端的翻译结果都会经过这道归一**。
4. `kunlun.translate_error` 显式归一（与 `ascend` 的既有做法对齐）。

**本地无硬件验证**：归一后 `disposition` 可取、幂等成立、`coerce_category` 各类输入行为正确，
并**复现了修复前的 `KeyError`**（同一段验证里跑）。

### 建议条文

```
§2 Backend 接入规范新增一条约束：

  后端 translate_error() 返回的 FlagosError，其 category 必须是 api 层 ErrorCategory
  枚举的实例（不得使用本模块或其他模块自行定义的、取值相同的枚举/整数）。

  框架侧保证：api.translate_via_backend() 会对任何返回结果做一次归一化（coerce_category +
  normalize_error），因此即使后端未严格归一也不会崩；但后端仍应显式归一，
  以免丢失"分级来源"等语义。

§4 变更记录新增：错误对象跨模块类归一（框架层兜底 + 后端显式归一），
  附本次崩溃现象与复现证据。
```

### 兼容性

**已落地、无破坏**：归一化对 `ascend` 是幂等操作（它本来就返回 api 层枚举），
对 `flagos` / `kunlun` 是修复。建议在 v0.1.x 内补进契约文本。

---

## 5. 后端不支持有界同步时的声明与降级契约（前瞻性条款，本次未触发）

> ⚠️ **诚实标注**：本条**不是实测暴露的缺口**——两家实例（`ascend` / `kunlun`）都**支持**有界同步
> （conformance `e2_host_timeout` 实测：未 record 的 `wait_host(200ms)` 返回 `False`、耗时 201 ms，不永久阻塞）。
> 本条属于**前瞻性补强**，理由是"多流 16 项基线要逐后端比对"，下一个后端未必支持。

### 现状

接口约定 §1.3 把 `stream.synchronize(timeout_ms)` 与 `event.wait_host(timeout_ms)` 定为**有界**接口
（"长驻服务必须用有界同步，防整体 hang"），§2 的插件规范也把
`synchronize_stream（有界）/ wait_event_host（有界）` 列为必须实现的三个多流支撑方法之一。
但 `supports()` 的能力键里**没有** `bounded_sync`。

### 问题

若某厂商运行时**不提供超时原语**（只能无限等待），后端作者面临两难：
要么写一个有界接口但内部无法真正有界（**静默违约**），要么直接不实现（**上层调用即崩**）。
当前契约没有给出"如何声明、上层如何降级"的路径。

### 建议条文

```
1. 新增能力位 supports("bounded_sync") -> bool：
   后端不支持时如实声明 False，并实现如下降级语义（而不是静默违约）：
     · synchronize_stream(stream, timeout_ms)：在 timeout_ms 内以轮询可查询的完成标志实现
       "有界返回"；若底层连可查询标志都没有，则必须先记录一条 known_issues
       并在超时后抛 TimeoutError（宁可有界失败，不可永久阻塞）；
     · wait_event_host(event, timeout_ms)：同上。
2. 上层约定：调用方在长驻服务路径上必须先判 supports("bounded_sync")，
   为 False 时自行选择"保守同步 + 独立看护线程"，不得依赖永久阻塞的语义。
3. conformance：不支持有界同步的后端，e2 用例生成 stub-skip 报告并注明"降级实现方式"。
```

### 兼容性

纯新增（能力位 + 降级约定），对现有两家实例无影响。

---

## 6. `known_issues()` 纳入契约

### 现状

本次接入中新增了可选方法 `backends/base.py::known_issues()`（默认返回 `[]`，**零回归**），
并在 `kunlun` 后端落地为**12 字段结构化声明**（`id / summary / trigger / repro_rate / irrelevant /
layer / evidence / workaround / workaround_cost / report_to / affects` 等）。
`proto_train_leg.py` 还据此加了 `_preflight_env_check()`：命中已声明缺陷的环境条件时**开跑前告警**。
但接口约定 §2 的插件规范里**没有这个方法**。

### 问题

这是本次最有实际价值的一个字段——它让"下游读到后端即知坑"，避免每个使用者重复踩一遍
（例如"设了某个环境变量会导致概率性挂死"这种事，写在后端里比写在聊天记录里有价值得多）。
不写进契约，第 3 家接入者不会知道要声明它。

### 建议条文

```
§2 Backend 接入规范新增（可选但强烈建议）：

  known_issues() -> list[dict]
    结构化声明本后端**已定性但需厂商侧解决**的问题。建议字段：
      id / summary / trigger（触发条件）/ repro_rate（复现率）
      / irrelevant（已排除的无关因素）/ layer（责任层）/ evidence（函数级证据）
      / workaround（规避方式）/ workaround_cost（规避代价）/ report_to（上报对象）
      / affects（影响面：哪些场景受影响、哪些不受影响）

  约定：
    · 默认返回 []（不声明视为"无已知问题"）；
    · info() 须包含 known_issues；
    · 校验自检（smoke）应打印本后端的能力声明与已知问题摘要，
      使接入方在跑第一条业务命令前就看见。
```

### 兼容性

**已实现且零回归**（`base.py` 默认实现返回 `[]`，`ascend` / `flagos` 未声明即空列表）。
仅需补入契约文本。

---

## 7. 能力降级必须"整组一致"（可观测字段不得只改其一）

### 现状

统一错误对象 `FlagosError` 有三个**能力相关**的可观测字段（`conformance/errors.py`）：

```
mapped     : True = 错误码命中映射表（**确定分级**）；False = 关键词/兜底（保守分级）
graded_by  : "code_map" | "message_hint" | "default"（F5 分级来源）
error_code : 厂商原始错误码（若有）
```

共享翻译器的码表 `ACL_ERR_TO_CATEGORY` 是**单一厂商（昇腾 ACL）码表**，
但错误码**抽取规则却是通用的**：

```python
m = re.search(r"ret\s*=\s*(\d+)", msg)
if not m:
    m = re.search(r"error code is\s*(\d+)", msg)
```

### 问题（实测，三家实例）

只要异常消息里**恰好出现一个码表内的数字**，翻译器就返回 `graded_by="code_map"` **且 `mapped=True`**。
对**没有厂商码表**的后端（`kunlun` / `cambricon`：厂商码不透出为数字码），这不是本厂商的码表命中 ⇒
必须降级。但它们原先**只改了 `graded_by` 一个字段**：

```python
if graded_by == "code_map":
    graded_by = "message_hint_unexpected"     # ← 只改这个
...
mapped=bool(getattr(fe, "mapped", False)),    # ← 仍然是 True
```

实测产出的记录（MLU590 真机，错误闭环 `l4_by_code` 一条）：

```json
{ "category": "L4_FATAL", "mapped": true, "graded_by": "message_hint_unexpected" }
```

⇒ `mapped=True` 的含义是**确定分级**，而 `graded_by` 同时说"不是码表命中"：**记录自相矛盾**。
下游只读 `mapped` 就会把 `L4_FATAL / device_recovery` 当成有码表依据的定论，
而它只是"消息里碰巧出现了别家的数字" ⇒ 可能触发一次**不必要**的设备恢复。
这正是不变式 **I2（禁止伪造）** 要防的形态：**字段齐全、业务继续、闭环全绿，但结论是假的**。

### 建议条文

> **能力降级整组一致性**：任何因"本后端不具备该能力"而做的降级，必须把**该能力涉及的
> 全部可观测字段一起降级**，不得只改其中一个；若某字段无法如实表达（如 `error_code` 拿到的
> 是**他厂**码），应置空并在 `root_cause` 中保留原文（F4 不受影响）。
> 特别地：**共享的厂商码表被多后端复用时，无码表后端负有降级义务**，
> 并且必须把这层意图写进实现的注释，避免下一位接入者照抄。

### 兼容性

**已修且零回归**：`kunlun` / `cambricon` 三字段一起降级；`ascend` / `flagos`（**声明了** `error_map`）
行为不变。防回归判据 4 条（离线自检 3 条 + smoke 3 条，其中含**他厂码串入负向测试**），
并已做**非空转验证**（直接对共享翻译器投喂同一消息，实测返回 `mapped=True`）。

---

## 8. 声明的能力必须可调用；`info()["supports"]` 键集合按能力全集呈现

### 现状

`device_state`（设备四态查询）**不在** `BaseBackend` 的 13 个抽象方法里 ——
它是可选能力。**910C `flagos` 后端在 `_capabilities` 里声明了它，却没有实现该方法。**

同一家的 `info()` 又手写了**第二份键名清单**：

| | 后果 |
|---|---|
| 缺 10 键 | 已声明的 `recovery_probe` / `error_map` / `device_state` 等在 `info()` 里**恒显 False** |
| 多 3 键 | `device_rebuild_probe` / `device_rebuild_real` / `error_code_map` **根本不存在**于能力命名空间 |

### 问题（实测）

```
AttributeError: 'FlagosBackend' object has no attribute 'device_state'. Did you mean: 'device_type'?
```

- **ABC 拦不住**：抽象方法检查只能拦"没实现的抽象方法"，声明了可选能力却不实现**不会被拦**；
- **smoke 拦不住**：`device_state` 判据只对**被选中的后端**跑（910C 该节选中 `ascend`），`flagos` 从未被覆盖；
- **键名漂移会骗过既有判据**：smoke 已有"`info.supports` 各项取值与 `supports()` 一致"，
  但两边都取 `False` ⇒ 取值一致性照样成立，而读者从 `info()` 会得出**完全相反**的结论。

⇒ **"声明即承诺"**：能力声明是给下游看的契约，声明了不可调用就是契约违约，
且它藏得比"抽象方法没实现"更深。

### 建议条文

> **声明即承诺**：后端 `_capabilities` 中出现的每一项能力，都必须有**可调用的实现**
> （`device_state` 这类可选能力同样适用）；conformance 应包含一条
> "**遍历已声明能力并逐一验证其调用入口存在**"的判据。
> **`info()["supports"]` 的键集合必须等于能力全集**（按 `_CAPABILITY_KEYS` 逐项呈现 True/False），
> 不得手写第二份键名清单；同一份清单只能有一个来源。

### 兼容性

**已修且零回归**：`flagos` 补上 `device_state`（复用共享四态机，**必须标准 `import` 以共享单例**）
并按 `_CAPABILITY_KEYS` 派生 `info()["supports"]`（与 `kunlun` / `cambricon` 同款）。
新增自检判据 2 条（`device_state 可调用` / `键集合 == 能力全集`），并已做**非空转验证**。
⚠️ **`flagos` 的修复尚未在 910C 真机验证**（当日 910C SSH 不可达），待网络恢复补跑。

---

## 10. 厂商扩展懒加载：拼厂商专有字符串前必须先触碰设备

**问题**：统一 API 的 `use(backend)` **不触发**厂商扩展导入（后端刻意懒加载，避免在无该扩展的环境里
import 失败）；而容器通常设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，torch 也**不会**自动注册厂商后端。
于是**两条路径**都会炸，且报错文案都指向"框架/环境不支持"：

| 拼什么 | 报错 | 后果 |
|---|---|---|
| **设备串**（`"npu:0"`） | `RuntimeError: Expected one of cpu, cuda, ipu, xpu, … : npu` | **conformance 整轮 ABORT**（后端初始化阶段即崩，不是某条用例失败） |
| **集合通信后端名**（`"hccl"`） | `AssertionError: Unknown backend type hccl` | 训练腿起不来（`init_process_group` 阶段） |

**为什么只在部分后端暴露**：`kunlun` 的 `device_type` 是 `"cuda"`，属 torch **内置**命名空间、无需注册 ⇒
一直正常。**这是跨后端不对称的典型**：同类代码在部分后端上不报错，不能据此推断"我的路径没问题"。

**建议条款**（写进 Backend 契约）：

> 统一 API 的 `use()` / `current()` **不保证**厂商命名空间已注册。
> **任何构造厂商专有字符串（设备串、集合通信后端名、厂商 API 调用）的调用点，前一步必须先经后端
> 触碰一次设备**；推荐的最小触碰是 `backend.device_count()`（13 个抽象方法之一，所有后端都必须实现，
> 且语义无害）。

**为什么推荐 `device_count()` 而非"显式 import 厂商模块"**：显式 import 会把"某厂商的模块名"写进
上层的通用代码（正是要消除的耦合）；而 `device_count()` 只依赖统一抽象，后端换厂商不用改调用方。

**配套建议**：**厂商集合通信后端名必须有默认值**，且**未探测过的芯片宁可报错退出也不给兜底** ——
否则会落到 `gloo`，造成"**静默退化为纯 CPU 集合通信**"（训练照样跑完、loss 照样降，
但**设备侧通信根本没被验证**）。本项目已对 `cambricon` 采用该做法，本次已给 `ascend` 补
`DC_DIST_BT=hccl` 默认值。

**已在 910C 验证**：修复后 conformance **13/13 + 6/6**（此前整轮 ABORT）、
训练腿 `TRAIN_LEG_PASS 6/6`（`dist=hccl`）。

---

## 9. 验证资产的可达性与证据卫生（操作性条款）

### 现状

本轮的三处问题都不是"后端实现错"，而是**验证资产本身的缺陷**：

| 问题 | 后果 |
|---|---|
| 离线自检工具**只为 `cambricon` 一家内置 stub** | `--backend kunlun\|ascend\|flagos` 直接被拒 ⇒ **三家从未被自检过**（第 8 条因此长期未被发现） |
| `proto_error_recovery_loop.py` **硬编码昇腾错误码 `507046`** 在文案里 | 在无厂商码的后端上产出**假证据**（该码根本不会产生），且会被后续读者当成"这家也能报出 507046" |
| 同一脚本对**无码表后端**仍把 L4 注入描述为"按码表触发" | 记录描述与实际验证的东西**不符**（它实际验的是"诚实降级"） |

### 建议条文（写入《新芯片接入手册》，不进接口约定正文）

> ① **自检工具必须覆盖全部已接入后端**；新增一家 = 同时补它的 stub，
> 并且 stub 只能让"能真实验到的判据"参与判定 —— **stub 覆盖不到的判据必须显式 SKIP 并写明原因**，
> 既不误报 FAIL，也不混入"通过"计数（汇总行区分 通过 / 失败 / 跳过）。
> ② **验证脚本不得硬编码单一厂商的错误码或厂商专有文案**；
> 需要错误码时从**实际异常**中提取，拿不到就如实写"无厂商码"。
> ③ **注入用例必须自描述**：把"这条注入期望什么"写进结果记录（如 `expectation` / `expect_matched`），
> 且期望**参与判定** —— 不能只看"业务还在跑"就放过。

### 兼容性

已做：自检工具 stub 通用化（四家各一行注册）+ 显式 SKIP 机制 + 崩溃改判 FAIL；
清掉硬编码码；L4 注入按后端分化并写入期望。
⚠️ **stub 是语义空间的**，无法模拟真实算子与厂商运行时 ⇒ 昇腾的探活与有界同步仍以真机
`conformance` / `smoke` 为准（工具已在输出里显式标注这一边界）。

---

## 附：本建议与"月度计划 11 月交付物"的关系

本方向 9 月三件套之一即"接口约定修订建议"（新芯片首次检验规范的产出）。
上述 **9 条**中：

- **第 1、2、4、7、8 条**建议在 **9 月内**落进契约文本
  （第 4、7、8 条代码已修，均附防回归判据；第 2 条含一处具体缺陷需一并修）；
- **第 3、6 条**建议随 **v0.2（10 月）**发布；
- **第 5 条**留作 v0.2 的候选 —— 第 3 家（寒武纪）**同样声明了 `bounded_sync` 但流同步为"超时上报"语义**
  ⇒ 该条的现实性进一步增强，建议 v0.2 一并处理；
- **第 9 条**属操作性条款，建议**直接并入《新芯片接入手册》**（已做部分同步）。

> **第 3 家接入对第 5 条的补充证据**：寒武纪实测 `Stream.synchronize()` **不接受 `timeout`**
> （`TypeError: unexpected keyword argument`），与本层"有界 = 超时上报"的语义一致；
> 参照的昆仑芯亦同款。⇒ "不支持原生有界同步时的声明与降级契约"已是**两家的共同现状**，不再是前瞻条款。

修订落地路径遵循约定 §4 的变更流程：**本文档 → 接口约定更新（变更记录节）→ 知会全部下游 → conformance 回归全绿**。
