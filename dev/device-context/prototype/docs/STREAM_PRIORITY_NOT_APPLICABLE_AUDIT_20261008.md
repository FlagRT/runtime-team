# 「流优先级不适用」自证审计（910C / P800 · 2026-10-08）

> 触发：用户要求自查 —— 「910C 与 P800 的『如实不适用』，是**真的不存在**，还是**我自己没发现**」。
> 动机背景：本方向在 **2026-10-08 第十二轮**刚在寒武纪身上栽过一次同类跟头 ——
> cambricon 后端注释里「本家没有独立句柄式 C API」「无 `ExternalStream`」两句**外推来的话都是错的**，
> 实测 `libcnrt.so` 有 17 个 `cnrtQueue*` 符号、`torch.mlu.ExternalStream` 可用。
> ⇒ 因此对另两家的「不适用」**不能只引用既有结论**，必须**重新取证**。
>
> 机器：**昇腾 910C**（`npu1-27`，容器 `dc-audit-910c`，物理卡 2 独占）·
> **昆仑芯 P800**（`VM-0-2-ubuntu`，容器 `hliu553-device-context-p800`，逻辑卡 0）。
> 探针：`probes/audit_stream_priority_ascend.py` · `probes/audit_stream_priority_kunlun.py`（本轮新增）；
> 原始日志与 JSON：`../910C/probes/audit_20261008_out/`、`../P800/probes/audit_20261008_out/`。

---

## 0 结论（先行）

| 实例 | 原结论 | 审计后 | 变化 |
|---|---|---|---|
| **910C** | 如实不适用（不声明 `stream_priority_control`） | ✅ **成立** | **根因表述升级**（从笼统的「没有入口」→ 精确到「C++ 有、Python 绑定缺失，且所有绕行被插件自身校验拒绝」） |
| **P800** | 如实不适用（优先级空间退化为单点） | ✅ **成立** | **证据面加宽 4 项**（逐卡 ×7 · runtime API 交叉 · 显式上下文 · 无原生 XPU API） |

**并且审计过程本身查出我此前的 4 处不到位**（见 §3）—— 都**没有改变结论**，但有两处**足以让结论不可靠**，
已修并记入纪律。**唯一仍未验证的候选路径**见 §4。

---

## 1 「不适用」自证六问（可复用清单）

> 一个「不适用」结论要站得住，必须能回答下面 6 问。**任何一问答不上来，结论就不能写「如实不适用」**，
> 只能写「未验证」。本轮两家的审计就是照这 6 问逐条取证。

| # | 问 | 这一问防的是什么 | 本轮对应动作 |
|---|---|---|---|
| 1 | **只测了单实例 / 单卡 / 单档吗？** | 把「一张卡的属性」当成「全机/全设备族的属性」 | P800 **逐卡 7/8 张**全查（第 3 张有他人 61 GB 作业，**如实标注未测**） |
| 2 | **只看了「一层」吗？** | 兼容层 vs 原生层；**driver API vs runtime API**；torch 属性 vs **设备 C API** | P800 补测 runtime API（`cuda*`）并与 driver API 交叉；回读一律走**设备 C API** |
| 3 | **只「挑了几个名字」还是扫了全量符号？** | 「没找到」可能是「没搜到」 | 两台都做**全量 `nm -D` 扫描**（并按大小写不敏感 + 变体名） |
| 4 | **有没有「接受但忽略 / 接受但使用点才失败」的假通路？** | 「构造成功」被当成「这条路可用」 | 910C：`Stream(stream_ptr=…)` **静默忽略**；`Stream(stream_id=…)` **接受但使用点断言** |
| 5 | **「另两家做不到 ⇒ 这家也做不到」的外推又出现了吗？** | 注释代替事实（寒武纪那次的教训） | 两台都**重新实测**，不引用既有结论 |
| 6 | **取证时跑的是当前版本吗？** | 「命令跑通」≠「跑的是当前代码」 | 本轮**自己就踩了**（§3-4）⇒ 每次取证先核版本 |

---

## 2 逐条取证

### 2.1 昇腾 910C —— 结论成立

| 检查 | 实测（原始读数） |
|---|---|
| **A. ACL 有没有 priority setter** | ❌ 头文件 `aclrtSetStreamAttribute` 的枚举只有 `ACL_STREAM_ATTR_FAILURE_MODE / FLOAT_OVERFLOW_CHECK / USER_CUSTOM_TAG / CACHE_OP_INFO`（**无 priority**）；`libascendcl.so` 里名字带 `Priority` 的导出符号**只有两个 Get**：`aclrtDeviceGetStreamPriorityRange`、`aclrtStreamGetPriority` |
| **B. ACL 层是否保留优先级** | ✅ `aclrtCreateStreamWithConfig(0/3/7)` 全部 `create_rc=0`，回读 **priority = 请求值**（0/3/7）⇒ **不是设备限制**；回读是**真设备读数**（`aclrtStreamGetPriority` / `GetFlags` / `GetId` 三个独立口径 `rc` 全 0） |
| **C. `torch.npu.Stream(priority=p)` 逐档位** | ⚠️ p=0..7 得到 **8 条互不相同的句柄**（torch 侧 `stream_id` 97→104），但**设备回读 priority 全部 = 0** ⇒ **参数确实没进设备**（此表是首次逐档位留证） |
| **D. kwarg 穷举** | `priority` / `stream_ptr` / `device_index` / `device_type` / `is_sync_launch` 被接受（`flags` 被拒）；**所有被接受者**的设备 priority 都是 0；`stream_id` 单独传因缺 `device_type` 报 `PTA invalid type` |
| **E. `Stream(stream_id=<裸 ACL 句柄>)`** | ⚠️ **构造不报错**，且 `stream_id` **逐位等于裸句柄**；但此后**任一步使用**都抛：<br>`RuntimeError: 0 INTERNAL ASSERT FAILED at "../torch_npu/csrc/core/npu/NPUStream.cpp":371, please report a bug to PyTorch. Unrecognized stream stream 187651927616464 on device npu:0`<br>—— 读 `.npu_stream` / `repr(s)` / `s.synchronize()` / `with torch.npu.stream(s)` **四处全部同一断言** |
| **F. `_npu_setStream(stream_id=<裸句柄>)`** | ❌ 同一断言（厂商原话见上）。**对照**：用「池内 id」走同一条路**正常生效**（`became_other=True`）⇒ **链路本身没坏，坏的是「外部句柄没被登记」** |
| **G. C++ 面是否**有**正确入口** | ✅ **有** —— `libtorch_npu.so` 导出 `c10_npu::getStreamFromExternal(void*, signed char)` 与 `c10_npu::setCurrentNPUStream(c10_npu::NPUStream)`；<br>❌ 但 `torch.npu` **无** `ExternalStream`，`torch_npu._C` **无任何 `*External*` 符号**，包内 Python 对 `getStreamFromExternal` **零引用**（仅 `npu/utils.py` 用了 `_npu_setStream`）；<br>❌ 也没有别的 `.so` 导入 `getStreamFromExternal`（**全量扫 .so**：只有 `libtorch_npu.so` 定义它，无任何导入方） |
| **H. 统一面门禁** | ✅ `create_stream(priority=0/7)` 均抛 `NotImplementedError`（**文案含能力键名**）⇒ **显式拒绝**，不是静默给一条无优先级的流 |

**⇒ 根因（精确版）**：**不是设备限制**（ACL 侧 0/3/7 全部保留），而是
**「插件 C++ 已有 `getStreamFromExternal`，但未暴露到 Python；所有 Python 可见的绕行路径都被插件自身的
`Unrecognized stream` 校验拒绝」**。这与原 `known_issues` 的方向一致，但**更精确、且不再依赖"没有入口"这种笼统说法**。

### 2.2 昆仑芯 P800 —— 结论成立

| 检查 | 实测（原始读数） |
|---|---|
| **Q1/Q2. 逐卡 + 有上下文后查区间** | 卡 0/1/2/4/5/6/7 **全部 `rc=0`、`range=[0,0]`**；卡 3 因他人 61 GB 作业在用，**本轮不去创建上下文 ⇒ 如实标注「未测」**。⇒ 「只测一张卡就推广到全机」这个可能**已被排除** |
| **Q3. 原生 XPU 接口面** | 全量 `nm -D` 扫描：含 priority 的库**只有两个** —— `libcudart.so.12.9.1.kunlun`（**runtime API** `cudaDeviceGetStreamPriorityRange` / `cudaStreamCreateWithPriority` / `cudaStreamGetPriority`）与 `libxpucuda.so.515.58.kunlun`（**driver API** `cu*`，即 `libcuda.so.1` 的符号链接目标）；`libxpurt.so.12.9.1.kunlun` **含 priority 0 个、含 Stream 0 个**；**`xpu*` 前缀的原生流 API 零命中** |
| **Q4. torch 侧交叉** | `torch.cuda.Stream(priority=-1).priority` = **0**；`torch.cuda.ExternalStream` 可用；本 torch 版本**无** `torch.cuda.get_stream_priority_range` |
| **R3. driver API 逐值建流** | `cuStreamCreateWithPriority` 对 **-10/-5/-2/-1/0/1/5/10** 全部 `create_rc=0`，`cuStreamGetPriority` 回读**全部 0** |
| **S. runtime API 逐值 + 交叉** | `cudaDeviceGetStreamPriorityRange` = **(0,0)**（`rc=0`）；`cudaStreamCreateWithPriority` 对 **-2/-1/0/1/2** 全部 `create_rc=0`、回读**全部 0**；**与 driver API 读数一致**（`S3_runtime_vs_driver_一致 = True`） |
| **N. 反向核对** | 全量扫描 `xpu*`/`XPU*` 前缀的 `Stream`/`Queue` API ⇒ **零命中** ⇒ 优先级**只有 CUDA 兼容层这一套**，不存在「原生层另有一套」 |

**⇒ 结论**：「优先级空间退化为单点」既不是「只测了一张卡」、也不是「只看了一层」、也不是「只搜了几个名字」
—— 它是**兼容运行时上报的设备属性**；两条独立实现路径（driver/runtime）读数一致。

---

## 3 ⚠️ 审计查出的**我自己的 4 处不到位**（都不改变结论，但两处足以让结论不可靠）

| # | 问题 | 后果 | 处置 |
|---|---|---|---|
| **1** | **只看了 driver API（`cu*`），漏了 runtime API（`cuda*`）** | 若两者读数不同，「设备属性」这个结论就会错 | 补测；**读数一致** ⇒ 结论不变，但证据面加宽 |
| **2** | **起初只在 1 张卡上测区间** | 「全机退化」可能是「一张卡退化」 | 逐卡补测 7/8（跳过在用卡），**全部一致** |
| **3** | ⚠️ **我的探针仪器在撒谎**（第一轮 910C）：用 `getattr(s, "priority", 默认值)` 探测，而该属性**会抛 `RuntimeError: NPU dose not support Stream.get_priority()`** ⇒ 被我的 `except` **误记成「构造被拒绝」**，整段 kwarg 结论作废。<br>同族：**`hasattr(s, "priority")` 也抛**（`hasattr` 只吞 `AttributeError`） | 会把「读属性失败」写成「这条路没有入口」 | 整段重做：**探存在性一律 `try/except BaseException` 并分类回报**（不存在 / 存在但抛 / 存在可读）；并把「构造接受」与「使用点生效」拆成两个判据 |
| **4** | ⚠️ **改了本机脚本却没同步就出结论**（P800 那次）：远端还是旧版 ⇒ 新加的 **S/N 两段静默没跑**，日志里看不出来。**且我第一眼用 `grep "a\|b"`（BRE）复核版本，得到静默 0 命中**，差点把「没同步」掩盖过去 | 「跑完了」≠「跑的是当前版本」；**静默漏跑**是最危险的一种 | 同步后重跑，并**同时落 `.log` 与 `.json`**（同版本）；复核一律 `grep -E` |

⭐ **共同点**：4 处都不是「能力判断错了」，而是**「取证过程本身可能不可靠」**——
即本项目反复出现的那一家族：**仪器/判据在替被测对象说话**。

---

## 4 唯一仍未验证的候选路径（如实登记 · 待裁定）

**路线**：自编译一个 torch 扩展，直接调用 `c10_npu::getStreamFromExternal(void*, signed char)`
（或 `getStreamFromPool(priority, device)`），把 pyACL 建的带优先级裸流包成 torch 可用流。

**可行性**：**工具链齐备** —— 容器内 `g++/gcc/c++` 都在，`torch/include` 与 `torch_npu/include`（**743 个头文件**）都在。

**为什么本轮不做**（三条，按重要性排序）：

1. **不保证成功**：插件对「未登记流」的校验（`Unrecognized stream … (I didn't recognize the stream type)`）
   **未必**因走 `getStreamFromExternal` 而消失 —— 那取决于它是否把该流注册进内部表。
   在没验证之前，**不能把它写成「可行」**。
2. **改变部署面**：会引入自编译 `.so`（ABI 随 torch_npu 版本变化）⇒ 与「本层只做接口、不引入自有二进制」的定位冲突。
3. **它是上游诉求的替代实现**：正解是 torch_npu 暴露 Python 绑定（或让 `Stream(stream_ptr=…)` 真正接上）。
   本层已把该诉求写进 `known_issues().report_to`。

⇒ 因此**列为待定**：**若上层确实需要 910C 上的优先级效果测量**，再评估是否走这条路（含 ABI 与维护成本）。

---

## 5 证据清单与复现

```
../910C/probes/audit_stream_priority_ascend.py        # 910C 审计探针（合并版）
../910C/probes/audit_20261008_out/ascend_audit.log    # 原始输出（28 条 [A] 记录）
../910C/probes/audit_20261008_out/ascend_audit.json   # 同次运行的 JSON
../P800/probes/audit_stream_priority_kunlun.py        # P800 审计探针（合并版）
../P800/probes/audit_20261008_out/kunlun_audit.log    # 原始输出（16 条 [P] 记录）
../P800/probes/audit_20261008_out/kunlun_audit.json   # 同次运行的 JSON
```

复现（均为**容器内**执行）：

- 910C：`python probes/audit_stream_priority_ascend.py --out <out>.json`
  ⚠️ 需先自建只挂**一张**空闲卡的容器（性能类/取证类实验独占卡）；
      `ASCEND_RT_VISIBLE_DEVICES` 用**容器内 0 基索引**；
      本层需要的 `runtime` 入口靠 `DC_ROOT` + `DC_BACKEND=ascend` 给足。
- P800：`python probes/audit_stream_priority_kunlun.py --skip-devices 3 --out <out>.json`
  （`--skip-devices` 用于**别人正在用的卡**：本轮卡 3 有他人 61 GB 作业，**不去动它的上下文**）

**用卡与现场**：910C 用物理卡 2（自建容器，收工即删）；P800 用逻辑卡 0（他人占卡 3）。
收工核验：910C 我的容器已删、他人容器未动、无残留进程；P800 容器恢复 Exited。
