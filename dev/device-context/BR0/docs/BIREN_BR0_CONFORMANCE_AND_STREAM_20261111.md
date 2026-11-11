# 壁仞 BR_0：conformance 13+6 与多流 16 项基线实测（2026-11-11）

> 采集时刻 2026-11-11 14:49–14:56 CST ｜ 机器 `wz-server008` ｜ 容器 `dc-biren-hliu553`
> 范围：**只做设备上下文 + 多流**（训练腿 / 推理腿按用户裁定本轮不做）。所有读数均为当次真机实测。

## 0 结论

| 判据 | 结果 | 判定串 / rc |
|---|---|---|
| conformance 设备上下文与多流 13 例 | **13 / 13 通过** | `CONFORMANCE_PASS`，`rc=0` |
| conformance 推理 6 例 | **6 / 6 通过** | `CONFORMANCE_PASS`，`rc=0` |
| 多流 Stream 16 项基线 | **16 项已逐项对照**：12 项实测通过、1 项实测「不具备有效优先级」、1 项标注不适用、2 项本轮未做（如实列） | `STREAM_SEMANTICS_PASS 8/8`（探针覆盖的 8 项） |
| smoke 自检（上一轮） | 46 / 0 | — |

三条要单独讲的：①**流优先级**构造收参但**回读恒 0**（单档），范围 API 触发上游断言 ⇒ 三键全不声明（§5.1）；②i6 那条 **-75237%** 的负重叠率不是语义缺陷，是**一次性编译 + 工作量太小**，放大工作量后重叠率收敛到 **±2% 内**（§5.2）；③公共探针里 S12 的结论文案此前**把昆仑芯的上游缺陷套到了别的后端头上**，已改为按后端分支（§5.3）。

另有一条**与接入无关但必须登记的现场事实**：今天机器上**掉了两张卡、另有两张新发故障**（8 张→6 张），本方向使用的两张卡（`85:00.0` / `a5:00.0`）日志干净（§6）。

---

## 1 载体与环境（自包含）

| 项 | 读数 | 取自 |
|---|---|---|
| 容器 | `dc-biren-hliu553`，`Up 3 hours`，`RestartCount=0`，StartedAt `2026-11-11T03:32:05Z` | `docker inspect` |
| 镜像 | `birensupa-smartinfer-vllm:26.07.09-py310-pt2.8.0-vllm0.21.0-c066s001t001b24145-br1xx` | 同上 |
| 镜像 ID | `sha256:081634a335d48760cb10fb240bf3655530b0e2748a46d836e097e6006953281a`（本地仓库**无 digest**） | `docker images --digests` |
| 设备节点 | `/dev/biren/card_5`、`/dev/biren/card_7`（容器内可见仅此两个） | `ls /dev/biren/` |
| 对应 PCI | `card_5 → 0000:85:00.0`、`card_7 → 0000:a5:00.0` | 宿主 `sysfs`：`/sys/class/biren/card_N/device` |
| torch / 厂商插件 | `torch 2.8.0+cpu`、`torch_br 1.8.7.20800+br1xx` | 探针首行 `[env]` |
| 设备命名空间 | `torch.supa`（`PrivateUse1`），`device_count=2` | 同上 |
| 后端 | `runtime:biren=0.1.0`（`__version__`），`supports()` 声明 9 键 | `conf13.json` 的 `env` |

**运行前置（缺一即假失败，不是芯片问题）**：容器内任何 python 调用前必须
`. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set`。
漏掉时 `import torch_br` 报 `ImportError: libsupti.so.1: cannot open shared object file` ——
原因是镜像把库路径/PYTHONPATH 放在入口脚本里，而 `docker exec` 起的是新进程、不继承入口进程的环境。

---

## 2 conformance 设备上下文与多流 13 例

```
[env] runtime:biren=0.1.0 devices=2
  [PASS] e1_event_record_wait         统一事件 record/wait 可用
  [PASS] e2_host_timeout              未 record wait_host(200ms) 返回 False，耗时 200ms（<1s，不永久阻塞）
  [PASS] e2_wait_before_record        record→wait 正常路径通过；边界：wait 未 record 事件可能永久阻塞（无超时），需统一层收敛（如实标注）
  [PASS] e3_query_unrecorded          未 record query=False（不崩溃）
  [PASS] f1_error_translation         统一错误对象: L2_PARAM(code=None, graded_by=message_hint) 位置=stream:0/op:matmul 根因保留
  [PASS] r_recovery                   评估=available 隔离=True 重建=True 在途重放集合=True 事件=['isolated','available']
  [PASS] s1_stream_order              流内顺序近似：链式运算结果与逐 CPU 参考相对误差 2.26e-05（<1e-3）
  [PASS] s2_explicit_dependency       两条独立张量链结果正确（无隐式同步破坏）
  [PASS] s3_result_visibility         跨流显式依赖可见性: z 全 3 = True
  [PASS] s4_explicit_transfer         显式 wait_stream 传递: b 全 10 = True
  [PASS] t1_pinned_async_copy         pinned->device non_blocking 拷贝数据一致
  [PASS] t2_inflight_protection       拷贝完成后释放源缓冲无异常
  [PASS] t3_topology_path             跨设备直接传输数据一致=True；拓扑接口: 统一面未暴露拓扑查询（如实标注）
=== 汇总: 13/13 通过 ===  结论: CONFORMANCE_PASS
```

两点读法提示（这两条是**结果文本里的如实标注**，不是失败）：

- `e3_query_unrecorded` 之所以能过，是因为**本层** `Event` 适配器用 `_recorded` 标志把「未 record」修正为 `False`；
  原生 `torch.supa.Event()` 未 record 时 `query()` **返回 True（误报已完成）** —— 这是本栈的既有缺陷，已在 `known_issues` 登记（`BIREN-EVENT-UNRECORDED-QUERY-TRUE`）。
- `e2_wait_before_record` 指出的「未 record 就 wait 可能永久阻塞」是**跨后端共性**（原生无超时入口），本层用有界轮询兜住；不算本家缺陷。

错误翻译（`f1`）分级来源是 `message_hint`、`code=None`：本栈**未做厂商错误码表探测**，故不给 `error_map` 声明——
「未探测」不是「不具备」，两者在台账里分开写。

---

## 3 conformance 推理 6 例

```
[env] runtime:biren=0.1.0 devices=2
  [PASS] i1_device_context_after_load 加载后上下文可用: logits.shape=(4, 64)
  [PASS] i2_infer_forward_stream_order 多轮前向同流顺序: 固定输入逐轮一致(误差=0.00e+00)
  [PASS] i3_kv_buffer_visibility      KV 跨流可见性: 写入流→事件→计算流, 读回全 8 = True
  [PASS] i4_d2h_sample_transfer       D2H 采样回传: logits.cpu()→topk, shape=(4, 3)
  [PASS] i5_longrun_device_state      长驻 20 轮: 累计 logits=-4.54e+00, 无 NaN/Inf = True
  [PASS] i6_pipeline_overlap          流水线依赖链: 末轮数据正确(rel_err=7.04e-05)=True | 观测: pipe=0.390s serial=0.002s 重叠率=-21150%
=== 汇总: 6/6 通过 ===  结论: CONFORMANCE_PASS
```

⚠️ `i6` 判据本身是**数据正确性**（通过）；后面那串 `重叠率=-21150%` 是**观测值**，
其成因已单独定位，见 §5.2 —— **不得把这个负值读成「流水线语义不成立」，也不得引它当性能数据**。

---

## 4 多流 Stream 16 项基线：逐项对照

16 项的定义域（4 域 × 分组）：基础语义 4 项、并发与绑定 3 项、工程安全 4 项、能力与限制 5 项。

| # | 子项 | 判定 | 证据（当次读数） |
|---|---|---|---|
| 1 | 流内顺序（FIFO） | ✅ 实测 | 探针 S1：流内 4 op 串行 → `5.000000`（期望 5.0）；conformance `s1` 相对误差 `2.26e-05` |
| 2 | 显式依赖 / 无隐式同步 | ✅ 实测 | 探针 S2：两条独立流 A=`7.0000`、B=`11.0000`；conformance `s2` |
| 3 | 跨流可见性（event） | ✅ 实测 | conformance `s3`：写入流 record → 读流 wait → `z 全 3`；i3 KV 缓冲同构 |
| 4 | `wait_stream` 传递 | ✅ 实测 | conformance `s4`：`b 全 10 = True` |
| 5 | 多流并发重叠 | ✅ 语义通过 / ⚠️ 无性能收益 | 探针 S12 并发段：hi=`5.0000`、lo=`8.0000`；重叠率实测 **±2% 内**（§5.2 缩放表） |
| 6 | 集合通信与流绑定 | ⛔ **本轮未做** | 属训练腿范畴（用户裁定本轮不做训推）；集合通信后端名**尚未探测**，不得给兜底结论 |
| 7 | 图捕获流语义 | ⛔ **本轮未做** | `graph_capture` **未声明**（未实测）；`torch.supa` 有 `SUPAGraph` 类但不等于图捕获语义已验证 |
| 8 | 默认流 vs 命名流 + `record_stream` | ✅ 实测 | 探针 S8：默认流=`2.0000`、命名流=`4.0000`；跨流复用（`record_stream` 后）=`10.0000` |
| 9 | 错误隔离分层 | ✅ API 级实测 / 设备级为语义推断 | 探针 S9：流 A 注入参数类异常（`RuntimeError`，shape 不匹配）后流 B 仍 `6.0000`；设备级（芯片故障影响该设备全部流）**未真实触发**，仅按错误定义推断 |
| 10 | 流/事件生命周期与配额 | ✅ 实测 | 探针 S10：**1000 次**创建销毁（`1.01 s`）后新建流仍正常（`9.0000`） |
| 11 | 跨流内存分配 | ✅ 实测 | 探针 S11：流 A 分配、事件依赖后在流 B 使用 → `6.0000` |
| 12 | 流优先级 | ⚠️ **实测：不具备有效优先级** | 构造 `Stream(priority=N)` 对 N∈{0,−5,−1,−2,1,7} 全部回读 `0`；`Stream.priority_range()` 抛 `RuntimeError: INTERNAL ASSERT FAILED (greatest_priority <= -1)`（§5.1） |
| 13 | 多设备流绑定 | ✅ 实测（双卡） | 探针 S13：`set_device(0/1)` 各建独立流 → `[2.0, 3.0]`（期望 `[2,3]`） |
| 14 | 同步超时 | ✅ 有界语义实测 / 原生不支持 | `e2_host_timeout`：未 record 的 `wait_host(200ms)` 返回 `False`、耗时 200ms（不永久阻塞）；原生 `synchronize()`/`Stream.synchronize()` **无 timeout 形参**，本层做成「到点抛 `TimeoutError`」的**上报**语义，**不保证中断底层执行** ⇒ 不声明 `bounded_sync` / `sync_timeout` |
| 15 | 跨进程共享（IPC） | ⚪ 标注不适用 | 与 PPU 同口径：本方向接入期不做跨进程共享；**不写成「不支持」**，写「本轮不适用/未验证」 |
| 16 | 流数量配额 | ✅ 实测（与 #10 同证据） | 1000 次创建销毁无泄漏；与其他后端同口径（而非按厂商文档给的数字） |

**计数**：✅ 实测通过 **12** 项（1/2/3/4/5/8/9/10/11/13/14/16）｜⚠️ 实测不具备 **1** 项（12）｜⚪ 不适用 **1** 项（15）｜⛔ 本轮未做 **2** 项（6/7）。

> 口径说明：`STREAM_SEMANTICS_PASS 8/8` 与上表 16 项**不是同一套计数**——探针只覆盖「必须真正创建流才能验证」的 8 项
> （S1/S2/S8–S13），其余项由 conformance 用例与上表所列证据覆盖。引用时不要把 8/8 说成 16/16。

探针逐项输出（`stream_biren2.log`）：

```
[env] backend=biren dev_api=supa devices=2 torch=2.8.0+cpu runtime=n/a
[S1]  流内顺序: 5.000000（期望 5.0）✅
[S2]  无隐式同步: A=7.0000 B=11.0000 ✅
[S8]  默认流 vs 命名流: 2.0000/4.0000；跨流分配器安全(record_stream)=10.0000 ✅
[S9]  流错误隔离(API级): 流A rc=RuntimeError 流B=6.0000 ✅  ⚠️设备级错误为语义推断
[S10] 生命周期: 1000 次创建销毁（1.01s）后新流仍可用 ✅
[S11] 跨流内存: 6.0000（期望6）✅
[S12] 流优先级: 未声明（参数被接收但回读恒 0，未生效） | range=None | 并发 hi=5.0000 lo=8.0000 ✅
[S13] 多设备流绑定: [2.0, 3.0]（期望 [2,3]）✅
STREAM_SEMANTICS_PASS: 8/8 子项通过
```

---

## 5 三处需要单独说明的实测结论

### 5.1 流优先级：能设、能回读，但回读恒 0（单档）

三条独立事实（`biren_priority_probe.py` / `biren_priority_range_probe.py`，输出见 `facts_s12_i6_*.log`）：

| 问 | 实测 |
|---|---|
| 有没有范围查询入口 | `torch.supa.stream_priority_range` **不存在**；`torch.supa.Stream.priority_range` **存在**；`torch.supa.Stream.priority` 存在 |
| 范围 API 能调吗 | `Stream.priority_range()` → `RuntimeError: greatest_priority <= -1 INTERNAL ASSERT FAILED`（`BR_PACKAGE/buildSystem/...`），进程不崩、`rc=0` |
| 设置能落地吗 | `Stream(priority=0/1/-1/-2/-5/7)` 构造**全部成功**，`stream.priority` **逐个回读都是 0**；`current_stream().priority` 也是 0 |

⇒ 结论：**参数被接收、优先级未生效（单档 0）**。因此三键全不声明：

- `stream_priority`：设置不落地；
- `stream_priority_control`：同上；
- `stream_priority_readback`：**回读接口存在但恒 0、不反映设置** —— 声明它等于给自己一个假能力（这正是「能设 ≠ 有效果」的教科书形态）。

「`priority_range` 报 `greatest_priority <= -1` 断言」与昆仑芯侧观测到的是**同形的上游缺陷**（那边命中 `c10/cuda/CUDAStream.h:188`），
但两边是各自实测、**未做同源性证明**，只登记为「同形」，不合并归因。

### 5.2 i6 的负重叠率：一次性编译 + 工作量太小，不是语义缺陷

分三步定位：

**(a) 逐段计时（原语都不慢，`biren_ov_scale.py` 前身）**：

| 单步 | 耗时（8 次均值） |
|---|---|
| 流上下文切换 | 0.058 ms |
| `Stream()` / `Event()` 创建 | 0.011 / 0.004 ms |
| `event.record(s)` / 已完成事件 `wait_event` | 0.006 / 0.003 ms |
| `copy_ non_blocking`（H2D 512×512） | 0.098 ms |
| matmul 512 + sum | 0.214 ms |

按此推算每轮应约 0.5 ms，而 i6 实测每轮约 97 ms ⇒ 存在一次性开销。

**(b) 逐段计时抓到元凶**：循环里**第一轮** `(512,512)@(512,512)` 耗时 **2153.67 ms**，第 2–4 轮降到 **0.29–0.50 ms**。
即「**某形状首次在命名流上执行**」要付一次性编译/建内核代价。i6 的计时区间把这段包了进去。

**(c) 预热后 + 放大工作量（`ov_scale.py`，两次独立运行）**：

| n / batches | 运行① 重叠率 | 运行② 重叠率 |
|---|---|---|
| 512 / 4（= i6 口径） | −48.1% | −44.7% |
| 1024 / 8 | +0.5% | +1.7% |
| 2048 / 8 | +2.0% | +1.1% |
| 2048 / 16 | +1.0% | −0.1% |
| 4096 / 8 | +4.5% | +0.1% |

⇒ 两条结论，都要如实写：
1. i6 的 −21150%（首次跑）/−75237%（冷启动复测）**全部来自一次性编译与主机侧调度开销**，工作量越大越收敛；
2. 放大后重叠率只在 **±2% 内波动**，说明**本栈这套「3 流 + 4 事件」的流水线在当前实现下没有可测的重叠收益**。
   这属**性能域**（对外提交），本方向不以它为验收判据；也**不据此推断厂商实现有缺陷**——只登记现象与测量方法。

纪律备注：吞吐/重叠类读数**必须预热后再计时**（同 MLU590 那次「首请求 86 s 编译开销污染吞吐」的教训），
否则会把编译开销当成稳态性能。

### 5.3 探针文案错归属已修正（公共资产）

公共探针 `probes/probe_stream_semantics_full.py` 的 S12 分支原本是二选一：`BACKEND == "ascend"` 走 pyACL，
**其余全部**落到「昆仑芯」那段结论文案里。于是壁仞的结果里出现了「昆仑芯后端已主动拦截」这种**错归属**描述。

修法：拆成 `kunlun` / `biren` / 其他三支，各自只陈述**本家实测到的事实**；`biren` 支写明
「构造收参但回读恒 0、范围 API 抛上游断言、故三键不声明」；其余后端退化为「未声明 = 未实测，不等同于不具备」。
`kunlun` 支文本**一字未改**（保持其归档口径可复现）。

首跑日志（`biren_stream_..._145025/`）保留未删，作为「修正前」的反证；本报告引用的是修正后重跑结果。

---

## 6 现场事实：今天卡况发生了变化（8 张 → 6 张）

与接入动作无关，但会影响后续所有人取卡，必须登记（2026-11-11 14:47 读数）。

**当前 `brsmi` 只剩 6 张**，且**显示索引与设备节点名不再一一对应**：

| brsmi 显示索引 | Bus-Id | 设备节点（`/dev/biren/card_N`） | 状态 |
|---|---|---|---|
| 0 | `05:00.0` | `card_1` | ⚠️ `N/A` 读数、128 MiB 残留（**今日 13:34 新故障**） |
| 1 | `65:00.0` | `card_3` | ✅ 正常，被他人占用 38720 MiB |
| 2 | `83:00.0` | `card_4` | ⚠️ `N/A` 读数、128 MiB 残留（**今日 14:22 新故障**） |
| 3 | `85:00.0` | `card_5` | ✅ 正常，0 MiB（**本方向在用**） |
| 4 | `a3:00.0` | `card_6` | ✅ 正常，被他人占用 8320 MiB |
| 5 | `a5:00.0` | `card_7` | ✅ 正常，0 MiB（**本方向在用**） |

变化经过（`dmesg` 原文时间）：

- `03:00.0` 与 `63:00.0`（原 0 号 / 2 号坏卡）今日 **彻底从总线移除**：`device remove` 分别出现在 `11:59:58`、`12:11:32`、`12:12:37`（03:00.0）与 `11:59:52`、`14:17:23`（63:00.0）。
- 剩余卡中又有两张今日新发故障，链条与昨日同型：`smmu context fault(Read) fsr=0x402` → `PQM hang recovery fail` → `found spc hang, mask 0xffffffff` → `hot reset has been triggered`（05:00.0 于 `13:34:59`；83:00.0 于 `14:24:59`）。
- 昨天（11-10）的故障是 2 张，今天波及 **4 张**（2 张已掉总线 + 2 张 N/A）。**这是扩散，不是收敛**。

我们两张卡的日志干净：`85:00.0` / `a5:00.0` 最后的事件是 `11:59:50 ... receive SIGTERM`（我们自己重建容器时的正常终止），无 ERR / 无 hang / 无 reset。

**两条取卡铁律（今天就踩到了）**：

1. `BIREN_VISIBLE_DEVICES` 选的是**设备节点序号**（`card_N`），**不是 `brsmi` 显示索引**。卡掉线后 `brsmi` 会重排索引而节点名不变 ⇒ 照 `brsmi` 索引挑卡可能挑到别人的卡。
2. `brsmi` 读数正常 **≠** 卡可用（新故障卡在复位后仍显示 128 MiB 残留 + `N/A`）；**取卡前跑一次最小算子探针**。

---

## 7 未做与边界（本轮范围裁定）

| 项 | 状态 | 归属 |
|---|---|---|
| 训练腿（两 rank、50 步） | 未做 | 用户裁定本轮不做训推 |
| 推理腿（前向 13 项 / 服务化 10 项） | 未做 | 同上 |
| 集合通信后端名探测 | **未探测** | 属训练腿前置；**宁可报错不给兜底**（纪律：不探测就写「未验证」） |
| 图捕获（`graph_capture`） | 未验证 | `torch.supa` 有 `SUPAGraph` 类，但**类存在 ≠ 语义已验证**，不声明 |
| 错误码表（`error_map`） | 未探测 | 只观察到 `ErrorCode: 100`（无设备）与 `SUDNN` 错误名，未做码级注入 |
| 设备级错误隔离 | 语义推断 | 真实触发风险高，未做；已在 #9 写明「推断」而非实测 |
| 卡 0/2（现 03/63）故障处置 | 待上游 | 已出专项报告；今日又新增 05/83 两张，需一并提给提供方 |

---

## 8 证据与复跑

- 证据索引与 md5：`BR0/probes/biren_conf_stream_20261111_out/MANIFEST.md`
- conformance：`biren_conf_20261111_144929/{conf13,conf6}.{log,json}`
- 多流：`biren_stream2_20261111_145225/{stream_biren2.log, stream_semantics_full_result_biren.json, i6w.log}`
- S12 + i6 事实：`biren_facts_20261111_145533/facts_s12_i6_20261111_145533.log`
- 复跑脚本：`biren_conf_stream_20261111_out/scripts/`（`biren_conf_run.sh` / `biren_stream_rerun.sh` / `biren_facts_run.sh`，均过 `bash -n`）
- 后端实现：`prototype/runtime/backends/biren/backend.py`（`supports()` 9 键，`known_issues()` 结构化）
