# net.cc chunk 流水线 1/10 偶发数据错：源码级调研与修复方案

> 调查对象：集合级 10 轮 `test_ag_hetero.py` 中 rank0(CUDA) 间歇性 `sum=1.0`（AR 的 tmpDev[1]=0，即 rank0 收到的 rank1 数据为 0）
> 结论先行：**问题几乎可以确定在 rank1(910C, CANN) 的 send 侧——`aclrtStreamQuery` 的"D2H 拷贝完成"判定早于 CPU 读可见性，导致 isend 偶发发出未写全的 pinned host buffer**。修复用 eventSynchronize 精确等待（两侧通用）。
> 日期：2026-08-28 · 调研人：Kistich + AI

---

## 1. 问题定义与已固化的证据链

### 1.1 现象
| 项 | 值 |
|---|---|
| 触发场景 | 集合级 10 轮循环（AG int64 8B ×2 + AR float32 4000B） |
| 失败率 | 约 1/10 轮 |
| 失败侧 | **rank0(4090-1, CUDA) 恒为失败侧**；rank1(910C, CANN) 恒对（sum=3.0） |
| 失败表现 | rank0 的 AR `sum=1.0`（期望 3.0）；诊断实锤 `tmpDev[1]=0.0`（rank0 收到的 rank1 数据是 0） |
| 不触发场景 | 50 步真实训练（3GB 梯度、24576 chunk/op）全程无异常 |

### 1.2 已做过的实验（历史结论，全部真实）
| 实验 | 结果 | 结论 |
|---|---|---|
| 诊断版 D2H probe（reduce 前读 tmpDev[0]/[1] + streamSynchronize） | 10/10 PASS | 时间缓冲侥幸；D2H 读强制同步 |
| streamSynchronize 替代（仅 rank0 侧 reduce 前） | 6/10 | 不够；commStream 同步不覆盖 cpStream |
| deviceSynchronize（rank0 侧 reduce 前） | 7/10 | **跨机无效**——问题在 rank1，rank0 全设备同步管不到 rank1 |
| 禁用设备 reduce（host reduce）+ COMPILE_KERNEL=1 | 8/10 FAIL | 排除设备 reduce；**COMPILE_KERNEL_HOST 干扰 proxy** |
| Makefile 拆分 COMPILE_KERNEL/COMPILE_KERNEL_HOST | 9/10 | 修复干扰源，剩余 1/10 为原生存量 |
| recv 侧 eventQuery 替代 streamQuery | 7/10 更差 | event 环形复用导致状态查询误判（保守方向无效） |
| streamQuery 回退 | 9/10（当前状态） | 维持 |

### 1.3 关键矛盾（推导核心）
- 失败时 `tmpDev[1]=0`，且 rank0 已做 `deviceSynchronize`（所有流完成）→ **H2D 的源（pinned host buffer）本身就是 0** → 数据在 socket 层就是 0。
- rank0(CUDA) 的 send 恒对（rank1 收到的 rank0 数据总是正确）→ **CUDA 的 streamQuery 语义可靠**。
- rank1(CANN) 的 recv 恒对 → **rank1 的 recv 侧 H2D/streamQuery 路径可靠**。
- 排除 → **问题锁定在 rank1(CANN) 的 send 侧：D2H → streamQuery → isend 这条链**。

---

## 2. 全链路代码解剖（本轮深挖）

### 2.1 op 生命周期总览
```
uniRunnerAllGather (uni_runner.cc:398)
  └─ flagcxHeteroGroupStart()
  └─ for r in 0..nRanks-1:
  │    flagcxHeteroSend(sendbuff, peer=r)   // 内部 GroupStart/End（depth 嵌套）
  │    flagcxHeteroRecv(recvbuff+r*size, peer=r)
  └─ flagcxHeteroGroupEnd()
       └─ flagcxGroupEndInternal (group.cc:554) — depth==0 时触发一次 groupLaunch
            ├─ Round0：self peer 的 send/recv task 匹配 → self-copy op（P2P transport）
            ├─ Round1..nRanks-1：recv task → op(pattern=Recv, opId=-roundOpId)
            │                     send task → op(pattern=Send, opId=+roundOpId)
            ├─ 每个 op：semaphore->addCounter(opId)；eventRecord；按 step 分组入 proxyOps
            ├─ flagcxProxySaveOp → proxy 生产队列
            ├─ launchHostFunc(launchStream, cpuAsyncKernel, semaphore)  // P2 修复：只 signalStart
            └─ semaphore->wait()   // P2 修复：主线程等待全部 subCounter
```
- **groupLaunch 的 depth 嵌套正确**（GroupEnd 在 depth==0 才 launch）→ 一个 allgather 的全部 send/recv op 合并进一次调度。
- **semaphore 计数正确**：self op(opId=0)、send(+roundOpId)、recv(-roundOpId) 各 addCounter 一次、subCounter 一次，wait 收敛（nDone==nOps）。**已排除 opId/信号量计数竞态**。

### 2.2 proxy 线程调度（proxy.cc:512 flagcxProxyProgress → progressOps:305）
- 单 proxy 线程，每轮 `progressOps` **对每个 peer 的 sendQueue/recvQueue 只处理队列头一个 op**（FIFO）。
- head op `done==1 && semaphore->pollEnd()` 才出队；**head 未完成会阻塞同队列后续 op**（本场景队列各 1 op，无影响）。
- 一个 peer 的 send 与 recv 是**独立连接**（各自 transportResources / cpStream / buffers）。

### 2.3 send 侧状态机（net.cc:145 flagcxProxySend，regBufFlag=0 主路径）
```
① waitCopy：D2H(data→buffers[step]) 提交到 cpStream → eventRecord(cpEvents[step]) → waitCopy++
② posted：streamQuery(cpStream)==SUCCESS → copied++/done → isend(buffers[posted&mask]) → posted++
③ transmitted：netAdaptor->test(req) done → transmitted++
④ copied==chunkSteps → subCounter(opId)
```
- **SOCKET 下 buffers[0] = pinned host 内存（flagcxMemHost / aclrtMallocHost）**；D2H = 显存→pinned host。
- **isend 的前提 = `streamQuery(cpStream)` 返回 SUCCESS**。

### 2.4 recv 侧状态机（net.cc:291 flagcxProxyRecv）
```
① posted：irecv(buffers[step]) → posted++
② postFlush：test(req) done → SOCKET 置 0x1 → postFlush++
③ waitCopy：H2D(buffers[step]→data+off) 提交 cpStream → eventRecord → waitCopy++
④ copied：streamQuery(cpStream)==SUCCESS → copied++
⑤ copied==chunkSteps → subCounter(opId)
```

### 2.5 socket 握手协议（socket_adaptor.cc:514 flagcxNetSocketTest）
- **无 tag 匹配**：收发匹配完全依赖 ctrlSock 上"先交换 4 字节 size"的 FIFO 握手 + 数据 sock 顺序。
- size 交换成功后按 `nSocks` 分 subtask，交 `persistentSocketThread`（helper 线程）实际收发。
- **done 判定**：全部 subtask `offset==size`（helper 线程"先写数据、后更新 offset"，顺序正确 → 排除 helper 线程竞态）。
- recv 侧 `r->size = data`（被发送方报的 size 覆盖）——**若 send 侧 isend 的 size 与 recv 期望一致则安全；无 size 校验机制**。

### 2.6 两侧 streamQuery 实现对照
| 侧 | 实现 | 语义 |
|---|---|---|
| CUDA (cuda_adaptor.cc:303) | `cudaStreamQuery(stream)` | `cudaSuccess`=流中所有命令**实际执行完成**（含拷贝数据落位，CUDA 有明确保证） |
| CANN (cann_adaptor.cc:150) | `aclrtStreamQuery(stream, &status)` | `ACL_STREAM_STATUS_COMPLETE`=**"Stream 上的所有任务已完成"**（官方文档未明确保证"数据对 CPU 可见"） |

---

## 3. 竞态候选逐项分析

| # | 候选 | 证据 | 判定 |
|---|---|---|---|
| A | **rank1(CANN) send 侧 `aclrtStreamQuery` 误判 D2H 完成 → isend 发出未写全的 pinned buffer** | 失败侧恒为 rank0 recv；rank0 的 deviceSynchronize 无效（跨机）；CUDA send 恒对；CANN recv 恒对；官方文档语义缺口（§2.6） | **最强假设** |
| B | rank0(CUDA) recv 侧 H2D 与 copied 判定的跨流可见性 | deviceSynchronize 后 tmpDev[1] 仍 0 → H2D 源就是 0，数据在 socket 层已错 | **排除**（数据源头错） |
| C | socket helper 线程 offset 更新与数据写入竞态 | 源码确认"先写数据后更新 offset" | **排除** |
| D | send/recv 握手 size 错位（无 tag 匹配） | 同连接 FIFO，单 op 场景顺序一致；但**多 op 排队 + send 延迟时存在理论风险** | **次级候选**（与 A 叠加时放大） |
| E | opId/semaphore 计数竞态 | 源码确认 opId 唯一、wait 收敛 | **排除** |
| F | COMPILE_KERNEL_HOST 干扰 | 已拆分修复（8→9/10） | **已修复，非当前根因** |

---

## 4. 最强假设的理论依据（为什么 CANN 可能误判）

### 4.1 官方文档语义缺口
- CUDA：`cudaStreamQuery` 返回 `cudaSuccess` 时，**所有先前提交的命令（包括 DMA 拷贝）已完成执行**（NV 文档明确）。
- CANN：`ACL_STREAM_STATUS_COMPLETE` = "Stream 上的所有任务已完成"——**未明确"任务完成"是否等价于"DMA 数据已写入目标内存且对 CPU 可见"**。

### 4.2 平台相关窗口（ARM + DMA 缓存一致性）
- 910C 是 **aarch64**：D2H 由 DMA 引擎写入 pinned host 内存，`aclrtStreamQuery` 的完成标记与 **CPU 侧 cache 失效（invalidate）的顺序**存在平台实现窗口。
- 若 stream 状态置 COMPLETE 早于 cache 失效完成，**socket helper 线程（CPU 读）会命中陈旧缓存行**（读到 0/旧值）。
- x86(4090) 的 DMA/CPU 一致性由硬件保证（PCIe 一致性协议），无此窗口 → 解释"CUDA 侧恒对、CANN 侧偶发"。

### 4.3 为什么诊断版 D2H probe 能 10/10（时间缓冲假说）
- 诊断在 rank0 reduce 前读 tmpDev（D2H）+ streamSynchronize → **放慢了 rank0 的完成节奏**，间接给 rank1 的 D2H 更多时间落位，isend 提前发出的概率被压缩到 0。
- deviceSynchronize（rank0 侧）**管不到 rank1 的时序** → 无效。两者对比恰好指向"问题在 rank1 send"。

---

## 5. 决定性验证实验（二选一，建议先做实验 1）

### 实验 1：rank1(CANN) send 侧 isend 前探针（实锤/证伪候选 A）
在 910C 的 `flagcxProxySend` 的 isend 分支（net.cc:235 附近 `resources->netAdaptor->isend(...)` 前）插入：
```cpp
// 仅当 stepBuff 是 host 指针（SOCKET 下恒为 pinned host）时直接 CPU 读
long v0 = 0; memcpy(&v0, args->subs[args->posted & stepMask].stepBuff, sizeof(long));
fprintf(stderr, "[DBG-SEND] rank=%d opId=%d posted=%d size=%zu first8=%ld\n",
        comm->rank, args->opId, args->posted,
        (size_t)args->subs[args->posted & stepMask].stepSize, v0);
fflush(stderr);
```
- 若失败轮 `first8 == 0`（或 != 期望值 2.0）→ **实锤候选 A：isend 前数据未就绪**。
- 若恒等于期望值 → 候选 A 证伪，转向实验 2 排查 recv/网络层。

### 实验 2：交换角色（910C 作 rank0）
- 910C 侧 `NODE_ROLE=npu RANK=0 MASTER_ADDR=10.120.72.27`，4090-1 侧 `RANK=1`。
- 若失败仍表现为"**4090 的 recv 收到 910C 的错数据**"→ 与角色无关，实锤 910C send 侧。
- 若失败跟随 rank0 变化 → 需重新定位。

---

## 6. 分级修复方案

### S1（推荐，最小改动，两侧通用）：eventSynchronize 精确等待
- **send 侧**（net.cc flagcxProxySend posted 段）：`streamQuery(cpStream)` 改为 `eventSynchronize(cpEvents[step])`——cpEvents[step] 是 D2H 提交后立即 record 的事件，**eventSynchronize 阻塞等该事件完成（= 事件前所有任务含 D2H 实际完成、数据落位）**。
- **recv 侧**（net.cc flagcxProxyRecv copied 段）：同样改为 `eventSynchronize(cpEvents[step])`。
- 为什么可靠：eventSynchronize 是**阻塞等待**，不依赖状态查询语义；即使 cpEvents[step] 被环形复用（下一轮 op 覆盖 record），等到的是"最近一次 record 的完成"（**保守方向，只会慢不会错**）。
- 为什么之前 eventQuery（recv 侧）失败：eventQuery 是**非阻塞查询**，依赖 event 状态，环形复用 + 查询语义导致误判"完成"→ 提前 copied。**eventSynchronize 无此问题**。
- 改动量：net.cc 两处，各 1 行语义替换。编译两侧即可。

### S2（保守兜底）：send 侧 isend 前 deviceSynchronize
- 与 S1 同思路但更重：等本机所有流完成。实测过"rank0 侧"无效（跨机），但 **send 侧（rank1 本机）有效**——因为 D2H 就在本机 cpStream。
- 代价：每次 allgather 一次全设备同步（3GB 数据下 ms 级，可接受）。适合先验证正确性再用 S1 精化。

### S3（最稳、性能等价）：send 的 D2H 改同步拷贝
- `flagcxProxySend` 的 D2H 分支（非 IBRC）用**同步拷贝**（cudaMemcpy / aclrtMemcpy 同步版，或 stream=NULL）替代 async + streamQuery。
- 正确性 100%（无异步完成判定）；性能与现状等价（proxy 本来就在等拷贝完成）。
- 注意：同步拷贝会阻塞 proxy 线程——若对端 recv 在等，可能引入握手串行化（小张量场景无感）。

### S4（长期根治）：socket 协议加固 + 完成判定重构
- socket 握手加 **opId/序列号**校验（防多 op 排队时 size 错位，消除"无 tag"缺陷）。
- 完成判定统一为 per-chunk event + eventSynchronize（S1 的完整版），处理 event 环形复用。
- 改动大，建议作为独立任务。

### 推荐组合
1. 先做**实验 1** 实锤候选 A；
2. 若实锤 → **S1（eventSynchronize 两侧）** 直接落地，跑 10 轮循环验证；
3. 若 S1 仍有残余 → S2（send 侧 deviceSynchronize）兜底对照；
4. S4 作为长期项排期。

---

## 7. 开放问题

1. **CANN 官方是否有已知 issue**：`aclrtStreamQuery` 对异步拷贝（D2H）的完成判定精度，建议向华为 CANN 支持确认或查昇腾社区 issue。
2. **50 步训练为何不触发**：3GB 梯度（24576 chunk/op）下 proxy 线程几乎每轮都在推进 D2H/isend 流水线，isend 时 D2H 大概率早已完成；小张量（1 chunk/op）时 isend 紧跟 D2H 提交，更容易撞上"完成判定窗口"——**与观察一致**，但需实验 1 佐证。
3. **S1 的 eventSynchronize 是否引入性能回归**：阻塞等待 vs 现在的 streamQuery 轮询——语义上两者都等拷贝完成，S1 只消除了"误判窗口"，不增加等待时间；需实测单步 sync 对比（预期无回归或微增）。
4. **RoCE 打通后该问题是否消失**：IBRC 路径走 GDR/iflush，完成判定机制不同（IBRC 分支用 iflush），候选 A 的窗口可能不适用——届时需回归验证。

---

## 8. 实验 1 实锤（2026-08-28 完成）—— 历史日志自证，无需新探针

**发现**：`flagcxProxySend` 的 isend 前本就有 P4-SEND-DATA 打点（CPU 直读 stepBuff 前 16 字节），它就是现成的实验 1 探针。系统对比历史循环日志（`/tmp/loop8_r*_*.log`）：

| 轮次 | 910C AR isend 前 buffer (v0) | 4090 sum |
|---|---|---|
| 3/4/5/7/10（PASS） | `0x4000000040000000` = **2.0**（正确） | 3.0 |
| **6/8/9（FAIL）** | `0x000000000000000b` = **11（AG 旧数据残留）** | **1.0** |

- 失败轮 910C isend 前 buffer 是**旧数据（AG 残留）**而非期望的 2.0 → **实锤候选 A：910C(CANN) 的 D2H 未执行完成，isend 发出旧 buffer**。
- 排除方向：4090 的 P4-SEND-DATA 恒正确（CUDA 侧无问题）；910C 的 recv 恒对。

## 9. S1 修复落地与验证（2026-08-28）

**改动**（net.cc 两侧，仅 send 侧，recv 侧保留 streamQuery 以隔离变量）：
```cpp
// 旧：if (deviceAdaptor->streamQuery(resources->cpStream) == flagcxSuccess) { args->copied++; done = 1; }
// 新：if (deviceAdaptor->eventSynchronize(resources->cpEvents[step]) == flagcxSuccess) { args->copied++; done = 1; }
```
- `cpEvents[step]` 是 D2H 提交后立即 `eventRecord` 的事件；`eventSynchronize` **阻塞等到事件完成 = D2H 数据实际落位**，不依赖 `aclrtStreamQuery` 的状态查询语义。
- 即使 cpEvents 环形复用（下一轮 op 覆盖 record），等到的是"最近一次 record 的完成"——**保守方向，只会慢不会错**。

**结果**：
| 版本 | 10 轮循环 |
|---|---|
| 修复前（streamQuery） | 9/10（早期 7/10、8/10） |
| **修复后（eventSynchronize）** | **两轮连续 10/10（20/20）全过，0 死锁** |

**下一步建议**：① recv 侧是否同步改（当前无暴露，可留作防御性修复，风险为零）；② 跑 50 步真实训练确认无回归与吞吐；③ 提交流程（本修复 + P8 合并进同一干净补丁，同步 GitHub/OpenI）。

## 10. recv 侧同步修复 + 50 步训练验证（2026-08-28）

**recv 侧（防御性）**：`flagcxProxyRecv` copied 段同样 `streamQuery(cpStream)` → `eventSynchronize(cpEvents[step])`（H2D 事件完成 = 数据落位），彻底消除同源竞态。改动后 10/10 循环通过，无回归。

**50 步训练（eventSynchronize 全量生效）**：
| 项 | 结果 |
|---|---|
| s0 loss | **2.8891**（与基线逐位一致） |
| 单步 sync | **~26.6s/步**（sync_total=1331s/50 步；P8 时代 total=1624s、~32s/步——**不降反升 ~17%**，诊断打点减少 + eventSynchronize 无轮询开销） |
| loss 收敛 | s20=2.1327 vs gloo 基线终点 2.1312，两 rank 同步 |
| 死锁/数据异常 | 全程无 |
| ckpt | 双侧正常保存退出（rank0 total=1350s） |

**结论**：P9（send/recv eventSynchronize）修复了集合级 1/10 偶发数据错，且对训练性能零影响。20/20 + 10/10 循环 + 50 步训练全通过。

---

## 附：本调研涉及的源码位置速查
| 文件 | 关键位置 |
|---|---|
| `flagcx/core/net.cc` | flagcxProxySend:145 / flagcxProxyRecv:291 / streamQuery 判定:231,440 |
| `flagcx/core/group.cc` | groupLaunch:110 / task→op 转换:169-465 / GroupEndInternal:554 |
| `flagcx/core/proxy.cc` | flagcxProxyProgress:512 / progressOps:305 / flagcxProxyGetPostedOps:437 |
| `flagcx/core/include/launch_kernel.h` | flagcxHostSemaphore:44 / wait:142 |
| `flagcx/adaptor/net/socket_adaptor.cc` | flagcxNetSocketTest:514 / GetTask:471 / persistentSocketThread:205 |
| `flagcx/adaptor/device/cuda_adaptor.cc` | streamQuery:303 / eventSynchronize:382 |
| `flagcx/adaptor/device/cann_adaptor.cc` | streamQuery:150 / eventSynchronize:205 / deviceMalloc(flagcxMemHost):51 |
| `flagcx/runner/uni_runner.cc` | uniRunnerAllGather:398 / AllReduce 分片:121-165 |
