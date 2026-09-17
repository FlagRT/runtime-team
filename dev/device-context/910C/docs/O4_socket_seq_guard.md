# O4：socket 协议无 tag 匹配——暴露点分析与加固方案

> 工作项：O4（socket 协议无 tag 匹配加固，P3 防御性）
> 日期：2026-08-31 ｜ 状态：**✅ 已完成并提交上游**（commit `08535e1`，FlagRT/FlagCX `kistich/ascend-dev1.0`）
> 一句话：**socket 层发送/接收完全靠"ctrlSock 上先交换 4 字节 size"的 FIFO 顺序配对，无任何 op 身份与校验——错位一旦发生就是静默数据损坏。**

---

## 1. 当前协议全貌（源码级）

### 1.1 连接拓扑（socket_adaptor.cc）
```
send 侧（flagcxNetSocketConnect, :336）    recv 侧（flagcxNetSocketAccept, :390）
  ┌─ comm{ ctrlSock + socks[0..nSocks-1] }    ┌─ comm{ ctrlSock + socks[0..nSocks-1] }
  │   nSocks 个数据 socket（并行传输）          │   同构
  └─ 与对端 recv 侧一一连接                     └─ 与对端 send 侧一一连接
```
- send 与 recv 是**两条独立 socket 连接**（各自 ctrlSock + 数据 socks）。
- 连接建立阶段有迭代握手（`Connect` 里传 `uint8_t i` 序号，:369），**但数据传输阶段没有任何序号/身份**。

### 1.2 Request 结构（:158）—— 无身份字段
```cpp
struct flagcxNetSocketRequest {
  int op;              // FLAGCX_SOCKET_SEND / RECV（仅方向）
  void *data;
  int size;            // 本地期望（recv）/ 实际（send）
  struct flagcxSocket *ctrlSock;
  int offset;
  int used;            // 1=交换 size → 2=传输数据
  struct flagcxNetSocketComm *comm;
  struct flagcxNetSocketTask *tasks[MAX_SOCKETS];
  int nSubs;
};
```
**没有 tag/opId/seq**——socket 层无法区分"这是第几个 collective 的哪个 op"。

### 1.3 握手与传输状态机（flagcxNetSocketTest, :514）
```
used == 1：交换 size
  flagcxSocketProgress(op, ctrlSock, &data, sizeof(int), &offset)  // ctrlSock 上收发 4 字节
  ...（阻塞等 4 字节）
  if (r->op == RECV && data > r->size) { WARN truncated; return InvalidUsage; }  // 仅"大于"报错！
  r->size = data;      // ← 接收方用对端报的 size 覆盖本地期望
  used = 2 → 分 subtask（nSocks 个，helper 线程在数据 sock 传输）
used == 2：轮询 subtask 完成（offset == size 全完成 → done）
```

---

## 2. 暴露的问题（错位场景逐条列出）

| # | 暴露点 | 后果 | 当前是否触发 |
|---|--------|------|------------|
| **E1** | **recv 侧 `r->size = data` 单向覆盖**：对端报的 size 与本地期望不同时，**只有 `data > 期望` 才报错**（WARN truncated），`data < 期望` **静默接受** | 若错位使 recv 收到更小的 size → 按错误大小收数据 → 本 op 数据残缺 + 后续 op 全部错位（静默） | 否（P9 后 FIFO 保序），但**校验缺失是硬伤** |
| **E2** | **无 op 身份（无 seq/tag）**：若两侧 op 顺序不一致（send 跳过/延迟/取消某 op、未来多 op 并发调度、失败重试路径），recv 侧无法发现"收的是错配的 size" | 跨 op 数据错位，**无法检测** | 理论风险（当前单 op FIFO 不触发） |
| **E3** | **数据阶段无校验**：数据 socks 传输原始字节，无长度/校验和 | 错位/截断后**不可检测**，静默损坏（loss 悄悄变差） | 理论风险 |
| **E4** | **依赖 proxy 单线程保序**：send 连接与 recv 连接各自独立，两侧 op 顺序必须严格一一对应；任何一侧的进度差（如 P9 修复前的 isend 提前）都会打破配对 | 历史实证：P9 前 1/10 偶发数据错就是"isend 提前发旧 buffer"（侧配对未破，但数据内容错）；若配对被破则更严重 | P9 已消除实际竞态，但**协议无防线** |
| **E5** | **MAX_REQUESTS slot 复用**（:200）：request 用完 `used=0` 复位循环利用，无代际标识 | 极端并发下旧 request 误配 | 理论风险 |

> **历史关联**：P9（eventSynchronize）修复的是"isend 发出未就绪 buffer"（数据内容竞态）；**O4 修复的是"配对身份缺失"（协议结构竞态）**——两者正交，O4 是 P9 之上的协议级防线。

---

## 3. 加固方案：握手加 per-comm 单调递增 seq + 双向 size 校验

### 3.1 设计
| 改动 | 内容 |
|------|------|
| Request 加 `int seq` | 每个 request 绑定一个 per-comm 递增序号（op 身份） |
| Comm 加 `int sendSeq / recvSeq` | send 侧分配；recv 侧维护本地期望（收到必须 == 期望） |
| 握手扩展为 8 字节 header | `{ int seq; int size; }` 替代原 4 字节 size（发送方在 ctrlSock 上先发 header） |
| 接收方校验（双条件） | `seq == 本地期望 && size == 本地期望 size`；**任一不匹配 → WARN + 返回 flagcxInternalError**（暴露错位，绝不静默） |
| 数据阶段 | 维持现状（per-op 校验已能防错位；per-chunk checksum 为可选增强） |

### 3.2 伪代码
```cpp
// 发送方 isend：
struct flagcxNetSocketRequest *r;
flagcxNetSocketGetRequest(comm, FLAGCX_SOCKET_SEND, data, size, &r);
r->seq = comm->sendSeq++;          // 分配序号

// Test used==1 握手（发送方视角）：
struct { int seq; int size; } hdr = { r->seq, r->size };
flagcxSocketProgress(SEND, ctrlSock, &hdr, sizeof(hdr), &off);   // 先发 8 字节 header

// Test used==1 握手（接收方视角）：
struct { int seq; int size; } hdr;
flagcxSocketProgress(RECV, ctrlSock, &hdr, sizeof(hdr), &off);   // 收 8 字节 header
if (hdr.seq != comm->recvSeq || hdr.size != r->size) {
  WARN("seq mismatch: got %d expect %d, size %d expect %d", ...);
  return flagcxInternalError;                                     // 暴露错位
}
comm->recvSeq++;                                                   // 期望 +1
r->size = hdr.size;                                                // 校验通过后使用
```

### 3.3 兼容性
- 两侧同版本部署（dev 分支），无旧版兼容需求；协议 magic 不变。
- 若需兼容旧端：握手 header 加版本字节（可后续加）。

### 3.4 风险
- 握手从 4 字节变 8 字节：双方同时改，无半开状态（连接建立时的迭代握手已同步两侧实现）。
- 校验失败即报错：正常路径（FIFO 保序）永不触发，触发即暴露真实错位（这正是目的）。

---

## 4. 验证方案

| 步骤 | 内容 | 预期 |
|------|------|------|
| V1 | 两侧编译通过 | 无 error |
| V2 | 10 轮 `test_ag_hetero.py` 循环 | 0 死锁、sum=3.0 全对（正常路径 seq 严格递增，不误报） |
| V3 | **注入错位测试**：构造"send 侧 seq 与 recv 期望不一致"场景（临时改 recv 期望或手工破坏 header） | `flagcxGetLastError` 报 seq/size mismatch，**不再静默** |
| V4 | 50 步训练冒烟（可选） | 无回归 |

---

## 5. 提交与看板
- 提交 FlagCX `kistich/ascend-dev1.0`（bundle 中转）+ OpenI/GitHub 看板 O4 ✅。

---

## 6. 落地过程中的关键发现（2026-08-31 补）

实现 O4 时通过注入测试暴露了两个**更深层的错误处理缺陷**，一并修复：

### 6.1 net.cc 忽略 test() 返回值 → 检测到错位却卡死
- 现象：seq 校验检测到 mismatch（返回 InternalError），但 `flagcxProxySend/Recv` 里
  3 处 `netAdaptor->test()` **未检查返回值** → 错误被吞 → recv 永不完成 → 主线程
  `wait()` 永久阻塞（静默卡死）。
- 修复：3 处 test 调用加 `FLAGCXCHECK`。

### 6.2 proxy 线程错误无终结机制 → 无限重试
- 现象：校验失败后 request 停留在 used==1，proxy 重试 test 时从 ctrlSock 继续读字节
  （读到数据而非 header，hdr 变成 0x40000000=float 2.0）→ 数据流永久错位。
- 修复：
  1. **request 错误终态**：校验失败置 `used=3`，Test 开头对 used==3 直接返回错误
     （不再触碰 socket）；
  2. **semaphore error 机制**：`flagcxSemaphore` 基类加虚方法 `setError/hasError`，
     HostSemaphore 实现 error 标志，`wait()` 检测 error 提前退出；
     `progressOps` 里 flagcxProxySend/Recv 错误 → `semaphore->setError()`；
     `groupLaunch` wait() 后 `hasError()` → `flagcxInternalError`。
- 效果：注入错位时 rank0 报 `DistBackendError: flagcxInternalError` + Last error 显示
  真实调用点（O3 TLS 联动），**非卡死、非静默**。

### 6.3 二次剥离
- 8/28 O1 rebase 的 3-way merge 使 O5 剥离部分丢失（group/proxy/init/socket 的
  [HETERO-DBG] 复活）；本次补剥 20 处（含 cann_adaptor 3 处从未剥离的）。
- 教训：**rebase 后必须重新全量验证诊断标记归零**（O5 的验证在 rebase 前做的）。

---
