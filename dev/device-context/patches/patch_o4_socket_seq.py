#!/usr/bin/env python3
"""O4: socket 协议加固——握手加 per-comm seq + 双向 size 校验。

- Request 加 seq 字段（op 身份）
- Comm 加 sendSeq/recvSeq（per-comm 单调递增）
- Isend 分配 seq；Irecv 记录本地期望 seq
- Test 握手从 4 字节 size 扩展为 8 字节 header {seq, size}；
  接收方校验 seq == 期望 && size == 期望，不匹配即报错（不再静默错位）

用法：python3 patch_o4_socket_seq.py <FlagCX根>
"""
import sys, os

ROOT = sys.argv[1].rstrip("/")
p = os.path.join(ROOT, "flagcx/adaptor/net/socket_adaptor.cc")
s = open(p).read()

def rep(old, new, label):
    global s
    n = s.count(old)
    if n == 0:
        print(f"[FAIL] {label}: pattern not found")
        sys.exit(1)
    if n > 1:
        print(f"[FAIL] {label}: ambiguous ({n})")
        sys.exit(1)
    s = s.replace(old, new, 1)
    print(f"[ok] {label}")

# 1) Request 加 seq
rep("""struct flagcxNetSocketRequest {
  int op;
  void *data;
  int size;
  struct flagcxSocket *ctrlSock;
  int offset;
  int used;""",
    """struct flagcxNetSocketRequest {
  int op;
  void *data;
  int size;
  int seq; // O4: per-comm 单调递增 op 身份（发送侧分配，接收侧校验）
  struct flagcxSocket *ctrlSock;
  int offset;
  int used;""",
    "Request.seq")

# 2) Comm 加 sendSeq/recvSeq（在 requests 数组前）
rep("""  struct flagcxNetSocketRequest requests[MAX_REQUESTS];""",
    """  int sendSeq; // O4: 发送侧已分配序号（递增）
  int recvSeq; // O4: 接收侧期望序号（校验通过后 +1）
  struct flagcxNetSocketRequest requests[MAX_REQUESTS];""",
    "Comm.sendSeq/recvSeq")

# 3) 握手 header 结构（Task 结构之后）
rep("""struct flagcxNetSocketTask {
  int op;
  void *data;
  int size;
  struct flagcxSocket *sock;
  int offset;
  int used;
  flagcxResult_t result;
};""",
    """struct flagcxNetSocketTask {
  int op;
  void *data;
  int size;
  struct flagcxSocket *sock;
  int offset;
  int used;
  flagcxResult_t result;
};

// O4: 握手 header（替代原 4 字节 size，附 op 序号以校验配对）
struct flagcxSocketHdr {
  int seq;
  int size;
};""",
    "SocketHdr")

# 4) Isend 分配 seq
rep("""flagcxResult_t flagcxNetSocketIsend(void *sendComm, void *data, size_t size,
                                    int tag, void *mhandle, void *phandle,
                                    void **request) {
  struct flagcxNetSocketComm *comm = (struct flagcxNetSocketComm *)sendComm;
  FLAGCXCHECK(
      flagcxNetSocketGetRequest(comm, FLAGCX_SOCKET_SEND, data, size,
                                (struct flagcxNetSocketRequest **)request));
  return flagcxSuccess;
}""",
    """flagcxResult_t flagcxNetSocketIsend(void *sendComm, void *data, size_t size,
                                    int tag, void *mhandle, void *phandle,
                                    void **request) {
  struct flagcxNetSocketComm *comm = (struct flagcxNetSocketComm *)sendComm;
  FLAGCXCHECK(
      flagcxNetSocketGetRequest(comm, FLAGCX_SOCKET_SEND, data, size,
                                (struct flagcxNetSocketRequest **)request));
  // O4: 分配 per-comm 递增 seq（op 身份）
  (*(struct flagcxNetSocketRequest **)request)->seq = comm->sendSeq++;
  return flagcxSuccess;
}""",
    "Isend.seq")

# 5) Irecv 记录期望 seq
rep("""flagcxResult_t flagcxNetSocketIrecv(void *recvComm, int n, void **data,
                                    size_t *sizes, int *tags, void **mhandles,
                                    void **phandles, void **request) {
  struct flagcxNetSocketComm *comm = (struct flagcxNetSocketComm *)recvComm;
  if (n != 1)
    return flagcxInternalError;
  FLAGCXCHECK(
      flagcxNetSocketGetRequest(comm, FLAGCX_SOCKET_RECV, data[0], sizes[0],
                                (struct flagcxNetSocketRequest **)request));
  return flagcxSuccess;
}""",
    """flagcxResult_t flagcxNetSocketIrecv(void *recvComm, int n, void **data,
                                    size_t *sizes, int *tags, void **mhandles,
                                    void **phandles, void **request) {
  struct flagcxNetSocketComm *comm = (struct flagcxNetSocketComm *)recvComm;
  if (n != 1)
    return flagcxInternalError;
  FLAGCXCHECK(
      flagcxNetSocketGetRequest(comm, FLAGCX_SOCKET_RECV, data[0], sizes[0],
                                (struct flagcxNetSocketRequest **)request));
  // O4: 记录本地期望 seq（Test 校验通过后 comm->recvSeq++）
  (*(struct flagcxNetSocketRequest **)request)->seq = comm->recvSeq;
  return flagcxSuccess;
}""",
    "Irecv.seq")

# 6) Test 握手段：4 字节 size -> 8 字节 {seq,size} + 双向校验
rep("""  if (r->used == 1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    FLAGCXCHECK(
        flagcxSocketProgress(r->op, r->ctrlSock, &data, sizeof(int), &offset));

    if (offset == 0)
      return flagcxSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int))
      FLAGCXCHECK(
          flagcxSocketWait(r->op, r->ctrlSock, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == FLAGCX_SOCKET_RECV && data > r->size) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union flagcxSocketAddress addr;
      flagcxSocketGetAddr(r->ctrlSock, &addr);
      WARN(
          "NET/Socket : peer %s message truncated : receiving %d bytes instead of %d. If you believe your socket network is in healthy state, \\
          there may be a mismatch in collective sizes or environment settings (e.g. FLAGCX_PROTO, FLAGCX_ALGO) between ranks",
          flagcxSocketToString(&addr, line), data, r->size);
      return flagcxInvalidUsage;
    }
    r->size = data;
    r->offset = 0;
    r->used = 2; // done exchanging size""",
    """  if (r->used == 1) { /* try to send/recv {seq,size} header */
    struct flagcxSocketHdr hdr = {r->seq, r->size};
    int offset = 0;
    FLAGCXCHECK(
        flagcxSocketProgress(r->op, r->ctrlSock, &hdr, sizeof(hdr), &offset));

    if (offset == 0)
      return flagcxSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 8 bytes, but just in case ...
    if (offset < (int)sizeof(hdr))
      FLAGCXCHECK(
          flagcxSocketWait(r->op, r->ctrlSock, &hdr, sizeof(hdr), &offset));

    if (r->op == FLAGCX_SOCKET_RECV) {
      // O4: seq + size 双向校验——配对错位立即暴露，绝不静默。
      // 历史：原实现仅 data > r->size 报错，data < r->size 静默接受导致
      // 跨 op 数据错位不可检测（E1）。
      if (hdr.seq != r->seq || hdr.size != r->size) {
        char line[SOCKET_NAME_MAXLEN + 1];
        union flagcxSocketAddress addr;
        flagcxSocketGetAddr(r->ctrlSock, &addr);
        WARN(
            "NET/Socket : op mismatch on recv (peer %s): got seq=%d size=%d, expect seq=%d size=%d. \\
            Op ordering between ranks is broken — aborting instead of corrupting data.",
            flagcxSocketToString(&addr, line), hdr.seq, hdr.size, r->seq,
            r->size);
        return flagcxInternalError;
      }
      comm->recvSeq++; // O4: 校验通过，期望序号前进
    }
    r->size = hdr.size;
    r->offset = 0;
    r->used = 2; // done exchanging size""",
    "Test.handshake")

open(p, "w").write(s)
print("=== O4 PATCH DONE ===")
