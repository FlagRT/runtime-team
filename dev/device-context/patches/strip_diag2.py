#!/usr/bin/env python3
"""O5 二次剥离：清除 rebase 合并残留的诊断打点 + O4 INJECT。

背景：8/28 O1 rebase（3354cb3 onto 900f9d2）的 3-way merge 使
group.cc/proxy.cc/init.cc/socket_adaptor.cc 的 O5 剥离丢失；
cann_adaptor.cc 的 3 处错误路径诊断从未剥离；socket_adaptor.cc 含
O4 fault 验证的 INJECT-FAULT 临时破坏（必须移除）。

用法：python3 strip_diag2.py <FlagCX根>
"""
import sys, os

ROOT = sys.argv[1].rstrip("/")

def rep(path, pairs):
    p = os.path.join(ROOT, path)
    s = open(p).read()
    for i, (old, new) in enumerate(pairs):
        n = s.count(old)
        if n == 0:
            print(f"[FAIL] {path} #{i}: not found")
            sys.exit(1)
        if n > 1:
            print(f"[FAIL] {path} #{i}: ambiguous ({n})")
            sys.exit(1)
        s = s.replace(old, new, 1)
    open(p, "w").write(s)
    print(f"[ok] {path}: {len(pairs)}")

# ---- group.cc：6 处 ----
rep("flagcx/core/group.cc", [
    ("""               !flagcxIntruQueueEmpty(&tasks->peers[sendPeer].sendQueue)) {
          fprintf(stderr, "[HETERO-DBG] groupLaunch: processing recv task peer=%d\\n", recvPeer); fflush(stderr);
          // Process one recv task (for IPC register)""",
     """               !flagcxIntruQueueEmpty(&tasks->peers[sendPeer].sendQueue)) {
          // Process one recv task (for IPC register)"""),
    ("""            op->stream = p2p->stream;
            if (op->connection == NULL) {
              fprintf(stderr, "[HETERO-DBG] groupLaunch recv conn NULL: rank=%d peer=%d ch=%d connected=%d\\n",
                      comm->rank, peer, op->channelId,
                      comm->channels[op->channelId].peers[peer]->recv[0].connected); fflush(stderr);
              WARN("groupLaunch: recv proxyConn.connection is NULL for rank %d """,
     """            op->stream = p2p->stream;
            if (op->connection == NULL) {
              WARN("groupLaunch: recv proxyConn.connection is NULL for rank %d """),
    ("""  fprintf(stderr, "[HETERO-DBG] groupLaunch: tasks processed, proxyOps=%zu\\n", proxyOps.size()); fflush(stderr);
  // Save all proxy ops in step order""",
     """  // Save all proxy ops in step order"""),
    ("""  }
  fprintf(stderr, "[HETERO-DBG] groupLaunch: proxyOps saved\\n"); fflush(stderr);

  if (launchStream != nullptr && launchEvent != nullptr) {
    fprintf(stderr, "[HETERO-DBG] groupLaunch: launching kernel (device=%d)\\n", deviceAsyncKernel ? 1 : 0); fflush(stderr);
    if (deviceAsyncKernel) {""",
     """  }

  if (launchStream != nullptr && launchEvent != nullptr) {
    if (deviceAsyncKernel) {"""),
    ("""fail:
  fprintf(stderr, "[HETERO-DBG] groupLaunch FAILED ret=%d, cleaning up\\n", ret); fflush(stderr);
  groupCleanup(&flagcxGroupJobMainPtr->base);""",
     """fail:
  groupCleanup(&flagcxGroupJobMainPtr->base);"""),
])

# ---- proxy.cc：2 处 ----
rep("flagcx/core/proxy.cc", [
    ("""                                       void *respBuff, int respSize) {
  fprintf(stderr, "[HETERO-DBG] ProxyCallBlocking enter type=%d peer=%d\\n", type, proxyConn->tpRank); fflush(stderr);
  // Alloc some memory to act as a handle""",
     """                                       void *respBuff, int respSize) {
  // Alloc some memory to act as a handle"""),
    ("""  if (proxyConn->connection == NULL) {
    fprintf(stderr, "[HETERO-DBG] ProxyConnect NULL conn: rank=%d -> peer=%d transport=%d send=%d\\n",
            comm->rank, proxyRank, transport, send); fflush(stderr);
    WARN("flagcxProxyConnect: service thread returned NULL connection for rank """,
     """  if (proxyConn->connection == NULL) {
    WARN("flagcxProxyConnect: service thread returned NULL connection for rank """),
])

# ---- init.cc：12 处（行内 fprintf + fflush） ----
rep("flagcx/core/init.cc", [
    ("""  INFO(FLAGCX_INIT, "inside initTransportsRank");
  fprintf(stderr, "[HETERO-DBG] rank=%d inside initTransportsRank\\n", comm->rank); fflush(stderr);
  flagcxResult_t ret = flagcxSuccess;""",
     """  INFO(FLAGCX_INIT, "inside initTransportsRank");
  flagcxResult_t ret = flagcxSuccess;"""),
    ("""  // Question: where did we initialize comm->bootstrap?
  fprintf(stderr, "[HETERO-DBG] rank=%d before bootstrapAllGather peerInfo\\n", comm->rank); fflush(stderr);
  INFO(FLAGCX_INIT, "start bootstrapAllGather for peerInfo");""",
     """  // Question: where did we initialize comm->bootstrap?
  INFO(FLAGCX_INIT, "start bootstrapAllGather for peerInfo");"""),
    ("""                                         sizeof(struct flagcxPeerInfo)),
                  ret, fail);
  fprintf(stderr, "[HETERO-DBG] rank=%d peerInfo allgather done, before barrier\\n", comm->rank); fflush(stderr);
  FLAGCXCHECKGOTO(bootstrapCollBarrier(comm->bootstrap, rank, nranks, 0), ret,
                  fail);
  fprintf(stderr, "[HETERO-DBG] rank=%d barrier done\\n", comm->rank); fflush(stderr);
""",
     """                                         sizeof(struct flagcxPeerInfo)),
                  ret, fail);
  FLAGCXCHECKGOTO(bootstrapCollBarrier(comm->bootstrap, rank, nranks, 0), ret,
                  fail);
"""),
    ("""    comm->magic = magic;
    fprintf(stderr, "[HETERO-DBG] rank=%d before bootstrapCollInit\\n", comm->rank); fflush(stderr);
    FLAGCXCHECKGOTO(""",
     """    comm->magic = magic;
    FLAGCXCHECKGOTO("""),
    ("""                          comm->rank, comm->nRanks, magic, comm->abortFlag,
                          &comm->bootstrap),
        res, fail);
    fprintf(stderr, "[HETERO-DBG] rank=%d bootstrapCollInit OK\\n", comm->rank); fflush(stderr);
  }""",
     """                          comm->rank, comm->nRanks, magic, comm->abortFlag,
                          &comm->bootstrap),
        res, fail);
  }"""),
    ("""  fprintf(stderr, "[HETERO-DBG] rank=%d before flagcxNetInit\\n", comm->rank); fflush(stderr);
  FLAGCXCHECK(flagcxNetInit(comm));
  fprintf(stderr, "[HETERO-DBG] rank=%d flagcxNetInit OK\\n", comm->rank); fflush(stderr);
  INFO(FLAGCX_INIT, "Using network %s", comm->netAdaptor->name);
  fprintf(stderr, "[HETERO-DBG] rank=%d before getBusId\\n", comm->rank); fflush(stderr);
  INFO(FLAGCX_INIT, "getting busId for cudaDev %d", comm->cudaDev);
  FLAGCXCHECK(getBusId(comm->cudaDev, &comm->busId));
  fprintf(stderr, "[HETERO-DBG] rank=%d getBusId OK busId=%lx\\n", comm->rank, (unsigned long)comm->busId); fflush(stderr);
  INFO(FLAGCX_INIT, "getting commHash for rank %d", comm->rank);""",
     """  FLAGCXCHECK(flagcxNetInit(comm));
  INFO(FLAGCX_INIT, "Using network %s", comm->netAdaptor->name);
  INFO(FLAGCX_INIT, "getting busId for cudaDev %d", comm->cudaDev);
  FLAGCXCHECK(getBusId(comm->cudaDev, &comm->busId));
  INFO(FLAGCX_INIT, "getting commHash for rank %d", comm->rank);"""),
    ("""  fprintf(stderr, "[HETERO-DBG] rank=%d before initTransportsRank\\n", comm->rank); fflush(stderr);
  INFO(FLAGCX_INIT, "start initTransportsRank");
  FLAGCXCHECKGOTO(initTransportsRank(comm, NULL), res, fail);
  fprintf(stderr, "[HETERO-DBG] rank=%d initTransportsRank OK\\n", comm->rank); fflush(stderr);
""",
     """  INFO(FLAGCX_INIT, "start initTransportsRank");
  FLAGCXCHECKGOTO(initTransportsRank(comm, NULL), res, fail);
"""),
])

# ---- socket_adaptor.cc：3 处 enter + INJECT 移除 ----
rep("flagcx/adaptor/net/socket_adaptor.cc", [
    ("""                                     void **listenComm) {
  fprintf(stderr, "[HETERO-DBG] SocketListen enter dev=%d\\n", dev); fflush(stderr);
  if (dev < 0 ||""",
     """                                     void **listenComm) {
  if (dev < 0 ||"""),
    ("""                                      void **sendComm) {
  fprintf(stderr, "[HETERO-DBG] SocketConnect enter dev=%d\\n", dev); fflush(stderr);
  if (dev < 0 ||""",
     """                                      void **sendComm) {
  if (dev < 0 ||"""),
    ("""flagcxResult_t flagcxNetSocketAccept(void *listenComm, void **recvComm) {
  fprintf(stderr, "[HETERO-DBG] SocketAccept enter\\n"); fflush(stderr);
  struct flagcxNetSocketListenComm *lComm =""",
     """flagcxResult_t flagcxNetSocketAccept(void *listenComm, void **recvComm) {
  struct flagcxNetSocketListenComm *lComm ="""),
    ("""  // O4: 记录本地期望 seq（Test 校验通过后 comm->recvSeq++）
  // [INJECT-FAULT] 故意 +1 制造 seq 错位，验证 O4 检测能力（验证后恢复）
  (*(struct flagcxNetSocketRequest **)request)->seq = comm->recvSeq + 1;""",
     """  // O4: 记录本地期望 seq（Test 校验通过后 comm->recvSeq++）
  (*(struct flagcxNetSocketRequest **)request)->seq = comm->recvSeq;"""),
])

# ---- cann_adaptor.cc：3 处错误路径诊断 ----
rep("flagcx/adaptor/device/cann_adaptor.cc", [
    ("""    if (serr != ACL_SUCCESS) {
      fprintf(stderr,
              "[HETERO-DBG] aclrtSubscribeReport failed aclErr=%d, "
              "fallback to direct host call\\n",
              (int)serr);
      fflush(stderr);
      fn(args);
      return flagcxSuccess;
    }""",
     """    if (serr != ACL_SUCCESS) {
      fn(args);
      return flagcxSuccess;
    }"""),
    ("""  if (err != ACL_SUCCESS) {
    fprintf(stderr,
            "[HETERO-DBG] aclrtLaunchCallback failed aclErr=%d, "
            "fallback to direct host call\\n",
            (int)err);
    fflush(stderr);
    fn(args);
    return flagcxSuccess;
  }""",
     """  if (err != ACL_SUCCESS) {
    fn(args);
    return flagcxSuccess;
  }"""),
    ("""  aclError perr = aclrtProcessReport(-1);
  if (perr != ACL_SUCCESS) {
    fprintf(stderr, "[HETERO-DBG] aclrtProcessReport aclErr=%d\\n", (int)perr);
    fflush(stderr);
  }
  return flagcxSuccess;""",
     """  aclError perr = aclrtProcessReport(-1);
  (void)perr;
  return flagcxSuccess;"""),
])

print("=== STRIP2 DONE ===")
