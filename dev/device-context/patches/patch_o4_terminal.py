import sys
ROOT = sys.argv[1].rstrip("/")

# 1) socket_adaptor.cc：Test 错误终态
p = ROOT + "/flagcx/adaptor/net/socket_adaptor.cc"
s = open(p).read()
# 1a. Test 开头：used==3 直接返回错误（防重试读流）
old1 = """  *done = 0;
  struct flagcxNetSocketRequest *r = (struct flagcxNetSocketRequest *)request;"""
new1 = """  *done = 0;
  struct flagcxNetSocketRequest *r = (struct flagcxNetSocketRequest *)request;
  if (r->used == 3) /* O4: 协议错位终态——不再触碰 socket */
    return flagcxInternalError;"""
n1 = s.count(old1)
if n1 != 1:
    print(f"[FAIL] test-head: matches={n1}"); sys.exit(1)
s = s.replace(old1, new1, 1)
print("[ok] test-head used==3 guard")
# 1b. mismatch 分支：置 used=3（终态）
old2 = """        return flagcxInternalError;
      }
      r->comm->recvSeq++; // O4: 校验通过，期望序号前进"""
new2 = """        r->used = 3; // O4: 协议错位终态——停止触碰 socket，错误持续上报
        return flagcxInternalError;
      }
      r->comm->recvSeq++; // O4: 校验通过，期望序号前进"""
n2 = s.count(old2)
if n2 != 1:
    print(f"[FAIL] mismatch-terminal: matches={n2}"); sys.exit(1)
s = s.replace(old2, new2, 1)
print("[ok] mismatch used=3 terminal")
# 1c. 移除 O4-DBG 调试打印
dbg = '    fprintf(stderr, "[O4-DBG] hdr seq=%d size=%d | r seq=%d size=%d op=%d\\n", hdr.seq, hdr.size, r->seq, r->size, r->op); fflush(stderr);\n'
if dbg in s:
    s = s.replace(dbg, "", 1)
    print("[ok] removed O4-DBG")
else:
    print("[skip] no O4-DBG")
open(p, "w").write(s)

# 2) proxy.cc：progressOps 传播 flagcxProxySend/Recv 错误
p2 = ROOT + "/flagcx/core/proxy.cc"
s2 = open(p2).read()
pairs = [
    ("              flagcxProxySend(resources, op->recvbuff, op->nbytes, &op->args);",
     "              FLAGCXCHECK(flagcxProxySend(resources, op->recvbuff, op->nbytes,\n                                  &op->args));"),
    ("              flagcxProxyRecv(resources, op->recvbuff, op->nbytes, &op->args);",
     "              FLAGCXCHECK(flagcxProxyRecv(resources, op->recvbuff, op->nbytes,\n                                  &op->args));"),
]
for i, (old, new) in enumerate(pairs):
    n = s2.count(old)
    if n != 1:
        print(f"[FAIL] proxy.cc #{i}: matches={n}"); sys.exit(1)
    s2 = s2.replace(old, new, 1)
    print(f"[ok] proxy.cc #{i}")
open(p2, "w").write(s2)
print("=== O4 TERMINAL DONE ===")
