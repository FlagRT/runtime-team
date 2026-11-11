"""在 P800 上按范围清单输出「补丁包」到 stdout。

补丁包格式（小端）：
  uint64  段数
  每段: uint64 偏移, uint32 长度, 长度字节的数据
"""

import json
import struct
import sys

GOOD = ("/data1/dinghaisong/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/"
        "blobs/0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd")

spans = json.load(open(sys.argv[1]))["pages"]
out = sys.stdout.buffer
out.write(struct.pack("<Q", len(spans)))
sent = 0
with open(GOOD, "rb") as f:
    for off, ln in spans:
        f.seek(off)
        data = f.read(ln)
        if len(data) != ln:
            raise SystemExit("短读 @%d" % off)
        out.write(struct.pack("<QI", off, ln))
        out.write(data)
        sent += ln
out.flush()
sys.stderr.write("sent %d segments, %d bytes\n" % (len(spans), sent))
