#!/bin/bash
# 变体 C/D：对逐字副本做「整块移除」与「只改一处」两种二分
set -uo pipefail
NAME=hliu553-dc-dev
HOST_P=/bmcp_lvm_fs/hliu553/runtime-team/dev/device-context/prototype
CTR_P=/workspace/runtime-team/dev/device-context/prototype

/usr/bin/python3 - "$HOST_P/runtime/proto/proto_train_leg.py" "/bmcp_lvm_fs/hliu553/probes_ppu" <<'PY'
import sys, pathlib
src, dst = sys.argv[1], sys.argv[2]
s = pathlib.Path(src).read_text(encoding="utf-8")

BLOCK = '''    runtime.use(BACKEND)
    runtime.set_device(local_rank)
    res["device_type"] = runtime.current().device_type
    dev_id = f"{runtime.current().device_type}:{local_rank}"
    n_dev = runtime.device_count()
    st = runtime.create_stream()
'''
assert s.count(BLOCK) == 1, f"BLOCK count={s.count(BLOCK)}"

SYNC = "    runtime.synchronize(local_rank)\n"
assert s.count(SYNC) == 1, f"SYNC count={s.count(SYNC)}"

STREAM = "    st = runtime.create_stream()\n"
assert s.count(STREAM) == 1, f"STREAM count={s.count(STREAM)}"

# C：整块移除原型运行时调用（同步改用 torch 原生）
c = s.replace(BLOCK, '''    res["device_type"] = "cuda"
    dev_id = f"cuda:{local_rank}"
    n_dev = torch.cuda.device_count()
    st = None
''').replace(SYNC, "    torch.cuda.synchronize()\n")
pathlib.Path(dst, "ptl_noruntime.py").write_text(c, encoding="utf-8")
print("WROTE ptl_noruntime.py")

# D：只把「持有统一流」取消（其余原型调用保留）
d = s.replace(STREAM, "    st = None\n")
pathlib.Path(dst, "ptl_nostream.py").write_text(d, encoding="utf-8")
print("WROTE ptl_nostream.py")
PY

for V in ptl_noruntime ptl_nostream; do
  echo "=========== 变体 $V ==========="
  docker exec "$NAME" bash -lc "
    set -u
    OUT=/workspace/probes_ppu/segv_isolate/e13_$V
    rm -rf \$OUT; mkdir -p \$OUT
    cd /workspace/probes_ppu
    export PYTHONFAULTHANDLER=1
    CUDA_VISIBLE_DEVICES=0,1 DC_BACKEND=ppu DC_DIST_BT=nccl DC_ROOT=$CTR_P \
      DC_MODEL=/workspace/models/Qwen3-Embedding-0.6B DC_OUT_DIR=\$OUT MAX_STEPS=20 BATCH=4 SEQ=128 \
      timeout 400 python3 -u -m torch.distributed.run --standalone --nproc_per_node=2 --tee 3 \
        --log-dir \$OUT/logs $V.py > \$OUT/run.log 2>&1
    echo RC=\$?
    grep -nE 'TRAIN_LEG_(PASS|FAIL)|Segmentation|exitcode' \$OUT/run.log | head -8
  "
done
