#!/bin/bash
set -e
cd /workspace/proto
PY="python3 protocol_runner.py --out /workspace/proto --tag ON --port 18320"
for L in 256 1024 3072; do for R in 1 2 3; do $PY --action S1 --length $L --round $R; done; done
for R in 1 2 3 4; do $PY --action S2 --round $R; done
for C in 17 35 63 77; do for R in 1 2; do $PY --action B --c $C --round $R; done; done
echo ON910_DONE
