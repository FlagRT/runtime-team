#!/bin/bash
# 910C 框架格修复验证：torch_npu 2.10.0(基线) vs 2.10.0.post7(环境层 overlay)
# 判据（对齐 Ascend 回归测试）：weakref 回调是否真触发 + 进程退出码
set -u
C=dc-probe-910c-1104
PY=/usr/local/python3.11.15/bin/python3
W=/workspace/wr1104
O=/home/hliu553/dc-rerun-1027/wr1104_out
mkdir -p "$O"

run_arm() { local tag=$1 pp=$2
  echo "=== 臂 $tag (PYTHONPATH=${pp:-<空>}) ==="
  docker exec -e ASCEND_RT_VISIBLE_DEVICES=2 ${pp:+-e PYTHONPATH=$pp} -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $W && $PY -u env_probe.py" > "$O/${tag}_env.log" 2>&1
  echo "  env rc=$?  $(grep -aoE 'torch_npu [0-9a-z.]+' "$O/${tag}_env.log" | tail -1)"
  docker exec -e ASCEND_RT_VISIBLE_DEVICES=2 ${pp:+-e PYTHONPATH=$pp} -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $W && $PY -u wr_cb_ctrl.py" > "$O/${tag}_ctrl.log" 2>&1
  echo "  ctrl(空白对照) rc=$?  $(grep -aoE 'plain_object_callback_fired: [A-Za-z]+' "$O/${tag}_ctrl.log" | tail -1)"
  docker exec -e ASCEND_RT_VISIBLE_DEVICES=2 ${pp:+-e PYTHONPATH=$pp} -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $W && $PY -u wr_callback.py" > "$O/${tag}_cb.log" 2>&1
  echo "  cb rc=$?  $(grep -aoE 'callback_fired: [A-Za-z]+ \(called=[0-9]+\)' "$O/${tag}_cb.log" | tail -2 | tr '\n' ' ')"
  echo "$(grep -aoE 'callback_fired: [A-Za-z]+' "$O/${tag}_cb.log" | tail -1)" > "$O/${tag}_verdict.txt"
  echo "rc=$?" >> "$O/${tag}_verdict.txt"
}

echo "########## 基座与介质核对 ##########"
docker exec "$C" bash -lc "$PY -m pip show torch_npu 2>/dev/null | grep -E '^Version'; echo '--- post7 包树 ---'; ls /workspace/pkgs_npu_post7/torch_npu-2.10.0.post7.dist-info/ 2>/dev/null | head -3"

run_arm A_base ""
echo "--- A 臂进程退出码实采 ---"
docker exec -e ASCEND_RT_VISIBLE_DEVICES=2 "$C" bash -lc "cd $W && $PY -u wr_callback.py >/dev/null 2>&1; echo rc=\$?" > "$O/A_rc.txt"; cat "$O/A_rc.txt"

run_arm B_post7 "/workspace/pkgs_npu_post7"
echo "--- B 臂进程退出码实采 ---"
docker exec -e ASCEND_RT_VISIBLE_DEVICES=2 -e PYTHONPATH=/workspace/pkgs_npu_post7 "$C" bash -lc "cd $W && $PY -u wr_callback.py >/dev/null 2>&1; echo rc=\$?" > "$O/B_rc.txt"; cat "$O/B_rc.txt"

echo "=========== 汇总 ==========="
printf "%-8s %-24s %-28s %s\n" 臂 torch_npu 回调 退出码
for t in A_base B_post7; do
  printf "%-8s %-24s %-28s %s\n" "$t" \
    "$(grep -aoE 'torch_npu [0-9a-z.]+' "$O/${t}_env.log" | tail -1)" \
    "$(grep -aoE 'callback_fired: [A-Za-z]+' "$O/${t}_cb.log" | tail -1)" \
    "$(cat $O/${t}_rc.txt 2>/dev/null)"
done
