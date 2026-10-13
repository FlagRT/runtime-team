#!/bin/bash
# 910C · T1 本轮复跑（容器内执行；cwd = 原型根）
# 目的：跨后端基类改动（2026-10-10：_owned_streams 3 元组 + 类型复核）的破坏面覆盖
set -u

PROTO=/mnt/raid/hliu553/dc_t1_20261013/prototype
OUT=/mnt/raid/hliu553/dc_t1_20261013/out
PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python

export DC_BACKEND=ascend
export DC_ROOT=$PROTO
export DC_OUT_DIR=$OUT
export PYTHONDONTWRITEBYTECODE=1
export ASCEND_RT_VISIBLE_DEVICES=0        # 容器内 0 基索引（本容器只挂 davinci5）
mkdir -p "$OUT"
cd "$PROTO" || exit 1

echo "########## [0] 跑的是当前版本吗（版本自证）##########"
echo "类型复核 tname 命中（期望 4）  = $(grep -c tname runtime/backends/base.py)"
echo "审计项数 @item（期望 78）      = $(grep -c '@item(' scripts/duty_response_audit.py)"
echo "registry 后端表                = $(grep -oE '_KNOWN_BACKENDS = \([^)]*' runtime/backends/registry.py)"
echo "契约不变式入口                 = $(ls runtime/conformance/contract_invariants.py 2>/dev/null && echo 存在)"
echo "@ $(hostname) $(date '+%F %T')  ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES（容器内 0 基）"

echo
echo "########## [1] 设备可见性 ##########"
$PY -c "
import torch, torch_npu
print('  torch     =', torch.__version__)
print('  torch_npu =', torch_npu.__version__)
print('  npu_count =', torch.npu.device_count())
print('  dev0      =', torch.npu.get_device_name(0))
" 2>&1 | tail -6

echo
echo "########## [2] 契约不变式 I1–I4（真机）##########"
date "+[%H:%M:%S] start"
$PY -u runtime/conformance/runner.py --backend ascend --cases contract_invariants \
    --out "$OUT/t1_coninvariants_ascend.json" > "$OUT/t1_coninvariants_ascend.log" 2>&1
echo "rc=$?"
grep -E "^  \[(PASS|FAIL|SKIP)|=== 汇总|结论:" "$OUT/t1_coninvariants_ascend.log" | sed 's/^/  /' | tail -8
date "+[%H:%M:%S] end"

echo
echo "########## [3] 职责响应审计 78 项（主判据）##########"
date "+[%H:%M:%S] start"
$PY -u scripts/duty_response_audit.py --backend ascend \
    --out "$OUT/t1_duty_ascend.json" > "$OUT/t1_duty_ascend.log" 2>&1
echo "rc=$?"
date "+[%H:%M:%S] end"

echo
echo "########## [4] 汇总 ##########"
$PY -c "
import json
try:
    d = json.load(open('$OUT/t1_duty_ascend.json'))
    print('  duty     : ok=%s fail=%s skip=%s total=%s verdict=%s' % (d.get('ok'), d.get('fail'), d.get('skip'), d.get('total'), d.get('verdict')))
    bad = [(r.get('sid'), str(r.get('detail'))[:120]) for r in (d.get('rows') or []) if str(r.get('status','')).upper() == 'FAIL']
    for sid, det in bad:
        print('  FAIL     :', sid, '|', det)
    print('  L1 行    :', [(r.get('sid'), r.get('status')) for r in (d.get('rows') or []) if str(r.get('sid')) in ('L1','L2','L3','L4','L5')])
except Exception as e:
    print('  duty JSON 解析失败:', e)
try:
    c = json.load(open('$OUT/t1_coninvariants_ascend.json'))
    print('  coninv   : passed=%s/%s verdict=%s' % (c.get('passed'), c.get('total'), c.get('verdict')))
except Exception as e:
    print('  coninv JSON 解析失败:', e)
"
echo "T1_910C_DONE"
