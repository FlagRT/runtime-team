#!/bin/bash
# P800 · T1-re（2026-10-14）：跨后端基类改动**第二次**落共享层的破坏面复跑
# 与 10-13 的区别：本轮 `backends/base.py` 又改了（T11 修法：无上下文不登记 / 释放表强持有 / 登记表上限）
# 目的：确认这道改动在 P800 上**无回归**（判据 + 逐项状态与 10-13 基线比对）
set -u

PROTO=/workspace/dc_t1re_20261014/prototype
OUT=/workspace/dc_t1re_20261014/out
source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda
PY=python

export DC_BACKEND=kunlun
export DC_ROOT=$PROTO
export DC_OUT_DIR=$OUT
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0          # 避开故障卡 dev1（UUID b3509946）
mkdir -p "$OUT"
cd "$PROTO" || exit 1

echo "########## [0] 跑的是当前版本吗（版本自证）##########"
echo "  sha256(base.py) 前 16 位 = $(sha256sum runtime/backends/base.py | cut -c1-16)（本机 4dca3469d99ebf81）"
echo "  _ledger_put 命中（期望 6）  = $(grep -c '_ledger_put' runtime/backends/base.py)"
echo "  cid is not None（期望 3）   = $(grep -c 'cid is not None' runtime/backends/base.py)"
echo "  释放表强持有（期望 1）      = $(grep -c '_o=native_stream' runtime/backends/base.py)"
echo "  审计项数 @item（期望 78）   = $(grep -c '@item(' scripts/duty_response_audit.py)"
echo "  registry 后端表            = $(grep -oE '_KNOWN_BACKENDS = \([^)]*' runtime/backends/registry.py)"
echo "  @ $(hostname) $(date '+%F %T')  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo
echo "########## [0b] 离线自检（无设备；回归基线 111/0/1 → 期望 114/0/1）##########"
$PY scripts/backend_offline_check.py --backend kunlun 2>&1 | grep -aE "离线自检结果|\[FAIL\]" | sed 's/^/  /'

echo
echo "########## [1] 设备可见性 ##########"
$PY -c "
import torch
print('  torch     =', torch.__version__)
print('  visible   =', torch.cuda.device_count())
print('  dev0      =', torch.cuda.get_device_name(0))
" 2>&1 | tail -4

echo
echo "########## [2] 契约不变式 I1–I4（真机）##########"
date "+[%H:%M:%S] start"
$PY -u runtime/conformance/runner.py --backend kunlun --cases contract_invariants \
    --out "$OUT/t1_coninvariants_kunlun_20261014.json" > "$OUT/t1_coninvariants_kunlun_20261014.log" 2>&1
echo "  rc=$?"
grep -aE "^  \[(PASS|FAIL|SKIP)|=== 汇总" "$OUT/t1_coninvariants_kunlun_20261014.log" | sed 's/^/  /' | tail -8
date "+[%H:%M:%S] end"

echo
echo "########## [3] 职责响应审计 78 项（主判据）##########"
date "+[%H:%M:%S] start"
$PY -u scripts/duty_response_audit.py --backend kunlun \
    --out "$OUT/t1_duty_kunlun_20261014.json" > "$OUT/t1_duty_kunlun_20261014.log" 2>&1
echo "  rc=$?"
date "+[%H:%M:%S] end"

echo
echo "########## [4] 汇总 ##########"
$PY -c "
import json
try:
    d = json.load(open('$OUT/t1_duty_kunlun_20261014.json'))
    print('  duty     : ok=%s fail=%s skip=%s total=%s verdict=%s' % (d.get('ok'), d.get('fail'), d.get('skip'), d.get('total'), d.get('verdict')))
    L = {r.get('sid'): r.get('status') for r in (d.get('rows') or []) if str(r.get('sid')) in ('L1','L2','L3','L4','L5')}
    print('  L 域     :', sorted(L.items()))
except Exception as e:
    print('  duty JSON 解析失败:', e)
try:
    c = json.load(open('$OUT/t1_coninvariants_kunlun_20261014.json'))
    print('  coninv   : passed=%s/%s verdict=%s' % (c.get('passed'), c.get('total'), c.get('verdict')))
except Exception as e:
    print('  coninv JSON 解析失败:', e)
"
echo "T1RE_P800_DONE"
