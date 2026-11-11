#!/bin/bash
# MLU590 · T1-re（2026-10-14 第二轮）：共享层**再次**改动后的破坏面复跑
# 与 10-14 首轮的区别：`backends/base.py` 又改了（T11 修法：无上下文不登记 / 释放表强持有 / 登记表上限）
# 目的：确认该改动在 MLU590 上**无回归**（判据 + 与首轮基线**逐项**比对）
set -u

PROTO=/work/dc_t1re_20261014/prototype
OUT=/work/dc_t1re_20261014/out
PY=/flagos/bin/python3

export DC_BACKEND=cambricon
export DC_ROOT=$PROTO
export DC_OUT_DIR=$OUT
export PYTHONDONTWRITEBYTECODE=1
export MLU_VISIBLE_DEVICES=0           # 容器内 0 基索引
mkdir -p "$OUT"
cd "$PROTO" || exit 1

echo "########## [0] 跑的是当前版本吗（版本自证）##########"
echo "  sha256(base.py) 前 16 位 = $(sha256sum runtime/backends/base.py | cut -c1-16)（本机 4dca3469d99ebf81）"
echo "  _ledger_put 命中（期望 6）  = $(grep -c '_ledger_put' runtime/backends/base.py)"
echo "  cid is not None（期望 3）   = $(grep -c 'cid is not None' runtime/backends/base.py)"
echo "  释放表强持有（期望 1）      = $(grep -c '_o=native_stream' runtime/backends/base.py)"
echo "  审计项数 @item（期望 78）   = $(grep -c '@item(' scripts/duty_response_audit.py)"
echo "  registry 后端表            = $(grep -oE '_KNOWN_BACKENDS = \([^)]*' runtime/backends/registry.py)"
echo "  @ $(hostname) $(date '+%F %T')  MLU_VISIBLE_DEVICES=$MLU_VISIBLE_DEVICES"

echo
echo "########## [0b] 离线自检（无设备；首轮基线 100/0/0 → 期望 103/0/0）##########"
$PY scripts/backend_offline_check.py --backend cambricon 2>&1 | grep -aE "离线自检结果|\[FAIL\]" | sed 's/^/  /'

echo
echo "########## [1] 设备可见性 ##########"
$PY -c "
import torch, torch_mlu
print('  torch     =', torch.__version__)
print('  torch_mlu =', torch_mlu.__version__)
print('  mlu_count =', torch.mlu.device_count())
print('  dev0      =', torch.mlu.get_device_name(0))
" 2>&1 | tail -6

echo
echo "########## [2] 契约不变式 I1–I4（真机）##########"
date "+[%H:%M:%S] start"
$PY -u runtime/conformance/runner.py --backend cambricon --cases contract_invariants \
    --out "$OUT/t1_coninvariants_cambricon_20261014re.json" > "$OUT/t1_coninvariants_cambricon_20261014re.log" 2>&1
echo "  rc=$?"
grep -aE "^  \[(PASS|FAIL|SKIP)|=== 汇总" "$OUT/t1_coninvariants_cambricon_20261014re.log" | sed 's/^/  /' | tail -8
date "+[%H:%M:%S] end"

echo
echo "########## [3] 职责响应审计 78 项（主判据）##########"
date "+[%H:%M:%S] start"
$PY -u scripts/duty_response_audit.py --backend cambricon \
    --out "$OUT/t1_duty_cambricon_20261014re.json" > "$OUT/t1_duty_cambricon_20261014re.log" 2>&1
echo "  rc=$?"
date "+[%H:%M:%S] end"

echo
echo "########## [4] 汇总 ##########"
$PY -c "
import json
try:
    d = json.load(open('$OUT/t1_duty_cambricon_20261014re.json'))
    print('  duty     : ok=%s fail=%s skip=%s total=%s verdict=%s' % (d.get('ok'), d.get('fail'), d.get('skip'), d.get('total'), d.get('verdict')))
    L = {r.get('sid'): r.get('status') for r in (d.get('rows') or []) if str(r.get('sid')) in ('L1','L2','L3','L4','L5')}
    print('  L 域     :', sorted(L.items()))
    for r in (d.get('rows') or []):
        if str(r.get('status','')).upper() == 'FAIL':
            print('  FAIL     :', r.get('sid'), '|', str(r.get('detail'))[:130])
except Exception as e:
    print('  duty JSON 解析失败:', e)
try:
    c = json.load(open('$OUT/t1_coninvariants_cambricon_20261014re.json'))
    print('  coninv   : passed=%s/%s verdict=%s' % (c.get('passed'), c.get('total'), c.get('verdict')))
except Exception as e:
    print('  coninv JSON 解析失败:', e)
"
echo "T1RE_MLU590_DONE"
