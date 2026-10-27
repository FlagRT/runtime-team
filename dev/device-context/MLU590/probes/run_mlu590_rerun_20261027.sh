#!/bin/bash
# MLU590 共享层复跑（2026-10-27）：验证四处共享层改动在本实例零回归 ——
#   ① create_event(enable_timing) 新入口  ② _stream_ctx 强持有（退出期段错误第五载体）
#   ③ 职责审计 C13 方向/量纲/三态         ④ proto_infer_leg 模型路径自动探测
#
# 与 910C / P800 同口径（同六段 + [0.5] 模型权重预检，仅 backend/解释器/选卡变量分叉）。
# 用法（**宿主上跑**，脚本自己 docker exec）：
#   bash run_mlu590_rerun_20261027.sh
#
# 环境事实（沿用既有实测约定，不另立）：
#   容器 dc-mlu590-hliu553（自 2026-09-22 Up）· 容器内 /work == 宿主 /srv/hliu553
#   解释器 /flagos/bin/python3（py3.10 / torch 2.7.1+cpu / torch_mlu 1.29.2）
#   选卡 MLU_VISIBLE_DEVICES（**容器内 0 基**）；MLU 无独占卡模型，靠变量自律
#   模型 /hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B（HF 缓存形态，由 resolve_model 解析到 snapshots/<hash>）
set -u
WH=/srv/hliu553/dc_rerun_20261027
WC=/work/dc_rerun_20261027
C=dc-mlu590-hliu553
PY=/flagos/bin/python3
DEV=${DEV:-0}
MODEL=${MODEL:-/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B}
# ⚠️ 训练腿必须收 **snapshot 路径**：`proto_infer_leg` 有 resolve_model（把 HF 缓存根解析到
#    snapshots/<hash>），而 `proto_train_leg` **没有**该解析（共享层两腿不一致，2026-10-27 实测
#    `ValueError: Unrecognized model in .../models--Qwen--Qwen3-Embedding-0.6B`）⇒ 按历史用法先解析。
#    ⚠️ 解析必须**在容器内**做：`/hf_cache` 是容器内挂载点，宿主上是 `/srv/data/hf_cache`
#    （第一版在宿主算，结果为空 ⇒ 回落到缓存根 ⇒ 仍然报同一个错，实测踩过）。
MODEL_SNAP_EXPR='export DC_MODEL=$(ls -d "$DC_MODEL"/snapshots/*/ 2>/dev/null | head -1 || echo "$DC_MODEL")'

PROTO=$WC/dev/device-context/prototype
O="$WH/out"
mkdir -p "$O"

say() { echo "$@"; }

say "############################################################"
say "# MLU590 共享层复跑  $(date '+%F %T')"
say "# 容器: $C ｜ 解释器: $PY ｜ MLU_VISIBLE_DEVICES 单段=$DEV"
say "############################################################"

say ""
say "=== [0] 版本自证（三处改动都应在）==="
grep -c "enable_timing" "$WH/dev/device-context/prototype/runtime/__init__.py" 2>/dev/null | sed 's/^/  runtime\/__init__ enable_timing 命中: /'
grep -c "lambda _o=native_stream" "$WH/dev/device-context/prototype/runtime/backends/base.py" 2>/dev/null | sed 's/^/  _stream_ctx 强持有: /'
grep -c 'DC_MODEL") or next' "$WH/dev/device-context/prototype/runtime/proto/proto_infer_leg.py" 2>/dev/null | sed 's/^/  模型自动探测: /'
grep -c '@item(' "$WH/dev/device-context/prototype/scripts/duty_response_audit.py" 2>/dev/null | sed 's/^/  审计项数（期望 78）: /'

say ""
say "=== [0.5] 模型权重预检（防 nan 伪装成判据失败；910C 上真踩过）==="
docker exec -e DC_MODEL="$MODEL" "$C" bash -lc "$PY - <<'PY'
import os, glob, torch
from safetensors import safe_open
root = os.environ['DC_MODEL']
cands = glob.glob(os.path.join(root, 'model.safetensors')) or glob.glob(os.path.join(root, 'snapshots', '*', 'model.safetensors'))
if not cands:
    print('  预检跳过：未找到单文件 model.safetensors'); raise SystemExit(0)
p = sorted(cands)[0]
bad = 0
with safe_open(p, framework='pt') as f:
    for k in f.keys():
        t = f.get_tensor(k)
        if t.is_floating_point():
            bad += int(torch.isnan(t).sum()) + int(torch.isinf(t).sum())
print(f'  文件 = {p}')
print(f'  权重非有限值总数 = {bad}')
raise SystemExit(0 if bad == 0 else 3)
PY" > "$O/model_precheck.log" 2>&1
echo "  rc=$? (3 = 权重含非有限值 ⇒ 换模型副本，别往下跑)"
grep -aE "文件 =|非有限值|预检跳过" "$O/model_precheck.log" | tail -2 | sed 's/^/    /'

say ""
say "=== [1] 离线契约自检（无设备段）==="
docker exec -e DC_BACKEND=cambricon -e DC_ROOT="$PROTO" "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/backend_offline_check.py --backend cambricon" \
  > "$O/offline.log" 2>&1
echo "  rc=$?"
grep -aE "通过|失败|跳过|总计" "$O/offline.log" | tail -2 | sed 's/^/    /'

say ""
say "=== [2] smoke（卡 $DEV）==="
docker exec -e MLU_VISIBLE_DEVICES="$DEV" -e DC_BACKEND=cambricon -e DC_ROOT="$PROTO" "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -u runtime/smoke_runtime.py --backend cambricon" \
  > "$O/smoke.log" 2>&1
echo "  rc=$?"
grep -aE "结果:|通过 /" "$O/smoke.log" | tail -2 | sed 's/^/    /'

say ""
say "=== [3] 契约不变式 I1–I4（关键看**退出码**）==="
docker exec -e MLU_VISIBLE_DEVICES="$DEV" -e DC_BACKEND=cambricon -e DC_ROOT="$PROTO" \
  -e PYTHONFAULTHANDLER=1 "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -X faulthandler -u runtime/conformance/runner.py \
     --backend cambricon --cases contract_invariants --out $WC/out/ci.json" > "$O/invariants.log" 2>&1
echo "  rc=$?"
grep -aE "汇总:|结论:|CONTRACT_INVARIANTS|Fatal Python" "$O/invariants.log" | tail -3 | sed 's/^/    /'

say ""
say "=== [4] 职责审计 78 项（归档 62/0/16，C13 原为 SKIP·CNRT 建不出计时事件）==="
docker exec -e MLU_VISIBLE_DEVICES="$DEV" -e DC_BACKEND=cambricon -e DC_ROOT="$PROTO" "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/duty_response_audit.py --backend cambricon --out $WC/out/duty.json" \
  > "$O/duty.log" 2>&1
echo "  rc=$?"
grep -aE "DUTY_RESPONSE|verdict" "$O/duty.log" | tail -2 | sed 's/^/    /'
echo "  C13 行："; grep -a -A1 "C13" "$O/duty.log" | tail -2 | sed 's/^/    /'

say ""
say "=== [5] 推理腿：不传 DC_MODEL（验证自动探测命中 HF 缓存候选）==="
docker exec -e MLU_VISIBLE_DEVICES="$DEV" -e DC_BACKEND=cambricon -e DC_ROOT="$PROTO" "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -u runtime/proto/proto_infer_leg.py --backend cambricon --out $WC/out/infer_leg_autodetect.json" \
  > "$O/infer_leg_autodetect.log" 2>&1
echo "  rc=$?"
grep -aE "INFER_LEG|区分度|句/s" "$O/infer_leg_autodetect.log" | tail -2 | sed 's/^/    /'

say ""
say "--- [5b] 同一腿，显式 DC_MODEL（与 910C 权威跑同口径）---"
docker exec -e MLU_VISIBLE_DEVICES="$DEV" -e DC_BACKEND=cambricon -e DC_ROOT="$PROTO" -e DC_MODEL="$MODEL" "$C" bash -lc \
  "cd $PROTO && PYTHONDONTWRITEBYTECODE=1 $PY -u runtime/proto/proto_infer_leg.py --backend cambricon --out $WC/out/infer_leg.json" \
  > "$O/infer_leg.log" 2>&1
echo "  rc=$?"
grep -aE "INFER_LEG|区分度|句/s" "$O/infer_leg.log" | tail -2 | sed 's/^/    /'

say ""
say "=== [6] 训练腿 50 步（2 卡，容器内 0/1）==="
# ⚠️ DC_DIST_BT 必须显式给：本原型对寒武纪**拒绝兜底 gloo**（那会静默退化成纯 CPU 集合通信，
#    训练照样跑完但设备侧通信没被验证）。归档基线（20260922/20261008）实测用的是 `cncl`。
docker exec -e MLU_VISIBLE_DEVICES=0,1 -e DC_BACKEND=cambricon -e DC_ROOT="$PROTO" -e DC_DIST_BT=cncl \
  -e DC_MODEL="$MODEL" -e DC_OUT_DIR="$WC/out" -e MAX_STEPS=50 -e BATCH=4 -e SEQ=128 "$C" bash -lc \
  "cd $PROTO && $MODEL_SNAP_EXPR && echo \"[model] DC_MODEL=\$DC_MODEL\" && PYTHONDONTWRITEBYTECODE=1 $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
     runtime/proto/proto_train_leg.py --backend cambricon" > "$O/train_leg.log" 2>&1
echo "  rc=$?"
grep -aE "TRAIN_LEG|loss=|tok/s" "$O/train_leg.log" | tail -4 | sed 's/^/    /'

say ""
say "=== 汇总 ==="
for f in offline smoke invariants duty infer_leg train_leg; do
  printf "  %-12s " "$f"
  v=$(grep -aoE "TRAIN_LEG_PASS|TRAIN_LEG_FAIL|INFER_LEG_PASS|INFER_LEG_FAIL|CONTRACT_INVARIANTS_PASS|CONTRACT_INVARIANTS_FAIL|DUTY_RESPONSE_PASS|DUTY_RESPONSE_FAIL" "$O/$f.log" 2>/dev/null | tail -1)
  if [ -z "$v" ]; then
    # 中文汇总行（离线自检 / smoke 没有英文 verdict）：失败数为 0 即 PASS
    line=$(grep -aoE "结果: [0-9]+ 通过 / [0-9]+ 失败|自检结果: [0-9]+ 通过 / [0-9]+ 失败" "$O/$f.log" 2>/dev/null | tail -1)
    if [ -n "$line" ]; then
      fail=$(echo "$line" | grep -oE "/ [0-9]+ 失败" | grep -oE "[0-9]+")
      [ "${fail:-1}" = "0" ] && v="PASS（$line）" || v="FAIL（$line）"
    fi
  fi
  echo "${v:-(无 verdict)}"
done
say ""
say "证据目录（宿主）: $O"
