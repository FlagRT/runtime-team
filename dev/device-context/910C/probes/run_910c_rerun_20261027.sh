#!/bin/bash
# 910C 共享层复跑（2026-10-27）：验证四处共享层改动在本实例零回归 ——
#   ① create_event(enable_timing) 新入口  ② _stream_ctx 强持有（退出期段错误第五载体）
#   ③ 职责审计 C13 方向/量纲/三态         ④ proto_infer_leg 模型路径自动探测
#
# 与 P800 同口径（同脚本结构，仅 backend/路径/启动方式分叉）。
# 用法（宿主上跑，脚本自己 docker exec）：
#   bash run_910c_rerun_20261027.sh
#
# 纪律：容器只挂 davinci6/7（与接入手册训练腿示例同一对）；单卡段用容器内 0 基索引。
set -u
WH=/home/hliu553/dc-rerun-1027          # 宿主侧工作区（本机盘 /home）
WC=/workspace                            # 容器侧（宿主 $WH 即容器 /workspace）
C=dc-rerun-1027-910c
DEV=${DEV:-0}                            # 容器内 0 基索引（只挂 2 张卡）
MODEL=${MODEL:-$WC/models/Qwen3-Embedding-0.6B}
IMG=flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64
O="$WH/out"
mkdir -p "$O"

say() { echo "$@"; }
run_in() {   # run_in <logname> <docker exec 参数...> -- <容器内命令>
  local log="$1"; shift
  docker exec "$@" > "$O/$log.log" 2>&1
  local rc=$?
  echo "  rc=$rc  (日志 out/$log.log)"
  return $rc
}

say "############################################################"
say "# 910C 共享层复跑  $(date '+%F %T')"
say "# 镜像: $IMG"
say "# 容器: $C（只挂 davinci6/7）"
say "############################################################"

say ""
say "=== [0] 代码版本核对（三处改动都应在）==="
grep -c "enable_timing" "$WH/dev/device-context/prototype/runtime/__init__.py" 2>/dev/null | sed 's/^/  runtime/__init__ enable_timing 命中: /'
grep -c "lambda _o=native_stream" "$WH/dev/device-context/prototype/runtime/backends/base.py" 2>/dev/null | sed 's/^/  _stream_ctx 强持有: /'
grep -c "DC_MODEL\") or next" "$WH/dev/device-context/prototype/runtime/proto/proto_infer_leg.py" 2>/dev/null | sed 's/^/  模型自动探测: /'
echo "  git 版自身份（宿主侧工作区非 git，用文件指纹）："
( cd "$WH/dev/device-context" && find prototype/runtime -name "*.py" -newer prototype/README.md 2>/dev/null | head -3 )

say ""
say "=== [0.5] 模型权重预检（**防 nan 伪装成判据失败**）==="
# 2026-10-27 实测：迁移快照里 model.safetensors 有 410 个 NaN + 1 个 inf（同一快照另一份副本干净）
# ⇒ 训练腿 loss 从第 0 步就 nan、5/6 失败，看着像栈回归，其实是权重坏了。开跑前先扫一遍。
docker exec -e DC_MODEL="$MODEL" "$C" bash -lc "python3 - <<'PY'
import os, torch
from safetensors import safe_open
p = os.path.join(os.environ[\"DC_MODEL\"], \"model.safetensors\")
if not os.path.isfile(p):
    print(\"  预检跳过：无单文件 model.safetensors（分片模型请自行确认）\"); raise SystemExit(0)
bad = 0
with safe_open(p, framework=\"pt\") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        if t.is_floating_point():
            bad += int(torch.isnan(t).sum()) + int(torch.isinf(t).sum())
print(f\"  权重非有限值总数 = {bad}\")
raise SystemExit(0 if bad == 0 else 3)
PY" > "$O/model_precheck.log" 2>&1
echo "  rc=$? (3 = 权重含非有限值 ⇒ 换模型副本，别往下跑)"
grep -aE "非有限值|预检跳过" "$O/model_precheck.log" | tail -2 | sed 's/^/    /'

say ""
say "=== [1] 离线契约自检（无设备段）==="
run_in offline -e DC_BACKEND=ascend -e DC_ROOT="$WC/dev/device-context/prototype" \
  "$C" bash -lc "cd $WC/dev/device-context/prototype && python3 -u scripts/backend_offline_check.py --backend ascend"
grep -aE "通过|失败|跳过|总计" "$O/offline.log" | tail -2 | sed 's/^/    /'

say ""
say "=== [2] smoke（容器内卡 0）==="
run_in smoke -e ASCEND_RT_VISIBLE_DEVICES="$DEV" -e CUDA_VISIBLE_DEVICES="$DEV" \
  -e DC_BACKEND=ascend -e DC_ROOT="$WC/dev/device-context/prototype" \
  "$C" bash -lc "cd $WC/dev/device-context/prototype && python3 -u runtime/smoke_runtime.py --backend ascend"
grep -aE "结果:|通过 /|失败" "$O/smoke.log" | tail -2 | sed 's/^/    /'

say ""
say "=== [3] 契约不变式 I1–I4（关键看**退出码**：PPU 上曾是 rc=139）==="
run_in invariants -e ASCEND_RT_VISIBLE_DEVICES="$DEV" -e CUDA_VISIBLE_DEVICES="$DEV" \
  -e DC_BACKEND=ascend -e DC_ROOT="$WC/dev/device-context/prototype" -e PYTHONFAULTHANDLER=1 \
  "$C" bash -lc "cd $WC/dev/device-context/prototype && python3 -X faulthandler -u runtime/conformance/runner.py --backend ascend --cases contract_invariants --out $WC/out/ci.json"
grep -aE "汇总:|结论:|CONTRACT_INVARIANTS|Fatal Python" "$O/invariants.log" | tail -3 | sed 's/^/    /'

say ""
say "=== [4] 职责审计 78 项（C13 修正后应 OK，且不得负值）==="
run_in duty -e ASCEND_RT_VISIBLE_DEVICES="$DEV" -e CUDA_VISIBLE_DEVICES="$DEV" \
  -e DC_BACKEND=ascend -e DC_ROOT="$WC/dev/device-context/prototype" \
  "$C" bash -lc "cd $WC/dev/device-context/prototype && python3 -u scripts/duty_response_audit.py --backend ascend --out $WC/out/duty.json"
grep -aE "DUTY_RESPONSE|verdict" "$O/duty.log" | tail -2 | sed 's/^/    /'
echo "  C13 行："; grep -a -A1 "C13" "$O/duty.log" | tail -2 | sed 's/^/    /'

say ""
say "=== [5] 推理腿：不传 DC_MODEL（验证自动探测；910C 原路径 /mnt/raid 已因 RAID 重建变化）==="
run_in infer_leg -e ASCEND_RT_VISIBLE_DEVICES="$DEV" -e CUDA_VISIBLE_DEVICES="$DEV" \
  -e DC_BACKEND=ascend -e DC_ROOT="$WC/dev/device-context/prototype" -e DC_MODEL="$MODEL" \
  "$C" bash -lc "cd $WC/dev/device-context/prototype && python3 -u runtime/proto/proto_infer_leg.py --backend ascend --out $WC/out/infer_leg.json"
grep -aE "INFER_LEG|区分度|model_source|句/s" "$O/infer_leg.log" | tail -3 | sed 's/^/    /'
python3 - "$O/infer_leg.json" <<'PY' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print("    model_source =", d.get("model_source"))
    print("    model        =", d.get("model"))
except Exception as e:
    print("    (json 读取失败:", type(e).__name__, ")")
PY

say ""
say "=== [6] 训练腿 50 步（2 卡，容器内 0/1）==="
run_in train_leg -e ASCEND_RT_VISIBLE_DEVICES=0,1 -e CUDA_VISIBLE_DEVICES=0,1 \
  -e DC_BACKEND=ascend -e DC_ROOT="$WC/dev/device-context/prototype" \
  -e DC_MODEL="$MODEL" -e DC_OUT_DIR="$WC/out" \
  -e MAX_STEPS=50 -e BATCH=4 -e SEQ=128 \
  "$C" bash -lc "cd $WC/dev/device-context/prototype && python3 -m torch.distributed.run --standalone --nproc_per_node=2 runtime/proto/proto_train_leg.py --backend ascend"
grep -aE "TRAIN_LEG|loss|tok/s|overall" "$O/train_leg.log" | tail -4 | sed 's/^/    /'

say ""
say "=== 汇总 ==="
for f in offline smoke invariants duty infer_leg train_leg; do
  printf "  %-12s " "$f"
  grep -aoE "TRAIN_LEG_PASS|TRAIN_LEG_FAIL|INFER_LEG_PASS|INFER_LEG_FAIL|CONTRACT_INVARIANTS_PASS|CONTRACT_INVARIANTS_FAIL|DUTY_RESPONSE_PASS|DUTY_RESPONSE_FAIL|离线自检结果|PASS|FAIL" "$O/$f.log" 2>/dev/null | tail -1 || echo "(无 verdict)"
done
say ""
say "证据目录（宿主）: $O"
