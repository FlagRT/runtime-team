#!/bin/bash
# P800 第 7 轮回归（2026-09-30 · (A) 方案：流优先级统一 API 落地轮）
#
# 破坏面同 910C r9（共享层 + 三家后端 + 入口映射 + 离线判据 + stub）；
# 本机覆盖 kunlun 这一支。`create_stream()` 的全部消费方至少覆盖一次。
# 免跑（理由可复核）：serve_standard.sh 不经过本层。
set -u
PROTO=/workspace/dc_regress_20260929/prototype
OUT=/data2/hliu553/dc_regress_20260929/out_r7        # ⚠️ 宿主路径（本脚本在宿主上跑）
OUT_C=/workspace/dc_regress_20260929/out_r7          # 同一目录的容器内路径
PYENV="source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda && cd $PROTO"
MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
C=hliu553-device-context-p800
# 选卡：dev4（292 MiB / 0%，最空闲）+ dev0（1970 MiB，无进程）；**dev1 为已知故障卡必须避开**
export CUDA_VISIBLE_DEVICES=4,0
mkdir -p "$OUT"
cd "$OUT" || exit 1

echo "===== [0] 跑的是当前版本吗 ====="
grep -c "def _create_stream_raw" $PROTO/runtime/backends/base.py
grep -c "def _create_stream_raw" $PROTO/runtime/backends/kunlun/backend.py
grep -c "cuCtxGetStreamPriorityRange" $PROTO/runtime/backends/kunlun/backend.py
grep -c "\[8b\]" $PROTO/scripts/backend_offline_check.py

step () {
  local name="$1"; shift
  echo "===== $name ====="
  docker exec -e CUDA_VISIBLE_DEVICES=4,0 -e DC_BACKEND=kunlun -e DC_OUT_DIR=$OUT_C \
    "$C" bash -lc "$PYENV && $*" > "$OUT/$name.log" 2>&1
  echo "rc=$?"
}

step r7_offline_kunlun        "python3 scripts/backend_offline_check.py --backend kunlun"
step r7_symmetry_all          "python3 scripts/backend_offline_check.py --all"
step r7_smoke_kunlun          "python3 runtime/smoke_runtime.py --backend kunlun"
step r7_conf13_kunlun         "python3 runtime/conformance/runner.py --backend kunlun --out $OUT_C/r7_conf13_kunlun.json"
step r7_confinfer6_kunlun     "python3 runtime/conformance/runner.py --backend kunlun --cases infer_cases --out $OUT_C/r7_confinfer6_kunlun.json"
step r7_coninvariants_kunlun  "python3 runtime/conformance/runner.py --backend kunlun --cases contract_invariants --out $OUT_C/r7_coninvariants_kunlun.json"
step r7_duty_kunlun           "python3 scripts/duty_response_audit.py --backend kunlun --out $OUT_C/r7_duty_kunlun.json"
step r7_errorloop_kunlun      "python3 runtime/proto/proto_error_recovery_loop.py --backend kunlun --out $OUT_C/r7_errorloop_kunlun.json"
step r7_bc_probe_kunlun       "python3 probes/probe_bc_contract.py --backend kunlun --out $OUT_C/r7_bc_probe_kunlun.json"
step r7_prio_api_kunlun       "python3 probes/probe_stream_priority_api.py --backend kunlun --dev 0 --out $OUT_C/r7_prio_api_kunlun.json"
step r7_entry_verify_kunlun   "python3 probes/recover_entry_verify.py --backend kunlun --dev 0 --out $OUT_C/r7_entry_verify_kunlun.json"
step r7_stream_semantics      "python3 probes/probe_stream_semantics_full.py"
step r7_stream_quota          "DC_QUOTA_N=2000 python3 probes/probe_stream_quota.py"
step r7_demo_unified          "python3 runtime/demos/demo_unified.py"

echo "===== r7_train_leg ====="
docker exec -e CUDA_VISIBLE_DEVICES=4,0 -e DC_ROOT=$PROTO -e DC_BACKEND=kunlun -e DC_MODEL=$MODEL \
  -e DC_OUT_DIR=$OUT_C/train_kunlun -e MAX_STEPS=50 -e BATCH=4 -e SEQ=128 "$C" bash -lc \
  "$PYENV && cd $PROTO/runtime/proto && mkdir -p $OUT_C/train_kunlun && python3 -m torch.distributed.run --standalone --nproc_per_node=2 --master_addr=127.0.0.1 proto_train_leg.py" \
  > "$OUT/r7_train_leg.log" 2>&1
echo "rc=$?"

echo "===== r7_infer_leg ====="
docker exec -e CUDA_VISIBLE_DEVICES=4 -e DC_ROOT=$PROTO -e DC_BACKEND=kunlun -e DC_MODEL=$MODEL \
  -e DC_OUT_DIR=$OUT_C -e DC_ROUNDS=5 "$C" bash -lc \
  "$PYENV && python3 runtime/proto/proto_infer_leg.py" > "$OUT/r7_infer_leg.log" 2>&1
echo "rc=$?"

echo "===== ALL DONE ====="
