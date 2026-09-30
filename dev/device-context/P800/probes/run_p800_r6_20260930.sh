#!/bin/bash
# P800 第 6 轮全套复跑（2026-09-30）—— 共享层改动后的覆盖
# 破坏面：752551b（A2 三处缺陷修复）· 2647370（补公开入口）· 57cb9dd（info 收口 + 2 条判据）· 本轮（B2 探针）
# 全部这些改动**从未在 P800 上跑过**（P800 最后全套停在 1f26633 轮 r5）
set -u
C=hliu553-device-context-p800
PROTO=/workspace/dc_regress_20260929/prototype
OUT=/data2/hliu553/dc_regress_20260929/out_r6           # ⚠️ 宿主路径（本脚本在宿主上跑）
OUT_C=/workspace/dc_regress_20260929/out_r6             # 同一目录的容器内路径
# 选卡：dev4（292MiB / 0%，最空闲）；**dev1 为已知故障卡（UUID b3509946）必须避开**
export CUDA_VISIBLE_DEVICES=4
PYENV="source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda && cd $PROTO"
mkdir -p "$OUT"

echo "===== [0] 跑的是当前版本吗 ====="
docker exec "$C" bash -lc "$PYENV && \
  echo -n 'set_device_state: '; grep -c 'def set_device_state' runtime/backends/base.py; \
  echo -n 'handle_error: ';    grep -c 'def handle_error'    runtime/backends/base.py; \
  echo -n 'info device_type: '; grep -c '\"device_type\": self.device_type,' runtime/backends/kunlun/backend.py; \
  echo -n 'base_info_fields: '; grep -c 'self._base_info_fields()' runtime/backends/kunlun/backend.py; \
  echo -n 'judge min-keys: ';  grep -c 'info() 含契约最小键集' scripts/backend_offline_check.py; \
  echo -n 'probes: ';          ls probes/ | wc -l"

step () {
  local name="$1"; shift
  local cmd="$*"
  echo "===== $name ====="
  docker exec "$C" bash -lc "$PYENV && $cmd" > "$OUT/$name.log" 2>&1
  echo "rc=$?"
}

step r6_offline_kunlun      "python3 scripts/backend_offline_check.py --backend kunlun"
step r6_symmetry_all        "python3 scripts/backend_offline_check.py --all"
step r6_smoke_kunlun        "python3 runtime/smoke_runtime.py --backend kunlun"
step r6_conf13_kunlun       "python3 runtime/conformance/runner.py --backend kunlun --out $OUT_C/r6_conf13_kunlun.json"
step r6_confinfer6_kunlun   "python3 runtime/conformance/runner.py --backend kunlun --cases infer_cases --out $OUT_C/r6_confinfer6_kunlun.json"
step r6_coninvariants_kunlun "python3 runtime/conformance/runner.py --backend kunlun --cases contract_invariants --out $OUT_C/r6_coninvariants_kunlun.json"
step r6_duty_kunlun         "python3 scripts/duty_response_audit.py --backend kunlun --out $OUT_C/r6_duty_kunlun.json"
step r6_errorloop_kunlun    "python3 runtime/proto/proto_error_recovery_loop.py --backend kunlun --out $OUT_C/r6_errorloop_kunlun.json"
step r6_bc_probe_kunlun     "python3 probes/probe_bc_contract.py --backend kunlun --out $OUT_C/r6_bc_probe_kunlun.json"
step r6_entry_verify_kunlun "python3 probes/recover_entry_verify.py --backend kunlun --dev 0 --out $OUT_C/r6_entry_verify_kunlun.json"
step r6_stress_kunlun       "python3 probes/recover_multiproc_stress.py --backend kunlun --devices 0 --rounds 2 --out $OUT_C/stress_out --max-seconds 300"
echo "===== ALL DONE ====="
