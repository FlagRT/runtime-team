#!/usr/bin/env bash
# Host entry. Setup restarts happen only before each independent trial.
set -euo pipefail
backend=milvus
run=0
profiles=2048:32
warmup=1024:32
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) run=1; shift ;;
    --backend) backend="${2:?Missing backend}"; shift 2 ;;
    --profiles) profiles="${2:?Missing profiles}"; shift 2 ;;
    --warmup-profile) warmup="${2:?Missing warmup profile}"; shift 2 ;;
    *) echo 'Usage: run_measured_acceptance.sh [--run] [--backend milvus|elasticsearch] [--profiles 2048:32] [--warmup-profile 1024:32]' >&2; exit 2 ;;
  esac
done
case "$backend" in milvus|elasticsearch) ;; *) echo 'Invalid backend' >&2; exit 2 ;; esac
runner=(docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c .venv/bin/python -u)
if [[ "$run" == 0 ]]; then
  "${runner[@]}" scripts/runtime/restart_memory_api.py --retrieval-backend "$backend"
  exit 0
fi
log_dir=$(mktemp -d /tmp/rag-memory-suite.XXXXXX)
trap 'rm -rf -- "$log_dir"' EXIT
phase_output=''
run_phase() {
  "${runner[@]}" "$@" | tee "$log_dir/phase.log"
  phase_output=$(sed -n 's/^OUTPUT=//p' "$log_dir/phase.log" | tail -n 1)
  [[ "$phase_output" == /workspace/dev/rag_ljy/experiment_outputs/* ]] || { echo 'Missing experiment output path' >&2; return 1; }
}
deployments=()
trials=()
for mode in burst staggered continuous; do
  run_phase scripts/runtime/restart_memory_api.py --run --retrieval-backend "$backend"
  deployments+=("$phase_output")
  run_phase scripts/checks/check_measured_watermarks.py --run --backend "$backend" \
    --modes "$mode" --count 2 --warmup-profile "$warmup" --profiles "$profiles"
  trials+=("$phase_output")
done
run_phase scripts/checks/check_stale_memory_live.py --run --backend "$backend"
"${runner[@]}" scripts/checks/summarize_measured_acceptance.py --backend "$backend" \
  --trials "${trials[@]}" --deployments "${deployments[@]}" --stale "$phase_output"
