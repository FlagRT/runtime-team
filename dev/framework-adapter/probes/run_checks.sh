#!/usr/bin/env bash
# Explicit entry point. No installation, SSH, or container lifecycle actions.
set -euo pipefail
probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-help}
if (( $# > 0 )); then shift; fi

case "$mode" in
  help|-h|--help)
    printf '%s\n' \
      'Usage: bash probes/run_checks.sh MODE [arguments]' \
      '  local            Syntax only; no torch import, network, or device access' \
      '  metadata-tests   Model-probe control tests; no torch or device needed' \
      '  model-baseline   Explicit --model --device --dtype --out; no automatic fallback' \
      '  qwen-controls    CPU scoped-adapter tests; requires torch/transformers/pytest' \
      '  qwen-gems        NPU scoped replacement; explicit --model --runtime-root --out' \
      '  rms-shadow       Same-input RMSNorm observations; not model replacement' \
      '  rms-ablation     Experimental Q/K vs hidden RMSNorm replacement' \
      '  legacy-pytorch   Existing personal 910C container; NOT monthly acceptance' \
      '  legacy-vllm      Existing personal 910C container; NOT monthly acceptance' \
      '  cross-vendor     Container-side probe; pass --device and --mode explicitly'
    ;;
  local)
    if (( $# != 0 )); then echo 'local takes no arguments' >&2; exit 2; fi
    python3 - "$probe_dir" <<'PY'
import ast
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
python_files = sorted(root.glob('*.py'))
shell_files = sorted(root.glob('*.sh'))
for path in python_files:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
for path in shell_files:
    subprocess.run(['bash', '-n', str(path)], check=True)
print(f'Syntax PASS: {len(python_files)} Python, {len(shell_files)} shell files.')
print('No functional, model, or device tests were run.')
PY
    ;;
  legacy-pytorch|legacy-vllm)
    if (( $# != 0 )); then echo "$mode takes no arguments" >&2; exit 2; fi
    echo 'Historical environment only. Confirm free resources and your running container first.' >&2
    if [[ "$mode" == legacy-pytorch ]]; then
      exec bash "$probe_dir/run_pytorch_eager_checks.sh"
    else
      exec bash "$probe_dir/run_910c_checks.sh"
    fi
    ;;
  metadata-tests)
    if (( $# != 0 )); then echo 'metadata-tests takes no arguments' >&2; exit 2; fi
    exec python3 -m unittest discover -s "$probe_dir" -p 'test_qwen_probe_metadata.py' -v
    ;;
  model-baseline)
    exec python3 "$probe_dir/qwen_embedding_baseline.py" "$@"
    ;;
  qwen-controls)
    exec python3 -m pytest -q "$probe_dir/test_qwen_scoped_adapter.py" "$@"
    ;;
  qwen-gems)
    exec python3 "$probe_dir/qwen_gems_validation.py" "$@"
    ;;
  rms-shadow)
    exec python3 "$probe_dir/qwen_rms_shadow.py" "$@"
    ;;
  rms-ablation)
    exec python3 "$probe_dir/qwen_rms_ablation.py" "$@"
    ;;
  cross-vendor)
    exec python3 "$probe_dir/cross_vendor_smoke.py" "$@"
    ;;
  *) echo "Unknown mode: $mode (use --help)" >&2; exit 2 ;;
esac
