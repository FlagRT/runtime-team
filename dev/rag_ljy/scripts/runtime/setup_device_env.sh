#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${project_dir}/../device-context/prototype${PYTHONPATH:+:$PYTHONPATH}"
export DC_BACKEND="${DC_BACKEND:-ascend}"

python_bin="${RAG_PYTHON_BIN:-$(command -v python3)}"
venv_dir="${RAG_VENV_DIR:-${project_dir}/.venv}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python was not found at ${python_bin}" >&2
  echo "Set RAG_PYTHON_BIN to the selected vendor image's Python executable." >&2
  exit 1
fi

# Resolve the image interpreter even when the existing RAG venv is activated.
python_bin="$("${python_bin}" -I -c 'import sys; print(sys._base_executable)')"
"${python_bin}" -c 'import sys; assert sys.version_info >= (3, 11), "RAG requires Python >=3.11; use a compatible vendor environment"'
"${python_bin}" - <<'PYTHON'
import os
import runtime

backend = runtime.use(os.environ["DC_BACKEND"])
# Load the vendor stack without allocating model weights.
torch = backend.torch
print("Using backend:", backend.name, "device type:", backend.device_type, "PyTorch:", torch.__version__)
PYTHON

# Keep the image's inference stack compatible. These constraints also repair
# older RAG venvs that shadow the image with downgraded Transformers/Hub wheels.
constraints_file="$(mktemp)"
trap 'rm -f "$constraints_file"' EXIT
"${python_bin}" -I - > "${constraints_file}" <<'PY'
from importlib.metadata import PackageNotFoundError, version

for name in (
    "torch", "torch-npu", "transformers", "huggingface-hub", "tokenizers",
    "vllm", "vllm-ascend",
):
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        if name in {"torch", "transformers", "huggingface-hub"}:
            raise
PY

"${python_bin}" -m venv --system-site-packages "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install --constraint "${constraints_file}" --editable "${project_dir}[test,serve]"
"${venv_dir}/bin/python" -m pip check

echo "RAG environment ready: ${venv_dir}"
echo "Activate with: source ${venv_dir}/bin/activate"
echo "For existing containers, add ${project_dir}/../device-context/prototype to PYTHONPATH in your calling shell."
echo "Select the same backend when launching RAG: DC_BACKEND=${DC_BACKEND}"
