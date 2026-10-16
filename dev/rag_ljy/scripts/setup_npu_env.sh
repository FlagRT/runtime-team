#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${RAG_PYTHON_BIN:-$(command -v python3)}"
venv_dir="${RAG_VENV_DIR:-${project_dir}/.venv}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python was not found at ${python_bin}" >&2
  echo "Set RAG_PYTHON_BIN to the image's Python executable with torch_npu." >&2
  exit 1
fi

# Resolve the image interpreter even when the existing RAG venv is activated.
python_bin="$("${python_bin}" -I -c 'import sys; print(sys._base_executable)')"
"${python_bin}" -I -c 'import sys, torch, torch_npu; assert sys.version_info >= (3, 11), "Python >=3.11 is required"; print("Using image PyTorch:", torch.__version__, "torch_npu:", torch_npu.__version__)'

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
        if name in {"torch", "torch-npu", "transformers", "huggingface-hub"}:
            raise
PY

"${python_bin}" -m venv --system-site-packages "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install --constraint "${constraints_file}" --editable "${project_dir}[test,serve]"
"${venv_dir}/bin/python" -m pip check

echo "RAG environment ready: ${venv_dir}"
echo "Activate with: source ${venv_dir}/bin/activate"
