#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${RAG_PYTHON_BIN:-/usr/local/python3.11.15/bin/python3.11}"
venv_dir="${RAG_VENV_DIR:-${project_dir}/.venv}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python 3.11 was not found at ${python_bin}" >&2
  echo "Set RAG_PYTHON_BIN to the Python 3.11 executable in the NPU container." >&2
  exit 1
fi

"${python_bin}" -m venv --system-site-packages "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install --editable "${project_dir}[test]"

echo "RAG environment ready: ${venv_dir}"
echo "Activate with: source ${venv_dir}/bin/activate"
