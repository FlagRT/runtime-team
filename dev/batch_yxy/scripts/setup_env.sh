#!/usr/bin/env bash
# batch_yxy 开发环境初始化/重建脚本（在容器内以 root 执行，幂等，可重复跑）
# 用法：bash /workspace/dev/batch_yxy/scripts/setup_env.sh
#
# 约定：
#   - venv 放在 /workspace 挂载盘（容器重建后保留），依赖见 requirements.txt
#   - ssh-client / opencode 装在容器内（不在挂载盘，容器重建后重跑本脚本即可）
#   - 本脚本与 requirements.txt 入 git，保证环境可复现
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BATCH_PYTHON_BIN:-/usr/local/python3.12.13/bin/python3}"
venv_dir="${BATCH_VENV_DIR:-${project_dir}/.venv}"

# 1. 基础工具：ssh（git over ssh 需要）
if ! command -v ssh >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq openssh-client
fi

# 2. git 身份（容器内 global 配置；key 通过 compose 挂载宿主 ~/.ssh/id_rsa）
git config --global user.name  "${GIT_USER_NAME:-YoannFang}"
git config --global user.email "${GIT_USER_EMAIL:-yuanc1511@gmail.com}"

# 3. venv（--system-site-packages 复用镜像自带 CANN 相关系统包）
if [[ ! -x "${python_bin}" ]]; then
  echo "未找到 Python：${python_bin}" >&2
  echo "可用 BATCH_PYTHON_BIN 指定，例如 BATCH_PYTHON_BIN=/usr/bin/python3 bash $0" >&2
  exit 1
fi
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_bin}" -m venv --system-site-packages "${venv_dir}"
fi
"${venv_dir}/bin/python" -m pip install --upgrade pip

# 4. Python 依赖
if [[ -f "${project_dir}/requirements.txt" ]]; then
  "${venv_dir}/bin/python" -m pip install -r "${project_dir}/requirements.txt"
fi

# 5. opencode（不在挂载盘，重建后重跑即可）
if [[ ! -x "${HOME}/.opencode/bin/opencode" ]]; then
  curl -fsSL https://opencode.ai/install | bash
fi

echo
echo "batch_yxy 环境就绪："
echo "  venv 激活：source ${venv_dir}/bin/activate"
echo "  opencode：${HOME}/.opencode/bin/opencode"
