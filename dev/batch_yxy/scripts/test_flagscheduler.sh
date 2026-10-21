#!/usr/bin/env bash
# Run in an existing test environment; never install/modify external packages.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_python="${FLAGSCHEDULER_TEST_PYTHON:-python}"
mkdir -p "${project_dir}/.tmp"
export TMPDIR="${project_dir}/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "${project_dir}"
exec "${test_python}" -B -m pytest tests/ -q -p no:cacheprovider \
  --basetemp="${project_dir}/.tmp/pytest" "$@"
