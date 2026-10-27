#!/usr/bin/env bash
# Compatibility entry point; defaults to Ascend via the shared setup script.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_device_env.sh" "$@"
