#!/usr/bin/env bash
# Same memory policy/criteria as Milvus; the API is explicitly switched to ES.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_measured_acceptance.sh" --backend elasticsearch "$@"
