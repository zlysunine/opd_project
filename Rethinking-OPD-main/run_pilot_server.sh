#!/usr/bin/env bash
# Online server entry: prepare an isolated environment, fetch weights, then run.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-python3}
exec "$BOOTSTRAP_PYTHON" "${PROJECT_ROOT}/scripts/pilot/server_bootstrap.py" "$@"
