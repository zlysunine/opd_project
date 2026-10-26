#!/usr/bin/env bash
set -euo pipefail
OPSD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$OPSD_ROOT"
export PYTHONPATH="$OPSD_ROOT:$OPSD_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "${OPSD_PYTHON:-python}" -m scripts.research.pipeline "$@"
