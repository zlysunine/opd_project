#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/verl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Default config: 200 OPSD updates, online metrics and checkpoint representation analysis.
# GRPO remains available explicitly with --method grpo (or --method both).
exec python -m scripts.research.compare run --method opsd "$@"
