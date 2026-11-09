#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/verl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DEFAULT_WORKDIR="research_runs/controlled_deepmath20"
case "${1:-}" in
  status|logs|stop) stage="$1"; shift; exec python -m scripts.research.job_control "$stage" --workdir "$DEFAULT_WORKDIR" "$@" ;;
  prepare|report) stage="$1"; shift; exec python -m scripts.research.compare "$stage" --workdir "$DEFAULT_WORKDIR" "$@" ;;
  configure) shift; exec python -m scripts.research.compare configure --workdir "$DEFAULT_WORKDIR" --config configs/controlled_deepmath20.json "$@" ;;
  train|analyze) stage="$1"; shift; exec python -m scripts.research.compare "$stage" --workdir "$DEFAULT_WORKDIR" --method all "$@" ;;
esac
exec python -m scripts.research.compare run --workdir "$DEFAULT_WORKDIR" --config configs/controlled_deepmath20.json --method all "$@"
