#!/usr/bin/env bash
# A budget-limited, two-teacher Section 3.1 pilot. Models must already be local.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${PROJECT_ROOT}/verl:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SWANLAB_MODE=offline
export WANDB_MODE=disabled
exec "${PYTHON:-python3}" "${PROJECT_ROOT}/scripts/pilot/thinking_consistency.py" "$@"
