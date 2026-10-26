#!/usr/bin/env bash
# Qwen3-1.7B LoRA: full-vocabulary forward KL, batch 8 x 8 rollouts, 500 steps.
# Print the full workflow by default; --execute runs it in a prepared environment.
set -euo pipefail
OPSD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$OPSD_ROOT/opsd_pipeline.sh" run "$@"
