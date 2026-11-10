# Source this file in bash; it does not start training.
source /root/autodl-tmp/envs/sft_grpo/bin/activate
cd /root/autodl-tmp/opd_project/Rethinking-OPD-main
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export HF_HOME=/root/autodl-tmp/sft_grpo_cache/huggingface
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/sft_grpo_cache/torch_extensions
export TRITON_CACHE_DIR=/root/autodl-tmp/sft_grpo_cache/triton
export RAY_TMPDIR=/tmp/sft_grpo_ray
export MPLCONFIGDIR=/root/autodl-tmp/sft_grpo_cache/matplotlib
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
mkdir -p "$HF_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$RAY_TMPDIR" "$MPLCONFIGDIR"
