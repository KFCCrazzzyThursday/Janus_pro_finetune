#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${JANUS_ROOT_DIR:-/home/user/Janus_pro_finetune}"
export JANUS_VENV="${JANUS_VENV:-${ROOT_DIR}/.venv}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${XDG_CACHE_HOME}/modelscope}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"
# PyTorch 2.6 defaults torch.load() to weights_only=True. Transformers loads
# RNG state without an explicit mode, while checkpoint-270 contains the normal
# NumPy/Python RNG tuple produced by the older training host. The transferred
# checkpoint is hash-verified and trusted, so permit that full RNG-state load
# to preserve exact continuation instead of silently reseeding the run.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

export JANUS_CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1}"
export JANUS_NPROC_PER_NODE="${JANUS_NPROC_PER_NODE:-2}"
export JANUS_NUMA_AFFINITY="${JANUS_NUMA_AFFINITY:-0}"
export JANUS_STAGE_MODEL_TO_RAM="${JANUS_STAGE_MODEL_TO_RAM:-0}"
export JANUS_MAX_RAM_USED_GIB="${JANUS_MAX_RAM_USED_GIB:-256}"
export JANUS_MAX_SWAP_USED_GIB="${JANUS_MAX_SWAP_USED_GIB:-0.25}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-8}"

export JANUS_STAGE1_SFT_MODEL="${JANUS_STAGE1_SFT_MODEL:-${ROOT_DIR}/models/Janus-Pro-7B-stage1-sft}"
export JANUS_STAGE1_GRPO_DATA="${JANUS_STAGE1_GRPO_DATA:-${ROOT_DIR}/data/processed/tqa/train_prompt_model_difficulty.remote.jsonl}"
export JANUS_GRPO_VAL_DATASET="${JANUS_GRPO_VAL_DATASET:-${ROOT_DIR}/data/processed/tqa/val_prompt.remote.jsonl}"
export JANUS_STAGE1_GRPO_OUTPUT="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/stage1/tqa_grpo_accfmt_a100_from270_managed30}"

# This continuation phase intentionally makes strict format 75% of the scalar
# reward coefficients. Accuracy spans [-1, 1], while format spans [0, 1], so
# their maximum weighted swings are 0.50 and 0.75 respectively.
export JANUS_GRPO_REWARD_FUNCS="janus_accuracy janus_format"
export JANUS_GRPO_REWARD_WEIGHTS="${JANUS_GRPO_REWARD_WEIGHTS:-0.25 0.75}"

# Two A100s retain the paper-shaped global rollout: 8 prompts x G=16 = 128
# completions. Rollout forward chunks stay small, so this changes duration but
# not the peak activation shape used by one forward pass.
export JANUS_GRPO_NUM_GENERATIONS="${JANUS_GRPO_NUM_GENERATIONS:-16}"
export JANUS_GRPO_PER_DEVICE_BATCH="${JANUS_GRPO_PER_DEVICE_BATCH:-1}"
export JANUS_GRPO_GRAD_ACCUM="${JANUS_GRPO_GRAD_ACCUM:-64}"
export JANUS_GRPO_GENERATION_BATCH="${JANUS_GRPO_GENERATION_BATCH:-128}"
export JANUS_GRPO_STEPS_PER_GENERATION="${JANUS_GRPO_STEPS_PER_GENERATION:-64}"
export JANUS_LOCAL_ROLLOUT_FORWARD_BATCH_SIZE="${JANUS_LOCAL_ROLLOUT_FORWARD_BATCH_SIZE:-4}"
export JANUS_GRPO_DATASET_NUM_PROC="${JANUS_GRPO_DATASET_NUM_PROC:-4}"
export JANUS_GRPO_DATALOADER_WORKERS="${JANUS_GRPO_DATALOADER_WORKERS:-2}"

export JANUS_GRPO_CHECKPOINT_INTERVAL="${JANUS_GRPO_CHECKPOINT_INTERVAL:-30}"
export JANUS_GRPO_SAVE_STEPS="${JANUS_GRPO_SAVE_STEPS:-30}"
export JANUS_GRPO_SAVE_TOTAL_LIMIT="${JANUS_GRPO_SAVE_TOTAL_LIMIT:-2}"
export JANUS_GRPO_MAX_STEPS="${JANUS_GRPO_MAX_STEPS:-3000}"
export JANUS_GRPO_SEGMENT_MAX_RETRIES="${JANUS_GRPO_SEGMENT_MAX_RETRIES:-10}"
export JANUS_GRPO_VALIDATION_MAX_RETRIES="${JANUS_GRPO_VALIDATION_MAX_RETRIES:-3}"
export JANUS_GRPO_RETRY_BASE_SECONDS="${JANUS_GRPO_RETRY_BASE_SECONDS:-30}"
export JANUS_GRPO_RETRY_MAX_SECONDS="${JANUS_GRPO_RETRY_MAX_SECONDS:-300}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MODELSCOPE_CACHE}" \
  "${TORCH_HOME}" "${JANUS_STAGE1_GRPO_OUTPUT}"

if [[ "${1:-}" == "--managed" ]]; then
  shift
  exec bash "${ROOT_DIR}/scripts/run_stage1_grpo_managed.sh" "$@"
fi
exec bash "${ROOT_DIR}/scripts/run_stage1_grpo.sh" "$@"
