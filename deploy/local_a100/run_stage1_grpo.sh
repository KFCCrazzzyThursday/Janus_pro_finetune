#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OFFLINE=0
RUN_MODE="formal"
if [[ "${1:-}" == "--offline" ]]; then
  OFFLINE=1
  RUN_MODE="offline"
  shift
elif [[ "${1:-}" == "--memory-smoke" ]]; then
  RUN_MODE="memory-smoke"
elif [[ "${1:-}" == "--smoke" ]]; then
  RUN_MODE="smoke"
fi

export JANUS_VENV="${JANUS_VENV:-${ROOT_DIR}/.venv}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${XDG_CACHE_HOME}/modelscope}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"

export JANUS_CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export JANUS_NPROC_PER_NODE="${JANUS_NPROC_PER_NODE:-8}"
export JANUS_GRPO_BACKEND="${JANUS_GRPO_BACKEND:-ddp}"
export JANUS_TUNER_TYPE="${JANUS_TUNER_TYPE:-lora}"
export JANUS_LORA_RANK="${JANUS_LORA_RANK:-32}"
export JANUS_LORA_ALPHA="${JANUS_LORA_ALPHA:-64}"
export JANUS_LORA_DROPOUT="${JANUS_LORA_DROPOUT:-0.05}"
export JANUS_GRPO_LEARNING_RATE="${JANUS_GRPO_LEARNING_RATE:-1e-5}"
export JANUS_OPTIM="${JANUS_OPTIM:-adamw_torch}"
# The A100 experiment uses the stabilized variance curriculum documented in
# docs/reward_variance_weighting_revision.md.  Set this to "paper" for an exact
# equations-(3.9)-(3.10) ablation.
export JANUS_REWARD_WEIGHTING="${JANUS_REWARD_WEIGHTING:-stabilized}"
export JANUS_REWARD_VARIANCE_MIX="${JANUS_REWARD_VARIANCE_MIX:-0.5}"
export JANUS_STAGE_MODEL_TO_RAM="${JANUS_STAGE_MODEL_TO_RAM:-0}"
export JANUS_KEEP_HTTP_PROXY="${JANUS_KEEP_HTTP_PROXY:-1}"
export JANUS_STAGE1_SFT_MODEL="${JANUS_STAGE1_SFT_MODEL:-${ROOT_DIR}/models/Janus-Pro-7B-stage1-sft}"
export JANUS_STAGE1_GRPO_DATA="${JANUS_STAGE1_GRPO_DATA:-${ROOT_DIR}/data/processed/tqa/train_prompt_model_difficulty.jsonl}"

# Preserve the paper's eight prompts / 128 completions per optimizer update.
# Eight 40 GiB ranks use one prompt per device and sixteen accumulation steps.
export JANUS_GRPO_PER_DEVICE_BATCH="${JANUS_GRPO_PER_DEVICE_BATCH:-1}"
export JANUS_GRPO_GRAD_ACCUM="${JANUS_GRPO_GRAD_ACCUM:-16}"
export JANUS_GRPO_GENERATION_BATCH="${JANUS_GRPO_GENERATION_BATCH:-128}"
export JANUS_GRPO_STEPS_PER_GENERATION="${JANUS_GRPO_STEPS_PER_GENERATION:-16}"
export JANUS_LOCAL_ROLLOUT_FORWARD_BATCH_SIZE="${JANUS_LOCAL_ROLLOUT_FORWARD_BATCH_SIZE:-4}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-7}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-7}"
# Each rank owns a full CPU-side checkpoint before FSDP2 shards it.  Forking
# dataset/DataLoader workers at that point multiplies the committed address
# space and exhausts this 251 GiB host before the first GPU step.
export JANUS_GRPO_DATASET_NUM_PROC="${JANUS_GRPO_DATASET_NUM_PROC:-1}"
export JANUS_GRPO_DATALOADER_WORKERS="${JANUS_GRPO_DATALOADER_WORKERS:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

case "${RUN_MODE}" in
  offline)
    export JANUS_OFFLINE_JUDGE_STUB=1
    DEFAULT_OUTPUT="${ROOT_DIR}/outputs/stage1/tqa_grpo_a100x8_offline"
    ;;
  memory-smoke)
    DEFAULT_OUTPUT="${ROOT_DIR}/outputs/smoke/stage1_tqa_grpo_a100x8_full_shape"
    ;;
  smoke)
    DEFAULT_OUTPUT="${ROOT_DIR}/outputs/smoke/stage1_tqa_grpo_a100x8"
    ;;
  formal)
    DEFAULT_OUTPUT="${ROOT_DIR}/outputs/stage1/tqa_grpo_lora_ddp_a100x8"
    ;;
esac
export JANUS_STAGE1_GRPO_OUTPUT="${JANUS_STAGE1_GRPO_OUTPUT:-${DEFAULT_OUTPUT}}"

mkdir -p "${TMPDIR}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${JANUS_STAGE1_GRPO_OUTPUT}"
exec bash "${ROOT_DIR}/scripts/run_stage1_grpo.sh" "$@"
