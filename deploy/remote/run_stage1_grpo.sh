#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${JANUS_ROOT_DIR:-/workspace/Janus_pro_finetune}"
SECRET_ENV="${JANUS_SECRET_ENV:-/dev/shm/janus-grpo-secret.env}"
if [[ ! -s "${SECRET_ENV}" ]]; then
  echo "Missing runtime-only judge credential: ${SECRET_ENV}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${SECRET_ENV}"

export JANUS_VENV="${ROOT_DIR}/.venv"
export HF_HOME="${ROOT_DIR}/.hf_home"
export HF_XET_CACHE="${HF_HOME}/xet"
export TMPDIR="${ROOT_DIR}/.tmp"
export JANUS_CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export JANUS_NPROC_PER_NODE="${JANUS_NPROC_PER_NODE:-4}"
export JANUS_STAGE_MODEL_TO_RAM="${JANUS_STAGE_MODEL_TO_RAM:-0}"
export JANUS_STAGE1_SFT_MODEL="${JANUS_STAGE1_SFT_MODEL:-${ROOT_DIR}/models/Janus-Pro-7B-stage1-sft}"
export JANUS_STAGE1_GRPO_DATA="${JANUS_STAGE1_GRPO_DATA:-${ROOT_DIR}/data/processed/tqa/train_prompt_model_difficulty.remote.jsonl}"
export JANUS_STAGE1_GRPO_OUTPUT="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/stage1/tqa_grpo_blackwell}"

# Four 96 GiB cards can use the paper-shaped two-prompt micro-batch. There are
# 8 prompts and 128 completions per optimizer batch (16 generations/prompt).
export JANUS_GRPO_PER_DEVICE_BATCH="${JANUS_GRPO_PER_DEVICE_BATCH:-2}"
export JANUS_GRPO_GRAD_ACCUM="${JANUS_GRPO_GRAD_ACCUM:-16}"
export JANUS_GRPO_GENERATION_BATCH="${JANUS_GRPO_GENERATION_BATCH:-128}"
export JANUS_GRPO_STEPS_PER_GENERATION="${JANUS_GRPO_STEPS_PER_GENERATION:-16}"
export JANUS_LOCAL_ROLLOUT_FORWARD_BATCH_SIZE="${JANUS_LOCAL_ROLLOUT_FORWARD_BATCH_SIZE:-2}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
export JANUS_REASONING_JUDGE_MODEL="${JANUS_REASONING_JUDGE_MODEL:-deepseek-v4-flash-vision-exp}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
mkdir -p "${TMPDIR}" "${JANUS_STAGE1_GRPO_OUTPUT}"

exec bash "${ROOT_DIR}/scripts/run_stage1_grpo.sh" "$@"
