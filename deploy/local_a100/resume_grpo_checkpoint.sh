#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SNAPSHOT_ROOT="${1:?Usage: resume_grpo_checkpoint.sh HUGGINGFACE_SNAPSHOT [extra trainer args...]}"
shift
MANIFEST="${SNAPSHOT_ROOT}/metadata/manifest.json"

if [[ ! -s "${MANIFEST}" ]]; then
  echo "Missing resume manifest: ${MANIFEST}" >&2
  exit 2
fi

read_manifest() {
  "${ROOT_DIR}/.venv/bin/python" -c \
    'import json,sys; value=json.load(open(sys.argv[1]));
for key in sys.argv[2].split("."): value=value[key]
print(value)' \
    "${MANIFEST}" "$1"
}

EXPECTED_CODE_COMMIT="$(read_manifest code.commit)"
EXPECTED_SWIFT_COMMIT="$(read_manifest code.ms_swift_commit)"
EXPECTED_JANUS_COMMIT="$(read_manifest code.deepseek_janus_commit)"
CHECKPOINT_NAME="$(read_manifest checkpoint)"
MAX_STEPS="$(read_manifest max_steps)"
WORLD_SIZE="$(read_manifest distributed.world_size)"
NUM_GENERATIONS="$(read_manifest grpo.num_generations)"
GENERATION_BATCH_SIZE="$(read_manifest grpo.generation_batch_size)"
STEPS_PER_GENERATION="$(read_manifest grpo.steps_per_generation)"
GRADIENT_ACCUMULATION_STEPS="$(read_manifest grpo.gradient_accumulation_steps)"
REWARD_WEIGHTING="$(read_manifest reward_weighting.mode)"
REWARD_VARIANCE_MIX="$(read_manifest reward_weighting.variance_mix)"
OFFLINE_REASONING_STUB="$(read_manifest emergency_offline_reasoning_stub)"

MICRO_BATCH_DENOMINATOR=$((WORLD_SIZE * STEPS_PER_GENERATION))
if ((GENERATION_BATCH_SIZE % MICRO_BATCH_DENOMINATOR != 0)); then
  echo "Manifest has an invalid generation batch/world-size/step combination" >&2
  exit 2
fi
PER_DEVICE_BATCH_SIZE=$((GENERATION_BATCH_SIZE / MICRO_BATCH_DENOMINATOR))

project_training_tree_unchanged() {
  local expected="$1"
  local actual="$2"
  git -C "${ROOT_DIR}" cat-file -e "${expected}^{commit}" 2>/dev/null &&
    git -C "${ROOT_DIR}" merge-base --is-ancestor "${expected}" "${actual}" &&
    git -C "${ROOT_DIR}" diff --quiet "${expected}..${actual}" -- \
      configs \
      patches \
      src \
      training \
      scripts/run_stage1_grpo.sh \
      deploy/local_a100/run_stage1_grpo.sh
}

require_commit() {
  local repository="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(git -C "${repository}" rev-parse HEAD)"
  if [[ "${actual}" != "${expected}" ]]; then
    if [[ "${label}" == project ]] && project_training_tree_unchanged "${expected}" "${actual}"; then
      echo "Project HEAD is a training-compatible descendant of ${expected}: ${actual}"
    elif [[ "${JANUS_ALLOW_CODE_MISMATCH:-0}" != "1" ]]; then
      echo "${label} commit mismatch: expected ${expected}, got ${actual}" >&2
      exit 2
    fi
  fi
}

require_commit "${ROOT_DIR}" "${EXPECTED_CODE_COMMIT}" project
require_commit "${ROOT_DIR}/upstream/ms-swift" "${EXPECTED_SWIFT_COMMIT}" ms-swift
require_commit "${ROOT_DIR}/upstream/deepseek-janus" "${EXPECTED_JANUS_COMMIT}" deepseek-janus

SOURCE_DATA="${SNAPSHOT_ROOT}/data/train_prompt_model_difficulty.jsonl"
RESUME_DATA="${SNAPSHOT_ROOT}/data/train_prompt_model_difficulty.relocated.jsonl"
if [[ ! -s "${RESUME_DATA}" ]]; then
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/relocate_jsonl_paths.py" \
    "${SOURCE_DATA}" "${RESUME_DATA}" \
    --from-prefix /data1/home/a100/Janus \
    --to-prefix "${SNAPSHOT_ROOT}" \
    --check-images
fi

CHECKPOINT="${JANUS_RESUME_CHECKPOINT:-${SNAPSHOT_ROOT}/${CHECKPOINT_NAME}}"
if [[ ! -s "${CHECKPOINT}/adapter_model.safetensors" || ! -s "${CHECKPOINT}/optimizer.pt" ]]; then
  echo "Incomplete resume checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

export JANUS_STAGE1_SFT_MODEL="${JANUS_STAGE1_SFT_MODEL:-${SNAPSHOT_ROOT}}"
export JANUS_STAGE1_GRPO_DATA="${JANUS_STAGE1_GRPO_DATA:-${RESUME_DATA}}"
export JANUS_STAGE1_GRPO_OUTPUT="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/stage1/grpo-resumed}"
export JANUS_GRPO_MAX_STEPS="${JANUS_GRPO_MAX_STEPS:-${MAX_STEPS}}"
export JANUS_GRPO_SAVE_STEPS="${JANUS_GRPO_SAVE_STEPS:-5}"
export JANUS_NPROC_PER_NODE="${JANUS_NPROC_PER_NODE:-${WORLD_SIZE}}"
export JANUS_GRPO_NUM_GENERATIONS="${JANUS_GRPO_NUM_GENERATIONS:-${NUM_GENERATIONS}}"
export JANUS_GRPO_PER_DEVICE_BATCH="${JANUS_GRPO_PER_DEVICE_BATCH:-${PER_DEVICE_BATCH_SIZE}}"
export JANUS_GRPO_GENERATION_BATCH="${JANUS_GRPO_GENERATION_BATCH:-${GENERATION_BATCH_SIZE}}"
export JANUS_GRPO_STEPS_PER_GENERATION="${JANUS_GRPO_STEPS_PER_GENERATION:-${STEPS_PER_GENERATION}}"
export JANUS_GRPO_GRAD_ACCUM="${JANUS_GRPO_GRAD_ACCUM:-${GRADIENT_ACCUMULATION_STEPS}}"
export JANUS_REWARD_WEIGHTING="${JANUS_REWARD_WEIGHTING:-${REWARD_WEIGHTING}}"
export JANUS_REWARD_VARIANCE_MIX="${JANUS_REWARD_VARIANCE_MIX:-${REWARD_VARIANCE_MIX}}"

RUN_MODE_ARGS=()
if [[ "${OFFLINE_REASONING_STUB,,}" == true ]]; then
  RUN_MODE_ARGS+=(--offline)
fi

exec bash "${ROOT_DIR}/deploy/local_a100/run_stage1_grpo.sh" \
  "${RUN_MODE_ARGS[@]}" \
  --resume_from_checkpoint "${CHECKPOINT}" \
  "$@"
