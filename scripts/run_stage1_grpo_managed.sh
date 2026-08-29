#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
OUTPUT_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/stage1/tqa_grpo_lora_managed30}"
INTERVAL="${JANUS_GRPO_CHECKPOINT_INTERVAL:-30}"
TOTAL_STEPS="${JANUS_GRPO_MAX_STEPS:-3000}"
WORLD_SIZE="${JANUS_NPROC_PER_NODE:-5}"
STATE_TOOL=("${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/grpo_run_state.py")

if (( INTERVAL <= 0 || TOTAL_STEPS <= 0 || WORLD_SIZE <= 0 )); then
  echo "Checkpoint interval, total steps and world size must be positive." >&2
  exit 2
fi
if (( TOTAL_STEPS % INTERVAL != 0 )); then
  echo "TOTAL_STEPS=${TOTAL_STEPS} must be divisible by INTERVAL=${INTERVAL}." >&2
  exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for managed GRPO training." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
exec 9>"${OUTPUT_DIR}/.managed.lock"
if ! flock -n 9; then
  echo "Another managed GRPO supervisor already owns ${OUTPUT_DIR}." >&2
  exit 3
fi

latest_checkpoint=""
if latest_checkpoint="$("${STATE_TOOL[@]}" latest "${OUTPUT_DIR}" --world-size "${WORLD_SIZE}")"; then
  :
else
  latest_checkpoint=""
fi

if [[ -z "${latest_checkpoint}" ]]; then
  unexpected="$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 \
    ! -name '.managed.lock' ! -name 'launcher.log' -print -quit)"
  if [[ -n "${unexpected}" ]]; then
    echo "No verified checkpoint exists, but ${OUTPUT_DIR} is not empty." >&2
    echo "Refusing to mix a fresh run with old observability data: ${unexpected}" >&2
    exit 3
  fi
  current_step=0
else
  current_step="${latest_checkpoint##*-}"
  "${STATE_TOOL[@]}" prepare-resume "${OUTPUT_DIR}" --world-size "${WORLD_SIZE}" >/dev/null
fi

validation_done() {
  local step=$1
  local history="${OUTPUT_DIR}/validation/history.jsonl"
  [[ -s "${history}" ]] && jq -s -e --argjson step "${step}" \
    'any(.[]; .step == $step)' "${history}" >/dev/null
}

validate_checkpoint() {
  local checkpoint=$1
  local step="${checkpoint##*-}"
  if validation_done "${step}"; then
    echo "Validation already recorded for step ${step}."
    return
  fi
  JANUS_STAGE1_GRPO_OUTPUT="${OUTPUT_DIR}" \
    JANUS_CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,2,3,4}" \
    JANUS_NPROC_PER_NODE="${WORLD_SIZE}" \
    bash "${ROOT_DIR}/scripts/run_stage1_grpo_validation.sh" "${checkpoint}"
}

if [[ -n "${latest_checkpoint}" ]]; then
  validate_checkpoint "${latest_checkpoint}"
fi

while (( current_step < TOTAL_STEPS )); do
  target_step=$((current_step + INTERVAL))
  echo "Managed GRPO segment: ${current_step} -> ${target_step} (final ${TOTAL_STEPS})."

  JANUS_STAGE1_GRPO_OUTPUT="${OUTPUT_DIR}" \
    JANUS_GRPO_AUTO_RESUME=0 \
    JANUS_RESUME_FROM_CHECKPOINT="${latest_checkpoint}" \
    JANUS_GRPO_SAVE_STEPS="${INTERVAL}" \
    JANUS_GRPO_SAVE_TOTAL_LIMIT=2 \
    JANUS_GRPO_MAX_STEPS="${target_step}" \
    JANUS_TRAIN_LOGGING_DIR="${OUTPUT_DIR}/runs/trainer" \
    JANUS_REPORT_TO=tensorboard \
    bash "${ROOT_DIR}/scripts/run_stage1_grpo.sh"

  expected_checkpoint="${OUTPUT_DIR}/checkpoint-${target_step}"
  "${STATE_TOOL[@]}" verify "${expected_checkpoint}" \
    --world-size "${WORLD_SIZE}" --write-manifest >/dev/null
  "${STATE_TOOL[@]}" prepare-resume "${OUTPUT_DIR}" \
    --world-size "${WORLD_SIZE}" >/dev/null
  validate_checkpoint "${expected_checkpoint}"

  latest_checkpoint="${expected_checkpoint}"
  current_step="${target_step}"
done

echo "Managed GRPO completed ${TOTAL_STEPS} optimizer steps."
