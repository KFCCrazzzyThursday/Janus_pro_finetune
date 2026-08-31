#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
OUTPUT_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/stage1/tqa_grpo_lora_managed30}"
INTERVAL="${JANUS_GRPO_CHECKPOINT_INTERVAL:-30}"
TOTAL_STEPS="${JANUS_GRPO_MAX_STEPS:-3000}"
WORLD_SIZE="${JANUS_NPROC_PER_NODE:-5}"
SEGMENT_MAX_RETRIES="${JANUS_GRPO_SEGMENT_MAX_RETRIES:-10}"
VALIDATION_MAX_RETRIES="${JANUS_GRPO_VALIDATION_MAX_RETRIES:-3}"
RETRY_BASE_SECONDS="${JANUS_GRPO_RETRY_BASE_SECONDS:-30}"
RETRY_MAX_SECONDS="${JANUS_GRPO_RETRY_MAX_SECONDS:-300}"
STATE_TOOL=("${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/grpo_run_state.py")

if (( INTERVAL <= 0 || TOTAL_STEPS <= 0 || WORLD_SIZE <= 0 \
      || SEGMENT_MAX_RETRIES < 0 || VALIDATION_MAX_RETRIES < 0 \
      || RETRY_BASE_SECONDS < 0 || RETRY_MAX_SECONDS < RETRY_BASE_SECONDS )); then
  echo "Managed interval, steps, world size and retry settings are invalid." >&2
  exit 2
fi
if (( TOTAL_STEPS % INTERVAL != 0 )); then
  echo "TOTAL_STEPS=${TOTAL_STEPS} must be divisible by INTERVAL=${INTERVAL}." >&2
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
  local attempt=0
  local delay
  local status
  if validation_done "${step}"; then
    echo "Validation already recorded for step ${step}."
    return
  fi
  while true; do
    if JANUS_STAGE1_GRPO_OUTPUT="${OUTPUT_DIR}" \
      JANUS_CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,2,3,4}" \
      JANUS_NPROC_PER_NODE="${WORLD_SIZE}" \
      bash "${ROOT_DIR}/scripts/run_stage1_grpo_validation.sh" "${checkpoint}"; then
      return
    else
      status=$?
    fi
    attempt=$((attempt + 1))
    if (( VALIDATION_MAX_RETRIES > 0 && attempt > VALIDATION_MAX_RETRIES )); then
      echo "Validation for step ${step} failed after ${attempt} attempts (status ${status})." >&2
      return "${status}"
    fi
    delay=$((RETRY_BASE_SECONDS * attempt))
    (( delay > RETRY_MAX_SECONDS )) && delay="${RETRY_MAX_SECONDS}"
    echo "Validation for step ${step} failed (status ${status}); retry ${attempt} in ${delay}s." >&2
    sleep "${delay}"
  done
}

quarantine_incomplete_future_checkpoints() {
  local stable_step=$1
  local attempt=$2
  local candidate
  local name
  local step
  local quarantine_dir="${OUTPUT_DIR}/failed_checkpoints"
  while IFS= read -r candidate; do
    name="${candidate##*/}"
    [[ "${name}" =~ ^checkpoint-([0-9]+)$ ]] || continue
    step="${BASH_REMATCH[1]}"
    (( step > stable_step )) || continue
    mkdir -p "${quarantine_dir}"
    mv -- "${candidate}" "${quarantine_dir}/${name}.retry-${attempt}-$(date +%s)"
    echo "Quarantined incomplete checkpoint after failed segment: ${candidate}"
  done < <(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print)
}

if [[ -n "${latest_checkpoint}" ]]; then
  validate_checkpoint "${latest_checkpoint}"
fi

while (( current_step < TOTAL_STEPS )); do
  target_step=$((current_step + INTERVAL))
  echo "Managed GRPO segment: ${current_step} -> ${target_step} (final ${TOTAL_STEPS})."
  segment_attempt=0
  while true; do
    if JANUS_STAGE1_GRPO_OUTPUT="${OUTPUT_DIR}" \
      JANUS_GRPO_AUTO_RESUME=0 \
      JANUS_RESUME_FROM_CHECKPOINT="${latest_checkpoint}" \
      JANUS_GRPO_SAVE_STEPS="${INTERVAL}" \
      JANUS_GRPO_SAVE_TOTAL_LIMIT=2 \
      JANUS_GRPO_MAX_STEPS="${target_step}" \
      JANUS_TRAIN_LOGGING_DIR="${OUTPUT_DIR}/runs/trainer" \
      JANUS_REPORT_TO=tensorboard \
      bash "${ROOT_DIR}/scripts/run_stage1_grpo.sh"; then
      break
    else
      segment_status=$?
    fi

    segment_attempt=$((segment_attempt + 1))
    recovered_checkpoint=""
    if recovered_checkpoint="$("${STATE_TOOL[@]}" latest "${OUTPUT_DIR}" --world-size "${WORLD_SIZE}")"; then
      recovered_step="${recovered_checkpoint##*-}"
    else
      recovered_checkpoint=""
      recovered_step=-1
    fi
    if (( recovered_step == target_step )); then
      echo "Segment process exited ${segment_status}, but checkpoint-${target_step} is complete; continuing with verification."
      latest_checkpoint="${recovered_checkpoint}"
      break
    fi
    if (( recovered_step != current_step )); then
      echo "Cannot safely retry segment: expected stable step ${current_step}, found ${recovered_step}." >&2
      exit "${segment_status}"
    fi
    if (( SEGMENT_MAX_RETRIES > 0 && segment_attempt > SEGMENT_MAX_RETRIES )); then
      echo "Segment ${current_step}->${target_step} failed after ${segment_attempt} attempts (status ${segment_status})." >&2
      exit "${segment_status}"
    fi

    quarantine_incomplete_future_checkpoints "${current_step}" "${segment_attempt}"
    "${STATE_TOOL[@]}" prepare-resume "${OUTPUT_DIR}" \
      --world-size "${WORLD_SIZE}" >/dev/null
    latest_checkpoint="${recovered_checkpoint}"
    retry_delay=$((RETRY_BASE_SECONDS * segment_attempt))
    (( retry_delay > RETRY_MAX_SECONDS )) && retry_delay="${RETRY_MAX_SECONDS}"
    echo "Segment ${current_step}->${target_step} failed (status ${segment_status}); retry ${segment_attempt} from ${latest_checkpoint} in ${retry_delay}s." >&2
    sleep "${retry_delay}"
  done

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
