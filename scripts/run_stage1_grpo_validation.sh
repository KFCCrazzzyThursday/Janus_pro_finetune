#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
CHECKPOINT="${1:-}"

if [[ -z "${CHECKPOINT}" || ! -d "${CHECKPOINT}" ]]; then
  echo "usage: $0 /path/to/checkpoint-N" >&2
  exit 2
fi
CHECKPOINT="$(cd "${CHECKPOINT}" && pwd)"
checkpoint_name="${CHECKPOINT##*/}"
if [[ ! "${checkpoint_name}" =~ ^checkpoint-([0-9]+)$ ]]; then
  echo "Checkpoint must be named checkpoint-N: ${CHECKPOINT}" >&2
  exit 2
fi
step="${BASH_REMATCH[1]}"
RUN_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${CHECKPOINT%/*}}"
RUN_DIR="$(mkdir -p "${RUN_DIR}" && cd "${RUN_DIR}" && pwd)"
MODEL_SOURCE="${JANUS_STAGE1_SFT_MODEL:-$(jq -r '.base_model_name_or_path' "${CHECKPOINT}/adapter_config.json")}"
VAL_DATASET="${JANUS_GRPO_VAL_DATASET:-${ROOT_DIR}/data/processed/tqa/val_prompt.jsonl}"
VAL_ROOT="${JANUS_GRPO_VALIDATION_ROOT:-${RUN_DIR}/validation}"
printf -v step_padded '%06d' "${step}"
VAL_DIR="${VAL_ROOT}/checkpoint-${step_padded}/tqa_val"
SUMMARY="${VAL_DIR}/summary.json"

if [[ ! -d "${MODEL_SOURCE}" ]]; then
  echo "Base model recorded by the adapter does not exist: ${MODEL_SOURCE}" >&2
  exit 2
fi
if [[ ! -s "${VAL_DATASET}" ]]; then
  echo "Validation dataset is missing or empty: ${VAL_DATASET}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
IFS=',' read -r -a physical_gpus <<<"${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${JANUS_NPROC_PER_NODE:-${#physical_gpus[@]}}"
if (( NPROC_PER_NODE != ${#physical_gpus[@]} )); then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} does not match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-/root/nfs/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

mkdir -p "${VAL_DIR}"
reuse=0
if [[ -s "${SUMMARY}" ]]; then
  recorded_adapter="$(jq -r '.adapter // empty' "${SUMMARY}")"
  recorded_input="$(jq -r '.input // empty' "${SUMMARY}")"
  recorded_samples="$(jq -r '.num_samples // 0' "${SUMMARY}")"
  recorded_prefix_conditioning="$(jq -r '.response_prefix_conditioning // empty' "${SUMMARY}")"
  expected_samples="$(wc -l < "${VAL_DATASET}")"
  if [[ "${recorded_adapter}" == "${CHECKPOINT}" \
        && "${recorded_input}" == "$(realpath "${VAL_DATASET}")" \
        && "${recorded_samples}" == "${expected_samples}" \
        && "${recorded_prefix_conditioning}" == "assistant_context_without_terminal_eos" \
        && -z "${JANUS_GRPO_VAL_MAX_SAMPLES:-}" ]]; then
    reuse=1
  fi
fi

if (( reuse )); then
  echo "Reusing completed validation: ${SUMMARY}"
else
  eval_args=(
    --model "${MODEL_SOURCE}"
    --adapter "${CHECKPOINT}"
    --model-source "${MODEL_SOURCE}"
    --input "${VAL_DATASET}"
    --output-dir "${VAL_DIR}"
    --batch-size "${JANUS_GRPO_VAL_BATCH_SIZE:-1}"
    --max-new-tokens "${JANUS_GRPO_VAL_MAX_NEW_TOKENS:-384}"
    --response-prefix "<think>"
    --seed "${JANUS_GRPO_VAL_SEED:-42}"
  )
  if [[ -n "${JANUS_GRPO_VAL_MAX_SAMPLES:-}" ]]; then
    eval_args+=(--max-samples "${JANUS_GRPO_VAL_MAX_SAMPLES}")
  fi
  echo "Validating ${CHECKPOINT} on ${VAL_DATASET} with ${NPROC_PER_NODE} GPUs."
  "${PYTHON_ENV}/bin/torchrun" --standalone --nproc-per-node="${NPROC_PER_NODE}" \
    "${ROOT_DIR}/scripts/evaluate_vqa.py" "${eval_args[@]}" \
    2>&1 | tee "${VAL_DIR}/validation.log"
fi

"${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/grpo_run_state.py" \
  record-validation "${CHECKPOINT}" "${SUMMARY}" \
  --world-size "${NPROC_PER_NODE}"

echo "Validation summary: ${SUMMARY}"
jq '{accuracy,strict_format_rate,parse_failure_rate,mean_reasoning_tokens,mean_completion_tokens,num_samples,runtime_seconds}' "${SUMMARY}"
