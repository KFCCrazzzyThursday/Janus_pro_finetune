#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
MODEL_SOURCE_DIR="${JANUS_MODEL_DIR:-${ROOT_DIR}/models/Janus-Pro-7B}"
OUTPUT="${JANUS_TQA_DIFFICULTY_OUTPUT:-${ROOT_DIR}/data/processed/tqa/train_prompt_model_difficulty.jsonl}"
INPUT="${JANUS_TQA_PROMPT_DATA:-${ROOT_DIR}/data/processed/tqa/train_prompt.jsonl}"
EXTRA_ARGS=()

if [[ "${1:-}" == "--smoke" ]]; then
  shift
  OUTPUT="${ROOT_DIR}/outputs/smoke/tqa_difficulty.jsonl"
  EXTRA_ARGS+=(--max-samples 16)
fi

source "${ROOT_DIR}/scripts/lib/runtime.sh"
janus_stage_model_to_ram "${MODEL_SOURCE_DIR}"

export CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,3,4}"
export NPROC_PER_NODE="${JANUS_NPROC_PER_NODE:-4}"
export HF_HOME="${HF_HOME:-/root/nfs/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "Annotating TQA difficulty on physical GPUs ${CUDA_VISIBLE_DEVICES} (${NPROC_PER_NODE} ranks)."
"${PYTHON_ENV}/bin/torchrun" --standalone --nproc-per-node="${NPROC_PER_NODE}" \
  "${ROOT_DIR}/scripts/annotate_tqa_difficulty.py" \
  --model "${JANUS_ACTIVE_MODEL_DIR}" \
  --model-source "${MODEL_SOURCE_DIR}" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}" "$@"
