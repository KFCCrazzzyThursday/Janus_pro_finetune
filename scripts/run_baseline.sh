#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
MODEL_SOURCE_DIR="${JANUS_MODEL_DIR:-${ROOT_DIR}/models/Janus-Pro-7B}"

source "${ROOT_DIR}/scripts/lib/runtime.sh"

export CUDA_VISIBLE_DEVICES=0,1,3,4
export HF_HOME="${HF_HOME:-/root/nfs/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"

run_eval() {
  local dataset="$1"
  local split="$2"
  "${PYTHON_ENV}/bin/torchrun" --standalone --nproc-per-node=4 \
    "${ROOT_DIR}/scripts/evaluate_vqa.py" \
    --model "${JANUS_ACTIVE_MODEL_DIR}" \
    --model-source "${MODEL_SOURCE_DIR}" \
    --input "${ROOT_DIR}/data/processed/${dataset}/${split}_prompt.jsonl" \
    --output-dir "${ROOT_DIR}/outputs/baseline/${dataset}_${split}" \
    --max-new-tokens 384 \
    "${@:3}"
}

if [[ $# -ge 2 ]]; then
  janus_stage_model_to_ram "${MODEL_SOURCE_DIR}"
  run_eval "$@"
else
  janus_stage_model_to_ram "${MODEL_SOURCE_DIR}"
  run_eval tqa val
  run_eval tqa test
  run_eval scienceqa/full test
  run_eval scienceqa test
fi
