#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
BASE_MODEL_DIR="${JANUS_MODEL_DIR:-${ROOT_DIR}/models/Janus-Pro-7B}"
OUTPUT_DIR="${ROOT_DIR}/outputs/stage1/tqa_synthesis_raw"
EXTRA_ARGS=()

latest_grpo_checkpoint() {
  find "${ROOT_DIR}/outputs/stage1/tqa_grpo" -maxdepth 1 -type d \
    -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1
}

MODEL_SOURCE_DIR="${JANUS_STAGE1_GRPO_MODEL:-$(latest_grpo_checkpoint)}"
if [[ "${1:-}" == "--smoke" ]]; then
  shift
  OUTPUT_DIR="${ROOT_DIR}/outputs/smoke/stage1_tqa_synthesis_raw"
  EXTRA_ARGS+=(--max-samples 16 --max-new-tokens 64)
  if [[ -z "${MODEL_SOURCE_DIR}" ]]; then
    echo "No GRPO checkpoint yet; smoke-testing synthesis from the base model."
    MODEL_SOURCE_DIR="${BASE_MODEL_DIR}"
  fi
fi

if [[ -z "${MODEL_SOURCE_DIR}" || ! -d "${MODEL_SOURCE_DIR}" ]]; then
  echo "No stage-1 GRPO checkpoint found. Run scripts/run_stage1_grpo.sh first," >&2
  echo "or set JANUS_STAGE1_GRPO_MODEL to its NFS checkpoint directory." >&2
  exit 2
fi

source "${ROOT_DIR}/scripts/lib/runtime.sh"
if [[ "${MODEL_SOURCE_DIR}" == "${BASE_MODEL_DIR}" ]]; then
  export JANUS_RAM_MODEL_NAME=Janus-Pro-7B
else
  export JANUS_RAM_MODEL_NAME=Janus-Pro-7B-stage1-grpo
fi
janus_stage_model_to_ram "${MODEL_SOURCE_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,3,4
export HF_HOME="${HF_HOME:-/root/nfs/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

"${PYTHON_ENV}/bin/torchrun" --standalone --nproc-per-node=4 \
  "${ROOT_DIR}/scripts/evaluate_vqa.py" \
  --model "${JANUS_ACTIVE_MODEL_DIR}" \
  --model-source "${MODEL_SOURCE_DIR}" \
  --input "${ROOT_DIR}/data/processed/tqa/train_prompt.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --response-prefix '<think>' \
  --batch-size 1 \
  --max-new-tokens 384 \
  "${EXTRA_ARGS[@]}" "$@"
