#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
BASE_MODEL_DIR="${JANUS_MODEL_DIR:-${ROOT_DIR}/models/Janus-Pro-7B}"
CHECKPOINT="${1:-}"
OUTPUT_ROOT="${ROOT_DIR}/outputs/stage1/scienceqa_sft_validation"

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(find "${ROOT_DIR}/outputs/stage1/scienceqa_sft" -maxdepth 1 \
    -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1)"
fi
if [[ -z "${CHECKPOINT}" || ! -d "${CHECKPOINT}" ]]; then
  echo "A completed stage-1 SFT checkpoint is required." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,3,4}"
export HF_HOME="${HF_HOME:-/root/nfs/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

run_eval() {
  local model="$1"
  local input="$2"
  local output="$3"
  if [[ -s "${output}/summary.json" ]]; then
    echo "Reusing completed validation: ${output}/summary.json"
    return
  fi
  "${PYTHON_ENV}/bin/torchrun" --standalone --nproc-per-node=4 \
    "${ROOT_DIR}/scripts/evaluate_vqa.py" \
    --model "${model}" \
    --model-source "${model}" \
    --input "${input}" \
    --output-dir "${output}" \
    --batch-size 1 \
    --max-new-tokens 384
}

# The existing TQA base validation is reused. ScienceQA val was not part of the
# earlier paper-table baseline, so create the missing exact control here.
run_eval \
  "${BASE_MODEL_DIR}" \
  "${ROOT_DIR}/data/processed/scienceqa/val_prompt.jsonl" \
  "${OUTPUT_ROOT}/base/scienceqa_val"
run_eval \
  "${CHECKPOINT}" \
  "${ROOT_DIR}/data/processed/scienceqa/val_prompt.jsonl" \
  "${OUTPUT_ROOT}/sft/scienceqa_val"
run_eval \
  "${CHECKPOINT}" \
  "${ROOT_DIR}/data/processed/tqa/val_prompt.jsonl" \
  "${OUTPUT_ROOT}/sft/tqa_val"

"${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/assess_stage1_sft.py" \
  --scienceqa-base "${OUTPUT_ROOT}/base/scienceqa_val/summary.json" \
  --scienceqa-sft "${OUTPUT_ROOT}/sft/scienceqa_val/summary.json" \
  --tqa-base "${ROOT_DIR}/outputs/baseline/tqa_val/summary.json" \
  --tqa-sft "${OUTPUT_ROOT}/sft/tqa_val/summary.json" \
  --output "${OUTPUT_ROOT}/assessment.json"

"${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/build_sft_qualitative_audit.py" \
  --scienceqa-prompts "${ROOT_DIR}/data/processed/scienceqa/val_prompt.jsonl" \
  --scienceqa-base "${OUTPUT_ROOT}/base/scienceqa_val/predictions.jsonl" \
  --scienceqa-sft "${OUTPUT_ROOT}/sft/scienceqa_val/predictions.jsonl" \
  --tqa-prompts "${ROOT_DIR}/data/processed/tqa/val_prompt.jsonl" \
  --tqa-base "${ROOT_DIR}/outputs/baseline/tqa_val/predictions.jsonl" \
  --tqa-sft "${OUTPUT_ROOT}/sft/tqa_val/predictions.jsonl" \
  --samples-per-bucket 2 \
  --output-json "${OUTPUT_ROOT}/qualitative_audit.json" \
  --output-markdown "${OUTPUT_ROOT}/qualitative_audit.md"

"${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/build_sft_sample_gallery.py" \
  --prompts "${ROOT_DIR}/data/processed/scienceqa/val_prompt.jsonl" \
  --base "${OUTPUT_ROOT}/base/scienceqa_val/predictions.jsonl" \
  --sft "${OUTPUT_ROOT}/sft/scienceqa_val/predictions.jsonl" \
  --count-per-outcome 5 \
  --output "${OUTPUT_ROOT}/sample_cases.md" \
  --assets-dir "${OUTPUT_ROOT}/sample_cases_assets"
