#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
RAW_DIR="${ROOT_DIR}/outputs/stage1/tqa_synthesis_raw"
FILTER_DIR="${ROOT_DIR}/outputs/stage1/tqa_synthesis_filter"
SFT_OUTPUT="${ROOT_DIR}/data/processed/tqa/train_synthetic_filtered.jsonl"
EXTRA_ARGS=()

if [[ "${1:-}" == "--smoke" ]]; then
  shift
  RAW_DIR="${ROOT_DIR}/outputs/smoke/stage1_tqa_synthesis_raw"
  FILTER_DIR="${ROOT_DIR}/outputs/smoke/stage1_tqa_synthesis_filter"
  SFT_OUTPUT="${ROOT_DIR}/outputs/smoke/train_synthetic_filtered.jsonl"
  EXTRA_ARGS+=(--smoke-stub)
elif [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for the DeepSeek replacement judge." >&2
  echo "Export it in this shell; do not save it in the repository." >&2
  exit 2
fi

if [[ ! -s "${RAW_DIR}/predictions.jsonl" ]]; then
  echo "Missing synthesized predictions: ${RAW_DIR}/predictions.jsonl" >&2
  echo "Run scripts/run_stage1_synthesis.sh first." >&2
  exit 2
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
export JANUS_REASONING_JUDGE_MODEL="${JANUS_REASONING_JUDGE_MODEL:-deepseek-v4-flash-vision-exp}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

"${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/filter_tqa_synthesis.py" \
  --input "${ROOT_DIR}/data/processed/tqa/train_prompt.jsonl" \
  --predictions "${RAW_DIR}/predictions.jsonl" \
  --audit-output "${FILTER_DIR}/decisions.jsonl" \
  --sft-output "${SFT_OUTPUT}" \
  --summary "${FILTER_DIR}/summary.json" \
  "${EXTRA_ARGS[@]}" "$@"
