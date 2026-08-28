#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JANUS_VENV="${JANUS_VENV:-${ROOT_DIR}/.venv}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export JANUS_CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export JANUS_NPROC_PER_NODE="${JANUS_NPROC_PER_NODE:-8}"
export JANUS_MODEL_DIR="${JANUS_MODEL_DIR:-${ROOT_DIR}/models/Janus-Pro-7B-stage1-sft}"
export JANUS_STAGE_MODEL_TO_RAM="${JANUS_STAGE_MODEL_TO_RAM:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"

exec bash "${ROOT_DIR}/scripts/run_annotate_tqa_difficulty.sh" "$@"
