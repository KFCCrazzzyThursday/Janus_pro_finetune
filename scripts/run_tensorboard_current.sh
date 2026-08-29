#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
RUN_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/stage1/tqa_grpo_lora_managed30}"

# Scope TensorBoard to the canonical managed run. Historical smoke, aborted and
# archived event files therefore cannot appear as extra or conflicting curves.
logdir_spec="train:${RUN_DIR}/runs/trainer,val:${RUN_DIR}/runs/validation,resources:${RUN_DIR}/runs/resources"
exec "${PYTHON_ENV}/bin/tensorboard" \
  --logdir_spec "${logdir_spec}" \
  --host "${JANUS_TENSORBOARD_HOST:-0.0.0.0}" \
  --port "${JANUS_TENSORBOARD_PORT:-6006}" \
  --reload_interval "${JANUS_TENSORBOARD_RELOAD_SECONDS:-5}" \
  --purge_orphaned_data true
