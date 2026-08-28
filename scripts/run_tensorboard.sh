#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
HOST="${JANUS_TENSORBOARD_HOST:-0.0.0.0}"
PORT="${JANUS_TENSORBOARD_PORT:-6006}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

exec "${PYTHON_ENV}/bin/tensorboard" \
  --logdir "${ROOT_DIR}/outputs" \
  --host "${HOST}" \
  --port "${PORT}" \
  --reload_interval 5
