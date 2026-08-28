#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${JANUS_TENSORBOARD_TMUX_SESSION:-janus_tensorboard}"
LOG_FILE="${JANUS_TENSORBOARD_LOG:-${ROOT_DIR}/outputs/tensorboard.log}"
LOG_DIR="${JANUS_TENSORBOARD_LOGDIR:-${ROOT_DIR}/outputs}"
VENV_DIR="${JANUS_VENV:-${ROOT_DIR}/.venv}"
HOST="${JANUS_TENSORBOARD_HOST:-0.0.0.0}"
PORT="${JANUS_TENSORBOARD_PORT:-6006}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required. Install it with the host package manager." >&2
  exit 2
fi
if [[ ! -x "${VENV_DIR}/bin/tensorboard" ]]; then
  echo "TensorBoard is missing from ${VENV_DIR}. Run scripts/setup_local_a100.sh." >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_FILE}")"
command_q="$(printf '%q ' "${VENV_DIR}/bin/tensorboard" --logdir "${LOG_DIR}" --host "${HOST}" --port "${PORT}" --reload_interval 5)"
log_q="$(printf '%q' "${LOG_FILE}")"
inner="set -o pipefail; ${command_q}2>&1 | tee ${log_q}"
inner_q="$(printf '%q' "${inner}")"
tmux new-session -d -s "${SESSION}" -c "${ROOT_DIR}" "bash -lc ${inner_q}"

echo "Started TensorBoard in tmux session: ${SESSION}"
echo "URL on the host: http://127.0.0.1:${PORT}"
echo "Log: ${LOG_FILE}"
