#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
RUN_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/stage1/tqa_grpo_lora_managed30}"
EVENT_WATCH_SECONDS="${JANUS_TENSORBOARD_EVENT_WATCH_SECONDS:-5}"

if ! [[ "${EVENT_WATCH_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JANUS_TENSORBOARD_EVENT_WATCH_SECONDS must be a positive integer." >&2
  exit 2
fi

# Scope TensorBoard to the canonical managed run. Historical smoke, aborted and
# archived event files therefore cannot appear as extra or conflicting curves.
# Keep the continuously appended resource stream out of the primary server by
# default: reloading that stream over NFS can delay discovery of a new trainer
# event file after checkpoint resume. The canonical resource CSV and event data
# remain on disk and can be enabled explicitly when needed.
logdir_spec="train:${RUN_DIR}/runs/trainer,val:${RUN_DIR}/runs/validation"
watched_event_dirs=("${RUN_DIR}/runs/trainer" "${RUN_DIR}/runs/validation")
if [[ "${JANUS_TENSORBOARD_INCLUDE_RESOURCES:-0}" == "1" ]]; then
  logdir_spec+=",resources:${RUN_DIR}/runs/resources"
  watched_event_dirs+=("${RUN_DIR}/runs/resources")
fi

mkdir -p "${watched_event_dirs[@]}"

event_file_snapshot() {
  find "${watched_event_dirs[@]}" -maxdepth 1 -type f \
    -name 'events.out.tfevents.*' -printf '%p\n' | LC_ALL=C sort
}

tensorboard_pid=""
cleanup() {
  trap - EXIT
  if [[ -n "${tensorboard_pid}" ]] && kill -0 "${tensorboard_pid}" 2>/dev/null; then
    kill "${tensorboard_pid}" 2>/dev/null || true
    wait "${tensorboard_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# A managed resume atomically rebuilds the canonical history and then opens a
# fresh live event file. TensorBoard retains data from deleted files in memory,
# so merely discovering the new file would accumulate duplicate history at
# every 30-step boundary. Supervise the server and restart only when the event
# filename set changes; ordinary appends do not cause a restart.
while true; do
  event_files="$(event_file_snapshot)"
  "${PYTHON_ENV}/bin/tensorboard" \
    --logdir_spec "${logdir_spec}" \
    --host "${JANUS_TENSORBOARD_HOST:-0.0.0.0}" \
    --port "${JANUS_TENSORBOARD_PORT:-6006}" \
    --reload_interval "${JANUS_TENSORBOARD_RELOAD_SECONDS:-5}" \
    --reload_multifile "${JANUS_TENSORBOARD_RELOAD_MULTIFILE:-true}" \
    --purge_orphaned_data false &
  tensorboard_pid=$!

  restart_requested=0
  while kill -0 "${tensorboard_pid}" 2>/dev/null; do
    sleep "${EVENT_WATCH_SECONDS}"
    updated_event_files="$(event_file_snapshot)"
    if [[ "${updated_event_files}" != "${event_files}" ]]; then
      echo "TensorBoard event file set changed; restarting the dashboard cache."
      restart_requested=1
      kill "${tensorboard_pid}" 2>/dev/null || true
      wait "${tensorboard_pid}" 2>/dev/null || true
      tensorboard_pid=""
      break
    fi
  done

  if (( restart_requested == 0 )); then
    set +e
    wait "${tensorboard_pid}"
    status=$?
    set -e
    tensorboard_pid=""
    exit "${status}"
  fi
done
