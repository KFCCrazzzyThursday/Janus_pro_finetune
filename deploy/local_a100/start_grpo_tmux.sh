#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${JANUS_GRPO_TMUX_SESSION:-janus_grpo_lora_ddp}"
LOG_FILE="${JANUS_GRPO_TMUX_LOG:-${ROOT_DIR}/outputs/stage1/a100x8_lora_ddp.log}"
RUN_ARGS=("$@")

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required. Install it with the host package manager." >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_FILE}")"
launcher_q="$(printf '%q' "${ROOT_DIR}/deploy/local_a100/run_stage1_grpo.sh")"
log_q="$(printf '%q' "${LOG_FILE}")"
run_args_q=""
for arg in "${RUN_ARGS[@]}"; do
  run_args_q+=" $(printf '%q' "${arg}")"
done

needs_key=1
case "${RUN_ARGS[0]:-}" in
  --offline|--smoke|--memory-smoke)
    needs_key=0
    ;;
esac

if (( needs_key )); then
  secret="${OPENAI_API_KEY:-}"
  if [[ -z "${secret}" ]]; then
    if [[ ! -t 0 ]]; then
      echo "Run this script from an interactive terminal so it can read OPENAI_API_KEY." >&2
      exit 2
    fi
    read -r -s -p "OPENAI_API_KEY: " secret
    printf '\n'
  fi
  if [[ -z "${secret}" ]]; then
    echo "OPENAI_API_KEY cannot be empty for a formal run." >&2
    exit 2
  fi

  # The pane reads the key from stdin. It is not embedded in the tmux command,
  # process arguments, repository, or log. The temporary tmux buffer is deleted
  # immediately after pasting.
  inner="trap 'stty echo' EXIT; stty -echo; printf 'JANUS_KEY_READY\\n'; IFS= read -r OPENAI_API_KEY; stty echo; trap - EXIT; printf '\\n'; export OPENAI_API_KEY; set -o pipefail; bash ${launcher_q}${run_args_q} 2>&1 | tee ${log_q}"
  inner_q="$(printf '%q' "${inner}")"
  tmux new-session -d -s "${SESSION}" -c "${ROOT_DIR}" "bash -lc ${inner_q}"

  ready=0
  for _ in {1..50}; do
    if tmux capture-pane -p -t "${SESSION}:0.0" | grep -q '^JANUS_KEY_READY$'; then
      ready=1
      break
    fi
    sleep 0.1
  done
  if (( ! ready )); then
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    echo "tmux pane did not become ready for secure key injection." >&2
    exit 2
  fi

  buffer="janus-secret-$$"
  printf '%s' "${secret}" | tmux load-buffer -b "${buffer}" -
  tmux paste-buffer -b "${buffer}" -d -t "${SESSION}:0.0"
  tmux send-keys -t "${SESSION}:0.0" Enter
  unset secret OPENAI_API_KEY
else
  inner="set -o pipefail; bash ${launcher_q}${run_args_q} 2>&1 | tee ${log_q}"
  inner_q="$(printf '%q' "${inner}")"
  tmux new-session -d -s "${SESSION}" -c "${ROOT_DIR}" "bash -lc ${inner_q}"
fi

echo "Started tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: ${LOG_FILE}"
