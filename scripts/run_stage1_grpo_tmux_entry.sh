#!/usr/bin/env bash
set -Eeuo pipefail

# Read the API key once from a private FIFO so it never appears in the tmux
# command line, repository, or training log. The caller owns creating/writing
# the FIFO; this process removes it immediately after reading it.
if [[ $# -ne 1 ]]; then
  echo "usage: $0 /tmp/janus-grpo-secret.XXXXXX/key" >&2
  exit 2
fi

key_fifo=$1
secret_dir=${key_fifo%/*}

case "$key_fifo" in
  /tmp/janus-grpo-secret.*/key) ;;
  *)
    echo "refusing unexpected API-key FIFO path: $key_fifo" >&2
    exit 2
    ;;
esac

cleanup_secret() {
  rm -f -- "$key_fifo"
  rmdir -- "$secret_dir" 2>/dev/null || true
}
trap cleanup_secret EXIT HUP INT TERM

if [[ ! -p "$key_fifo" ]]; then
  echo "API-key FIFO does not exist: $key_fifo" >&2
  exit 2
fi

IFS= read -r OPENAI_API_KEY < "$key_fifo"
cleanup_secret
trap - EXIT HUP INT TERM

if [[ -z "$OPENAI_API_KEY" ]]; then
  echo "received an empty OPENAI_API_KEY" >&2
  exit 2
fi
export OPENAI_API_KEY

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
exec bash scripts/run_stage1_grpo.sh
