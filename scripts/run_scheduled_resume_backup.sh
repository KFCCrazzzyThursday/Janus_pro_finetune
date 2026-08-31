#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-${ROOT_DIR}/.venv}"
RUN_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/stage1/tqa_grpo_accfmt_a100_from270_managed30}"
TARGET_EPOCH="${JANUS_BACKUP_TARGET_EPOCH:-1788228000}"
BACKUP_ID="${JANUS_BACKUP_ID:-a100-resume-20260901-1000-cst}"
GITHUB_BRANCH="${JANUS_BACKUP_GITHUB_BRANCH:-backup/a100-resume-20260901-1000-cst}"
SECRET_ENV="${JANUS_BACKUP_SECRET_ENV:-/dev/shm/janus-scheduled-backup.env}"
SNAPSHOT_ROOT="${JANUS_BACKUP_SNAPSHOT_ROOT:-${RUN_DIR}/scheduled_backups}"
LOG_DIR="${SNAPSHOT_ROOT}/logs"
LOCK_FILE="${SNAPSHOT_ROOT}/.${BACKUP_ID}.scheduler.lock"

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another scheduler already owns ${LOCK_FILE}." >&2
  exit 3
fi

while true; do
  now="$(date +%s)"
  remaining=$((TARGET_EPOCH - now))
  (( remaining <= 0 )) && break
  delay=300
  (( remaining < delay )) && delay="${remaining}"
  echo "$(date -u '+%FT%TZ') waiting ${remaining}s for backup target ${TARGET_EPOCH}."
  sleep "${delay}"
done

if [[ -s "${SECRET_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${SECRET_ENV}"
else
  echo "Warning: credential file is absent at ${SECRET_ENV}; uploads will report failures." >&2
fi

DATA_ARGS=()
for source in \
  "${ROOT_DIR}/data/processed/tqa/train_prompt_model_difficulty.remote.jsonl" \
  "${ROOT_DIR}/data/processed/tqa/val_prompt.remote.jsonl"; do
  [[ -f "${source}" ]] && DATA_ARGS+=(--data-file "${source}")
done

exec "${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/snapshot_and_upload_resume.py" \
  --repo-dir "${ROOT_DIR}" \
  --run-dir "${RUN_DIR}" \
  --snapshot-root "${SNAPSHOT_ROOT}" \
  --backup-id "${BACKUP_ID}" \
  --world-size "${JANUS_NPROC_PER_NODE:-2}" \
  --github-branch "${GITHUB_BRANCH}" \
  --hf-repo-id "${JANUS_HF_REPO_ID:-}" \
  --hf-revision "${JANUS_HF_REVISION:-backup/a100-resume-20260901-1000-cst}" \
  "${DATA_ARGS[@]}"
