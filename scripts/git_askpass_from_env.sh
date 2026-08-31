#!/usr/bin/env bash
set -eu

case "${1:-}" in
  *Username*) printf '%s\n' 'x-access-token' ;;
  *Password*) printf '%s\n' "${JANUS_GITHUB_TOKEN:?JANUS_GITHUB_TOKEN is required}" ;;
  *) exit 1 ;;
esac
