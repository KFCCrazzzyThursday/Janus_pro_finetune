#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${ROOT_DIR}/upstream"

# GitHub and package downloads are substantially faster on the target hosts
# without their generic proxy. This affects only this process and its children.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

clone_and_patch() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local patch_file="${4:-}"
  local target="${UPSTREAM_DIR}/${name}"

  if [[ ! -d "${target}/.git" ]]; then
    echo "Cloning ${name} at ${revision}"
    git clone --filter=blob:none "${url}" "${target}"
  fi

  local actual
  actual="$(git -C "${target}" rev-parse HEAD)"
  if [[ "${actual}" != "${revision}" ]]; then
    if [[ -n "$(git -C "${target}" status --porcelain)" ]]; then
      echo "Refusing to change revision of dirty upstream tree: ${target}" >&2
      return 1
    fi
    git -C "${target}" fetch origin "${revision}"
    git -C "${target}" checkout --detach "${revision}"
  fi

  if [[ -z "${patch_file}" ]]; then
    return 0
  fi
  if git -C "${target}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "Patch already applied: $(basename "${patch_file}")"
  elif git -C "${target}" apply --check "${patch_file}"; then
    git -C "${target}" apply "${patch_file}"
    echo "Applied patch: $(basename "${patch_file}")"
  else
    echo "Patch does not apply cleanly to ${target}: ${patch_file}" >&2
    return 1
  fi
}

mkdir -p "${UPSTREAM_DIR}"
clone_and_patch \
  ms-swift \
  https://github.com/modelscope/ms-swift.git \
  087aa1cc481f97fb7f69f12dc7c224fb95857e86 \
  "${ROOT_DIR}/patches/ms-swift-087aa1c-janus-fsdp2.patch"
clone_and_patch \
  deepseek-janus \
  https://github.com/deepseek-ai/Janus.git \
  1daa72fa409002d40931bd7b36a9280362469ead \
  "${ROOT_DIR}/patches/deepseek-janus-1daa72f-transformers-fsdp2.patch"
clone_and_patch \
  ScienceQA \
  https://github.com/lupantech/ScienceQA.git \
  2cbf8318e07b9ece895bb2ae605e71e38d623264

echo "Pinned upstream sources are ready under ${UPSTREAM_DIR}."
