#!/usr/bin/env bash

# Shared runtime helpers. The canonical checkpoint always stays on NFS.  The
# optional staging target is tmpfs (/dev/shm), so no model copy is written to a
# local disk.  Keeping one tmpfs copy also prevents four workers from faulting
# the same large PyTorch shards through the SeaweedFS FUSE mount concurrently.

janus_model_manifest() {
  local source_dir="$1"
  find "${source_dir}" -maxdepth 1 -type f \
    -printf '%f\t%s\t%T@\n' | LC_ALL=C sort | sha256sum | awk '{print $1}'
}

janus_directory_bytes() {
  local source_dir="$1"
  find "${source_dir}" -maxdepth 1 -type f -printf '%s\n' \
    | awk '{total += $1} END {printf "%.0f\n", total}'
}

janus_available_memory_bytes() {
  awk '/^MemAvailable:/ {printf "%.0f\n", $2 * 1024}' /proc/meminfo
}

janus_stage_model_to_ram() {
  local source_dir="$1"
  JANUS_ACTIVE_MODEL_DIR="${source_dir}"

  if [[ "${JANUS_STAGE_MODEL_TO_RAM:-1}" != "1" ]]; then
    echo "Using the canonical NFS checkpoint directly: ${source_dir}"
    return 0
  fi

  local ram_root="${JANUS_RAM_ROOT:-/dev/shm/janus-repro-${UID}}"
  local destination_name="${JANUS_RAM_MODEL_NAME:-$(basename "${source_dir}")}"
  if [[ "${ram_root}" != /dev/shm/janus-repro-* ]] || [[ "${ram_root}" == *..* ]] \
    || [[ -z "${destination_name}" ]] || [[ "${destination_name}" == */* ]]; then
    echo "Unsafe tmpfs staging path: root=${ram_root}, name=${destination_name}" >&2
    return 1
  fi
  local destination="${ram_root}/${destination_name}"
  local marker="${destination}/.janus-nfs-manifest"
  local source_manifest
  source_manifest="$(janus_model_manifest "${source_dir}")"

  if [[ -f "${marker}" ]] && [[ "$(<"${marker}")" == "${source_manifest}" ]]; then
    echo "Reusing verified RAM checkpoint: ${destination}"
    JANUS_ACTIVE_MODEL_DIR="${destination}"
    return 0
  fi

  local model_bytes available_bytes shm_available_bytes headroom_bytes
  model_bytes="$(janus_directory_bytes "${source_dir}")"
  available_bytes="$(janus_available_memory_bytes)"
  shm_available_bytes="$(df --output=avail -B1 /dev/shm | tail -n 1 | tr -d ' ')"
  headroom_bytes="${JANUS_MIN_RAM_HEADROOM_BYTES:-68719476736}"

  if (( available_bytes < model_bytes + headroom_bytes )); then
    echo "Refusing RAM staging: model=${model_bytes} B, MemAvailable=${available_bytes} B," \
      " required headroom=${headroom_bytes} B." >&2
    return 1
  fi
  if (( shm_available_bytes < model_bytes )); then
    echo "Refusing RAM staging: model=${model_bytes} B, /dev/shm available=${shm_available_bytes} B." >&2
    return 1
  fi

  mkdir -p "${ram_root}"
  if [[ -e "${destination}" ]]; then
    case "${destination}" in
      /dev/shm/janus-repro-*/*) rm -rf -- "${destination}" ;;
      *) echo "Unsafe RAM staging destination: ${destination}" >&2; return 1 ;;
    esac
  fi

  local partial="${destination}.partial.${BASHPID}"
  mkdir -p "${partial}"
  echo "Staging ${model_bytes} bytes from NFS into tmpfs: ${destination}"
  if ! cp -a --reflink=never "${source_dir}/." "${partial}/"; then
    case "${partial}" in
      /dev/shm/janus-repro-*/*.partial.*) rm -rf -- "${partial}" ;;
    esac
    return 1
  fi
  printf '%s\n' "${source_manifest}" >"${partial}/.janus-nfs-manifest"
  mv -- "${partial}" "${destination}"
  sync -f "${destination}"
  JANUS_ACTIVE_MODEL_DIR="${destination}"
  echo "RAM checkpoint ready; current memory state:"
  free -h
}
