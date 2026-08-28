#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${JANUS_VENV:-${ROOT_DIR}/.venv}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${ROOT_DIR}/.uv-python}"
export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"

mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" "${TMPDIR}"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it from https://docs.astral.sh/uv/ first." >&2
  exit 2
fi

bash "${ROOT_DIR}/scripts/bootstrap_upstreams.sh"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv --python 3.12 "${VENV_DIR}"
fi

uv pip install --python "${VENV_DIR}/bin/python" \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
uv pip install --python "${VENV_DIR}/bin/python" \
  --index-url https://pypi.org/simple \
  -r "${ROOT_DIR}/requirements/remote-blackwell.txt" \
  -r "${ROOT_DIR}/upstream/ms-swift/requirements/framework.txt"
uv pip install --python "${VENV_DIR}/bin/python" --no-deps \
  -e "${ROOT_DIR}/upstream/deepseek-janus" \
  -e "${ROOT_DIR}/upstream/ms-swift"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"
"${VENV_DIR}/bin/python" -c \
  'import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0) == (8, 0); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  "${VENV_DIR}/bin/python" -m pytest -q "${ROOT_DIR}/tests"
echo "A100 environment is ready at ${VENV_DIR}."
