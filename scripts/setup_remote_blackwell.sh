#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${JANUS_VENV:-${ROOT_DIR}/.venv}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${XDG_CACHE_HOME}/modelscope}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.uv-cache}"
export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
mkdir -p \
  "${HF_HOME}" "${HF_XET_CACHE}" "${XDG_CACHE_HOME}" \
  "${MODELSCOPE_CACHE}" "${TORCH_HOME}" "${UV_CACHE_DIR}" "${TMPDIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required on the remote host." >&2
  exit 2
fi

bash "${ROOT_DIR}/scripts/bootstrap_upstreams.sh"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv --python 3.12 "${VENV_DIR}"
fi

echo "Installing the CUDA 12.8 PyTorch build for Blackwell GPUs."
uv pip install --python "${VENV_DIR}/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1

echo "Installing the pinned training stack."
uv pip install --python "${VENV_DIR}/bin/python" \
  --index-url https://pypi.org/simple \
  -r "${ROOT_DIR}/requirements/remote-blackwell.txt" \
  -r "${ROOT_DIR}/upstream/ms-swift/requirements/framework.txt"
uv pip install --python "${VENV_DIR}/bin/python" --no-deps \
  -e "${ROOT_DIR}/upstream/deepseek-janus" \
  -e "${ROOT_DIR}/upstream/ms-swift"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"
"${VENV_DIR}/bin/python" - <<'PY'
import torch
import transformers
import trl
from torch.distributed.fsdp import FSDPModule
from janus.models import MultiModalityCausalLM
from swift import __version__ as swift_version

assert torch.cuda.is_available(), "CUDA is not available"
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) >= (12, 0), (major, minor)
assert torch.version.cuda and tuple(map(int, torch.version.cuda.split('.')[:2])) >= (12, 8)
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)} capability={major}.{minor}")
print(f"transformers={transformers.__version__} trl={trl.__version__} ms-swift={swift_version}")
print(f"fsdp2_api={FSDPModule.__name__} janus={MultiModalityCausalLM.__name__}")
PY

"${VENV_DIR}/bin/python" -m pytest -q "${ROOT_DIR}/tests"
echo "Remote Blackwell environment is ready at ${VENV_DIR}."
