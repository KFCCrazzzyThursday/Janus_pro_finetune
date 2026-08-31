#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${JANUS_VENV:-${ROOT_DIR}/.venv}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ROOT_DIR}/.pip-cache}"
export TMPDIR="${TMPDIR:-${ROOT_DIR}/.tmp}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# NVIDIA container images often inject an unreachable NGC extra index. The
# explicit PyPI index below is sufficient for this userspace-only install.
unset PIP_EXTRA_INDEX_URL
export PIP_CONFIG_FILE=/dev/null
mkdir -p "${HF_HOME}" "${HF_XET_CACHE}" "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}"

bash "${ROOT_DIR}/scripts/bootstrap_upstreams.sh"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip --isolated install --upgrade \
  --index-url https://pypi.org/simple pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip --isolated install \
  --index-url https://pypi.org/simple \
  -r "${ROOT_DIR}/requirements/remote-a100.txt" \
  -r "${ROOT_DIR}/upstream/ms-swift/requirements/framework.txt"
"${VENV_DIR}/bin/python" -m pip --isolated install --no-deps \
  -e "${ROOT_DIR}/upstream/deepseek-janus" \
  -e "${ROOT_DIR}/upstream/ms-swift"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"
"${VENV_DIR}/bin/python" - <<'PY'
import torch
import transformers
import trl
from janus.models import MultiModalityCausalLM
from swift import __version__ as swift_version

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) >= (8, 0), (major, minor)
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)} capability={major}.{minor}")
print(f"transformers={transformers.__version__} trl={trl.__version__} ms-swift={swift_version}")
print(f"janus={MultiModalityCausalLM.__name__}")
PY

"${VENV_DIR}/bin/python" -m pytest -q \
  "${ROOT_DIR}/tests/test_grpo_plugin.py" \
  "${ROOT_DIR}/tests/test_rewards.py"
echo "Remote A100 environment is ready at ${VENV_DIR}."
