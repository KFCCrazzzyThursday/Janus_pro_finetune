#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
# shellcheck disable=SC1091
. "${utils}/logging.sh" ""
# shellcheck disable=SC1091
. "${utils}/environment.sh"

cd /workspace/Janus_pro_finetune
exec pty /bin/bash /workspace/Janus_pro_finetune/deploy/remote/run_stage1_grpo.sh 2>&1
