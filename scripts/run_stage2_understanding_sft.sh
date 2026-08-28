#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
MODEL_SOURCE_DIR="${JANUS_MODEL_DIR:-${ROOT_DIR}/models/Janus-Pro-7B}"
RAW_DATA="${ROOT_DIR}/data/processed/tqa/train_sft.jsonl"
SYNTHETIC_DATA="${JANUS_SYNTHETIC_DATA:-${ROOT_DIR}/data/processed/tqa/train_synthetic_filtered.jsonl}"
OUTPUT_DIR="${ROOT_DIR}/outputs/stage2/understanding_sft"
SMOKE=0

if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
  OUTPUT_DIR="${ROOT_DIR}/outputs/smoke/stage2_understanding_sft"
fi

if (( ! SMOKE )) && [[ ! -s "${SYNTHETIC_DATA}" ]]; then
  echo "Missing filtered synthetic TQA data: ${SYNTHETIC_DATA}" >&2
  echo "Run scripts/run_stage1_synthesis.sh and run_filter_tqa_synthesis.sh first." >&2
  exit 2
fi

source "${ROOT_DIR}/scripts/lib/runtime.sh"
export JANUS_RAM_MODEL_NAME=Janus-Pro-7B
export JANUS_STAGE_MODEL_TO_RAM="${JANUS_STAGE_MODEL_TO_RAM:-0}"
janus_stage_model_to_ram "${MODEL_SOURCE_DIR}"

export CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,3,4}"
export NPROC_PER_NODE="${JANUS_NPROC_PER_NODE:-4}"
export HF_HOME="${HF_HOME:-/root/nfs/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

DATASETS=(--dataset "${RAW_DATA}")
if [[ -s "${SYNTHETIC_DATA}" ]]; then
  DATASETS+=("${SYNTHETIC_DATA}")
fi

COMMON_ARGS=(
  --model "${JANUS_ACTIVE_MODEL_DIR}"
  --model_type deepseek_janus_pro
  --template deepseek_janus_pro
  --local_repo_path "${ROOT_DIR}/upstream/deepseek-janus"
  --check_model false
  --external_plugins "${ROOT_DIR}/training/plugins/fsdp2_janus_compat.py"
  "${DATASETS[@]}"
  --split_dataset_ratio 0
  --dataset_num_proc 4
  --dataloader_num_workers 2
  --tuner_type full
  --freeze_llm false
  --freeze_vit true
  --freeze_aligner true
  --torch_dtype bfloat16
  --attn_impl sdpa
  --max_length 2048
  --truncation_strategy delete
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 16
  --gradient_checkpointing true
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --vit_gradient_checkpointing false
  --learning_rate 2e-5
  --lr_scheduler_type constant
  --weight_decay 0
  --max_grad_norm 1
  --adam_beta1 0.9
  --adam_beta2 0.95
  --use_galore true
  --galore_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
  --galore_rank 256
  --galore_update_proj_gap 128
  --galore_optim_per_parameter false
  --ddp_find_unused_parameters false
  --num_train_epochs 3
  --max_steps 555
  --save_strategy steps
  --save_steps 555
  --save_total_limit 1
  --logging_steps 1
  --logging_first_step true
  --logging_dir "${OUTPUT_DIR}/runs/trainer"
  --report_to tensorboard
  --include_num_input_tokens_seen true
  --include_tokens_per_second true
  --seed 42
  --data_seed 42
  --add_version false
  --output_dir "${OUTPUT_DIR}"
)

SFT_BACKEND="${JANUS_SFT_BACKEND:-fsdp2}"
if [[ "${JANUS_SFT_USE_DEEPSPEED:-0}" == "1" ]]; then
  SFT_BACKEND=deepspeed
fi
case "${SFT_BACKEND}" in
  fsdp2)
    COMMON_ARGS+=(--fsdp "${ROOT_DIR}/configs/fsdp2_galore.json")
    ;;
  ddp)
    ;;
  deepspeed)
    COMMON_ARGS+=(--deepspeed "${ROOT_DIR}/configs/deepspeed_zero2_galore.json")
    ;;
  *)
    echo "JANUS_SFT_BACKEND must be fsdp2, ddp, or deepspeed; got ${SFT_BACKEND}" >&2
    exit 2
    ;;
esac

if (( SMOKE )); then
  COMMON_ARGS+=(
    --gradient_accumulation_steps 1
    --num_train_epochs 1
    --max_steps 1
    --save_strategy no
  )
fi

echo "Launching stage-2 understanding SFT on physical GPUs ${CUDA_VISIBLE_DEVICES}."
echo "Distributed backend: ${SFT_BACKEND}"
echo "Canonical model: ${MODEL_SOURCE_DIR}"
echo "Synthetic data: ${SYNTHETIC_DATA}"
free -h
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/monitor_resources.py" \
  --output-dir "${OUTPUT_DIR}" \
  --physical-gpus "${CUDA_VISIBLE_DEVICES}" \
  --parent-pid "$$" &
RESOURCE_MONITOR_PID=$!
cleanup_resource_monitor() {
  kill "${RESOURCE_MONITOR_PID}" 2>/dev/null || true
  wait "${RESOURCE_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup_resource_monitor EXIT

"${PYTHON_ENV}/bin/swift" sft "${COMMON_ARGS[@]}" "$@"
