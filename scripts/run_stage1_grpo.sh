#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${JANUS_VENV:-/root/.venvs/janus-repro-py312}"
BASE_MODEL_DIR="${JANUS_MODEL_DIR:-${ROOT_DIR}/models/Janus-Pro-7B}"
GRPO_DATASET="${JANUS_STAGE1_GRPO_DATA:-${ROOT_DIR}/data/processed/tqa/train_prompt_model_difficulty.jsonl}"
SMOKE=0
MEMORY_SMOKE=0

if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
elif [[ "${1:-}" == "--memory-smoke" ]]; then
  MEMORY_SMOKE=1
  shift
fi

if (( SMOKE )); then
  OUTPUT_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/smoke/stage1_tqa_grpo}"
elif (( MEMORY_SMOKE )); then
  OUTPUT_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/smoke/stage1_tqa_grpo_full_shape}"
else
  OUTPUT_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/stage1/tqa_grpo}"
fi

if [[ ! -s "${GRPO_DATASET}" ]]; then
  echo "Missing stage-1 GRPO difficulty annotations: ${GRPO_DATASET}" >&2
  echo "Run scripts/run_annotate_tqa_difficulty.sh first." >&2
  exit 2
fi

latest_sft_checkpoint() {
  find "${ROOT_DIR}/outputs/stage1/scienceqa_sft" -maxdepth 1 -type d \
    -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1
}

MODEL_SOURCE_DIR="${JANUS_STAGE1_SFT_MODEL:-$(latest_sft_checkpoint)}"
if [[ -z "${MODEL_SOURCE_DIR}" ]]; then
  if (( SMOKE )); then
    echo "No stage-1 SFT checkpoint yet; smoke-testing GRPO plumbing from the base model."
    MODEL_SOURCE_DIR="${BASE_MODEL_DIR}"
  else
    echo "No stage-1 SFT checkpoint found. Run scripts/run_stage1_sft.sh first," >&2
    echo "or set JANUS_STAGE1_SFT_MODEL to its NFS checkpoint directory." >&2
    exit 2
  fi
fi

if (( SMOKE || MEMORY_SMOKE )); then
  export JANUS_REASONING_JUDGE_SMOKE_STUB=1
else
  unset JANUS_REASONING_JUDGE_SMOKE_STUB
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required for the configured reasoning reward judge." >&2
    echo "Export it in this shell (do not write it into a project file), then rerun." >&2
    exit 2
  fi
fi

source "${ROOT_DIR}/scripts/lib/runtime.sh"
export JANUS_STAGE_MODEL_TO_RAM="${JANUS_STAGE_MODEL_TO_RAM:-0}"
if [[ "${MODEL_SOURCE_DIR}" == "${BASE_MODEL_DIR}" ]]; then
  export JANUS_RAM_MODEL_NAME=Janus-Pro-7B
else
  export JANUS_RAM_MODEL_NAME=Janus-Pro-7B-stage1-sft
fi
janus_stage_model_to_ram "${MODEL_SOURCE_DIR}"

export CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,3,4}"
export NPROC_PER_NODE="${JANUS_NPROC_PER_NODE:-4}"
export HF_HOME="${HF_HOME:-/root/nfs/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"
export JANUS_MODEL_DIR="${JANUS_ACTIVE_MODEL_DIR}"
export JANUS_JUDGE_LOG_DIR="${OUTPUT_DIR}/judge_calls"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
export JANUS_REASONING_JUDGE_MODEL="${JANUS_REASONING_JUDGE_MODEL:-deepseek-v4-flash-vision-exp}"
export JANUS_REWARD_PRIOR="${JANUS_REWARD_PRIOR:-table}"
export JANUS_REWARD_DECAY_LAMBDA="${JANUS_REWARD_DECAY_LAMBDA:-0.00006666666666666667}"
export JANUS_KL_DECAY_STEPS="${JANUS_KL_DECAY_STEPS:-500}"
export JANUS_ADVANTAGE_THRESHOLD="${JANUS_ADVANTAGE_THRESHOLD:-0.2}"
# ms-swift's outer RLHF arguments expose local_rollout_forward_batch_size,
# but its internal TRL GRPOConfig currently drops that field. Pass the
# Transformers-engine chunk limit directly through the process environment.
export SWIFT_TRANSFORMERS_ROLLOUT_BATCH_SIZE="${JANUS_LOCAL_ROLLOUT_FORWARD_BATCH_SIZE:-1}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

COMMON_ARGS=(
  --rlhf_type grpo
  --model "${JANUS_ACTIVE_MODEL_DIR}"
  --model_type deepseek_janus_pro
  --template deepseek_janus_pro
  --local_repo_path "${ROOT_DIR}/upstream/deepseek-janus"
  --check_model false
  --dataset "${GRPO_DATASET}"
  --split_dataset_ratio 0
  --dataset_num_proc 4
  --dataloader_num_workers 2
  --external_plugins \
    "${ROOT_DIR}/training/plugins/fsdp2_janus_compat.py" \
    "${ROOT_DIR}/training/plugins/scienceqa_grpo.py"
  --reward_funcs janus_accuracy janus_length janus_format janus_reasoning
  --reward_weights 1 1 1 1
  --tuner_type full
  --freeze_llm false
  --freeze_vit true
  --freeze_aligner true
  --torch_dtype bfloat16
  --attn_impl sdpa
  --max_length 2048
  --max_completion_length "${JANUS_GRPO_MAX_COMPLETION_LENGTH:-384}"
  --response_prefix '<think>'
  --truncation_strategy delete
  --use_vllm false
  --temperature 1.0
  --top_p 1.0
  --num_generations "${JANUS_GRPO_NUM_GENERATIONS:-16}"
  # Four 46 GiB L40S cards need a one-sample multimodal micro-batch.  Doubling
  # accumulation and steps_per_generation preserves the paper's 128-completion
  # effective batch and one rollout batch per optimizer step.
  --per_device_train_batch_size "${JANUS_GRPO_PER_DEVICE_BATCH:-1}"
  --gradient_accumulation_steps "${JANUS_GRPO_GRAD_ACCUM:-32}"
  --generation_batch_size "${JANUS_GRPO_GENERATION_BATCH:-128}"
  --steps_per_generation "${JANUS_GRPO_STEPS_PER_GENERATION:-32}"
  # The local rollout contains 32 image/completion records. Generate them in
  # bounded chunks so 384-token KV caches do not contend with the FSDP shards.
  --gradient_checkpointing true
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --vit_gradient_checkpointing false
  --learning_rate 1e-6
  --lr_scheduler_type constant
  --warmup_ratio 0
  --weight_decay 0
  --max_grad_norm 1
  --adam_beta1 0.9
  --adam_beta2 0.95
  --use_galore true
  --galore_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
  --galore_rank 512
  --galore_update_proj_gap 128
  --galore_optim_per_parameter false
  --ddp_find_unused_parameters false
  --loss_type dapo
  --epsilon 0.2
  --epsilon_high 0.28
  --beta 0.04
  --kl_in_reward false
  --importance_sampling_level token
  --scale_rewards none
  --dynamic_sample true
  --max_resample_times "${JANUS_GRPO_MAX_RESAMPLE_TIMES:-10}"
  --num_train_epochs "${JANUS_GRPO_NUM_TRAIN_EPOCHS:-4}"
  --max_steps "${JANUS_GRPO_MAX_STEPS:-3000}"
  --save_strategy steps
  --save_steps "${JANUS_GRPO_SAVE_STEPS:-500}"
  --save_total_limit 2
  --logging_steps 1
  --log_completions true
  --log_entropy true
  --logging_first_step true
  --logging_dir "${OUTPUT_DIR}/runs/trainer"
  --report_to tensorboard
  # GRPO's generation collator returns a list of prompt records rather than a
  # standard {input_ids: tensor} batch. Transformers' generic token counters
  # assume the latter and fail before the first rollout. The GRPO plugin logs
  # completion length and rollout/training throughput directly instead.
  --include_num_input_tokens_seen false
  --include_tokens_per_second false
  --seed 42
  --data_seed 42
  --add_version false
  --output_dir "${OUTPUT_DIR}"
)

GRPO_BACKEND="${JANUS_GRPO_BACKEND:-fsdp2}"
case "${GRPO_BACKEND}" in
  fsdp2)
    COMMON_ARGS+=(--fsdp "${ROOT_DIR}/configs/fsdp2_galore.json")
    ;;
  ddp)
    ;;
  deepspeed)
    COMMON_ARGS+=(--deepspeed "${ROOT_DIR}/configs/deepspeed_zero3_galore.json")
    ;;
  *)
    echo "JANUS_GRPO_BACKEND must be fsdp2, ddp, or deepspeed; got ${GRPO_BACKEND}" >&2
    exit 2
    ;;
esac

if (( SMOKE )); then
  COMMON_ARGS+=(
    --max_completion_length 64
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 1
    --generation_batch_size 16
    --steps_per_generation 4
    --max_resample_times 1
    --num_train_epochs 1
    --max_steps 1
    --save_strategy no
  )
elif (( MEMORY_SMOKE )); then
  # Exercise the full completion length, generation batch, rollout chunks and
  # backward/optimizer path once, but do not spend external-judge calls or save
  # a checkpoint. This is the preflight used before the supervised GRPO job.
  COMMON_ARGS+=(
    --num_train_epochs 1
    --max_steps 1
    --save_strategy no
  )
fi

echo "Launching stage-1 TQA GRPO on physical GPUs ${CUDA_VISIBLE_DEVICES}."
echo "Distributed backend: ${GRPO_BACKEND}"
echo "Canonical input model: ${MODEL_SOURCE_DIR}"
echo "Active model path: ${JANUS_ACTIVE_MODEL_DIR}"
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

"${PYTHON_ENV}/bin/swift" rlhf "${COMMON_ARGS[@]}" "$@"
