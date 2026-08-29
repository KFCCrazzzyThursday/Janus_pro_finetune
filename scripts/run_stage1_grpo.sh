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
  OUTPUT_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/smoke/stage1_tqa_grpo_lora}"
elif (( MEMORY_SMOKE )); then
  OUTPUT_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/smoke/stage1_tqa_grpo_lora_full_shape}"
else
  OUTPUT_DIR="${JANUS_STAGE1_GRPO_OUTPUT:-${ROOT_DIR}/outputs/stage1/tqa_grpo_lora}"
fi

if [[ ! -s "${GRPO_DATASET}" ]]; then
  echo "Missing stage-1 GRPO difficulty annotations: ${GRPO_DATASET}" >&2
  echo "Run scripts/run_annotate_tqa_difficulty.sh first." >&2
  exit 2
fi

latest_sft_checkpoint() {
  local checkpoint_root="${ROOT_DIR}/outputs/stage1/scienceqa_sft"
  local checkpoint

  # The validated BF16 export is sharded and half the size of the legacy FP32
  # single-file export. Prefer it to keep NFS traffic and host RAM bounded.
  checkpoint="$(find "${checkpoint_root}" -maxdepth 1 -type d \
    -name 'checkpoint-*-bf16' -print 2>/dev/null | sort -V | tail -n 1)"
  if [[ -n "${checkpoint}" ]]; then
    echo "${checkpoint}"
    return
  fi

  find "${checkpoint_root}" -maxdepth 1 -type d \
    -name 'checkpoint-*' ! -name 'checkpoint-*-hf' -print 2>/dev/null | sort -V | tail -n 1
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

export CUDA_VISIBLE_DEVICES="${JANUS_CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export NPROC_PER_NODE="${JANUS_NPROC_PER_NODE:-5}"
export HF_HOME="${HF_HOME:-/root/nfs/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/upstream/deepseek-janus:${PYTHONPATH:-}"
export JANUS_MODEL_DIR="${JANUS_ACTIVE_MODEL_DIR}"
export JANUS_JUDGE_LOG_DIR="${OUTPUT_DIR}/judge_calls"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
export JANUS_REASONING_JUDGE_MODEL="${JANUS_REASONING_JUDGE_MODEL:-deepseek-v4-flash-vision-exp}"
# Judge optimizations ported from the H200 experiments.  Exact caching,
# pre-filtering groups that DAPO will discard, and higher request concurrency
# cut latency without approximating unscored candidates.  On this host,
# individual requests were faster and matched the appendix prompt exactly.
# Judge half of each G=16 group; the plugin tracks that observation mask so
# mean-imputed candidates no longer suppress the measured reasoning variance.
export JANUS_JUDGE_CACHE="${JANUS_JUDGE_CACHE:-1}"
export JANUS_JUDGE_CACHE_PATH="${JANUS_JUDGE_CACHE_PATH:-${ROOT_DIR}/outputs/judge_cache.sqlite3}"
export JANUS_JUDGE_BATCH_SIZE="${JANUS_JUDGE_BATCH_SIZE:-1}"
# DeepSeek applies one account-wide concurrency limit, while this value is
# per DDP rank. One request per rank keeps five-rank training below the current
# account limit of eight without requiring a cross-process semaphore.
export JANUS_JUDGE_CONCURRENCY="${JANUS_JUDGE_CONCURRENCY:-1}"
export JANUS_JUDGE_MAX_ATTEMPTS="${JANUS_JUDGE_MAX_ATTEMPTS:-5}"
# Keep retrying HTTP 429 with capped exponential backoff. Zero means unlimited;
# preserving the in-memory optimizer state is safer than aborting between saves.
export JANUS_JUDGE_RATE_LIMIT_MAX_ATTEMPTS="${JANUS_JUDGE_RATE_LIMIT_MAX_ATTEMPTS:-0}"
export JANUS_JUDGE_RATE_LIMIT_BASE_DELAY="${JANUS_JUDGE_RATE_LIMIT_BASE_DELAY:-5}"
export JANUS_JUDGE_RATE_LIMIT_MAX_DELAY="${JANUS_JUDGE_RATE_LIMIT_MAX_DELAY:-60}"
export JANUS_JUDGE_COMPACT_PROMPT="${JANUS_JUDGE_COMPACT_PROMPT:-0}"
export JANUS_JUDGE_MAX_REASONING_CHARS="${JANUS_JUDGE_MAX_REASONING_CHARS:-0}"
export JANUS_JUDGE_SAMPLE_FRACTION="${JANUS_JUDGE_SAMPLE_FRACTION:-0.5}"
export JANUS_JUDGE_ACTIVATION_MODE="${JANUS_JUDGE_ACTIVATION_MODE:-all_non_mastered}"
export JANUS_JUDGE_ACTIVATION_THRESHOLD="${JANUS_JUDGE_ACTIVATION_THRESHOLD:-0.60}"
export JANUS_JUDGE_SKIP_HOMOGENEOUS="${JANUS_JUDGE_SKIP_HOMOGENEOUS:-1}"
export JANUS_JUDGE_PRESAMPLE_FILTER="${JANUS_JUDGE_PRESAMPLE_FILTER:-1}"
export JANUS_JUDGE_PROMPT_VERSION="${JANUS_JUDGE_PROMPT_VERSION:-paper-batch-v1}"
export JANUS_REWARD_PRIOR="${JANUS_REWARD_PRIOR:-table}"
# Preserve the paper formula as an explicit ablation, but make the corrected
# range-normalized standard-deviation weighting unambiguous for this L40S run.
export JANUS_REWARD_WEIGHTING="${JANUS_REWARD_WEIGHTING:-stabilized}"
export JANUS_REWARD_VARIANCE_MIX="${JANUS_REWARD_VARIANCE_MIX:-${JANUS_REWARD_DYNAMIC_MIX:-0.5}}"
# Backward-compatible metric/config alias used by earlier local runs.
export JANUS_REWARD_DYNAMIC_MIX="${JANUS_REWARD_VARIANCE_MIX}"
export JANUS_REWARD_DECAY_LAMBDA="${JANUS_REWARD_DECAY_LAMBDA:-0.00006666666666666667}"
export JANUS_KL_DECAY_STEPS="${JANUS_KL_DECAY_STEPS:-500}"
export JANUS_ADVANTAGE_THRESHOLD="${JANUS_ADVANTAGE_THRESHOLD:-0.2}"
# ms-swift's outer RLHF arguments expose local_rollout_forward_batch_size,
# but its internal TRL GRPOConfig currently drops that field. Pass the
# Transformers-engine chunk limit directly through the process environment.
export SWIFT_TRANSFORMERS_ROLLOUT_BATCH_SIZE="${JANUS_LOCAL_ROLLOUT_FORWARD_BATCH_SIZE:-4}"

GRPO_PER_DEVICE_BATCH="${JANUS_GRPO_PER_DEVICE_BATCH:-1}"
GRPO_STEPS_PER_GENERATION="${JANUS_GRPO_STEPS_PER_GENERATION:-32}"
GRPO_GENERATION_BATCH_DEFAULT="$((NPROC_PER_NODE * GRPO_PER_DEVICE_BATCH * GRPO_STEPS_PER_GENERATION))"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# This host has GPUs 0/1/2 attached to NUMA node 0 and GPUs 3/4 attached to
# node 1. Give every rank a non-overlapping set of physical cores plus their
# SMT siblings. The external plugin applies these settings before model load;
# memory placement is preferred (fallback allowed), never a hard membind.
if [[ "${JANUS_NUMA_AFFINITY:-1}" == "1" ]]; then
  IFS=',' read -r -a JANUS_PHYSICAL_GPU_ARRAY <<<"${CUDA_VISIBLE_DEVICES}"
  JANUS_NUMA_CPUSET_LIST=()
  JANUS_NUMA_NODE_LIST=()
  for physical_gpu in "${JANUS_PHYSICAL_GPU_ARRAY[@]}"; do
    case "${physical_gpu}" in
      0) JANUS_NUMA_CPUSET_LIST+=("0-15,96-111"); JANUS_NUMA_NODE_LIST+=("0") ;;
      1) JANUS_NUMA_CPUSET_LIST+=("16-31,112-127"); JANUS_NUMA_NODE_LIST+=("0") ;;
      2) JANUS_NUMA_CPUSET_LIST+=("32-47,128-143"); JANUS_NUMA_NODE_LIST+=("0") ;;
      3) JANUS_NUMA_CPUSET_LIST+=("48-71,144-167"); JANUS_NUMA_NODE_LIST+=("1") ;;
      4) JANUS_NUMA_CPUSET_LIST+=("72-95,168-191"); JANUS_NUMA_NODE_LIST+=("1") ;;
      *)
        echo "No verified NUMA mapping for physical GPU ${physical_gpu}; set JANUS_NUMA_AFFINITY=0 or provide explicit mappings." >&2
        exit 2
        ;;
    esac
  done
  if (( ${#JANUS_PHYSICAL_GPU_ARRAY[@]} != NPROC_PER_NODE )); then
    echo "NPROC_PER_NODE=${NPROC_PER_NODE} does not match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
    exit 2
  fi
  export JANUS_NUMA_CPUSETS="$(IFS=';'; echo "${JANUS_NUMA_CPUSET_LIST[*]}")"
  export JANUS_NUMA_NODES="$(IFS=','; echo "${JANUS_NUMA_NODE_LIST[*]}")"
else
  unset JANUS_NUMA_CPUSETS JANUS_NUMA_NODES
fi

GRPO_BACKEND="${JANUS_GRPO_BACKEND:-ddp}"
EXTERNAL_PLUGINS=(
  "${ROOT_DIR}/training/plugins/numa_affinity.py"
  "${ROOT_DIR}/training/plugins/janus_lora_compat.py"
  "${ROOT_DIR}/training/plugins/scienceqa_grpo.py"
)
if [[ "${GRPO_BACKEND}" == "fsdp2" ]]; then
  EXTERNAL_PLUGINS+=("${ROOT_DIR}/training/plugins/fsdp2_janus_compat.py")
fi

COMMON_ARGS=(
  --rlhf_type grpo
  --model "${JANUS_ACTIVE_MODEL_DIR}"
  --model_type deepseek_janus_pro
  --template deepseek_janus_pro
  --local_repo_path "${ROOT_DIR}/upstream/deepseek-janus"
  --check_model false
  --dataset "${GRPO_DATASET}"
  --split_dataset_ratio 0
  --dataset_num_proc "${JANUS_GRPO_DATASET_NUM_PROC:-4}"
  --dataloader_num_workers "${JANUS_GRPO_DATALOADER_WORKERS:-2}"
  --external_plugins "${EXTERNAL_PLUGINS[@]}"
  --reward_funcs janus_accuracy janus_length janus_format janus_reasoning
  --reward_weights 1 1 1 1
  --tuner_type lora
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
  --lora_rank "${JANUS_LORA_RANK:-16}"
  --lora_alpha "${JANUS_LORA_ALPHA:-32}"
  --lora_dropout "${JANUS_LORA_DROPOUT:-0.05}"
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
  # L40S cards need a one-sample multimodal micro-batch. The default generation
  # batch is derived from world size so TRL's batch invariant remains exact.
  # Four ranks reproduce the paper's 128 completions; five ranks use 160 as an
  # explicit local-hardware adaptation while preserving 32 samples per rank.
  --per_device_train_batch_size "${GRPO_PER_DEVICE_BATCH}"
  --gradient_accumulation_steps "${JANUS_GRPO_GRAD_ACCUM:-32}"
  --generation_batch_size "${JANUS_GRPO_GENERATION_BATCH:-${GRPO_GENERATION_BATCH_DEFAULT}}"
  --steps_per_generation "${GRPO_STEPS_PER_GENERATION}"
  # Generate each rank's local rollout in bounded chunks so the multimodal
  # activations and 384-token KV caches remain below one L40S's 48 GiB limit.
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
  --optim adamw_torch
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
  --save_steps "${JANUS_GRPO_SAVE_STEPS:-30}"
  # Retain the latest two full-state checkpoints. The managed launcher also
  # keeps an independently copied best-validation checkpoint plus manifests.
  --save_total_limit "${JANUS_GRPO_SAVE_TOTAL_LIMIT:-2}"
  --save_only_model false
  # Validation is deterministic and external to Trainer so it can run after
  # the training process releases all five GPUs at each 30-step boundary.
  --eval_strategy no
  --logging_steps 1
  --log_completions true
  --log_entropy true
  --logging_first_step true
  --logging_dir "${JANUS_TRAIN_LOGGING_DIR:-${OUTPUT_DIR}/runs/trainer}"
  --report_to "${JANUS_REPORT_TO:-tensorboard}"
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

case "${GRPO_BACKEND}" in
  fsdp2)
    COMMON_ARGS+=(--fsdp "${ROOT_DIR}/configs/fsdp2_lora.json")
    ;;
  ddp)
    ;;
  deepspeed)
    COMMON_ARGS+=(--deepspeed "${ROOT_DIR}/configs/deepspeed_zero3_lora.json")
    ;;
  *)
    echo "JANUS_GRPO_BACKEND must be fsdp2, ddp, or deepspeed; got ${GRPO_BACKEND}" >&2
    exit 2
    ;;
esac

RESUME_CHECKPOINT="${JANUS_RESUME_FROM_CHECKPOINT:-}"
if [[ -z "${RESUME_CHECKPOINT}" ]] && [[ "${JANUS_GRPO_AUTO_RESUME:-1}" == "1" ]]; then
  if RESUME_CHECKPOINT="$("${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/grpo_run_state.py" \
      latest "${OUTPUT_DIR}" --world-size "${NPROC_PER_NODE}")"; then
    :
  else
    RESUME_CHECKPOINT=""
  fi
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
    echo "Resume checkpoint is not a directory: ${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
  "${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/grpo_run_state.py" \
    verify "${RESUME_CHECKPOINT}" --world-size "${NPROC_PER_NODE}" >/dev/null
  COMMON_ARGS+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
fi

if (( SMOKE )); then
  COMMON_ARGS+=(
    --max_completion_length 64
    --num_generations "${JANUS_GRPO_SMOKE_NUM_GENERATIONS:-4}"
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 1
    --generation_batch_size "$((NPROC_PER_NODE * 4))"
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
    --max_steps "${JANUS_GRPO_SMOKE_STEPS:-1}"
    --save_strategy no
  )
fi

echo "Launching stage-1 TQA GRPO on physical GPUs ${CUDA_VISIBLE_DEVICES}."
echo "Distributed backend: ${GRPO_BACKEND}"
echo "Tuner: LoRA (rank=${JANUS_LORA_RANK:-16}, alpha=${JANUS_LORA_ALPHA:-32}, dropout=${JANUS_LORA_DROPOUT:-0.05})"
echo "Reward weighting: ${JANUS_REWARD_WEIGHTING} (variance mix ${JANUS_REWARD_VARIANCE_MIX})"
echo "Local rollout forward batch: ${SWIFT_TRANSFORMERS_ROLLOUT_BATCH_SIZE}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "Resuming from: ${RESUME_CHECKPOINT}"
fi
echo "Canonical input model: ${MODEL_SOURCE_DIR}"
echo "Active model path: ${JANUS_ACTIVE_MODEL_DIR}"
if [[ "${JANUS_NUMA_AFFINITY:-1}" == "1" ]]; then
  echo "NUMA CPU sets by local rank: ${JANUS_NUMA_CPUSETS}"
  echo "NUMA preferred nodes by local rank: ${JANUS_NUMA_NODES}"
fi
free -h
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

mkdir -p "${OUTPUT_DIR}"
MAX_RAM_USED_GIB="${JANUS_MAX_RAM_USED_GIB:-115}"
MAX_SWAP_USED_GIB="${JANUS_MAX_SWAP_USED_GIB:-0.25}"
CURRENT_RAM_USED_GIB="$(awk '/^MemTotal:/ {total=$2} /^MemAvailable:/ {available=$2} END {printf "%.3f", (total-available)/1048576}' /proc/meminfo)"
if ! awk -v used="${CURRENT_RAM_USED_GIB}" -v limit="${MAX_RAM_USED_GIB}" 'BEGIN {exit !(used < limit)}'; then
  echo "Refusing launch: host RAM already uses ${CURRENT_RAM_USED_GIB} GiB; safety limit is ${MAX_RAM_USED_GIB} GiB." >&2
  exit 3
fi

"${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/memory_guard.py" \
  --output-dir "${OUTPUT_DIR}" \
  --parent-pid "$$" \
  --max-ram-used-gib "${MAX_RAM_USED_GIB}" \
  --max-swap-used-gib "${MAX_SWAP_USED_GIB}" &
MEMORY_GUARD_PID=$!

"${PYTHON_ENV}/bin/python" "${ROOT_DIR}/scripts/monitor_resources.py" \
  --output-dir "${OUTPUT_DIR}" \
  --physical-gpus "${CUDA_VISIBLE_DEVICES}" \
  --parent-pid "$$" &
RESOURCE_MONITOR_PID=$!
TRAINING_PID=""
cleanup_resource_monitor() {
  if [[ -n "${TRAINING_PID}" ]]; then
    kill -TERM -- "-${TRAINING_PID}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "${TRAINING_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${TRAINING_PID}" 2>/dev/null || true
    wait "${TRAINING_PID}" 2>/dev/null || true
  fi
  kill "${RESOURCE_MONITOR_PID}" 2>/dev/null || true
  wait "${RESOURCE_MONITOR_PID}" 2>/dev/null || true
  kill "${MEMORY_GUARD_PID}" 2>/dev/null || true
  wait "${MEMORY_GUARD_PID}" 2>/dev/null || true
}
trap cleanup_resource_monitor EXIT

setsid "${PYTHON_ENV}/bin/swift" rlhf "${COMMON_ARGS[@]}" "$@" &
TRAINING_PID=$!
wait "${TRAINING_PID}"
