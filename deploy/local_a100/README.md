# 8×A100 40GB LoRA GRPO deployment

This profile targets one host with eight NVIDIA A100-SXM4-40GB GPUs. The
validated formal configuration is LoRA plus DDP in BF16; GaLore is disabled.
See `docs/a100_lora_ddp_grpo_run.md` for the measured startup result and memory
and throughput snapshot.

## 1. Bootstrap

The setup script creates `.venv`, checks out pinned upstream repositories,
applies the versioned Janus/ms-swift patches, installs the CUDA 12.4 stack, and
runs the tests:

```bash
bash scripts/setup_local_a100.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/torchrun --standalone --nproc-per-node=8 \
  scripts/check_distributed.py
```

Download the SFT checkpoint (weights are intentionally excluded from Git):

```bash
HF_HOME="$PWD/.hf_home" .venv/bin/python scripts/download_sft_checkpoint.py \
  --local-dir "$PWD/models/Janus-Pro-7B-stage1-sft"
```

Prepare TQA and ScienceQA as described in the top-level README. The formal
stage-1 input is expected at:

```text
data/processed/tqa/train_prompt_model_difficulty.jsonl
```

Difficulty annotation remains a separate auditable step:

```bash
bash deploy/local_a100/run_annotate_tqa_difficulty.sh
```

## 2. Validate before paying for a full run

The memory smoke test keeps the production rollout dimensions, uses the offline
judge stub, runs one optimizer update, and saves no checkpoint:

```bash
bash deploy/local_a100/run_stage1_grpo.sh --memory-smoke
```

The expected PEFT summary is approximately 7.495B total parameters and 74.96M
trainable parameters. A successful test must pass generation, rewards,
reference/policy scoring, backward, and AdamW—not merely model loading.

## 3. Formal run in tmux

Use the helper so the API key never appears in the tmux command, process list,
repository, or log:

```bash
bash deploy/local_a100/start_grpo_tmux.sh
```

It prompts for `OPENAI_API_KEY`, starts `janus_grpo_lora_ddp`, injects the key
through pane stdin, and deletes the temporary tmux paste buffer. Attach and
detach with:

```bash
tmux attach -t janus_grpo_lora_ddp
# Detach without stopping: Ctrl-b d
```

The direct launcher is also available when the key already exists only in the
calling process environment:

```bash
bash deploy/local_a100/run_stage1_grpo.sh
```

The formal defaults are:

- 8 DDP ranks on GPUs `0,1,2,3,4,5,6,7`;
- LoRA r=32, alpha=64, dropout=0.05 on all Llama attention/MLP projections;
- AdamW, LR 1e-5, no GaLore;
- one prompt per device, 16 gradient-accumulation steps;
- 128 completions per update, 16 generations per prompt;
- Transformers rollout chunk 4 and maximum completion length 384;
- checkpoint every 500 optimizer steps, retaining the latest two.

Override any exposed value through its `JANUS_*` environment variable. Keep
`generation_batch_size = per_device_batch × world_size × steps_per_generation`.

## 4. Offline run and TensorBoard

The offline mode fixes reasoning reward to zero and is not numerically
comparable with the formal four-reward experiment:

```bash
JANUS_GRPO_MAX_STEPS=3 JANUS_GRPO_SAVE_STEPS=3 \
  JANUS_GRPO_TMUX_SESSION=janus_grpo_offline \
  bash deploy/local_a100/start_grpo_tmux.sh --offline
```

Start TensorBoard in its own tmux session:

```bash
bash deploy/local_a100/start_tensorboard_tmux.sh
```

It watches the complete `outputs/` tree on port 6006. Use an authenticated
platform port proxy or SSH tunnel where available. A public quick tunnel has no
authentication and can expose prompts, completions, and metrics.

## 5. Monitoring and resume

```bash
tmux list-sessions
tail -f outputs/stage1/a100x8_lora_ddp.log
nvidia-smi
```

To resume, point `JANUS_STAGE1_SFT_MODEL` at a saved adapter checkpoint and set
a new output directory. Do not place credentials in `.env`, Supervisor files,
shell scripts, or Git; runtime artifacts and secret-file patterns are ignored.

Optional FSDP2 compatibility remains available for research profiles via
`JANUS_GRPO_BACKEND=fsdp2` and `configs/fsdp2_cpu_efficient.json`, but it is not
the validated default for this LoRA A100 run.
